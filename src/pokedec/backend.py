import os
import json
from contextlib import asynccontextmanager

import anyio
import numpy as np
import onnxruntime
import wandb
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from PIL import Image
from google.cloud import storage
from image_analysis import calculate_image_characteristics
from datetime import datetime, timezone


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application.

    Args:
    - app (FastAPI): The FastAPI application instance.

    Yields:
    - None
    """
    global model, transform, imagenet_classes
    
    # Load the ONNX model
    run = wandb.init()
    artifact = run.use_artifact("pokedec_mlops/pokedec_train/pokedec_models_onnx:best", type="model")
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, "pokedec_model.onnx")
    model = onnxruntime.InferenceSession(model_path)

    yield


app = FastAPI(lifespan=lifespan)


def save_prediction_to_gcp(filepath: str, filename: str, prediction: int, probabilities: list[float]):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket("poke_store")
    time = datetime.now(tz=timezone.utc)

    image_characteristics = calculate_image_characteristics(filepath, rgb=True)

    # Prepare prediction data
    data = {
        "image_characteristics": image_characteristics,
        "filename": filename,
        "prediction": prediction,
        "probabilities": probabilities,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    blob = bucket.blob(f"prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")


def predict_image(image_path: str) -> str:
    """Predict image class (or classes) given image path and return the result.

    Args:
    - image_path (str): Path to the image file.

    Returns:
    - str: The prediction result.
    """

    img = Image.open(image_path).resize((128, 128)).convert("RGB")
    np_img = np.array(img, dtype=np.float32)
    np_img = np_img.transpose((2, 0, 1))  # Shape (128, 128, 3) -> (3, 128, 128)
    np_img = np.expand_dims(np_img, axis=0)  # Shape (1, 3, 128, 128)
    np_img = np_img / 255.0
    output = model.run(None, {"input": np_img})[0]
    output = output[0]  # Shape (1, 1000) -> (1000,)
    exp_out = np.exp(output - np.max(output))
    probabilities = exp_out / np.sum(exp_out)
    prediction = int(np.argmax(probabilities))
    return probabilities, prediction


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


# FastAPI endpoint to classify an image
@app.post("/classify/")
async def classify_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await file.read()
        async with await anyio.open_file(file.filename, "wb") as f:
            await f.write(contents)
        probabilities, prediction = predict_image(file.filename)

        background_tasks.add_task(
            save_prediction_to_gcp,
            filepath=file.filename,
            filename=file.filename,
            prediction=prediction,
            probabilities=probabilities.tolist(),
        )

        return {
            "filename": file.filename,
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

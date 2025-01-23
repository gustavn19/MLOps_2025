import json
import os
from pathlib import Path

import anyio
import pandas as pd
import numpy as np
from PIL import Image
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
from image_analysis import calculate_image_characteristics

# MNIST data as tensors
training_data_path = os.path.join(os.getcwd(), "data", "raw", "dataset")

# Load the training data
training_data_images = {}
for i, dir in enumerate(os.listdir(training_data_path)):
    # Open each file in the directory and save the data to a dictionary
    for image in (os.listdir(os.path.join(training_data_path, dir))):
        # The images are png, so we need to convert them to numpy arrays, (3, 128, 128) and reshape them to (128, 128, 3)
        training_data_images[i] =  np.array(Image.open(os.path.join(training_data_path, dir, image)).convert("RGB").resize((128, 128))).reshape(3,128,128)

# Get the image characteristics for the training data, take each image as a separate file and calculate the characteristics and save them in a dictionary
image_characteristics = {}
for i, image in training_data_images.items():
    image_characteristics[i] = {
        "image_characteristics": calculate_image_characteristics(image, rgb=True),
        "prediction": i,
    }


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    text_overview_report = Report(
        metrics=[
            DataDriftPreset(columns=["avg_brightness", "contrast", "avg_brightness_green"]),
            DataQualityPreset(),
            TargetDriftPreset(columns=["prediction"]),
        ]
    )
    text_overview_report.run(reference_data=reference_data, current_data=current_data)
    text_overview_report.save_html("monitoring.html")


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global training_data, class_names
    training_data = image_characteristics
    class_names = [str(i) for i in range(10)]

    yield

    del training_data, class_names


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    file_names = download_files(n=n)

    files = sorted(file_names, key=os.path.getctime)

    latest_files = files[-n:]

    # Load or process the files as needed
    charateristics, predictions = [], []
    for file in latest_files:
        with open(file, "r") as f:
            data = json.load(f)
            charateristics.append(data["image_characteristics"])
            predictions.append(data["prediction"])

    dataframe = pd.DataFrame({"image_characteristics": charateristics, "prediction": predictions})

    return dataframe


def download_files(n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""
    bucket = storage.Client().bucket("dtu_mlops_osquera")
    blobs = bucket.list_blobs(prefix="prediction_")
    blobs = sorted(blobs, key=lambda x: x.time_created, reverse=True)
    latest_blobs = blobs[:n]

    file_names = []
    for blob in latest_blobs:
        name_file = blob.name.replace(" ", "_").replace("+", "_").replace(":", "_")
        with open(name_file, "wb") as f:
            blob.download_to_file(file_obj=f)

        file_names.append(name_file)

    return file_names


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5):
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("."), n=n)

    # Turn training data into a DataFrame
    df_train = pd.DataFrame(training_data).T

    df_train = pd.concat(
        [df_train, pd.DataFrame(df_train["image_characteristics"].tolist()).set_index(df_train.index)], axis=1
    )
    df_train.drop("image_characteristics", axis=1, inplace=True)

    prediction_data = pd.concat(
        [
            prediction_data,
            pd.DataFrame(prediction_data["image_characteristics"].tolist()).set_index(prediction_data.index),
        ],
        axis=1,
    )
    prediction_data.drop("image_characteristics", axis=1, inplace=True)

    run_analysis(df_train, prediction_data)

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

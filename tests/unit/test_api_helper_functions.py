import numpy as np
import onnxruntime
import os

# from pokedec.backend import save_prediction_to_gcp, predict_image, model
import pokedec.backend as backend


def test_predict_image():
    """Test the predict_image function."""
    # global model
    backend.model = onnxruntime.InferenceSession(os.path.join(os.getcwd(), "models", "onnx", "model_best.onnx"))
    image_path = "tests/tests_data/abra_api_testing_img.png"
    probabilities, prediction = backend.predict_image(image_path)
    assert isinstance(probabilities, np.ndarray)
    assert isinstance(prediction, int)

    # Testing that total probability is 1
    assert np.isclose(np.sum(probabilities), 1.0)
    # assert prediction == 63
    # assert probabilities[prediction] > 0.9
    print("We're in business")

# TODO maybe test edge case where dim of image is not 128x128?

if __name__ == "__main__":
    test_predict_image()
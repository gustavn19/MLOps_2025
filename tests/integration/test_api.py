from fastapi.testclient import TestClient
from backend import app


#### BASIC TESTS ####
# Test the root endpoint
def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello from the backend with wandb!"}


# Test the classify endpoint
def test_classify_image():
    with TestClient(app) as client:
        file_path = "tests/tests_data/abra_api_testing_img.png"
        with open(file_path, "rb") as file:
            response = client.post("/classify/", files={"file": file})

        # assert status code
        assert response.status_code == 200

        # assert response content
        data = response.json()
        assert "filename" in data
        assert "prediction" in data
        assert "probabilities" in data

        # TODO: Add more assertions, maybe test the type of e.g. probabilities?


#### ADVANCED TESTS (e.g. edge cases) ####
def test_classify_image_no_file():
    with TestClient(app) as client:
        response = client.post("/classify/")
        assert response.status_code == 422


def test_classify_image_wrong_file_type():
    with TestClient(app) as client:
        file_path = "tests/tests_data/dummy_test_file.txt"
        with open(file_path, "rb") as file:
            response = client.post("/classify/", files={"file": file})
        assert response.status_code == 500
        # TODO: implement custom error message in the backend for this case and subsequently test for it here??

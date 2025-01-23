import random

from locust import HttpUser, between, task


class BackendUser(HttpUser):
    """A simple Locust user class used to load test the backend api."""

    wait_time = between(1, 3)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def classify_image(self) -> None:
        """A task that simulates a user uploading an image to the FastAPI app."""

        file_path = "tests/tests_data/abra_api_testing_img.png"
        with open(file_path, "rb") as file:
            self.client.post("/classify/", files={"file": file})    

    # https://backend-pokedec-228711502156.europe-west3.run.app 

    # TODO: A good use case for load testing in our case is to test that our API can handle a load right after it has been deployed..... to do? 
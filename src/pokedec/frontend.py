import json
import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/exalted-strata-447112-s0/locations/europe-west3"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "backend-pokedec":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    response = requests.post(predict_url, files={"file": image})
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    labels = json.load(open("pokemon_labels.json"))
    # reverse the prediction to get the class name
    labels = {value: key for key, value in labels["pokemon_labels"].items()}

    st.title("PokeDec: Pokemon Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            class_name = labels[int(prediction)]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.markdown(f"### Prediction: {class_name}")

            # make a nice bar chart
            
            data = {
                "Class": [labels[int(i)] for i in range(len(probabilities))],
                "Probability": probabilities,
            }
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df.nlargest(10, "Probability"), y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()

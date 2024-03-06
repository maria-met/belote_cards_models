import streamlit as st
import requests
from PIL import Image
from typing import Tuple
from io import BytesIO
import cv2
import numpy as np
from typing import Tuple


# def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]: # A function to read the image file as a numpy array
#     img = Image.open(data) # Open the image and convert it to RGB color space
#     img_resized = img.resize((180, 180), resample=Image.BICUBIC) # Resize the image to 180 x 180
#     image = np.array(img_resized) # Convert the image to a numpy array
#     return image # Return the image and its size

def main():
    st.set_page_config(page_title="Camera Image Capture and Prediction", page_icon=None, layout="wide")
    # Capture image
    st.write("1. Click the button to capture an image from the camera.")
    if st.button("Capture Image"):
        image = st.camera_input("Take a picture")
        st.image(image, channels="RGB", caption="Captured Image", use_column_width=True)
        pil_image = Image.fromarray(image)

        # Convert PIL Image to bytes
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format="JPEG")

        # Prepare files parameter for POST request
        files = {"image": ("image.jpg", img_bytes.getvalue(), "image/jpeg")}
        # image_processed = read_file_as_image(image.read()) # Read the image file
        # img_batch = np.expand_dims(image_processed, 0) # Add an extra dimension to the image so that it matches the input shape of the model

        # # Pass the image to the prediction function
        if st.button("Predict"):
        #     # Convert image to bytes
        #     # _, img_encoded = cv2.imencode(".jpg", image)
        #     # img_bytes = BytesIO(img_encoded.tobytes())

        #     # Send image to the FastAPI endpoint for prediction
        #     files = {"file": ("image.jpg", img_batch, "image/jpg")}
            response = requests.post("http://localhost:8000/predict", files=files)

        #     # Display prediction result
        #     st.write("2. Prediction Result:")
        #     st.write(response.json()["result"])
    else:
        st.info("Please upload an image.")

if __name__ == "__main__":
    main()

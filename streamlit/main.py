import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np


def main():

    st.set_page_config(page_title="Camera Image Capture and Prediction", page_icon=None, layout="wide")

    # Capture image
    st.write("1. Click the button to capture an image from the camera.")

    if st.button("Capture Image"):

        image = st.camera_input("Take a picture").getvalue()

        if image is not None:
            print(image)
            file = np.frombuffer(image, np.uint8)
            response = requests.post("http://localhost:8000/predict", files=file)

            # Display prediction result
            st.write("2. Prediction Result:")
            st.write(response.json())
        else:
            print('no picture taken')

    else:
        st.info("Please upload an image.")

if __name__ == "__main__":
    main()

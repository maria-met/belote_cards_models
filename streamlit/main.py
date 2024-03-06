import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np


def main():

    st.set_page_config(page_title="Camera Image Capture and Prediction", page_icon=None)

    # Capture image
    st.write("1. Click the button to capture an image from the camera.")

    picture = st.camera_input("Take a picture")

    if picture:
        img = Image.open(picture)
        # img.save('./last_img.jpg')


        # Convert PIL Image to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")

        # Prepare files parameter for POST request
        files = {"img": ("image.jpg", img_bytes.getvalue())}

        response = requests.post("", files=files)
        # Display prediction result
        # st.image(picture, caption=response.text)
        st.write(response.text)

        # ret, thresh = cv2.threshold(np.array(img), 150, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # # draw contours on the original image
        # image_copy = img.copy()
        # cv2.drawContours(image=np.array(image_copy), contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # # _, img_encoded = cv2.imencode(image_copy, eye)
        # # img_bytes = img_encoded.tobytes()
        # st.image(image_copy)
        # print(response)

    else:
        st.info("Please upload an image.")

if __name__ == "__main__":
    main()

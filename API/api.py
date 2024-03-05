from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Tuple # A library for type hints.
from io import BytesIO # A class for working with binary data in memory.
from PIL import Image # A library for image processing.
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from starlette.responses import Response

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

images = [
       '/home/mariame/code/maria-met/belote_cards_models/image/WIN_20240304_11_19_18_Pro.jpg',
       '/home/mariame/code/maria-met/belote_cards_models/image/WIN_20240304_11_20_05_Pro.jpg',
       '/home/mariame/code/maria-met/belote_cards_models/image/WIN_20240304_11_20_31_Pro.jpg',
       '/home/mariame/code/maria-met/belote_cards_models/image/WIN_20240304_15_04_52_Pro.jpg',
       '/home/mariame/code/maria-met/belote_cards_models/image/WIN_20240304_15_05_51_Pro.jpg',
       '/home/mariame/code/maria-met/belote_cards_models/image/WIN_20240304_15_07_28_Pro.jpg',
       '/home/mariame/code/maria-met/belote_cards_models/image/WIN_20240304_15_07_56_Pro.jpg'
       ]
model = YOLO('/home/mariame/code/maria-met/belote_cards_models/notebooks/runs/classify/train2/weights/best.pt')  # load a custo

def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]: # A function to read the image file as a numpy array
    img = Image.open(BytesIO(data)).convert('RGB') # Open the image and convert it to RGB color space
    img_resized = img.resize((180, 180), resample=Image.BICUBIC) # Resize the image to 180 x 180
    image = np.array(img_resized) # Convert the image to a numpy array
    return image, img_resized.size # Return the image and its size

# http://127.0.0.1:8000/predict
@app.post("/predict")
async def predict(img: UploadFile=File(...)):      # 1
    """
    make prediction based on the image captured
    """
    # image = await img.read() # Read the image file
    # nparr = np.fromstring(image, np.uint8)

    image, img_size = read_file_as_image(await img.read()) # Read the image file
    img_batch = np.expand_dims(image, 0) # Add an extra dimension to the image so that it matches the input shape of the model
    results = model(img_batch)
    names_dict = results[0].names

    probs = results[0].probs.data.tolist()

    #print(names_dict)
    #print(probs)
    label = names_dict[np.argmax(probs)]
    confidence = np.argmax(probs)
    print(label)
    return { # Return the prediction
            'class': label,
            'confidence': float(confidence)
        }

    # labels = []
    # for img in images:
    #     results = model(img)
    #     names_dict = results[0].names

    #     probs = results[0].probs.data.tolist()

    #     #print(names_dict)
    #     #print(probs)
    #     label = names_dict[np.argmax(probs)]
    #     confidence = np.argmax(probs)
    #     print(label)
    #     labels.append({ # Return the prediction
    #             'class': label,
    #             'confidence': float(confidence)
    #         })
    # return labels


@app.get("/")
def root():
    return {'greeting': 'Hello'}

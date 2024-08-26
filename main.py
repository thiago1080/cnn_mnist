from typing import List
import os
from fastapi import FastAPI, File, UploadFile, Form
import tensorflow as tf
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI()

# Alter the below path to the path of the saved model
model = tf.saved_model.load('models/base_model')

def predict(model, image_array):
    logits = model(image_array)
    prediction = tf.argmax(logits, axis=1)
    return prediction.numpy()[0]

@app.get("/")
async def root():
    return {"message": "Welcome! This service will predict an image based on the MNIST dataset"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predicts the class of an MNIST image.
    """
    image = Image.open(file.file).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255
    prediction = predict(model, image_array)
    return {"predicted_class": str(prediction)}

@app.post("/batch_predict")
async def batch_predict(directory: str = Form(...)):
    """
    Predicts the class of all MNIST images in a directory.
    """
    predictions = {}
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255
            prediction = predict(model, image_array)
            predictions[filename] = str(prediction)
    return predictions


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
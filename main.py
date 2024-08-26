from typing import List
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI()

model = tf.saved_model.load('module_with_signature')

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
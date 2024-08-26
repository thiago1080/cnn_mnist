from typing import List
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI()

# Carregar o modelo
model = tf.saved_model.load('module_with_signature')


def load_and_process_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255
    return image_array

def predict(model, image_array):
    logits = model(image_array)
    prediction = tf.argmax(logits, axis=1)
    return prediction.numpy()[0]

@app.get("/")
async def root():
    return {"message": "Welcome!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predicts the class of an MNIST image.
    """
    # Carregar a imagem enviada
    image = Image.open(file.file).convert('L')
    # Redimensionar a imagem para 28x28 pixels
    image = image.resize((28, 28))
    # Converter a imagem para um array NumPy
    image_array = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255
    # Fazer a previsão
    prediction = predict(model, image_array)
    return {"predicted_class": str(prediction)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
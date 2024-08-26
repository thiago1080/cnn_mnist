# MNIST Image Prediction Service

This FastAPI application provides endpoints to predict the class of MNIST images using a pre-trained TensorFlow model.

## Requirements

- Python 3.7+
- FastAPI
- TensorFlow
- NumPy
- Pillow
- Uvicorn

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```bash
    pip install fastapi tensorflow numpy pillow uvicorn
    ```

3. Ensure you have the pre-trained TensorFlow model saved in the `model` directory.

## Running the Application

Start the FastAPI application using Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### Root Endpoint

- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns a welcome message.

#### Example Request
```bash
curl -X GET "http://localhost:8000/"
```

#### Example Response
```json
{
  "message": "Welcome! This service will predict an image based on the MNIST dataset"
}
```

### Predict Image Endpoint

- **URL:** `/predict`
- **Method:** `POST`
- **Description:** Predicts the class of an MNIST image.

#### Example Request
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_your_image.png"
```

#### Example Response
```json
{
  "predicted_class": "7"
}
```

### Batch Predict Endpoint

- **URL:** `/batch_predict`
- **Method:** `POST`
- **Description:** Predicts the class of all MNIST images in a directory.

#### Example Request
```bash
curl -X POST "http://localhost:8000/batch_predict" -F "directory=/path_to_your_directory"
```

#### Example Response
```json
{
  "image1.png": "3",
  "image2.jpg": "5",
  "image3.jpeg": "1"
}
```

## Notes

- Ensure the images are in grayscale and resized to 28x28 pixels.
- The model should be compatible with the MNIST dataset.
- The model should be saved in the `model` directory.






# MNIST Convolutional Neural Network

This script trains a convolutional neural network (CNN) on the MNIST dataset using TensorFlow.

## Requirements

- Python 3.7+
- TensorFlow
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```bash
    pip install tensorflow numpy
    ```

## Running the Script

1. Ensure you have the MNIST dataset available. The script will automatically download it if not present.

2. Run the script:
    ```bash
    python main.py
    ```

## Script Overview

### Data Preparation

The script loads the MNIST dataset and preprocesses it by normalizing the pixel values and one-hot encoding the labels.

### Model Definition

A custom convolutional neural network (CNN) is defined using TensorFlow's low-level API. The model consists of multiple convolutional layers followed by fully connected layers.

### Training

The model is trained for a specified number of epochs using the Adam optimizer and categorical cross-entropy loss function.

### Evaluation

After training, the model's accuracy is evaluated on the test dataset.

### Saving the Model

The trained model is saved in TensorFlow's SavedModel format for later use.

## Example Output

During training, the script will print the loss for each epoch. After training, it will print the test accuracy.

```bash
Epoch 1, Loss: 0.1234
Epoch 2, Loss: 0.0987
...
Test accuracy: 0.9876
```


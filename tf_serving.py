import requests
import numpy as np
import json
from tensorflow.keras import layers, models, datasets

# Preparar os dados de teste
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Selecionar uma amostra para inferência
sample = x_test[0:1]

# Fazer a requisição para o servidor TensorFlow Serving
url = 'http://localhost:8501/v1/models/my_model:predict'
data = json.dumps({"instances": sample.tolist()})
headers = {"content-type": "application/json"}
response = requests.post(url, data=data, headers=headers)

# Processar a resposta
predictions = np.array(response.json()['predictions'])
print("Predicted class:", np.argmax(predictions))
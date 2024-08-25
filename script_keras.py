import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical

# Carregar e pré-processar o dataset MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Redimensionar as imagens para (28, 28, 1) e normalizar os valores dos pixels
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Converter os rótulos para one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definir a arquitetura da rede neural
input_layer = layers.Input(shape=(28, 28, 1))

# conv1: camada convolucional com 32 filtros
conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
conv1_pool = layers.MaxPooling2D((2, 2))(conv1)

# conv_2_1 e conv_2_2: camadas convolucionais com 32 filtros
conv_2_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_pool)
conv_2_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_pool)

conv_2_1_pool = layers.MaxPooling2D((2, 2))(conv_2_1)
conv_2_2_pool = layers.MaxPooling2D((2, 2))(conv_2_2)

# conv_3_1 e conv_3_2: camadas convolucionais com 256 filtros
conv_3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_2_1_pool)
conv_3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_2_2_pool)

# Concatenar as saídas das camadas conv_3_1 e conv_3_2
concat = layers.Concatenate()([conv_3_1, conv_3_2])

# Flatten a saída concatenada
flatten = layers.Flatten()(concat)

# Definir as camadas totalmente conectadas
fc_1 = layers.Dense(1000, activation='relu')(flatten)
fc_2 = layers.Dense(500, activation='relu')(fc_1)
output_layer = layers.Dense(10, activation='softmax')(fc_2)

# Definir o modelo
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
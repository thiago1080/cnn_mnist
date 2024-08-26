import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np


loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()
def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)
    return x_train, y_train, x_test, y_test

def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images)
        loss = loss_fn(labels, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# Definir o modelo com regularização L2
class ConvNetL2(tf.Module):
    def __init__(self):
        l2_regularizer = tf.keras.regularizers.l2(0.01)
        self.conv1 = tf.Variable(tf.random.normal([3, 3, 1, 32]), name='conv1', regularizer=l2_regularizer)
        self.conv2_1 = tf.Variable(tf.random.normal([3, 3, 32, 32]), name='conv2_1', regularizer=l2_regularizer)
        self.conv2_2 = tf.Variable(tf.random.normal([3, 3, 32, 32]), name='conv2_2', regularizer=l2_regularizer)
        self.conv3_1 = tf.Variable(tf.random.normal([3, 3, 32, 256]), name='conv3_1', regularizer=l2_regularizer)
        self.conv3_2 = tf.Variable(tf.random.normal([3, 3, 32, 256]), name='conv3_2', regularizer=l2_regularizer)
        self.fc1 = tf.Variable(tf.random.normal([7 * 7 * 512, 1000]), name='fc1', regularizer=l2_regularizer)
        self.fc2 = tf.Variable(tf.random.normal([1000, 500]), name='fc2', regularizer=l2_regularizer)
        self.fc3 = tf.Variable(tf.random.normal([500, 10]), name='fc3', regularizer=l2_regularizer)
        self.bias1 = tf.Variable(tf.zeros([1000]), name='bias1')
        self.bias2 = tf.Variable(tf.zeros([500]), name='bias2')
        self.bias3 = tf.Variable(tf.zeros([10]), name='bias3')
    @tf.function
    def __call__(self, x):
        x = tf.nn.conv2d(x, self.conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x1 = tf.nn.conv2d(x, self.conv2_1, strides=[1, 1, 1, 1], padding='SAME')
        x1 = tf.nn.relu(x1)
        x1 = tf.nn.max_pool2d(x1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x2 = tf.nn.conv2d(x, self.conv2_2, strides=[1, 1, 1, 1], padding='SAME')
        x2 = tf.nn.relu(x2)
        x2 = tf.nn.max_pool2d(x2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x1 = tf.nn.conv2d(x1, self.conv3_1, strides=[1, 1, 1, 1], padding='SAME')
        x1 = tf.nn.relu(x1)

        x2 = tf.nn.conv2d(x2, self.conv3_2, strides=[1, 1, 1, 1], padding='SAME')
        x2 = tf.nn.relu(x2)

        x = tf.concat([x1, x2], axis=3)
        x = tf.reshape(x, [-1, 7 * 7 * 512])

        x = tf.matmul(x, self.fc1) + self.bias1
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.fc2) + self.bias2
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.fc3) + self.bias3
        return x
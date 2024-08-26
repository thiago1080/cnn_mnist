import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns



# Definir o modelo com regularização L1
class ConvNetL1(tf.Module):
    def __init__(self):
        l1_regularizer = tf.keras.regularizers.l1(0.01)
        self.conv1 = tf.Variable(tf.random.normal([3, 3, 1, 32]), name='conv1', regularizer=l1_regularizer)
        self.conv2_1 = tf.Variable(tf.random.normal([3, 3, 32, 32]), name='conv2_1', regularizer=l1_regularizer)
        self.conv2_2 = tf.Variable(tf.random.normal([3, 3, 32, 32]), name='conv2_2', regularizer=l1_regularizer)
        self.conv3_1 = tf.Variable(tf.random.normal([3, 3, 32, 256]), name='conv3_1', regularizer=l1_regularizer)
        self.conv3_2 = tf.Variable(tf.random.normal([3, 3, 32, 256]), name='conv3_2', regularizer=l1_regularizer)
        self.fc1 = tf.Variable(tf.random.normal([7 * 7 * 512, 1000]), name='fc1', regularizer=l1_regularizer)
        self.fc2 = tf.Variable(tf.random.normal([1000, 500]), name='fc2', regularizer=l1_regularizer)
        self.fc3 = tf.Variable(tf.random.normal([500, 10]), name='fc3', regularizer=l1_regularizer)
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

model = ConvNet()

# Definir a função de perda e o otimizador
loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()

def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images)
        loss = loss_fn(labels, logits)
        # Adicionar a perda de regularização
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

epochs = 10
batch_size = 128
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        loss = train_step(model, x_batch, y_batch)
    print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')

def evaluate(model, images, labels):
    logits = model(images)
    prediction = tf.argmax(logits, axis=1)
    correct_prediction = tf.equal(prediction, tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

test_accuracy = evaluate(model, x_test, y_test)
print(f'Test accuracy: {test_accuracy.numpy()}')

module_with_signature_path ='model'
call = model.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
tf.saved_model.save(model, module_with_signature_path, signatures=call)

logits = model(x_test)
predictions = tf.argmax(logits, axis=1).numpy()
true_labels = tf.argmax(y_test, axis=1).numpy()
conf_matrix = confusion_matrix(true_labels, predictions)

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
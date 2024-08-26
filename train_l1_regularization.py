import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from improvements import ConvNetL1
from model_functions import get_mnist_data, train_step

MODEL_PATH = 'models/model_with_l1'
model = ConvNetL1()

x_train, y_train, x_test, y_test = get_mnist_data()


epochs = 10
batch_size = 128
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        loss = train_step(model, x_batch, y_batch)
    print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')
    
    



module_with_signature_path ='model'

call = model.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
tf.saved_model.save(model, MODULE_WITH_SIGNATURE_PATH, signatures=call)

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
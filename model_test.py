import os, json
import numpy as np
from nn_predict import nn_inference
from utils import mnist_reader

YOUR_MODEL_PATH = 'model/fashion_mnist' # Default format is h5
#TF_MODEL_PATH = f'{YOUR_MODEL_PATH}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_PATH}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_PATH}.json'

X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# === Load weights and architecture ===
weights = np.load(MODEL_WEIGHTS_PATH)
with open(MODEL_ARCH_PATH) as f:
    model_arch = json.load(f)
    
# Normalize input images
normalized_X = X_test / 255.0

# Perform inference for all test images
print('Classifying images...')
outputs = np.array([
    nn_inference(model_arch, weights, np.expand_dims(img, axis=0))
    for img in normalized_X
])
print('Done')

# Get predictions using argmax
predictions = np.argmax(outputs.squeeze(axis=1), axis=-1)

# Calculate number of correct predictions
correct = np.sum(predictions == y_test)

acc = correct / len(y_test)
print(f"Accuracy = {acc}")

def test_acc_50():
    assert acc > 0.5

def test_acc_80():
    assert acc > 0.8

def test_acc_90():
    assert acc > 0.9
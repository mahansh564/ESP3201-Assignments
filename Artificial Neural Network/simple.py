import numpy as np
import pandas as pd

# Define the activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Load the dataset
dataset = pd.read_csv('data/ANN_circle.csv')

# Define the split ratio
split = 0.75

# Split the dataset into training and testing sets
train = dataset.sample(frac=split, random_state=42)
test = dataset.drop(train.index)

# Separate features and labels for training and testing sets
X_train = train[['x1', 'x2']].values
Y_train = train[['label']].values
X_test = test[['x1', 'x2']].values
Y_test = test[['label']].values

# Initialize the neural network structure
input_layer_neurons = 2  # 2 input neurons
hidden_layer_neurons = 4  # 2-4 neurons
output_neuron = 1  # 1 output neuron

# Randomly initialize weights
np.random.seed(42)
W1 = np.random.randn(input_layer_neurons, hidden_layer_neurons) * 0.01
W2 = np.random.randn(hidden_layer_neurons, output_neuron) * 0.01

# Define the learning rate
learning_rate = 0.01

# Train the network
for epoch in range(10000):
    # Forward pass
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Calculate the error
    error = Y_train - A2

    # Backward pass
    dZ2 = error * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)

    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1)

    # Update weights
    W2 += learning_rate * dW2
    W1 += learning_rate * dW1

# Test the network
def predict(point):
    Z1 = np.dot(point, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    return 1 if A2 > 0.5 else 0

# Evaluate
correct_predictions = 0
for i in range(len(X_test)):
    prediction = predict(X_test[i])
    if prediction == Y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
print(f"Accuracy on the test set: {accuracy:.2f}")

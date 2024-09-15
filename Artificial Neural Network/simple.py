import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('data/ANN_circle.csv')

# Define the split ratio (75% train, 25% test)
split = 0.75
train = dataset.sample(frac=split, random_state=42)
test = dataset.drop(train.index)

# Separate features and labels for training and testing
X_train = train[['x1', 'x2']].values
Y_train = train[['label']].values
X_test = test[['x1', 'x2']].values
Y_test = test[['label']].values

# Initialize the neural network structure
input_neurons = 2  # x1 and x2
hidden_neurons = 4  # Start with 4 neurons
output_neurons = 1  # Binary output (0 or 1)

# Randomly initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_neurons, hidden_neurons) * 0.0001
W2 = np.random.randn(hidden_neurons, output_neurons) * 0.0001
b1 = np.zeros((1, hidden_neurons))
b2 = np.zeros((1, output_neurons))

# Set learning parameters
learning_rate = 0.6
epochs = 2000

# Training loop
losses = []
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_train, W1) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    # Compute loss (mean squared error)
    error = Y_train - A2
    loss = np.mean(np.square(error))
    losses.append(loss)

    # Backward pass
    dZ2 = error * A2 * (1 - A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W1 += learning_rate * dW1
    W2 += learning_rate * dW2
    b1 += learning_rate * db1
    b2 += learning_rate * db2


# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.show()

# Test the model and visualize the decision boundary
x_values = np.linspace(-2, 2, 100)
y_values = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_values, y_values)
X_flat = X.flatten()
Y_flat = Y.flatten()
test_points = np.column_stack((X_flat, Y_flat))

Z1 = np.dot(test_points, W1) + b1
A1 = 1 / (1 + np.exp(-Z1))
Z2 = np.dot(A1, W2) + b2
A2 = 1 / (1 + np.exp(-Z2))

decision_boundary = A2.reshape(X.shape)

# Plot the neural network's decision boundary
plt.contourf(X, Y, decision_boundary, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.6)
plt.colorbar()

# Plot the actual circle (radius=1, center=(0,0))
circle = plt.Circle((0, 0), 1, color='green', fill=False, linestyle='dashed', linewidth=2)
plt.gca().add_artist(circle)

# Scatter the test points
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test.flatten(), edgecolor='k', cmap='coolwarm', marker='o')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary vs Actual Circle')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Evaluate the accuracy on the test set
Z1_test = np.dot(X_test, W1) + b1
A1_test = 1 / (1 + np.exp(-Z1_test))
Z2_test = np.dot(A1_test, W2) + b2
A2_test = 1 / (1 + np.exp(-Z2_test))
predictions = (A2_test > 0.5).astype(int)

accuracy = np.mean(predictions == Y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # for reproducibility
n_hidden = 10
learning_rate = 0.1
epochs = 1000

# Randomly initialize weights and biases
W1 = np.random.randn(1, n_hidden)
b1 = np.random.randn(n_hidden)
W2 = np.random.randn(n_hidden, 1)
b2 = np.random.randn(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def forward_propagation(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    return a1, z2


def backward_propagation(X, Y, a1, z2):
    global W1, b1, W2, b2
    error = z2 - Y
    dW2 = np.dot(a1.T, error)
    db2 = np.sum(error)
    d_z1 = np.dot(error, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return np.mean(np.abs(error))

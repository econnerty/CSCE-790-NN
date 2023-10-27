import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)  # for reproducibility
n_hidden = 20
learning_rate = 0.0001
epochs = 1000

# Randomly initialize weights and biases
W1 = np.random.randn(1, n_hidden)
b1 = np.random.randn(n_hidden)
W2 = np.random.randn(n_hidden, 1)
b2 = np.random.randn(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

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
    d_z1 = np.dot(error, W2.T) * sigmoid_derivative(np.dot(X, W1) + b1)  # Change a1 to z1
    dW1 = np.dot(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return np.mean(np.abs(error))

X = np.array([[-1], [-0.9], [-0.8], [-0.7], [-0.6], [-0.5], [-0.4], [-0.3], [-0.2], [-0.1],
              [0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1]])
Y = np.array([[-0.96], [-0.577], [-0.073], [0.377], [0.641], [0.66], [0.461], [0.134],
              [-0.201], [-0.434], [-0.5], [-0.393], [-0.165], [0.099], [0.307], [0.396],
              [0.345], [0.182], [-0.031], [-0.219], [-0.321]])

errors = []

def plot_approximation(epoch):
    a1, z2 = forward_propagation(X)
    plt.figure()
    plt.plot(X, Y, label='Actual function')
    plt.plot(X, z2, label='NN approximation')
    plt.legend()
    plt.title(f'Function Approximation after {epoch} Epochs')
    plt.savefig(f'function_approximation_{epoch}_epochs.pdf')

for epoch in range(epochs):
    a1, z2 = forward_propagation(X)
    error = backward_propagation(X, Y, a1, z2)
    errors.append(error)
    if epoch+1 in [10,100,200,400,1000]:
        plot_approximation(epoch+1)

# (1) Training error vs epoch number
plt.figure()
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training Error vs Epoch Number')
plt.savefig('training_error_vs_epoch_number2.pdf')




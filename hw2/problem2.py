import numpy as np
import matplotlib.pyplot as plt

np.random.seed(15)  # for reproducibility
n_hidden = 100
learning_rate = 0.01
epochs = 10000

# Randomly initialize weights and biases
#he initialization
W1 = np.random.randn(1, n_hidden)
W2 = np.random.randn(n_hidden, 1)
b1 = np.random.randn(n_hidden)
b2 = np.random.randn(1)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def forward_propagation(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    return a1, z2,z1

def backward_propagation(X, Y, a1, z2,z1):
    global W1, b1, W2, b2
    n = len(Y)
    error = 2 * (z2 - Y) / n  # Derivative of MSE loss
    dW2 = np.dot(a1.T, error)
    db2 = np.sum(error)
    d_z1 = np.dot(error, W2.T) * relu_derivative(z1)
    dW1 = np.dot(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return mse_loss(Y, z2)  # Compute loss for monitoring

X = np.array([[-1], [-0.9], [-0.8], [-0.7], [-0.6], [-0.5], [-0.4], [-0.3], [-0.2], [-0.1],
              [0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1]])
Y = np.array([[-0.96], [-0.577], [-0.073], [0.377], [0.641], [0.66], [0.461], [0.134],
              [-0.201], [-0.434], [-0.5], [-0.393], [-0.165], [0.099], [0.307], [0.396],
              [0.345], [0.182], [-0.031], [-0.219], [-0.321]])

#sample points in the range [-1,1]
X2 = np.random.uniform(-1,1,(1000,1))

errors = []

def plot_approximation(epoch):
    # Forward propagate through the network
    a1, z2, z1 = forward_propagation(X2)
    
    # Sort the X2 and z2 arrays by the X2 values for proper plotting
    sorted_indices = np.argsort(X2[:, 0])
    sorted_X2 = X2[sorted_indices]
    sorted_z2 = z2[sorted_indices]
    
    plt.figure()
    plt.plot(X, Y, 'r-', label='Actual function')  # Plot as a red line
    plt.plot(sorted_X2, sorted_z2, 'b-', label='NN approximation')  # Plot as a blue line
    plt.legend()
    plt.title(f'Function Approximation after {epoch} Epochs')
    plt.xlabel('X value')
    plt.ylabel('Predicted Y value')
    plt.savefig(f'function_approximation_{epoch}_epochs.pdf')



for epoch in range(epochs):
    a1, z2,z1 = forward_propagation(X)
    error = backward_propagation(X, Y, a1, z2,z1)
    errors.append(error)
    if epoch+1 in [10,100,200,400,1000,10000]:
        plot_approximation(epoch+1)

# (1) Training error vs epoch number
plt.figure()
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training Error vs Epoch Number')
plt.savefig('training_error_vs_epoch_number2.pdf')




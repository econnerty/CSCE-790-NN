import numpy as np

np.random.seed(5)  # for reproducibility
input_dim = 2
output_dim = 2
learning_rate = 0.001
epochs = 10000

# Randomly initialize weights and biases
weights = np.random.randn(input_dim, output_dim)
biases = np.random.randn(output_dim)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(X):
    linear_combination = np.dot(X, weights) + biases
    output = sigmoid(linear_combination)
    return output

def backward_propagation(X, Y, output):
    error = output - Y
    d_weights = np.dot(X.T, error * sigmoid_derivative(output))
    d_biases = np.sum(error * sigmoid_derivative(output), axis=0)
    global weights, biases
    weights -= learning_rate * d_weights
    biases -= learning_rate * d_biases
    return error

X = np.array([[0.1, 1.2], [0.7, 1.8], [0.8, 1.6], [0.8, 0.6], [1.0, 0.8],
              [0.3, 0.5], [0.0, 0.2], [-0.3, 0.8], [-0.5, -1.5], [-1.5, -1.3]])

Y = np.array([[1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1]])

errors = []

for epoch in range(epochs):
    output = forward_propagation(X)
    error = backward_propagation(X, Y, output)
    errors.append(np.mean(np.abs(error)))

# Plotting
import matplotlib.pyplot as plt

# (1) Training error vs epoch number
plt.figure()
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training Error vs Epoch Number')
plt.savefig('training_error_vs_epoch_number.pdf')

def plot_decision_boundary(epoch):
    plt.figure(figsize=(8, 6))

    # Extracting weights and biases
    w1, w2 = weights[0], weights[1]
    b1, b2 = biases[0], biases[1]

    # Calculating the slope and intercept for the two decision boundaries
    m1 = -w1[0] / w1[1]
    c1 = -b1 / w1[1]
    m2 = -w2[0] / w2[1]
    c2 = -b2 / w2[1]

    # Generating the x-coordinates for the decision boundary lines
    x_coords = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 400)

    # Generating the y-coordinates for the decision boundary lines
    y_coords1 = m1 * x_coords + c1
    y_coords2 = m2 * x_coords + c2

    # Plotting the decision boundary lines
    plt.plot(x_coords, y_coords1, label='Decision Boundary 1')
    plt.plot(x_coords, y_coords2, label='Decision Boundary 2')

    # Plotting the data points
    Y_class = np.argmax(Y, axis=1) * 2 + np.argmin(Y, axis=1)
    plt.scatter(X[:, 0], X[:, 1], c=Y_class, edgecolors='k', marker='o', linewidth=1)

    plt.legend()
    plt.title(f'Decision Boundary and Data Points after {epoch} Epochs')
    plt.savefig(f'decision_boundary_and_data_points_after_{epoch}_epochs.pdf')

# Call the function after training
plot_decision_boundary(epochs)


# Reset weights and biases
np.random.seed(5)
weights = np.random.randn(input_dim, output_dim)
biases = np.random.randn(output_dim)

# Train for 3 epochs and plot decision boundary
for epoch in range(3):
    output = forward_propagation(X)
    error = backward_propagation(X, Y, output)
plot_decision_boundary(3)

# Continue training for 7 more epochs (total 10) and plot decision boundary
for epoch in range(7):
    output = forward_propagation(X)
    error = backward_propagation(X, Y, output)
plot_decision_boundary(10)

# Continue training for 90 more epochs (total 100) and plot decision boundary
for epoch in range(90):
    output = forward_propagation(X)
    error = backward_propagation(X, Y, output)
plot_decision_boundary(100)

# Continue training for 1000 more epochs (total 1100) and plot decision boundary
for epoch in range(1000):
    output = forward_propagation(X)
    error = backward_propagation(X, Y, output)
plot_decision_boundary(1100)


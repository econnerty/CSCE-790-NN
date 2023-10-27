import numpy as np

np.random.seed(10)  # for reproducibility
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

    # Generate a grid of points in the input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Flatten the grid to pass into the network, then reshape the output to match the grid
    Z = forward_propagation(np.c_[xx.ravel(), yy.ravel()])
    # Convert sigmoid outputs to binary (0 or 1) based on a 0.5 threshold
    Z_bin = (Z > 0.5).astype(int)
    # Combine the binary outputs of the two neurons to form a class label (0, 1, 2, or 3)
    Z_class = Z_bin[:, 0] * 2 + Z_bin[:, 1]
    Z_class = Z_class.reshape(xx.shape)

    # Plot the contour
    plt.contourf(xx, yy, Z_class, alpha=0.8)
    # Plot the data points
    # Convert target matrix Y to class labels
    Y_class = np.argmax(Y, axis=1) * 2 + np.argmin(Y, axis=1)
    plt.scatter(X[:, 0], X[:, 1], c=Y_class, edgecolors='k', marker='o', linewidth=1)

    plt.scatter(X[:, 0], X[:, 1], c=Y_class, edgecolors='k', marker='o', linewidth=1, label='_nolegend_')  # Hide legend entry for data points
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.jet(i/3), markersize=10, label=f'Class {i}') for i in range(4)]
    plt.legend(handles=handles, title='Classes')

    plt.title(f'Decision Boundary and Data Points after {epoch} Epochs')
    plt.savefig(f'decision_boundary_and_data_points_after_{epoch}_epochs.pdf')

# Reset weights and biases
np.random.seed(10)
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


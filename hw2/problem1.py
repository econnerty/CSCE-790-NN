import numpy as np

np.random.seed(15)  # for reproducibility
input_dim = 2
output_dim = 2
learning_rate = 0.4
epochs = 10000

# Randomly initialize weights and biases
weights = np.random.randn(input_dim, output_dim)
biases = np.random.randn(output_dim)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy_loss(y_true, y_pred):
    # Avoid division by zero
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Calculate the binary cross-entropy for each class
    loss = -np.mean((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))
    return loss

def forward_propagation(X):
    linear_combination = np.dot(X, weights) + biases
    output = sigmoid(linear_combination)
    return output

def backward_propagation(X, Y, output):
    global weights, biases
    n = X.shape[0]

    # Derivative of binary cross-entropy loss with respect to z (input to the sigmoid)
    d_loss_z = (output - Y)
    
    # Gradient of the loss with respect to the weights
    d_weights = np.dot(X.T, d_loss_z) / n
    
    # Gradient of the loss with respect to the biases
    d_biases = np.sum(d_loss_z, axis=0) / n
    
    # Update the weights and biases
    weights -= learning_rate * d_weights
    biases -= learning_rate * d_biases
    
    # Return the loss for monitoring
    return binary_cross_entropy_loss(Y, output)


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



def plot_decision_boundary(epoch, ylim=(-10, 10), colors=None):
    plt.figure(figsize=(8, 6))

    # Extracting weights and biases (assuming these are accessible in the function's scope)
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
    
    # If colors are not provided, use a default colormap
    if colors is None:
        colors = plt.cm.coolwarm(Y_class / float(max(Y_class)))

    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', marker='o', linewidth=1)

    plt.ylim(ylim)  # Set the vertical range

    plt.legend()
    plt.title(f'Decision Boundary and Data Points after {epoch} Epochs')
    plt.savefig(f'decision_boundary_and_data_points_after_{epoch}_epochs.pdf')

# Assuming X, Y, weights, and biases are defined elsewhere in the code


# Call the function after training
#plot_decision_boundary(epochs)


# Reset weights and biases
np.random.seed(25)
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
for epoch in range(900):
    output = forward_propagation(X)
    error = backward_propagation(X, Y, output)
plot_decision_boundary(1000)

def plot_final_decision_boundary(ylim=(-10, 10), colors=None):
    #clear previous plot
    plt.clf()
    plt.figure(figsize=(8, 6))

    # Extracting weights and biases (assuming these are accessible in the function's scope)
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
    #Y_class = np.argmax(Y, axis=1) * 2 + np.argmin(Y, axis=1)
    
    # If colors are not provided, use a default colormap
    #if colors is None:
    #    colors = plt.cm.coolwarm(Y_class / float(max(Y_class)))

    #plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', marker='o', linewidth=1)

    plt.ylim(ylim)  # Set the vertical range

    plt.legend()
    plt.title(f'Final Decision Boundaries')
    plt.savefig(f'final decision boundary.pdf')
plot_final_decision_boundary()


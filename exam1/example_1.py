#Matlab example from https://github.com/Maithilishetty/Neural-Net-Control/blob/master/OSLA/example1.m

import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(0)

# Activation function and its derivative
def tanh(x, a, b):
    return a * np.tanh(b * x)

def tanh_deriv(x, a, b):
    return a * (1 - np.tanh(b * x) ** 2)

# Parameters
a = 1
b = 1
m0 = 1
m1 = 20
m2 = 10
m3 = 1
eta1 = 0.1

# Initialize weights
w1 = 0.2 * np.random.rand(m1, m0 + 1)  # 20 x 2
w2 = 0.2 * np.random.rand(m2, m1 + 1)  # 10 x 21
w3 = 0.2 * np.random.rand(m3, m2 + 1)  # 1 x 11

# Training loop
for k in range(1, 50501):
    if k <= 500:
        u = np.sin(2 * np.pi * k / 250)
        p = np.array([1, u]).reshape(-1, 1)

        # Forward pass
        v1 = w1 @ p
        phi_v1 = tanh(v1, a, b)
        y1_k = np.vstack((1, phi_v1))
        v2 = w2 @ y1_k
        phi_v2 = tanh(v2, a, b)
        y2_k = np.vstack((1, phi_v2))
        v3 = w3 @ y2_k
        y3 = v3
        E = 0.6 * np.sin(np.pi * u) + 0.3 * np.sin(3 * np.pi * u) + 0.1 * np.sin(5 * np.pi * u) - y3

        # Backward pass
        phi_v3_diff = 1
        phi_v2_diff = tanh_deriv(v2, a, b)
        phi_v1_diff = tanh_deriv(v1, a, b)

        delta3 = E * phi_v3_diff
        delta_w3 = eta1 * delta3 @ y2_k.T
        delta2 = (w3[:, 1:].T @ delta3) * phi_v2_diff
        delta_w2 = eta1 * delta2 @ y1_k.T
        delta1 = (w2[:, 1:].T @ delta2) * phi_v1_diff
        delta_w1 = eta1 * delta1 @ p.T

        # Update weights
        w1 += delta_w1
        w2 += delta_w2
        w3 += delta_w3
    else:
        u = -1 + 2 * np.random.rand()
        p = np.array([1, u]).reshape(-1, 1)

        # Forward pass
        v1 = w1 @ p
        phi_v1 = tanh(v1, a, b)
        y1_k = np.vstack((1, phi_v1))
        v2 = w2 @ y1_k
        phi_v2 = tanh(v2, a, b)
        y2_k = np.vstack((1, phi_v2))
        v3 = w3 @ y2_k
        y3 = v3
        E = 0.6 * np.sin(np.pi * u) + 0.3 * np.sin(3 * np.pi * u) + 0.1 * np.sin(5 * np.pi * u) - y3

        # Backward pass
        phi_v3_diff = 1
        phi_v2_diff = tanh_deriv(v2, a, b)
        phi_v1_diff = tanh_deriv(v1, a, b)

        delta3 = E * phi_v3_diff
        delta_w3 = eta1 * delta3 @ y2_k.T
        delta2 = (w3[:, 1:].T @ delta3) * phi_v2_diff
        delta_w2 = eta1 * delta2 @ y1_k.T
        delta1 = (w2[:, 1:].T @ delta2) * phi_v1_diff
        delta_w1 = eta1 * delta1 @ p.T

        # Update weights
        w1 += delta_w1
        w2 += delta_w2
        w3 += delta_w3


# Arrays to store outputs
yp = np.zeros(1002)
ym = np.zeros(1002)

# Simulation loop for the outputs
for k in range(1, 1001):
    u = np.sin(2 * np.pi * k / 250)
    p = np.array([1, u]).reshape(-1, 1)

    # Desired plant output
    y1 = 0.3 * yp[k] + 0.6 * yp[k - 1] + 0.6 * np.sin(np.pi * u) + 0.3 * np.sin(3 * np.pi * u) + 0.1 * np.sin(5 * np.pi * u)

    # Neural network forward pass
    v1 = w1 @ p
    phi_v1 = tanh(v1, a, b)
    y1_k = np.vstack((1, phi_v1))
    v2 = w2 @ y1_k
    phi_v2 = tanh(v2, a, b)
    y2_k = np.vstack((1, phi_v2))
    v3 = w3 @ y2_k
    y2 = v3 + 0.3 * yp[k] + 0.6 * yp[k - 1]

    # Store outputs
    yp[k + 1] = y1
    ym[k + 1] = y2

# Calculate performance metrics
g = np.var(yp[2:] - ym[2:])
h = np.var(yp[2:])
perf = (1 - (g / h)) * 100
mse = np.sum((yp[2:] - ym[2:]) ** 2) / len(yp[2:])

# Display the performance
print(f"Variance of error: {g}")
print(f"Variance of plant output: {h}")
print(f"Performance: {perf}%")
print(f"Mean Squared Error: {mse}")

# Plot the outputs
plt.plot(yp[2:], label='Desired Output')
plt.plot(ym[2:], label='Neural Network Output')
plt.legend()
plt.title('Example 1 Single Sinusoidal Input')
plt.savefig('example1_single_sinusoidal_input.pdf')

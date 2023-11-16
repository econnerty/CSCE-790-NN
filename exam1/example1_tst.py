#Using code from as help https://github.com/Maithilishetty/Neural-Net-Control/blob/master/OSLA/example1.m

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Defining the Neural Network class
class NeuralNetwork:
    def __init__(self, m0, m1, m2, lambda_):
        self.w1 = np.random.rand(m1, m0 + 1)
        self.w2 = np.zeros((m2, m1 + 1))
        self.P = (1 / lambda_) * np.eye(m1 + 1)

    def forward_pass(self, u):
        p = np.array([1, u])
        v1 = self.w1 @ p
        phi_v1 = np.tanh(v1)  # 'a' and 'b' are both 1, so they were omitted
        y1 = np.concatenate(([1], phi_v1))
        v2 = self.w2 @ y1
        return y1, v2

    def backward_pass(self, y1, e):
        P1 = self.P - (self.P @ y1[:, None] @ y1[None, :] @ self.P) / (1 + y1 @ self.P @ y1)
        self.P = P1
        self.w2 = self.w2 + e * y1[None, :] @ P1

# Function to calculate the plant output with different types of input
def calculate_plant_output(yp, u, k):
    if k < 250:
        return 0.3 * yp[-1] + 0.6 * yp[-2] + 0.6 * np.sin(np.pi * u) + 0.3 * np.sin(3 * np.pi * u) + 0.1 * np.sin(5 * np.pi * u)
    else:
        # After k = 250, the input changes to a sum of two sinusoids.
        return 0.3 * yp[-1] + 0.6 * yp[-2] + np.sin(2 * np.pi * k / 250) + np.sin(2 * np.pi * k / 25)

# Initializations
m0, m1, m2, lambda_ = 1, 10, 1, 0.001
nn = NeuralNetwork(m0, m1, m2, lambda_)
n = np.arange(1001)
yp = np.array([0, 0])
yphat = np.array([0, 0])

# Main loop for the modified input
for k in n:
    if k < 250:
        u = np.sin(2 * np.pi * k / 250)
    else:
        u = np.sin(2 * np.pi * k / 250) + np.sin(2 * np.pi * k / 25)
    
    y1, y2 = nn.forward_pass(u)
    yphat1 = y2 + 0.3 * yp[-1] + 0.6 * yp[-2]
    yphat = np.append(yphat, yphat1)

    yp1 = calculate_plant_output(yp, u, k)
    yp = np.append(yp, yp1)

    e = yp1 - yphat1
    nn.backward_pass(y1, e)

# Plotting the output with the new type of input
plt.plot(n, yp[2:], label='Desired Output')
plt.plot(n, yphat[2:], label='Neural Network Output')
plt.legend()
plt.title('Example 1 Single sinusoidal input')
plt.show()

# Performance metrics with the new input
performance = (1 - np.var(yp[2:] - yphat[2:]) / np.var(yp[2:])) * 100
mse = np.mean((yp[2:] - yphat[2:])**2)

performance, mse

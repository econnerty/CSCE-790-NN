import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import numpy as np

np.random.seed(42)  # for reproducibility

# Define the functions f and g according to the plant model
def f_plant(yp_k):
    return yp_k / (1 + yp_k**2)

def g_plant(u_k):
    return u_k**3

def plant_equation(yp_k, u_k):
    return f_plant(yp_k) + g_plant(u_k)

# Generate random input data for u(k) in the interval [-2, 2]
num_data_points = 5000
u = np.random.uniform(-2, 2, num_data_points)
# Initialize the output array for y_p(k)
yp = np.zeros(num_data_points)

# Generate output data for y_p(k) using the plant equation
for k in range(1, num_data_points):
    yp[k] = f_plant(yp[k-1]) + g_plant(u[k-1])

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b1 = np.random.randn(hidden_size)
        self.b2 = np.random.randn(output_size)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2)+self.b2
        return self.z2

    def backward(self, x, y, o, learning_rate):
        # Backpropagation
        error = y - o
        d_output = error
        d_hidden_layer = np.dot(d_output, self.w2.T) * (1 - np.power(self.a1, 2))
        
        # Update weights
        self.w1 += np.dot(x.T, d_hidden_layer) * learning_rate
        self.w2 += np.dot(self.a1.T, d_output) * learning_rate
        self.b1 += np.sum(d_hidden_layer, axis=0) * learning_rate
        self.b2 += np.sum(d_output, axis=0) * learning_rate
        return self.mean_squared_error(y, o)


def train_network(nn, X, Y, epochs, learning_rate=0.001):
    losses = []
    for epoch in tqdm(range(epochs)):
        loss = 0
        for i in range(len(X)-1):
            o = nn.forward(X[i:i+1])  # Use slicing to keep dimensions
            loss +=nn.backward(X[i:i+1], Y[i], o, learning_rate)
        mean_loss = loss / (len(X)-1)
        losses.append(mean_loss)
    return losses
# Prepare the data for training
X = np.array([yp[:-1], yp[1:], u[:-1]]).T
Y = yp[2:].reshape(-1, 1)

# Initialize and train the network
nn_f = SimpleNN(input_size=1, hidden_size=10, output_size=1)
nn_g = SimpleNN(input_size=1, hidden_size=10, output_size=1)

# Prepare the data for training N_f
X_f = yp[:-1].reshape(-1, 1)
Y_f = np.array([f_plant(y) for y in yp[:-1]]).reshape(-1, 1)

# Prepare the data for training N_g
X_g = u[:-1].reshape(-1, 1)
Y_g = np.array([g_plant(u_val) for u_val in u[:-1]]).reshape(-1, 1)

# Train both networks
losses_f = np.array(train_network(nn_f, X_f, Y_f, epochs=10, learning_rate=0.1))
losses_g = np.array(train_network(nn_g, X_g, Y_g, epochs=10, learning_rate=0.1))
print(losses_f[-1])
print(losses_g[-1])

"""plt.plot(losses_f, label='Loss for Nf')
plt.title('Loss over time for Neural Network f')
plt.show()
plt.plot(losses_g, label='Loss for Ng')
plt.show()"""


# Generate new test data for u(k)
test_data_u = np.random.uniform(-2, 2, 50)

# Generate the test outputs for N_f and N_g
test_output_f = np.array([nn_f.forward(np.array([[y]])) for y in test_data_u]).flatten()
test_output_g = np.array([nn_g.forward(np.array([[u]])) for u in test_data_u]).flatten()
predicted_plant = test_output_f + test_output_g

# Calculate the actual values using the plant functions
actual_f = f_plant(test_data_u)
actual_g = g_plant(test_data_u)
actual_plant = actual_f + actual_g

# Plot the test outputs and actual values for comparison

plt.plot(actual_f, label='Actual Nf')
plt.plot(test_output_f, label='Predicted Nf')
plt.legend()
plt.title('Predicted Outputs for Nf')
plt.savefig('ex3_nf_predictedvs_actual.pdf')
plt.close()

plt.plot(actual_g,label='Actual Ng')
plt.plot(test_output_g, label='Predicted Ng')
plt.legend()
plt.title('Predicted Outputs for Ng')
plt.savefig('ex3_ng_predictedvs_actual.pdf')
plt.close()

plt.plot(actual_plant, label='Actual Plant')
plt.plot(predicted_plant, label='Predicted plant')
plt.legend()
plt.title('Predicted Outputs and Actual plant Values')
plt.savefig('ex3_predictedvs_actual.pdf')
plt.close()






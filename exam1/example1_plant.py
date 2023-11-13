import numpy as np
from tqdm.auto import tqdm

np.random.seed(42)  # for reproducibility

def f(u):
    return 0.6 * np.sin(np.pi * u) + 0.3 * np.sin(3 * np.pi * u) + 0.1 * np.sin(5 * np.pi * u)

def plant_equation(yp_k, yp_k_minus_1, u_k):
    return 0.3 * yp_k + 0.6 * yp_k_minus_1 + f(u_k)

# Generate data
num_data_points = 5000
yp = np.zeros(num_data_points)
yp[0] = 1.0  # Initial condition

u = np.sin(2 * np.pi * np.arange(num_data_points) / 250)

for k in range(2, num_data_points):
    yp[k] = plant_equation(yp[k-1], yp[k-2], u[k-1])


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


def train_network(nn, X, Y, epochs=50, learning_rate=0.001):
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
nn = SimpleNN(input_size=3, hidden_size=10, output_size=1)
losses = np.array(train_network(nn, X, Y, epochs=100, learning_rate=0.005))
print(losses[-1])
#Plot the loss over time
import matplotlib.pyplot as plt
"""plt.plot(losses.flatten())
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.show()"""



# Test the network
predicted = []
actual = []



# Generate sinusoidal test data
test_data = np.sin(2 * np.pi * np.arange(1000) / 250)
yp_test = np.zeros(len(test_data) + 2)  # Include initial conditions

# The test should start where the plant has some initial dynamic behavior
yp_test[0] = yp[-2]  # Second last value from the training data
yp_test[1] = yp[-1]  # Last value from the training data

# Run the test predictions
for k in range(2, len(yp_test)-1):
    # Create the input vector for the neural network
    input_vec = np.array([[yp_test[k-1], yp_test[k-2], test_data[k-1]]])
    # Predict the next output
    prediction = nn.forward(input_vec)
    # Update the plant using the actual equation for the next time step
    yp_test[k] = plant_equation(yp_test[k-1], yp_test[k-2], test_data[k-1])
    # Append to the lists
    predicted.append(prediction[0][0])  # Adjust if the output structure is different
    actual.append(yp_test[k])



# Compare the predicted and actual values
import matplotlib.pyplot as plt
plt.plot(predicted, label='predicted')
plt.plot(actual, label='actual')
plt.title('Predicted vs actual Plant Output')
plt.legend()
plt.savefig('ex1_predictedvs_actual.pdf')
plt.close()

# Generate sum of sinusoidal data




# Generate data
num_data_points = 5000
yp = np.zeros(num_data_points)

u = np.sin(2 * np.pi * np.arange(num_data_points) / 250) + np.sin(2 * np.pi * np.arange(num_data_points) / 25)

for k in range(2, num_data_points):
    yp[k] = plant_equation(yp[k-1], yp[k-2], u[k-1])

def train_network(nn, X, Y, epochs=50, learning_rate=0.001):
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
nn = SimpleNN(input_size=3, hidden_size=10, output_size=1)
losses = np.array(train_network(nn, X, Y, epochs=1000, learning_rate=0.005))
print(losses[-1])
#Plot the loss over time
import matplotlib.pyplot as plt
"""plt.plot(losses.flatten())
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.show()"""



# Test the network
predicted = []
actual = []


# Generate sinusoidal test data
test_data = np.sin(2 * np.pi * np.arange(1000) / 250) + np.sin(2 * np.pi * np.arange(1000) / 25)
yp_test = np.zeros(len(test_data) + 2)  # Include initial conditions

# The test should start where the plant has some initial dynamic behavior
yp_test[0] = yp[-2]  # Second last value from the training data
yp_test[1] = yp[-1]  # Last value from the training data

# Run the test predictions
for k in range(2, len(yp_test)-1):
    # Create the input vector for the neural network
    input_vec = np.array([[yp_test[k-1], yp_test[k-2], test_data[k-1]]])
    # Predict the next output
    prediction = nn.forward(input_vec)
    # Update the plant using the actual equation for the next time step
    yp_test[k] = plant_equation(yp_test[k-1], yp_test[k-2], test_data[k-1])
    # Append to the lists
    predicted.append(prediction[0][0])  # Adjust if the output structure is different
    actual.append(yp_test[k])



# Compare the predicted and actual values
import matplotlib.pyplot as plt
plt.plot(predicted, label='predicted')
plt.plot(actual, label='actual')
plt.title('Predicted vs actual Plant Output with Sum of Sinusoids')
plt.legend()
plt.savefig('ex1_sum_of_sinusoids.pdf')
plt.close()





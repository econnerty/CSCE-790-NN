import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the sigmoid function
def sigmoid(x):
    return (1 - np.exp(-100*x)) / (1 + np.exp(-100*x))

# Define the system of differential equations
def hopfield_network(t, x, W, b):
    return -0.5 * x + 0.5 * W.T @ sigmoid(x) + 0.5 * b

# Define the weight matrix W and bias vector b as given
W = np.array([[0, 1], [1, 0]])
b = np.array([0, 0])

# Create a grid of initial conditions
grid_values = np.linspace(-1, 1, 20)
initial_conditions = np.array(np.meshgrid(grid_values, grid_values)).T.reshape(-1, 2)

# Time span for the simulation
t_span = [0, 10]

# Solve the system for each initial condition and plot the phase-plane trajectories
plt.figure(figsize=(8, 8))

for x0 in initial_conditions:
    # Solve the differential equation
    sol = solve_ivp(hopfield_network, t_span, x0, args=(W, b), dense_output=True)
    
    # Plot the trajectory
    t_values = np.linspace(t_span[0], t_span[1], 300)
    x_values = sol.sol(t_values)
    plt.plot(x_values[0], x_values[1], linewidth=1)

# Plot settings
plt.title('Phase-plane trajectories of the Hopfield Network')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig('phase_plane_trajectories2.pdf')

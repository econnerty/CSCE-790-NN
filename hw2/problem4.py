import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the sigmoid function σ(x)
def sigmoid(x):
    return 1 / (1 + np.exp(-100 * x))

# Define the right-hand side of the differential equation system
def hopfield(t, x, W, b):
    sigma_x = sigmoid(x)
    dxdt = -21 * x + 12 * np.dot(W.T, sigma_x) + 12 * b
    return dxdt

# Define W and b
W = np.array([[0, 1], [1, 0]])
b = np.array([0, 0])

# Create a grid of initial conditions
x_init_vals = np.linspace(-1, 1, 10)
y_init_vals = np.linspace(-1, 1, 10)
initial_conditions = [(x_init, y_init) for x_init in x_init_vals for y_init in y_init_vals]


# Create a figure for the phase-plane plot
plt.figure()

for init_cond in initial_conditions:
    # Solve the differential equation system for this initial condition
    sol = solve_ivp(hopfield, [0, 10], init_cond, args=(W, b), t_eval=np.linspace(0, 10, 1000))
    # Plot the trajectory for this initial condition
    plt.plot(sol.y[0], sol.y[1])

# Configure the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase-Plane Trajectories')
plt.grid(True)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.savefig('phase_plane_trajectories.pdf')

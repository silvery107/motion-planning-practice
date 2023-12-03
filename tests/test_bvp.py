from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt

def pendulum_dynamics(t, y):
    g = 9.81  # acceleration due to gravity
    L = 1.0   # length of the pendulum
    m = 1.0   # mass of the pendulum bob
    b = 0.1   # damping coefficient

    dydt = np.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = y[4] / m - g * np.sin(y[2])
    dydt[2] = y[3]
    dydt[3] = -g / L * np.sin(y[2]) - b / (m * L**2) * y[3] + y[4] / (m * L**2)
    dydt[4] = 0  # No dynamics for control input
    
    return dydt

def boundary_conditions(ya, yb):
    # Define boundary conditions
    # Initial state: ya = [x0, v0, theta0, omega0, tau0]
    # Target state: yb = [xf, vf, thetaf, omegaf, tauf]
    return np.array([ya[0] - x0, ya[2] - theta0, yb[0] - xf, yb[2] - thetaf, yb[1] - vf])

# Initial and target states
x0, v0, theta0, omega0, tau0 = 0.0, 0.0, 0.1, 0.0, 0.0  # Initial state
xf, vf, thetaf, omegaf, tauf = 1.0, 0.0, np.pi, 0.0, 0.0  # Target state

# Time vector for integration
t_span = (0, 5)  # Adjust the time span based on your problem
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Initial guess for the solution
y_guess = np.zeros((5, t_eval.size))

# Set initial conditions based on the initial state
y_guess[:, 0] = [x0, v0, theta0, omega0, tau0]

# Control input (torque) as a function of time
# For simplicity, you can use a constant torque or define a function of time
tau_function = lambda t: 0.1  # Modify this function as needed

# Solve the boundary value problem using solve_bvp
solution = solve_bvp(lambda t, y: pendulum_dynamics(t, y), boundary_conditions, t_eval, y_guess, verbose=2)
print(f"Success: {solution.success}")
# Extract the solution
y_solution = solution.sol(t_eval)
# print(y_solution[4])

# Plot the results
plt.plot(t_eval, y_solution[0], label='x(t)')
plt.plot(t_eval, y_solution[2], label='theta(t)')
plt.plot(t_eval, y_solution[4], label='tau(t)')
plt.xlabel('Time')
plt.ylabel('State Variables and Control Input')
plt.legend()
plt.show()

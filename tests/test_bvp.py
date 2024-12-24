"""
Double integrator example for using the boundary value problem (BVP) solver.
The control law is derived from Pontryagin's Minimum Principle (PMP) via
minimizing a minimum energy continuous optimal control problem (OCP).
"""

from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt

def get_u(p):
    """ g(x, u) = u^2
    """
    u = 0
    if p >= 2:
        u = -1
    elif p <= -2:
        u = 1
    else:
        u = -0.5 * p

    return u

def fun(t, y):
    m = t.shape[0]
    dydt = np.zeros((n, m))

    x = y[:dim_x]
    p = y[dim_x:]
    u = np.zeros((dim_u, m))
    for idx in range(m):
        u[0, idx] = get_u(p[2, idx])
        u[1, idx] = get_u(p[3, idx])

    dydt[:dim_x] = A @ x + B @ u
    dydt[dim_x:] = - A.T @ p
    
    return dydt

def bc(ya, yb):

    return np.hstack(((ya - y0)[:dim_x], (yb - yf)[:dim_x]))


# Double integrator dynamics
dim_x = 4
dim_u = 2
A = np.zeros((dim_x, dim_x))
B = np.zeros((dim_x, dim_u))
A[:2, 2:] = np.eye(2)
B[2:, :] = np.eye(2)

# Number of state for the ODE system
n = dim_x*2

# Initial and target states
x0 = np.zeros(dim_x)
xf = np.array([1, 1, 0, 0])

y0 = np.zeros(n)
y0[:dim_x] = x0
yf = np.zeros(n)
yf[:dim_x] = xf

# Final time and discretization resolutions
tf = 2.1
num_points = 100
t_span = np.linspace(0, tf, num_points)

# Solve the boundary value problem using solve_bvp
y_init = np.linspace(y0, yf, num_points, axis=-1)
solution = solve_bvp(fun, bc, t_span, y_init, verbose=2)
print(f"Success: {solution.success}")
# Extract the solution
sol = solution.sol(t_span)

# Plot results
fig, ax = plt.subplots(n+1, 1)
ylabels = ["x1", "x2", "dx1", "dx2", "p1", "p2", "p3", "p4", "u"]
for idx in range(n):
    ax[idx].plot(t_span, sol[idx, :])
    ax[idx].set_ylabel(ylabels[idx])

ax[-1].plot(t_span, [get_u(p) for p in sol[-1, :]])
ax[-1].set_ylabel(ylabels[-1])
plt.show()

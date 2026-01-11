# We'll implement:
# 1) define interaction matrix A for 4 services
# 2) compute eigenvalues
# 3) simulate ODE system using RK4
# 4) create plots: load trajectories, phase-plane for two services, failure propagation example

import numpy as np
import matplotlib.pyplot as plt

# Define interaction matrix A (4 services)
A = np.array([
    [-0.6,  0.2,  0.0,  0.0],
    [ 0.1, -0.5,  0.3,  0.0],
    [ 0.0,  0.2, -0.4,  0.2],
    [ 0.0,  0.0,  0.3, -0.7]
])

# Compute eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# RK4 solver
def rk4(f, x0, t):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = f(t[i-1], x[i-1])
        k2 = f(t[i-1] + h/2, x[i-1] + h*k1/2)
        k3 = f(t[i-1] + h/2, x[i-1] + h*k2/2)
        k4 = f(t[i-1] + h, x[i-1] + h*k3)
        x[i] = x[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x

# System dynamics
def system(t, x):
    return A.dot(x)

# Time grid
t = np.linspace(0, 20, 400)

# Initial loads
x0 = np.array([1.0, 0.5, 0.2, 0.1])

# Simulate
trajectory = rk4(system, x0, t)

# Plot 1: Time evolution of all services
plt.figure()
for i in range(4):
    plt.plot(t, trajectory[:, i])
plt.xlabel("Time")
plt.ylabel("Service load")
plt.title("Load Evolution Over Time")
plt.show()

# Plot 2: Phase plane between Service 1 and Service 2
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.xlabel("Service 1 load")
plt.ylabel("Service 2 load")
plt.title("Phase Plane: Service 1 vs Service 2")
plt.show()

# Failure propagation scenario: spike on service 1
def system_with_spike(t, x):
    spike = 2.0 if 5 < t < 7 else 0.0
    return A.dot(x) + np.array([spike, 0, 0, 0])

trajectory_spike = rk4(system_with_spike, x0, t)

# Plot 3: Failure propagation after spike
plt.figure()
for i in range(4):
    plt.plot(t, trajectory_spike[:, i])
plt.xlabel("Time")
plt.ylabel("Service load")
plt.title("Failure Propagation After Traffic Spike")
plt.show()

# Return eigenvalues for user visibility
eigenvalues


# ======================================================================================================================
# A small example to solve the forward heat equation in two dimensions and time independent using DeepXDE
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================


import deepxde as dde
import numpy as np
import tensorflow as tf

# -----------------------
# Parameters
sigma = 0.1
mu_x = 0.5
mu_y = 0.5

# -----------------------
# Thermal diffusivity a(x,y)
def a(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return 1.0 + tf.exp(-((x1 - mu_x)**2 + (x2 - mu_y)**2) / (2 * sigma**2))

# -----------------------
# PDE Definition
def pde(x, u):
    grad_u = dde.grad.jacobian(u, x, i=0, j=0), dde.grad.jacobian(u, x, i=0, j=1)
    u_t = dde.grad.jacobian(u, x, i=0, j=2)

    ax = a(x[:, :2])
    flux_x = ax * grad_u[0]
    flux_y = ax * grad_u[1]
    div_flux_x = dde.grad.jacobian(flux_x, x, i=0, j=0)
    div_flux_y = dde.grad.jacobian(flux_y, x, i=0, j=1)

    return u_t - (div_flux_x + div_flux_y) - 1.0  # f = 1

# -----------------------
# Geometry and Time
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# -----------------------
# Initial and Boundary Conditions
def initial_func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

def boundary_func(x, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_func)
ic = dde.icbc.IC(geomtime, initial_func, lambda x, on_initial: on_initial)

# -----------------------
# Dataset
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=4000,
    num_boundary=100,
    num_initial=100,
    num_test=1000,
)

# -----------------------
# Neural Network
net = dde.nn.FNN([3] + [50] * 3 + [1], "tanh", "Glorot normal")

# -----------------------
# Model
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=15000)

# # Optional L-BFGS refinement
# model.compile("L-BFGS")
# model.train()

# -----------------------
# Evaluate
X, Y, T = np.meshgrid(np.linspace(0, 1, 100),
                      np.linspace(0, 1, 100),
                      [1.0])
xtest = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))
u_pred = model.predict(xtest).reshape(X.shape)

import matplotlib.pyplot as plt
plt.contourf(X[:, :, 0], Y[:, :, 0], u_pred[:, :, 0], levels=50, cmap='inferno')
plt.colorbar(label='u(x,y,t=1)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution u(x,y,t=1)')
plt.show()

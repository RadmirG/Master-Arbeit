# ======================================================================================================================
# Solving inverse problem for the heat equation
#                               ∂u − ∇⋅(a∇u) = f
# where ∂u time derivative of u(x,y,t), also solution for heat equation, a(x,y) is (unknown) heat diffusivity, which
# has to be calculated and f is some initial system input.
# ----------------------------------------------------------------------------------------------------------------------
# Objective: Infer a(x,y) such that the heat equation is satisfied for given observations u(x,y,t), where u is observed
# as synthetic or experimental function.
#
# Inverse Problem (PDE Residual): Using u(x,y,t), the inverse problem requires minimizing the residual of the
# heat equation:
#                               r(x,y,t,a) = ∂u − ∇⋅(a)∇u − f.
# The loss function for estimating a(x,y) is:
#                               L = (1/N ∑ ∣r∣^2) + λR(a),
# where:
#     r is the residual over all datapoints (x_i, y_i)
#     R(a) is a regularization term on a(x,y) (e.g., to enforce smoothness).
#     λ is the regularization weight.
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================

import deepxde as dde
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# ======================================================================================================================
# DEFINITION OF THE INVERSE PROBLEM AND DOMAIN
# ======================================================================================================================

# Define the spatial domain (x, y) ∈ [0, l] x [0, l] and time domain t ∈ [0, T]
L = 1
T = 10
spatial_domain = dde.geometry.Rectangle([0, 0], [L, L])
time_domain = dde.geometry.TimeDomain(0, T)
domain = dde.geometry.GeometryXTime(spatial_domain, time_domain)

# Observed data : u(x,y,t)=e^(−λt)sin(πx)sin(πy)
def u_obs(x_in):
    u = tf.exp(-x_in[:, 2:])*tf.sin(np.pi * x_in[:, 0:1])*tf.sin(np.pi * x_in[:, 1:2])
    return u

# Right side of heat PDE
def f_exact(x_in):
    f = ((2*np.pi*np.pi*(1 + x_in[:, 0:1] + x_in[:, 1:2]) - 1 - np.pi)
         *tf.exp(-x_in[:, 2:])*tf.sin(np.pi * x_in[:, 0:1])*tf.sin(np.pi * x_in[:, 1:2]))
    return f


# Generate 10 random (x, y) points in the spatial domain
num_points = 10
random_x = np.random.uniform(0, L, num_points)
random_y = np.random.uniform(0, L, num_points)
random_points = np.column_stack((random_x, random_y))
# Generate time steps from t = 0 to t = 10
time_values = np.linspace(0, T, 100)
# Create input for (x, y, t) evaluation
all_inputs = []
for t in time_values:
    t_array = np.full((10, 1), t)  # Set time for all spatial points
    inputs = np.hstack((random_points, t_array))  # Stack (x, y, t)
    all_inputs.append(inputs)

# Convert to TensorFlow tensor
all_inputs_tensor = tf.convert_to_tensor(np.vstack(all_inputs), dtype=tf.float32)

u_for_loss = u_obs(all_inputs_tensor)
f_for_loss = f_exact(all_inputs_tensor)


#-----------------------------------------------------------------------------------------------------------------------

# The heat equation residual for the inverse problem
def inverse_loss(x_in, outputs):
    u = outputs[:, 0:1]  # Learned u(x)
    a = outputs[:, 1:2]  # Learned a(x)
    f = outputs[:, 2:3]  # Learned f(x)

    u_x = dde.grad.jacobian(u, x_in, i=0, j=0)  # ∂u/∂x
    u_y = dde.grad.jacobian(u, x_in, i=0, j=1)  # ∂u/∂y
    u_t = dde.grad.jacobian(u, x_in, i=0, j=2)  # ∂u/∂t

    flux_x = a * u_x
    flux_y = a * u_y
    flux_xx = dde.grad.jacobian(flux_x, x_in, i=0, j=0)
    flux_yy = dde.grad.jacobian(flux_y, x_in, i=0, j=1)

    return u_t - (flux_xx + flux_yy) - f

#-----------------------------------------------------------------------------------------------------------------------
# Define boundary conditions (Neumann)
def boundary(_, on_boundary):
    return on_boundary  # True if x is on the domain boundary

bc = dde.icbc.DirichletBC(
    domain, lambda x_in: 0, boundary
)


# Define initial conditions
def initial_condition(x_in):
    return np.sin(np.pi * x_in[:, 0]) * np.sin(np.pi * x_in[:, 1])

test = initial_condition(all_inputs_tensor.numpy())

def initial(_, on_initial):
    return on_initial

ic = dde.icbc.IC(
    domain, initial_condition, lambda _, on_initial: on_initial )

#-----------------------------------------------------------------------------------------------------------------------
# Incorporate observed data constraints
u_data = dde.icbc.PointSetBC(
    all_inputs_tensor,
    u_for_loss,
    component=0
)

# Incorporate learned heat diffusivity a(x)
a_data = dde.icbc.PointSetBC(
    0,
    0,
    component=1
)

# Incorporate learned heat source f(x)
f_data = dde.icbc.PointSetBC(
    all_inputs_tensor,
    f_for_loss,
    component=2
)

# ======================================================================================================================
# TRAININGS PROCESS
# ======================================================================================================================

# Define the neural network to infer a(x)
pinn = dde.maps.FNN([3] + [50] * 3 + [3], "tanh", "Glorot uniform")
# pinn = dde.maps.FNN([1] + [100] * 5 + [3], "tanh", "Glorot uniform")


# Create training data for PINN
data = dde.data.PDE(
    domain,
    inverse_loss,   # Loss 0: PDE Residual
    [u_data,        # Loss 1: Observed Data
     #a_data,       # Loss 2: Heat diffusivity dependency
     f_data,        # Loss 3: Heat source dependency
     ic,            # Loss 4: Dirichlet BC : u(x,y,0) = sin(πx)sin(πy)
     bc],           # Loss 5: Dirichlet BC : u(x,y,t) = 0
    num_domain=4000,
    num_boundary=400,
)

# Define and train the model
model = dde.Model(data, pinn)
loss_weights=[2, 1.5, 0.1, 0.5, 0.5]
model.compile("adam", lr=1e-3, loss_weights=loss_weights)
loss_history, train_state = model.train(iterations=20000)
#model.compile("L-BFGS-B", loss_weights=loss_weights)
#loss_history, train_state = model.train(iterations=5000)


# ======================================================================================================================
# PREPARE PLOTS
# ======================================================================================================================
# Define a grid of points for plotting
num_points = 100
x = np.linspace(0, L, num_points)
y = np.linspace(0, L, num_points)
x_grid, y_grid = np.meshgrid(x, y)
xy = np.hstack((x_grid.flatten()[:, None], y_grid.flatten()[:, None]))

# Evaluate (u, a, f) using the trained model at time t=0
xy_with_zeros = np.hstack((xy, np.zeros((xy.shape[0], 1))))  # Append time=0 for evaluation
predictions = model.predict(xy_with_zeros)

# Extract predictions
u_pred = predictions[:, 0].reshape(num_points, num_points)  # u(x,y,0)
a_pred = predictions[:, 1].reshape(num_points, num_points)  # a(x,y)
f_pred = predictions[:, 2].reshape(num_points, num_points)  # f(x,y,0)

# ----------------------------------------------------------------------------------------------------------------------
# Plot u(x, y, 0) - Temperature Solution
plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, u_pred, levels=50, cmap="inferno")
plt.colorbar(label="$u(x, y, 0)$")
plt.title("Temperature $u(x, y, 0)$")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot a(x, y) - Heat Diffusivity
plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, a_pred, levels=50, cmap="viridis")
plt.colorbar(label="$a(x, y)$")
plt.title("Heat Diffusivity $a(x, y)$")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot f(x, y, 0) - Heat Source
plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, f_pred, levels=50, cmap="coolwarm")
plt.colorbar(label="$f(x, y, 0)$")
plt.title("Heat Source $f(x, y, 0)$")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 3D Plots
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(x_grid, y_grid, u_pred, cmap="inferno", edgecolor='none')
ax.set_title("Temperature $u(x, y, 0)$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("$u(x,y,0)$")

ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(x_grid, y_grid, a_pred, cmap="viridis", edgecolor='none')
ax.set_title("Heat Diffusivity $a(x, y)$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("$a(x,y)$")

ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(x_grid, y_grid, f_pred, cmap="coolwarm", edgecolor='none')
ax.set_title("Heat Source $f(x, y, 0)$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("$f(x,y,0)$")

plt.show()

# ======================================================================================================================
# LOSSES PLOT
# ======================================================================================================================

# Define loss labels based on the given loss indices
loss_labels = [
    r"$L_{PDE}$",
    r"$L_u$",
    #r"$L_a$",
    r"$L_f$",
    r"$L_{BC}$",
    r"$L_{IC}$"
]

# Convert loss_history.loss_train to a NumPy array for easier manipulation
loss_array = np.array(loss_history.loss_train)  # Shape: (epochs, num_losses)

plt.figure(figsize=(8, 5))
for i in range(loss_array.shape[1]):  # Iterate over loss components
    plt.semilogy(loss_history.steps, loss_array[:, i], label=loss_labels[i])

plt.grid()
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss deapXDE")
plt.show()
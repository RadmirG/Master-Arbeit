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


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # Ensure TensorFlow 1.x behavior


# ======================================================================================================================
# DEFINITION OF THE INVERSE PROBLEM AND DOMAIN
# ======================================================================================================================

# Define the spatial domain (x, y) ∈ [0, l] x [0, l] and time domain t ∈ [0, T]
l = 1
T = 1
spatial_domain = dde.geometry.Rectangle([0, 0], [l, l])
time_domain = dde.geometry.TimeDomain(0, T)
domain = dde.geometry.GeometryXTime(spatial_domain, time_domain)

# Observed data u_obs(x, y, t), synthetic data
# It can/should be experimental data
def u_obs(x_in):
    return (tf.cos(x_in[:, 2:] * np.pi * x_in[:, 0:1])
            + tf.sin(x_in[:, 2:] * np.pi * x_in[:, 1:2])) # * tf.exp(-x_in[:, 2:])


# Right side of heat PDE input
# Dirac delta function (as an example)
def dirac_delta(x_in):
    return tf.exp(-((x_in[:, 0:1] - 0.5) ** 2 + (x_in[:, 1:2] - 0.5) ** 2) / 0.01)


# The heat equation residual for the inverse problem
def inverse_loss(x_in, outputs):
    u = u_obs(x_in)  # Use the observed solution
    print(outputs.shape)
    a = outputs[:, 0:1]  # The output of the neural network for a(x, y)
    u_x = dde.grad.jacobian(u, x_in, i=0, j=0)  # ∂u/∂x
    u_y = dde.grad.jacobian(u, x_in, i=0, j=1)  # ∂u/∂y
    u_t = dde.grad.jacobian(u, x_in, i=0, j=2)  # ∂u/∂t
    flux_x = a * u_x
    flux_y = a * u_y
    flux_xx = dde.grad.jacobian(flux_x, x_in, i=0, j=0)
    flux_yy = dde.grad.jacobian(flux_y, x_in, i=0, j=1)
    f = dirac_delta(x_in)  # Dirac delta as the source term
    return u_t - (flux_xx + flux_yy) - f


# ======================================================================================================================
# TRAININGS PROCESS
# ======================================================================================================================

# Define a neural network to represent the unknown diffusivity a(x, y)
net_a = dde.maps.FNN([3] + [50] * 3 + [1], "tanh", "Glorot uniform")  # Note: 3 inputs (x, y, t)

# Create the inverse problem data
data = dde.data.TimePDE(
    domain,
    inverse_loss,
    [],
    num_domain=4000,
    num_boundary=200,
)

# Define the model for the inverse problem
model = dde.Model(data, net_a)

# Train the model to infer a(x,y)
model.compile("adam", lr=1e-3)
loss_history, train_state = model.train(epochs=10000)

# ======================================================================================================================
# PLOTS
# ======================================================================================================================

# Plot the inferred a(x,y)
dde.saveplot(loss_history, train_state, issave=True, isplot=True)

# Define a grid of points for plotting
num_points = 100
x = np.linspace(0, l, num_points)
y = np.linspace(0, l, num_points)
x_grid, y_grid = np.meshgrid(x, y)
xy = np.hstack((x_grid.flatten()[:, None], y_grid.flatten()[:, None]))

# Evaluate a(x,y) using the trained model
xy_with_zeros = np.hstack((xy, np.zeros((xy.shape[0], 1))))  # Append time=0 for evaluation
a_pred = model.predict(xy_with_zeros).reshape(num_points, num_points)

# Plot a(x, y)
plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, a_pred, levels=50, cmap="viridis")
plt.colorbar(label="$a(x, y)$")
plt.title("$a(x, y)$")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Define time points for u(x, y, t) plotting
times = [0.1, 0.5, 0.9]  # Example time points

# Plot u(x, y, t) for each time
for t in times:
    t_grid = np.full((xy.shape[0], 1), t)  # Create a NumPy array with the time value
    input_array = np.hstack((xy, t_grid))  # Combine x, y, and t
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)  # Convert to TensorFlow tensor

    # Evaluate Tensor within a session
    with tf.Session() as sess:
        u_tensor = u_obs(input_tensor)  # Calculate u(x, y, t) using TensorFlow operations
        u_pred = sess.run(u_tensor).reshape(num_points, num_points)  # Evaluate and reshape

    # Plot u(x, y, t)
    plt.figure(figsize=(8, 6))
    plt.contourf(x_grid, y_grid, u_pred, levels=50, cmap="plasma")
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Solution u(x, y, t) at t = {t}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# # Define the time points for the animation (e.g., 100 time steps between 0 and 1)
# times = np.linspace(0.0, 1.0, 100)  # 100 time points between 0 and 1
#
# # Define the mesh grid for x and y
# num_points = 50  # Adjust this for the resolution of the plot
# x = np.linspace(0, 1, num_points)  # Adjust num_points as needed
# y = np.linspace(0, 1, num_points)
# x_grid, y_grid = np.meshgrid(x, y)
# xy = np.column_stack((x_grid.ravel(), y_grid.ravel()))  # Flatten the grid for evaluation
#
# # Set up formatting for the movie files (using 'avconv' writer)
# Writer = animation.writers['avconv']  # Use AVConvWriter
# writer = Writer(fps=15)
#
# # Set up the figure for the animation
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # Initialize the contour plot
# c = ax.contourf(x_grid, y_grid, np.zeros_like(x_grid), levels=50, cmap="plasma")
# plt.colorbar(c, ax=ax, label="u(x, y, t)")
# ax.set_title('Solution u(x, y, t)')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
#
#
# # Function to update the plot for each frame (time step)
# def update_plot(t):
#     t_grid = np.full((xy.shape[0], 1), t)  # Create a NumPy array with the time value
#     input_array = np.hstack((xy, t_grid))  # Combine x, y, and t
#     input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)  # Convert to TensorFlow tensor
#
#     # Evaluate Tensor within a session
#     with tf.Session() as sess:
#         u_tensor = u_obs(input_tensor)  # Calculate u(x, y, t) using TensorFlow operations
#         u_pred = sess.run(u_tensor).reshape(num_points, num_points)  # Evaluate and reshape
#
#     # Remove old contours
#     for collection in ax.collections:
#         collection.remove()
#
#     # Create the new contour plot
#     c = ax.contourf(x_grid, y_grid, u_pred, levels=50, cmap="plasma")
#     return c.collections  # Return the contour collections to be animated
#
#
# # Create the animation
# ani = animation.FuncAnimation(fig, update_plot, frames=times, interval=100, blit=False)
#
# # Save the animation as an .avi video file
# ani.save('solution_movie.avi', writer=writer)
#
# print("Movie creation complete!")

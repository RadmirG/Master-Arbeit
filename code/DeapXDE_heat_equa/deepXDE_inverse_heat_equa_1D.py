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

import os

# Disables oneDNN optimizations
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import deepxde as dde
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d


# ======================================================================================================================
# DEFINITION OF THE INVERSE PROBLEM AND DOMAIN
# ======================================================================================================================

# # Define the 1D spatial domain
L = 1  # Length of the domain
domain = dde.geometry.Interval(0, L)

#-----------------------------------------------------------------------------------------------------------------------
# Observed data for u(x, t)
def u_obs(xml_file, points_number):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    x_obs = np.array([float(x) for x in root.find("Point").text.split()]).reshape(-1, 1)
    u_obs = np.array([float(u) for u in root.find("PointData").text.split()]).reshape(-1, 1)

    # Randomly select `num_points` while keeping the distribution diverse
    if len(x_obs) > points_number:
        # Evenly spaced selection
        indices = np.linspace(0, len(x_obs) - 1, points_number, dtype=int)
        x_obs = x_obs[indices]
        u_obs = u_obs[indices]

    return x_obs, u_obs

# Load observed data
x_obs, u_obs_values = u_obs("u_1D.xml", 11)

#-----------------------------------------------------------------------------------------------------------------------
# Interpolate u(x)
interp_func = interp1d(x_obs.flatten(), u_obs_values.flatten(), kind='cubic', fill_value="extrapolate")
# Generate fine grid for smooth curve
x_fine = np.linspace(min(x_obs), max(x_obs), 100)
# Compute interpolated values
u_interp = interp_func(x_fine)

f_exact = 1 + 4*x_obs

# Interpolate f(x)
f_interpolated = interp1d(x_obs.flatten(), f_exact.flatten(), kind='cubic', fill_value="extrapolate")

#-----------------------------------------------------------------------------------------------------------------------
# Define the inverse problem loss
@tf.autograph.experimental.do_not_convert
def inverse_loss(x_in, outputs):
    u = outputs[:, 0:1]  # Learned u(x)
    a = outputs[:, 1:2]  # Learned a(x)
    f = outputs[:, 2:3]  # Learned f(x)

    u_t = 0  # ∂u/∂t
    u_x = dde.grad.jacobian(u, x_in)  # ∂u/∂x
    flux_x = a * u_x
    flux_xx = dde.grad.jacobian(flux_x, x_in)  # ∂(a ∂u/∂x)/∂x

    return u_t - flux_xx - f # Residual of the PDE

#-----------------------------------------------------------------------------------------------------------------------
# Define boundary conditions (Dirichlet & Neumann)
@tf.autograph.experimental.do_not_convert
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

@tf.autograph.experimental.do_not_convert
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)

# Dirichlet BC: u(0) = 0, u(1) = 0
bc1 = dde.icbc.DirichletBC(domain, lambda x: 0, boundary_left, component=0)
bc2 = dde.icbc.DirichletBC(domain, lambda x: 0, boundary_right, component=0)

# Neumann BC: u'(0) = 1, u'(1) = -1
bc3 = dde.icbc.NeumannBC(domain, lambda x: 1, boundary_left, component=0)
bc4 = dde.icbc.NeumannBC(domain, lambda x: -1, boundary_right, component=0)

#-----------------------------------------------------------------------------------------------------------------------
# Incorporate observed data constraints
u_bc = dde.icbc.PointSetBC(
    x_obs,
    u_obs_values,
    component=0
)

# Incorporate learned heat diffusivity a(x)
a_bc = dde.icbc.PointSetBC(
    0,
    0,
    component=1
)

# Incorporate learned heat source f(x)
f_bc = dde.icbc.PointSetBC(
    x_obs,
    f_interpolated(x_obs),
    component=2
)

# ======================================================================================================================
# TRAININGS PROCESS
# ======================================================================================================================

# Define the neural network to infer a(x)
pinn = dde.maps.FNN([1] + [50] * 3 + [3], "tanh", "Glorot uniform")
# pinn = dde.maps.FNN([1] + [100] * 5 + [3], "tanh", "Glorot uniform")


# Create training data for PINN
data = dde.data.PDE(
    domain,
    inverse_loss,   # Loss 0: PDE Residual
    [u_bc,          # Loss 1: Observed Data
     #a_bc,          # Loss 2: Heat diffusivity dependency
     f_bc,          # Loss 3: Heat source dependency
     bc1,           # Loss 4: Dirichlet BC : u(0)  =  0
     bc2],          # Loss 5: Dirichlet BC : u(L)  =  0
     #bc3,           # Loss 6: Neumann BC   : u'(0) =  1
     #bc4],          # Loss 7: Neumann BC   : u'(L) = -1
    num_domain=4000,
    num_boundary=2,
)

# Define and train the model
model = dde.Model(data, pinn)
loss_weights=[1, 2, 0.1, 0.5, 0.5]
model.compile("adam", lr=1e-6, loss_weights=loss_weights)
loss_history, train_state = model.train(iterations=10000)
model.compile("L-BFGS-B", loss_weights=loss_weights)
loss_history, train_state = model.train(iterations=5000)

# ======================================================================================================================
# PREPARE PLOTS
# ======================================================================================================================

# Generate and plot the learned functions
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = model.predict(x_test)

a_plot = 1 + x_test
f_plot = f_interpolated(x_test)


# Plotting
plt.figure(figsize=(12, 6))

# Plot a(x)
plt.subplot(2, 2, 1)
plt.plot(x_test, a_plot, label=r"$a(x) = 1 + x$")
plt.plot(x_test, y_pred[:, 1], label=r"$a_{l}(x)$ : learned", color='green')
plt.title(r"Wärmeleitfähigkeit")
plt.xlabel("x")
plt.ylabel(r"$a(x)$")
plt.grid()
plt.legend()

# Plot f(x)
plt.subplot(2, 2, 2)
plt.plot(x_test, f_plot, label=r"$f(x) = 1 + 4x$")
plt.plot(x_test, y_pred[:, 2], label=r"$f_{l}(x)$ : learned", color='green')
plt.title(r"Source Term")
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid()
plt.legend()

# Plot the final solution u(x) and u_exact
plt.subplot(2, 1, 2)
plt.plot(x_test, y_pred[:, 0], label=r"$u_{l}(x)$ : learned", color='green')
plt.scatter(x_obs, u_obs_values, label="Observed u(x)", color='r', s=5)
plt.plot(x_fine, u_interp, color='blue', linestyle='--', label="Interpolated $u(x)$")
plt.title(r"Gelernte und interpolierte Lösungen")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x)$")
plt.grid()
plt.legend()

# Add an overall title
plt.suptitle(r"Implementierung in deapXDE", fontsize=14, fontweight="bold")
plt.tight_layout()
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
    r"$L_{DBC_l}$",
    r"$L_{DBC_r}$"
    #r"$L_{NBC_l}$",
    #r"$L_{NBC_r}$"
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
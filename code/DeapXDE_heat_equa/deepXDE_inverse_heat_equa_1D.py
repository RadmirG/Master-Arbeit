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
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import deepxde as dde
from deepxde.callbacks import Callback
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
def u_obs(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    x_obs = np.array([float(x) for x in root.find("Point").text.split()]).reshape(-1, 1)
    u_obs = np.array([float(u) for u in root.find("PointData").text.split()]).reshape(-1, 1)

    # Randomly select `num_points` while keeping the distribution diverse
    if len(x_obs) > 10:
        # Evenly spaced selection
        indices = np.linspace(0, len(x_obs) - 1, 10, dtype=int)
        x_obs = x_obs[indices]
        u_obs = u_obs[indices]

    return x_obs, u_obs

# Load observed data
x_obs, u_obs_values = u_obs("u_1D.xml")

#-----------------------------------------------------------------------------------------------------------------------
# Interpolate u(x)
interp_func = interp1d(x_obs.flatten(), u_obs_values.flatten(), kind='cubic', fill_value="extrapolate")
# Generate fine grid for smooth curve
x_fine = np.linspace(min(x_obs), max(x_obs), 100)
# Compute interpolated values
u_interp = interp_func(x_fine)


learned_values = {
    "x": x_obs,
    "a": np.random.random(size=x_obs.shape),
    "f": np.random.random(size=x_obs.shape)
}

def f_source(x_in):
    return 1 + 8*x_in

#-----------------------------------------------------------------------------------------------------------------------
# Define the inverse problem loss
@tf.autograph.experimental.do_not_convert
def inverse_loss(x_in, outputs):
    u = outputs[:, 0:1]  # Learned u(x)
    a = outputs[:, 1:2]  # Learned a(x)
    f = outputs[:, 2:3]  # Learned f(x)

    # Save a(x) and f(x) at x_obs for later use
    learned_values["x"] = x_in
    learned_values["a"] = outputs[:, 1:2]
    learned_values["f"] = outputs[:, 2:3]

    u_t = 0  # ∂u/∂t
    u_x = dde.grad.jacobian(u, x_in)  # ∂u/∂x
    flux_x = a * u_x
    flux_xx = dde.grad.jacobian(flux_x, x_in)  # ∂(a ∂u/∂x)/∂x

    return u_t - flux_xx - f # Residual of the PDE

#-----------------------------------------------------------------------------------------------------------------------
# Define boundary conditions (Dirichlet & Neumann)
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

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

class UpdateLearnedValuesCallback(Callback):
    def on_epoch_end(self):
        # Ensure values are updated at training points
        global learned_values
        #if learned_values["a"] is not None and learned_values["f"] is not None:
        #    print("Updated learned values a(x) and f(x)")


# Incorporate learned heat diffusivity a(x)
a_bc = dde.icbc.PointSetBC(
    learned_values["x"],
    learned_values["a"],
    component=1
)

# Incorporate learned heat source f(x)
f_bc = dde.icbc.PointSetBC(
    x_obs,
    f_source(x_obs),
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
loss_weights=[1, 2, 2, 0.5, 0.5] #, 0.5, 0.5]
model.compile("adam", lr=1e-5, loss_weights=loss_weights) # checkpoint_cb = dde.callbacks.ModelCheckpoint("checkpoints/model.keras", save_better_only=True)
loss_history, train_state = model.train(iterations=10000, callbacks=[UpdateLearnedValuesCallback()]) #,checkpoint_cb])
model.compile("L-BFGS-B", loss_weights=loss_weights)
loss_history, train_state = model.train(iterations=5000, callbacks=[UpdateLearnedValuesCallback()])

# ======================================================================================================================
# FILTERING FOR BEST SAVED MODEL
# ======================================================================================================================

# Load the best checkpoint after training
# ckpt = tf.train.get_checkpoint_state(".")  # Check current directory
#
# if ckpt and ckpt.model_checkpoint_path:
#     best_checkpoint = ckpt.model_checkpoint_path
#     print("Best checkpoint found:", best_checkpoint)
# else:
#     print("No checkpoint found.")
#
# model.restore(best_checkpoint)
# Find and load the best checkpoint
# # checkpoint_files = glob.glob("checkpoints/model.keras*")
# #
# # if checkpoint_files:
# #     best_checkpoint = max(checkpoint_files, key=os.path.getctime)
# #     print(f"Restoring best checkpoint: {best_checkpoint}")
# #     model.restore(best_checkpoint)
# # else:
# #     print("No checkpoint found, training from scratch.")

# # Get all checkpoint-related files
# all_ckpt_files = (glob.glob("checkpoints/model.keras*"))
# best_ckpt_prefix = best_checkpoint.split("\\")[-1]  # Extracts only the filename
# # Delete all except the best ones
# for file in all_ckpt_files:
#     if not any(file.endswith(ext) and best_ckpt_prefix in file for ext in [".index", ".meta", ".data-00000-of-00001"]):
#         os.remove(file)
#         print(f"Deleted: {file}")
# print("Cleanup complete! Only the best checkpoint is retained.")


# ======================================================================================================================
# PREPARE PLOTS
# ======================================================================================================================

# Generate and plot the learned functions
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = model.predict(x_test)

a_exact = 1 + x_test
f_exact = 1 + 4*x_test


# Plotting
plt.figure(figsize=(12, 6))

# Plot a(x)
plt.subplot(2, 2, 1)
plt.plot(x_test, a_exact, label=r"$a(x) = 1 + x$")
plt.plot(x_test, y_pred[:, 1], label=r"$a_{l}(x)$ : learned", color='green')
plt.title(r"Wärmeleitfähigkeit")
plt.xlabel("x")
plt.ylabel(r"$a(x)$")
plt.grid()
plt.legend()

# Plot f(x)
plt.subplot(2, 2, 2)
plt.plot(x_test, f_exact, label=r"$f(x) = 1 + 4x$")
plt.plot(x_test, y_pred[:, 2], label=r"$f_{l}(x)$ : learned", color='orange')
plt.title(r"Source Term")
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid()
plt.legend()

# Plot the final solution u(x) and u_exact
plt.subplot(2, 1, 2)
plt.plot(x_test, y_pred[:, 0], label=r"$u_{l}(x)$ : learned", color='green')
plt.scatter(x_obs, u_obs_values, label="Observed u(x)", color='r', s=5)
plt.plot(x_fine, u_interp, color='blue', linestyle='--', label="Interpolated Curve")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x)$")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# ======================================================================================================================
# LOSSES PLOT
# ======================================================================================================================
# dde.saveplot(loss_history, train_state, issave=True, isplot=True) # standard data loss plot, I guess

# Define loss labels based on the given loss indices
loss_labels = [
    "PDE Residual (inverse_loss)",
    "Observed Data (ic_bc)",
    #"Heat diffusivity dependency",
    "Heat source dependency",
    "Dirichlet BC : u(0) = 0",
    "Dirichlet BC : u(1) = 0",
    "Neumann BC (u'(0) = 1)",
    "Neumann BC (u'(1) = -1)"
]

# Convert loss_history.loss_train to a NumPy array for easier manipulation
loss_array = np.array(loss_history.loss_train)  # Shape: (epochs, num_losses)

plt.figure(figsize=(8, 5))
for i in range(loss_array.shape[1]):  # Iterate over loss components
    plt.semilogy(loss_history.steps, loss_array[:, i], label=loss_labels[i])

plt.grid()
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss")
plt.show()

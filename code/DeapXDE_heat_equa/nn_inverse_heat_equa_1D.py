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
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import xml.etree.ElementTree as ET

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ======================================================================================================================
# PREPARE OBSERVED DATA
# ======================================================================================================================

# load observed data for u(x, t)
def u_obs(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    x_obs = np.array([float(x) for x in root.find("Point").text.split()]).reshape(-1, 1)
    u_obs = np.array([float(u) for u in root.find("PointData").text.split()]).reshape(-1, 1)

    # Randomly select `num_points` while keeping the distribution diverse
    if len(x_obs) > 10:
        indices = np.linspace(0, len(x_obs) - 1, 10, dtype=int)
        x_obs = x_obs[indices]
        u_obs = u_obs[indices]

    return x_obs, u_obs


# Load observed data
x_obs, u_obs_values = u_obs("u_1D.xml")
# f(x) exact
f_exact = 1 + 4 * x_obs

# Dirichlet BC
u_0 = 0
u_L = 0
d_bc = [u_0, u_L]
d_bc_tf = tf.convert_to_tensor(d_bc, dtype=tf.float32)

# Neumann BC
du_0 = 1
du_L = -1
n_bc = [du_0, du_L]
n_bc_tf = tf.convert_to_tensor(n_bc, dtype=tf.float32)

# Generate fine grid for smooth curve
x_fine = np.linspace(min(x_obs), max(x_obs), 100)

# Interpolate u(x)
u_interpolated = interp1d(x_obs.flatten(), u_obs_values.flatten(), kind='cubic', fill_value="extrapolate")
# Interpolate f(x)
f_interpolated = interp1d(x_obs.flatten(), f_exact.flatten(), kind='cubic', fill_value="extrapolate")


# ======================================================================================================================
# COMPUTE LOSSES
# ======================================================================================================================

# Compute gradients
def compute_gradients(a, u, x, tape):
    tape.watch(x)
    u_x = tape.gradient(u, x)
    u_flux = tape.gradient(a * u_x, x)  # d/dx (a*u/dx)
    return u_x, u_flux


def pde_loss(x, outputs, tape):
    u = outputs[:, 0:1]  # u(x)
    a = outputs[:, 1:2]  # a(x)
    f = outputs[:, 2:3]  # f(x)

    # Compute gradients
    u_x, u_flux = compute_gradients(a, u, x, tape)

    # PDE residual
    residual = - u_flux - f
    return tf.reduce_mean(tf.square(residual)), u_x, u_flux


def data_loss(x, outputs):
    u = outputs[:, 0:1]
    a = outputs[:, 1:2]
    f = outputs[:, 2:3]

    u_interp = u_interpolated(x.numpy())
    f_interp = f_interpolated(x.numpy())

    loss_u = tf.reduce_mean(tf.square(u - u_interp))
    loss_a = tf.reduce_mean(tf.square(a - a))  # 'a' is to replace (maybe)
    loss_f = tf.reduce_mean(tf.square(f - f_interp))

    return [loss_u, loss_a, loss_f]

## das hier splitten!
def conditions_loss(u, u_x):
    loss_d_bc = tf.reduce_mean(tf.square([u[0].numpy(), u[-1].numpy()] - d_bc_tf))
    loss_n_bc = tf.reduce_mean(tf.square([u_x[0].numpy(), u_x[-1].numpy()] - n_bc_tf))
    return [loss_d_bc, loss_n_bc]


# General loss function
def loss_pinn(x, outputs, loss_weights, tape):
    loss_pde, u_x, u_flux = pde_loss(x, outputs, tape)
    loss_bc = conditions_loss(outputs[:, 0:1], u_x)
    loss_d_bc = loss_bc[0]
    loss_n_bc = loss_bc[1] * 0
    loss_data = data_loss(x, outputs)
    loss_u = loss_data[0]
    loss_a = loss_data[1]
    loss_f = loss_data[2]
    loss_general = (loss_weights[0] * loss_pde +
                    loss_weights[1] * loss_d_bc +
                    loss_weights[2] * loss_n_bc +
                    loss_weights[3] * loss_u +
                    loss_weights[4] * loss_a +
                    loss_weights[5] * loss_f)
    return loss_pde, loss_d_bc, loss_n_bc, loss_u, loss_a, loss_f, loss_general


# Training function
def train_pinn(model, x_train, iterations, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    losses = []     # [1, 2, 2, 0.5, 0.5]
    loss_weights = [1, 0.5, 0, 2, 0, 2]
    for epoch in range(iterations):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(x_train)  # NN outputs [u, a, f]
            general_loss = loss_pinn(x_train, outputs, loss_weights, tape)
        gradients = tape.gradient(general_loss[6], model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append([loss.numpy() for loss in general_loss[:6]])
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: [{', '.join(f'{float(loss):.3e}' for loss in general_loss[:6])}]")

        del tape

    return losses


# Define domain
domain_x = [0, 1]


# Generate training data
def generate_training_data(num_samples):
    x = np.random.uniform(domain_x[0], domain_x[1], num_samples)
    x = tf.convert_to_tensor(x[:, None], dtype=tf.float32)
    return x


# Neural network for approximating a(u), f(x), and u(x)
def create_nn(input_dim, output_dim, num_layers=3, num_neurons=50):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(num_neurons, activation="tanh"))
    model.add(tf.keras.layers.Dense(output_dim))
    return model


# Visualization
def visualize_solution(model, num_points=100):
    x = np.linspace(domain_x[0], domain_x[1], num_points)
    x_tf = tf.convert_to_tensor(x[:, None], dtype=tf.float32)
    outputs = model.predict(x_tf)
    u_pred = outputs[:, 0]
    a_pred = outputs[:, 1]
    f_pred = outputs[:, 2]

    # exact functions
    a_exact = 1 + x
    f_exact = 1 + 4 * x

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot a(x)
    plt.subplot(2, 2, 1)
    plt.plot(x, a_exact, label=r"$a(x) = 1 + x$")
    plt.plot(x, a_pred, label=r"$a_{l}(x)$ : learned", color='green')
    plt.title(r"Wärmeleitfähigkeit")
    plt.xlabel("x")
    plt.ylabel(r"$a(x)$")
    plt.grid()
    plt.legend()

    # Plot f(x)
    plt.subplot(2, 2, 2)
    plt.plot(x, f_exact, label=r"$f(x) = 1 + 4x$")
    plt.plot(x, f_pred, label=r"$f_{l}(x)$ : learned", color='green')
    plt.title(r"Source Term")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.grid()
    plt.legend()

    # Plot the final solution u(x) and u_exact
    plt.subplot(2, 1, 2)
    plt.plot(x, u_pred, label=r"$u_{l}(x)$ : learned", color='green')
    plt.scatter(x_obs, u_obs_values, label="Observed u(x)", color='r', s=5)
    plt.plot(x, u_interpolated(x), color='blue', linestyle='--', label="Interpolated $u(x)$")
    plt.title(r"Gelernte und interpolierte Lösungen")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main script
if __name__ == "__main__":
    start_time = time.time()

    pinn_model = create_nn(input_dim=1, output_dim=3)
    num_samples = 100
    iterations_number = 10000
    x_train = generate_training_data(num_samples)
    losses = train_pinn(pinn_model, x_train, iterations=iterations_number, learning_rate=1e-3)

    end_time = time.time()
    training_time = end_time - start_time
    # Convert to hours, minutes, seconds, and milliseconds
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    milliseconds = int((training_time % 1) * 1000)

    # Print the training time in the format HH:MM:SS.MMM
    print(f"Training time: {hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")

    visualize_solution(pinn_model)

# ======================================================================================================================
# LOSSES PLOT
# ======================================================================================================================

    # Define loss labels based on the given loss indices
    # loss_pde, loss_d_bc, loss_n_bc, loss_u, loss_a, loss_f
    loss_labels = [
        "PDE Residual (inverse_loss)",
        "Dirichlet BC",
        "Neumann BC",
        "Observed Data (ic_bc)",
        "Heat diffusivity dependency",
        "Heat source dependency",
    ]

    # Convert list to a NumPy array
    loss_array = np.array(losses).T  # Shape: (5000, 6)
    steps = np.arange(loss_array.shape[1])  # Number of training iterations
    plt.figure(figsize=(8, 5))
    for i in range(loss_array.shape[0]):  # Iterate over loss components
        plt.semilogy(steps, loss_array[i], label=loss_labels[i])

    plt.grid()
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss")
    plt.show()
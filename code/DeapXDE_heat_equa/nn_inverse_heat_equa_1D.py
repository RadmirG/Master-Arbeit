
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the domain and time range
domain_x = [0, 1]
domain_y = [0, 1]
time_range = [0.1, 1.0]  # t > 0


# Fundamental solution of the heat equation
def fundamental_solution(xyt):
    return (1 / (4 * np.pi * xyt[..., 2])) * tf.exp(-(xyt[..., 0] ** 2 + xyt[..., 1] ** 2) / (4 * xyt[..., 2]))


# Neural network for approximating a(u) and f(x, y, t)
def create_nn(input_dim, output_dim, num_layers=3, num_neurons=50):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(num_neurons, activation="tanh"))
    model.add(tf.keras.layers.Dense(output_dim))
    return model


# Compute gradients (helper function)
#@tf.function
def compute_gradients(a, u, xyt):
    # with tf.GradientTape(persistent=True) as tape:
    #     tape.watch(xyt)
    #     u_x = tape.gradient(u, xyt)[:, 0:1]  # du/dx
    #     if u_x is None:
    #         raise ValueError("Gradient computation failed. Check u and xyt.")
    #     u_y = tape.gradient(u, xyt)[:, 1:2]  # du/dy
    #     u_t = tape.gradient(u, xyt)[:, 2:3]  # du/dt
    #
    # flux_xx = tape.gradient(a * u_x, xyt)[:, 0:1]  # d²u/dx² * (a*du/dx)
    # flux_yy = tape.gradient(a * u_y, xyt)[:, 1:2]  # d²u/dy² * (a*du/dy)
    # del tape
    # return u_x, u_y, u_t, flux_xx, flux_yy
    # Ensure persistent tape and explicit dependencies
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xyt)  # Ensure that xyt is tracked
        u_pred = fundamental_solution(xyt)  # Direct dependency introduced
        u_x = tape.gradient(u_pred, xyt)[:, 0:1]  # du/dx
        u_y = tape.gradient(u_pred, xyt)[:, 1:2]  # du/dy
        u_t = tape.gradient(u_pred, xyt)[:, 2:3]  # du/dt

        # Ensure gradients are valid for computations of flux
        flux_xx = tape.gradient(a * u_x, xyt)[:, 0:1]  # d²u/dx² * (a*du/dx)
        flux_yy = tape.gradient(a * u_y, xyt)[:, 1:2]  # d²u/dy² * (a*du/dy)
    # Clean up tape
    del tape

    return u_x, u_y, u_t, flux_xx, flux_yy

# Loss function for the inverse problem
def inverse_loss(model, xyt, outputs, u_obs):

    #xyt = tf.random.uniform((10, 3), dtype=tf.float32)
    u = fundamental_solution(xyt)
    a = tf.ones_like(u)
    u_x, u_y, u_t, flux_xx, flux_yy = compute_gradients(a, u, xyt)

    u = u_obs(xyt)  # Use the fundamental solution for u(x, y, t)
    a = outputs[:, 0:1]  # a(u(x, y, t))
    f = outputs[:, 1:2]  # f(x, y, t)

    # Compute gradients of u
    u_x, u_y, u_t, flux_xx, flux_yy = compute_gradients(a, u, xyt)

    # Residual of the PDE
    residual = u_t - (flux_xx + flux_yy) - f
    loss = tf.reduce_mean(tf.square(residual))
    return loss


# Training function
def train_pinn(model, u_obs, xyt, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    losses = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            outputs = model(xyt)  # NN outputs [a, f]
            loss = inverse_loss(model, xyt, outputs, u_obs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append(loss.numpy())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    return losses


# Generate training data
def generate_training_data(num_samples):
    x = np.random.uniform(domain_x[0], domain_x[1], num_samples)
    y = np.random.uniform(domain_y[0], domain_y[1], num_samples)
    t = np.random.uniform(time_range[0], time_range[1], num_samples)
    xyt = np.stack((x, y, t), axis=1)
    return tf.convert_to_tensor(xyt, dtype=tf.float32)


# Visualization function for a(u) and f(x, y, t)
def visualize_solution(model, num_points=100, t_fixed=0.5):
    x = np.linspace(domain_x[0], domain_x[1], num_points)
    y = np.linspace(domain_y[0], domain_y[1], num_points)

    x_grid, y_grid = np.meshgrid(x, y)
    xyt = np.hstack((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), np.full_like(x_grid.reshape(-1, 1), t_fixed)))
    outputs = model.predict(xyt)
    a_pred = outputs[:, 0:1]
    f_pred = outputs[:, 1:2]

    plt.figure(figsize=(8, 6))
    plt.contourf(x_grid, y_grid, a_pred.reshape(num_points, num_points), levels=50, cmap="viridis")
    plt.colorbar(label="a(u(x, y, t))")
    plt.title("Predicted a(u(x, y, t))")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.contourf(x_grid, y_grid, f_pred.reshape(num_points, num_points), levels=50, cmap="viridis")
    plt.colorbar(label="f(x, y, t)")
    plt.title("Predicted f(x, y, t)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# Main script
if __name__ == "__main__":
    # Define the model (outputs [a(u), f])
    pinn_model = create_nn(input_dim=3, output_dim=2)

    # Generate synthetic training data
    num_samples = 1000
    xyt_train = generate_training_data(num_samples)

    # Fundamental solution as a known u(x, y, t)
    u_obs = lambda xyt: fundamental_solution(xyt)

    # Train the PINN
    losses = train_pinn(pinn_model, u_obs, xyt_train, epochs=2000, learning_rate=1e-3)

    # Visualize the loss propagation
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Propagation")
    plt.show()

    # Visualize the learned a(u) and f(x, y, t)
    visualize_solution(pinn_model)

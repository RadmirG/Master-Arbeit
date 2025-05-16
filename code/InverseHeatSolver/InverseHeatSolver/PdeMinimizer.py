import os

import tensorflow as tf
import numpy as np

from InverseHeatSolver.History import History


class PdeMinimizer(tf.keras.Model):
    def __init__(self, u_model, f_model=None, input_dim=1,
                 nn_dims={'num_layers': 2, 'num_neurons': 20}, lr=0.01,
                 time_dependent=False, two_dim=False):
        super().__init__()
        self.u_model = u_model
        self.f_model = f_model
        self.time_dependent = time_dependent
        self.two_dim = two_dim
        self.history = History()

        # build model
        self.a_model = tf.keras.Sequential()
        self.a_model.add(tf.keras.Input(shape=(input_dim,)))
        for _ in range(nn_dims['num_layers']):
            self.a_model.add(tf.keras.layers.Dense(nn_dims['num_neurons'], activation="tanh"))
        self.a_model.add(tf.keras.layers.Dense(1, activation="softplus"))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def call(self, inputs):
        if not self.two_dim and not self.time_dependent:
            a = self.a_model(inputs)
        elif not self.two_dim and self.time_dependent:
            x = inputs[:, :1]
            a = self.a_model(x)
        elif self.two_dim and not self.time_dependent:
            a = self.a_model(inputs)
        else:
            xy = inputs[:, :2]
            a = self.a_model(xy)
        return a

    def predict(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return self(x).numpy()

    def get_network(self):
        return self.a_model

    def pde_loss(self, inputs, inv_weight, tape):
        a = self(inputs)

        if self.u_model is None:
            raise ValueError("PdeMinimizer::inverse-loss() : u_model must be provided")
        u = self.u_model(inputs)
        if self.f_model is not None:
            f = self.f_model(inputs)
        else:
            f = 0.0

        # === Compute derivatives ===
        u_t = 0.0
        flux_yy = 0.0
        if self.two_dim:
            grads = tape.gradient(u, inputs)  # <=> (du/dx, du/dy)
            flux = tape.gradient(a * grads[:, 0:2], inputs)
            flux_xx = flux[:, 0:1]
            flux_yy = flux[:, 1:2]
            if self.time_dependent:
                u_t = grads[:, 2:3]
        else:
            grads = tape.gradient(u, inputs)
            flux_x = a * grads[:, 0:1]
            flux_xx = tape.gradient(flux_x, inputs)
            if self.time_dependent:
                u_t = grads[:, 1:2]

        res = u_t - (flux_xx + flux_yy) - f
        output = inv_weight * tf.reduce_mean(tf.square(res))

        return output

    def train(self, obs_domain, loss_weights, iterations=5000, print_every=100, regularize=False, best_loss=1e-1):
        if not isinstance(obs_domain, np.ndarray):
            raise TypeError("x_obs must be a NumPy array.")
        train_domain = tf.convert_to_tensor(obs_domain, dtype=tf.float32)

        best_losses = []
        best_weights = []
        best_steps = []

        for iteration in range(iterations):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(train_domain)
                losses = self.compute_losses(train_domain, loss_weights, tape, regularize)
            grads = tape.gradient(losses[0], self.a_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.a_model.trainable_variables))

            if iteration % print_every == 0 or iteration == iterations - 1:
                self.history.append_loss(losses[0].numpy())
                self.history.append_pde_loss(losses[1].numpy())
                self.history.steps.append(iteration + 1)
                tf.print(f"Step {iteration}: Loss = {losses[0]:.8f}")

            # Logic to hold the best models
            loss_val = losses[0].numpy()
            if loss_val < best_loss:
                best_loss = loss_val
                best_losses.append(best_loss)
                best_weights.append(self.a_model.get_weights())
                best_steps.append(iteration + 1)

            del tape

        # takes only the best model for saving
        if best_losses:
            min_index = np.argmin(best_losses)
            best_step = best_steps[min_index]
            tf.print(f"Best model on step {best_step} with loss = {best_losses[min_index]:.8f}")
            best_weights = best_weights[min_index]
            self.a_model.set_weights(best_weights)

        return self.history

    def compute_losses(self,train_domain, loss_weights, tape, regularize=False):
        pde_loss = self.pde_loss(train_domain, loss_weights['w_PDE_loss'], tape)
        a_grad_loss = tf.zeros_like(pde_loss)
        if regularize:
            # a_grad_loss = self.a_loss(train_domain, loss_weights['a_grad_loss'], tape).numpy()
            None
        loss = pde_loss + a_grad_loss
        return loss, pde_loss, a_grad_loss

    def save(self, save_dir, name="a_model.weights.h5"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.a_model.save_weights(os.path.join(save_dir, name))

    def restore(self, save_dir, name):
        if os.path.exists(save_dir):
            self.a_model.load_weights(os.path.join(save_dir, name))

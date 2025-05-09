import os
import numpy as np
import tensorflow as tf


class Interpolator(tf.keras.Model):
    def __init__(self, input_dim, hidden_units=20, lr=0.01, positive_output=False):
        super().__init__()
        layers = [
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_units, activation='tanh'),
            tf.keras.layers.Dense(hidden_units, activation='tanh'),
            tf.keras.layers.Dense(1)
        ]
        if positive_output:
            layers.append(tf.keras.layers.Activation('softplus'))

        self.model = tf.keras.Sequential(layers)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def call(self, x):
        return self.model(x)

    def fit(self, x_obs, y_obs, iterations=5000, print_every=100):
        # inputs are allways numpy vectors
        if not isinstance(x_obs, np.ndarray) or not isinstance(y_obs, np.ndarray):
            raise TypeError("Inputs x_obs and y_obs must be NumPy arrays.")
        x_train = tf.convert_to_tensor(x_obs, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_obs, dtype=tf.float32)

        history = []
        steps = []

        # temporals
        best_losses = []
        best_weights = []
        best_steps = []
        best_loss = 1e-2

        for iteration in range(iterations):
            with tf.GradientTape() as tape:
                y_pred = self(x_train)
                loss = self.loss_fn(y_train, y_pred)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            if iteration % print_every == 0 or iteration == iterations - 1:
                history.append(loss.numpy())
                steps.append(iteration+1)
                tf.print(f"Step {iteration}: Loss = {loss:.8f}")
            # Logic to hold the best models
            loss_val = loss.numpy()
            if loss_val < best_loss:
                best_loss = loss_val
                best_losses.append(best_loss)
                best_weights.append(self.model.get_weights())
                best_steps.append(iteration + 1)

            del tape

        # takes only the best model for saving
        if best_losses:
            min_index = np.argmin(best_losses)
            best_step = best_steps[min_index]
            tf.print(f"Best model on step {best_step} with loss = {best_losses[min_index]:.8f}")
            best_model = best_weights[min_index]
            self.model.set_weights(best_model)

        return history, steps

    def predict(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return self(x).numpy()

    def get_network(self):
        return self.model

    def save(self, save_dir, name="model.weights.h5"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_weights(os.path.join(save_dir, name))

    def restore(self, save_dir, name):
        if os.path.exists(save_dir):
            self.model.load_weights(os.path.join(save_dir, name))

    def l2_norm(self, x, y_exact):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y_exact = tf.convert_to_tensor(y_exact, dtype=tf.float32)
        pred = self(x)
        return tf.sqrt(tf.reduce_sum(tf.square(pred - y_exact))) / tf.cast(tf.shape(pred)[0], tf.float32)

import os
import numpy as np
import tensorflow as tf
from .History import History

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# seed = 42  # Can be any integer seed
# np.random.seed(seed)
# tf.random.set_seed(seed)

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
        self.rar_points = []  # Stores RAR-added points for later visualization

        # Build model
        self.a_model = tf.keras.Sequential()
        self.a_model.add(tf.keras.Input(shape=(input_dim,)))
        for _ in range(nn_dims['num_layers']):
            self.a_model.add(tf.keras.layers.Dense(nn_dims['num_neurons'], activation="tanh"))
        self.a_model.add(tf.keras.layers.Dense(1, activation="softplus"))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def call(self, inputs):
        if not self.time_dependent:
            return self.a_model(inputs)
        else:
            return self.a_model(inputs[:, :2] if self.two_dim else inputs[:, :1])

    def predict(self, x):
        return self(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()

    def get_network(self):
        return self.a_model

    def pde_residual(self, inputs, tape):
        a = self(inputs)
        u = self.u_model(inputs)
        f = self.f_model(inputs) if self.f_model is not None else 0.0

        u_t, flux_yy = 0.0, 0.0
        if self.two_dim:
            grads = tape.gradient(u, inputs)
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

        return u_t - (flux_xx + flux_yy) - f

    def pde_loss(self, residual, pde_loss_weight):
        return pde_loss_weight * tf.reduce_mean(tf.square(residual))

    def a_grad_loss(self, inputs, a_grad_loss_weight, tape):
        a = self(inputs)
        if not self.time_dependent:
            a_grad = tape.gradient(a, inputs)
        else:
            a_grad = tape.gradient(a, inputs[:, :2] if self.two_dim else inputs[:, :1])
        return a_grad_loss_weight * tf.reduce_mean(tf.square(a_grad))

    def gPINN_loss(self, residual, inputs, gPINN_loss_weight, tape):
        grad_res = tape.gradient(residual, inputs)
        grad_loss = tf.reduce_mean(tf.square(grad_res))
        return gPINN_loss_weight * grad_loss

    def compute_losses(self, inputs, loss_weights, tape, use_regularization=False, use_gPINN=False):
        residual = self.pde_residual(inputs, tape)
        pde_loss = self.pde_loss(residual, loss_weights['w_PDE_loss'])
        a_loss = self.a_grad_loss(inputs, loss_weights['a_grad_loss'], tape) if use_regularization else 0.0
        gpinn_loss = self.gPINN_loss(residual, inputs, loss_weights['gPINN_loss'], tape) if use_gPINN else 0.0
        total_loss = pde_loss + a_loss + gpinn_loss
        return total_loss, pde_loss, a_loss, gpinn_loss

    # RAR (Residual-based Adaptive Refinement) with logging of new points
    def rar(self, train_points, RAR_points_m=100):
        random_candidates = np.random.uniform(
            np.min(train_points, axis=0),
            np.max(train_points, axis=0),
            size=(train_points.shape[0], train_points.shape[1])
        )
        random_candidates_tf = tf.convert_to_tensor(random_candidates, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as rar_tape:
            rar_tape.watch(random_candidates_tf)
            residual = self.pde_residual(random_candidates_tf, rar_tape)
        residual_grad = rar_tape.gradient(residual, random_candidates_tf)
        grad_norm = tf.norm(residual_grad, axis=1)
        top_m_indices = tf.argsort(grad_norm, direction='DESCENDING')[:RAR_points_m]
        new_points = tf.gather(random_candidates_tf, top_m_indices).numpy()
        self.rar_points.append(new_points)
        updated_train_points = np.concatenate([train_points, new_points], axis=0)
        return updated_train_points

    def resample_train_points(self, train_points):
        new_point_set = np.random.uniform(np.min(train_points, axis=0), np.max(train_points, axis=0),
                                          size=(train_points.shape[0], train_points.shape[1]))

        if len(self.rar_points) > 0:
            rar_points_arr = np.concatenate(self.rar_points, axis=0)
            updated_points = np.concatenate([new_point_set, rar_points_arr], axis=0)
        else:
            updated_points = new_point_set
        return tf.convert_to_tensor(updated_points, dtype=tf.float32)

        #updated_points = np.concatenate([new_point_set, self.rar_points], axis=0)
        #return tf.convert_to_tensor(updated_points, dtype=tf.float32)

    def train(self, domain, loss_weights, iterations=5000, print_every=100, best_loss=1e-1,
              use_regularization=False, use_gPINN=False, use_RAR=False, RAR_cycles_n=500, RAR_points_m=100):
        if not isinstance(domain, np.ndarray):
            raise TypeError("x_obs must be a NumPy array.")
        train_domain = tf.convert_to_tensor(domain, dtype=tf.float32)

        best_losses = []
        best_weights = []
        best_steps = []

        for iteration in range(iterations):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(train_domain)
                losses = self.compute_losses(train_domain, loss_weights, tape, use_regularization, use_gPINN)
            grads = tape.gradient(losses[0], self.a_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.a_model.trainable_variables))

            # Residual-based Adaptive Refinement (RAR)
            if use_RAR and (RAR_cycles_n > 0) and (iteration + 1) % RAR_cycles_n == 0:
                train_domain = self.rar(train_domain.numpy(), RAR_points_m)
                train_domain = tf.convert_to_tensor(train_domain, dtype=tf.float32)

            if iteration % print_every == 0 or iteration == iterations - 1:
                # Test
                train_domain = self.resample_train_points(train_domain.numpy())

                self.history.append_loss(losses[0].numpy())
                self.history.append_pde_loss(losses[1].numpy())
                if use_regularization:
                    self.history.append_a_grad_loss(losses[2].numpy())
                if use_gPINN:
                    self.history.append_gPINN_loss(losses[3].numpy())
                self.history.steps.append(iteration + 1)
                tf.print(f"Step {iteration}: Loss = {losses[0]:.8f}")

            # Save best weights
            loss_val = losses[0].numpy()
            if loss_val < best_loss:
                best_loss = loss_val
                best_losses.append(best_loss)
                best_weights.append(self.a_model.get_weights())
                best_steps.append(iteration + 1)

            del tape

        # Set weights to best found
        if best_losses:
            min_index = np.argmin(best_losses)
            best_step = best_steps[min_index]
            tf.print(f"Best model on step {best_step} with loss = {best_losses[min_index]:.8f}")
            best_weights = best_weights[min_index]
            self.a_model.set_weights(best_weights)

        return self.history

    def save(self, save_dir, name="a_model.weights.h5"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.a_model.save_weights(os.path.join(save_dir, name))

    def restore(self, save_dir, name):
        if os.path.exists(save_dir):
            self.a_model.load_weights(os.path.join(save_dir, name))

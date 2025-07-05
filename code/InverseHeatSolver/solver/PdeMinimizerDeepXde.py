# ======================================================================================================================
# This class handles the second step of training using DeepXDE
#   2. Depending on the chosen solver (PdeMinimizer or PdeMinimizerDeepXde) the minimization of L will be solved.
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================

import os

import deepxde as dde
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

from .InputAdapter import InputAdapter
from .History import History


class PdeMinimizerDeepXde:
    def __init__(self, domain, obs_domain, u_model, f_model=None, input_dim=1,
                 nn_dims={'num_layers': 2, 'num_neurons': 20}, lr=0.01,
                 time_dependent=False, two_dim=False, reg_weight=1e-4):
        self.gPINN_weight = 0.0
        self.regularization_weight = 0.0
        self.use_gPINN = False
        self.use_regularization = False
        self.a_model = None
        self.pinn_domain = None
        self.u_model = u_model
        self.f_model = f_model
        self.time_dependent = time_dependent
        self.two_dim = two_dim
        self.history = History()
        self.input_dim = input_dim
        self.nn_dims = nn_dims
        self.prepare_domain(domain)

        self.obs_domain = obs_domain
        self.num_domain = 0

        self.prepare_model(obs_domain, reg_weight)
        self.learning_rate = lr

    def prepare_domain(self, domain):
        x_start, x_end = domain['x_domain'][:2]
        self.num_domain = domain['x_domain'][2]
        geometry_domain = dde.geometry.Interval(x_start, x_end)
        if self.two_dim:
            y_start, y_end = domain['y_domain'][:2]
            geometry_domain = dde.geometry.Rectangle([x_start, y_start], [x_end, y_end])
            self.num_domain = self.num_domain * domain['y_domain'][2]
        if self.time_dependent:
            t_start, t_end = domain['t_domain'][:2]
            time_domain = dde.geometry.TimeDomain(t_start, t_end)
            self.pinn_domain = dde.geometry.GeometryXTime(geometry_domain, time_domain)
            self.num_domain = self.num_domain * domain['t_domain'][2]
        else:
            self.pinn_domain = geometry_domain

    def prepare_model(self, obs_domain, reg_weight=1e-4):
        if self.time_dependent:
            data = dde.data.TimePDE(self.pinn_domain, self.pde_loss, [], num_domain=self.num_domain,
                                    num_boundary=5000, num_initial=5000, anchors=obs_domain, num_test=10000)
        else:
            data = dde.data.PDE(self.pinn_domain, self.pde_loss, [], num_domain=self.num_domain, num_boundary=1000,
                                anchors=obs_domain, num_test=1000)

        dim = (self.input_dim - 1) if self.time_dependent else self.input_dim
        a_net = dde.nn.FNN([dim]
                           + [self.nn_dims['num_neurons']] * self.nn_dims['num_layers']
                           + [1], "tanh", "Glorot normal")

        reg = dde.nn.regularizers.get(("l1l2", reg_weight, reg_weight))
        for layer in a_net.layers:
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = reg
                tf.print(layer)

        composite_net = InputAdapter(a_net, self.time_dependent, self.two_dim)
        self.a_model = dde.Model(data, composite_net)

    def predict(self, inputs):
        if not self.time_dependent:
            return self.a_model.predict(inputs)
        else:
            return self.a_model.predict(inputs[:, :2] if self.two_dim else inputs[:, :1])

    def get_network(self):
        return self.a_model.net

    def pde_loss(self, x_in, outputs):
        if self.u_model is not None:
            u = self.u_model(x_in)  # Interpolated u(x)
        else:
            raise ValueError('Keras Model NN for u(.) is None')

        if self.f_model is not None:
            f = self.f_model(x_in)  # Interpolated f(x)
        else:
            f = tf.zeros_like(u)

        a = outputs

        if not self.two_dim:
            u_x = dde.grad.jacobian(u, x_in, i=0, j=0)  # ∂u/∂x
            flux_x = a * u_x
            flux_xx = dde.grad.jacobian(flux_x, x_in, i=0, j=0)  # ∂(a ∂u/∂x)/∂x
        else:
            u_x = dde.grad.jacobian(u, x_in, i=0, j=0)  # ∂u/∂x
            u_y = dde.grad.jacobian(u, x_in, i=0, j=1)  # ∂u/∂y
            flux_x = a * u_x
            flux_y = a * u_y
            flux_xx = dde.grad.jacobian(flux_x, x_in, i=0, j=0)
            flux_yy = dde.grad.jacobian(flux_y, x_in, i=0, j=1)

        if not self.two_dim and not self.time_dependent:
            res = - flux_xx - f
        elif not self.two_dim and self.time_dependent:
            u_t = dde.grad.jacobian(u, x_in, i=0, j=1)  # ∂u/∂t
            res = u_t - flux_xx - f
        elif self.two_dim and not self.time_dependent:
            res = - (flux_xx + flux_yy) - f
        else:
            u_t = dde.grad.jacobian(u, x_in, i=0, j=2)  # ∂u/∂t
            res = u_t - (flux_xx + flux_yy) - f

        a_grad = tf.zeros_like(a)
        if self.use_regularization:
            a_x = dde.grad.jacobian(a, x_in, i=0, j=0)
            a_reg = tf.reduce_mean(tf.square(a_x))
            if self.two_dim:
                a_y = dde.grad.jacobian(a, x_in, i=0, j=1)
                a_reg += tf.reduce_mean(tf.square(a_y))
            a_grad= self.regularization_weight * a_reg
        gPINN = tf.zeros_like(res)
        if self.use_gPINN:
            res_x = dde.grad.jacobian(res, x_in, i=0, j=0)
            g_loss = tf.reduce_mean(tf.square(res_x))
            if self.two_dim:
                res_y = dde.grad.jacobian(res, x_in, i=0, j=1)
                g_loss += tf.reduce_mean(tf.square(res_y))
            gPINN = self.gPINN_weight * g_loss

        return res + a_grad + gPINN

    def train(self, loss_weights, iterations=5000, print_every=100, use_regularization=False, use_gPINN=False,
              use_RAR=False, RAR_cycles_n=500, RAR_points_m=100):
        if use_regularization:
            self.use_regularization=use_regularization
            if 'a_grad_loss' not in loss_weights:
                self.regularization_weight = 1
                print("Missing weight for 'a_grad_loss' in loss_weights. 'a_grad_loss' weight will be 1.")
            else:
                self.regularization_weight = loss_weights['a_grad_loss']
        if use_gPINN:
            self.use_gPINN=use_gPINN
            if 'gPINN_loss' not in loss_weights:
                self.gPINN_weight = 1
                print("Missing weight for 'gPINN_loss' in loss_weights. 'gPINN_loss' weight will be 1.")
            else:
                self.gPINN_weight = loss_weights['gPINN_loss']

        if use_RAR and False: # not implemented yet, cause of DeepXDE, there is no way
            return self.train_with_rar(loss_weights, iterations, print_every, use_regularization, use_gPINN,
                                       RAR_cycles_n, RAR_points_m)
        else:
            return self.dde_train(loss_weights, iterations, print_every, use_regularization, use_gPINN)

    def dde_train(self, loss_weights, iterations=5000, print_every=100, use_regularization=False, use_gPINN=False):
        self.use_regularization = use_regularization
        self.use_gPINN = use_gPINN
        pde_resampler = dde.callbacks.PDEPointResampler(period=print_every)
        callbacks = [pde_resampler]

        self.a_model.compile("adam", lr=self.learning_rate, loss_weights=list(loss_weights.values())[0])
        a_history, _ = self.a_model.train(iterations=iterations, callbacks=callbacks, display_every=print_every)

        self.history.losses['loss'] = np.array(a_history.loss_train).flatten().tolist()
        self.history.steps = a_history.steps
        return self.history

    #def train_with_rar(self, loss_weights, iterations=5000, print_every=100, use_regularization=False, use_gPINN=False,
    #                   RAR_cycles_n=500, RAR_points_m=100):
    #    total_iterations = 0
    #    history = History()
    #    current_obs_domain = self.obs_domain.copy()
    #    rar_round = 0
    #    while total_iterations < iterations:
    #        # 1. Train for RAR_cycles_n steps
    #        n_iter = min(RAR_cycles_n, iterations - total_iterations)
    #        history_part = self.dde_train(loss_weights, iterations=n_iter, print_every=print_every,
    #                                  use_regularization=use_regularization, use_gPINN=use_gPINN)
    #        # 2. Aggregate history
    #        if rar_round == 0:
    #            history.losses = history_part.losses.copy()
    #            history.steps = history_part.steps.copy()
    #        else:
    #            offset = history.steps[-1] if history.steps else 0
    #            history.losses['loss'].extend(history_part.losses['loss'])
    #            history.steps.extend([s + offset for s in history_part.steps])
    #        total_iterations += n_iter
    #        rar_round += 1
    #        # 3. RAR step: Add points with the largest residual gradient
    #        if total_iterations <= iterations:
    #            # --- 3a. Sample candidate points in the domain
    #            n_candidates = self.num_domain
    #            dim = current_obs_domain.shape[1]
    #            min_vals, max_vals = current_obs_domain.min(axis=0), current_obs_domain.max(axis=0)
    #            candidates = np.random.uniform(min_vals, max_vals, size=(n_candidates, dim))
    #            candidates_tf = tf.convert_to_tensor(candidates, dtype=tf.float32)
    #            # --- 3b. Evaluate residual at candidate points
    #            with tf.GradientTape() as tape:
    #                tape.watch(candidates_tf)
    #                res = self.pde_residual_for_rar(candidates_tf)
    #            grad_res = tape.gradient(res, candidates_tf)
    #            grad_norm = tf.norm(grad_res, axis=1).numpy()
    #            # --- 3c. Select top-m points with highest gradient-norm (RAR)
    #            top_idx = np.argsort(grad_norm)[-RAR_points_m:]
    #            rar_points = candidates[top_idx]
    #            # --- 3d. Add to training set
    #            current_obs_domain = np.concatenate([current_obs_domain, rar_points], axis=0)
    #            print(f"RAR round {rar_round}: added {RAR_points_m} points.")
    #        # 4. Always re-build model/data for new points!
    #        self.prepare_model(current_obs_domain)
    #    self.history = history
    #    return self.history

    #def pde_residual_for_rar(self, x):
    #    D = x.shape[1]
    #    # Unpack variables depending on config
    #    x_var = x[:, :1]  # always x
    #    y_var = x[:, 1:2] if self.two_dim else None
    #    t_var = x[:, -1:] if self.time_dependent else None
    #    with tf.GradientTape(persistent=True) as tape:
    #        tape.watch(x)
    #        u = self.u_model(x)
    #        a = self.a_model.net(x)
    #    # Prepare for fluxes and time derivatives
    #    if not self.two_dim and not self.time_dependent:
    #        # 1D, no time
    #        u_x = tape.gradient(u, x)  # (N, 1)
    #        flux = a * u_x
    #        flux_xx = tape.gradient(flux, x)
    #        f = self.f_model(x) if self.f_model is not None else 0.0
    #        residual = -flux_xx - f
    #    elif not self.two_dim and self.time_dependent:
    #        # 1D, time-dependent, x=[x, t]
    #        u_x = tape.gradient(u, x)[:, 0:1]
    #        u_t = tape.gradient(u, x)[:, 1:2]
    #        flux = a * u_x
    #        with tf.GradientTape() as tape2:
    #            tape2.watch(x)
    #            flux = a * u_x
    #        flux_xx = tape2.gradient(flux, x)[:, 0:1]
    #        f = self.f_model(x) if self.f_model is not None else 0.0
    #        residual = u_t - flux_xx - f
    #    elif self.two_dim and not self.time_dependent:
    #        # 2D, steady, x=[x, y]
    #        u_x = tape.gradient(u, x)[:, 0:1]
    #        u_y = tape.gradient(u, x)[:, 1:2]
    #        flux_x = a * u_x
    #        flux_y = a * u_y
    #        with tf.GradientTape(persistent=True) as tape2:
    #            tape2.watch(x)
    #            flux_x = a * u_x
    #            flux_y = a * u_y
    #        flux_xx = tape2.gradient(flux_x, x)[:, 0:1]
    #        flux_yy = tape2.gradient(flux_y, x)[:, 1:2]
    #        f = self.f_model(x) if self.f_model is not None else 0.0
    #        residual = -(flux_xx + flux_yy) - f
    #    else:
    #        # 2D time-dependent, x=[x, y, t]
    #        u_x = tape.gradient(u, x)[:, 0:1]
    #        u_y = tape.gradient(u, x)[:, 1:2]
    #        u_t = tape.gradient(u, x)[:, 2:3]
    #        flux_x = a * u_x
    #        flux_y = a * u_y
    #        with tf.GradientTape(persistent=True) as tape2:
    #            tape2.watch(x)
    #            flux_x = a * u_x
    #            flux_y = a * u_y
    #        flux_xx = tape2.gradient(flux_x, x)[:, 0:1]
    #        flux_yy = tape2.gradient(flux_y, x)[:, 1:2]
    #        f = self.f_model(x) if self.f_model is not None else 0.0
    #        residual = u_t - (flux_xx + flux_yy) - f
    #    return tf.reshape(residual, [-1])

    def save(self, save_dir, name="a_model.weights.h5"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.a_model.net.save_weights(os.path.join(save_dir, name))

    def restore(self, save_dir, name):
        if os.path.exists(save_dir):
            self.a_model.net(self.obs_domain)
            #if not self.a_model.net.built:
            input_shape = (None, self.input_dim)
            self.a_model.net.build(input_shape)
            self.a_model.compile("adam", lr=self.learning_rate, loss_weights=[1])
            self.a_model.net.load_weights(os.path.join(save_dir, name))

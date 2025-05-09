# ======================================================================================================================
# The current peace of code is presenting an approach for solving inverse problem for the heat equation over
# a geometric-time range. The heat equation is given as
#                               ∂u − ∇⋅(a∇u) = f,
# where:
# - ∂u time derivative of u(x,t), which is a temperature distribution.
# - f(x,t) is some initial system input
# - a(x) is (unknown) heat diffusivity and has to be calculated.
# ----------------------------------------------------------------------------------------------------------------------
# Infer a(x) such that the heat equation is satisfied for given observations u(x,t), where u is observed
# as synthetic or experimental function and the right side can be also given like measurements.
#
# Inverse Problem (PDE Residual): Using u(x,t), the inverse problem requires minimizing the residual of the
# heat equation:
#                               r(x,t,a) = ∂u − ∇⋅(a)∇u − f.
# The loss function for estimating a(x,y) is:
#                               L = (1/N ∑ ∣r∣^2) + λR(a),
# where:
#     r is the residual over all datapoints (x_i, t_i)
#     R(a) is a regularization term on a(x,y), not implemented in this version
#     λ is the regularization weight.
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================
import itertools
import json
import os
import pickle
import shutil
import time

import deepxde as dde
import numpy as np

from InverseHeatSolver.CompositeModel import CompositeModel
from InverseHeatSolver.Interpolator import Interpolator

from InverseHeatSolver.Visualizer import plot_1D


import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(4)
# tf.config.threading.set_inter_op_parallelism_threads(2)
#
# seed = 42  # Can be any integer seed
# np.random.seed(seed)
# tf.random.set_seed(seed)

# ======================================================================================================================
# ASSUMED STRUCTURE OF INITIALIZATION VARIABLES
# ---------------|-------------------------------------|----------------------------------------------------------------
#   VARS         |  VALUES                             |    EXPLANATION
# ---------------|-------------------------------------|----------------------------------------------------------------
# domain         |   ['x_domain' : [x_start,           |    not None
#                |                  x_end,             |
#                |                  x_num_samp],       |
#                |    'y_domain' : [ - " - ],          |    can be None
#                |    't_domain' : [ - " -]]           |    can be None
#                |                                     |----------------------------------------------------------------
#                |                                     |    Is currently only for rectangular geometries implemented.
# ---------------|-------------------------------------|----------------------------------------------------------------
# nn_dims        |   ['num_layers'  : nl,              |    hyperparameter
#                |    'num_neurons' : nn]              |    hyperparameter
# ---------------|-------------------------------------|----------------------------------------------------------------
# obs_values     |   ['dom_obs' : [x_1, ..., x_n]      |    not None
#                |    'u_obs' : [u_1, ..., u_k],       |    not None
#                |    'f_obs' : [f_1, ..., f_q]]       |    can be None
# ---------------|-------------------------------------|----------------------------------------------------------------
# u_dbc          |   @func(x,t)                        |    Dirichlet BC, not None
# ---------------|-------------------------------------|----------------------------------------------------------------
# u_ic           |   @func(x,t=0)                      |    Initial conditions, None, if time independent
# ---------------|-------------------------------------|----------------------------------------------------------------
# u_nbc          |   @func(x,t=0)                      |    Neumann BC, can be None
# ---------------|-------------------------------------|----------------------------------------------------------------
# loss_weights   |   ['w_PDE_loss'  : _,               |    not None
#                |    'w_BC_loss'   : _,               |    not None
#                |    'w_IC_loss'   : _,               |    can be None
#                |    'w_NBC_loss'  : _,               |    can be None
# ---------------|-------------------------------------|----------------------------------------------------------------
# model_optim... |  False by default                   |    if True the L-BFGS-B optimizer will be used
# ---------------|-------------------------------------|----------------------------------------------------------------
# save_plot      |  False by default                   |    Standard plot for learned values and losses of deepXDE
# ---------------|-------------------------------------|----------------------------------------------------------------
# num_domain     |
# ---------------|-------------------------------------|----------------------------------------------------------------
# num_boundary   |
# ---------------|-------------------------------------|----------------------------------------------------------------
# num_initial    |
# ---------------|-------------------------------------|----------------------------------------------------------------
# num_test       |
# ======================================================================================================================
# INTERNAL VARIABLES STRUCTURE
# ----------------------------------------------------------------------------------------------------------------------
# ...
# ======================================================================================================================


class InverseHeatSolver:
    def __init__(self, domain, nn_dims, obs_values, loss_weights, learning_rate, u_dbc, u_ic=None, u_nbc=None,
                 num_domain=1000, num_boundary=100, num_initial=None, num_test=1000):
        self.total_iterations = 0
        self.dimension = 1
        self.callbacks = []
        self.f_model = None
        self.u_model = None
        self.model_u_frozen = None
        self.domain = domain
        self.two_dim = False
        self.time_dependent = False
        self.observed_domain = None
        self.boundary_coordinates = None
        self.u_obs = None
        self.f_obs = None
        self.initial_coordinates = None
        self.pinn_domain = None
        self.model = None
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_initial = num_initial
        self.num_test = num_test

        # --------------------------------------------------------------------------------------------------------------
        # prepare initial objects
        self.prepare_domain()
        self.nn_dims = nn_dims
        self.prepare_observed_values(obs_values)
        self.u_dirichlet_bc = u_dbc
        self.u_neumann_bc = u_nbc
        self.u_initial_conditions = u_ic
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate

        # --------------------------------------------------------------------------------------------------------------
        self.loss_history = None
        self.train_state = None
        self.training_time = 0

# ----------------------------------------------------------------------------------------------------------------------
# preparing section

    def prepare_observed_values(self, obs_values):
        self.u_obs = obs_values['u_obs']
        self.f_obs = obs_values['f_obs']
        self.observed_domain = obs_values['dom_obs']
        self.dimension = self.observed_domain.shape[1]

    def prepare_domain(self):
        # Dimensions dependency
        if self.domain['y_domain'] is not None:
            self.two_dim = True
        if self.domain['t_domain'] is not None:
            self.time_dependent = True

        x_start, x_end = self.domain['x_domain'][:2]
        geometry_domain = dde.geometry.Interval(x_start, x_end)
        if self.two_dim:
            y_start, y_end = self.domain['y_domain'][:2]
            geometry_domain = dde.geometry.Rectangle([x_start, x_end], [y_start, y_end])
        if self.time_dependent:
            t_start, t_end = self.domain['t_domain'][:2]
            time_domain = dde.geometry.TimeDomain(t_start, t_end)
            self.pinn_domain = dde.geometry.GeometryXTime(geometry_domain, time_domain)
        else:
            self.pinn_domain = geometry_domain

    def prepare_test_domain(self):
        x_start, x_end, x_samp = self.domain['x_domain']
        # x_domain = np.linspace(x_start, x_end, x_samp)
        x_domain = np.random.uniform(x_start, x_end, size=(x_samp, 1))
        if self.domain['y_domain'] is not None:
            y_start, y_end, y_samp = self.domain['y_domain']
            y_domain = np.linspace(y_start, y_end, y_samp)
            if self.domain['t_domain'] is not None:
                t_start, t_end, t_samp = self.domain['t_domain']
                t_domain = np.linspace(t_start, t_end, t_samp)
                test_domain = np.column_stack((x_domain.flatten(), y_domain.flatten(), t_domain.flatten()))
            else:
                test_domain = np.column_stack((x_domain.flatten(), y_domain.flatten()))
        else:
            if self.domain['t_domain'] is not None:
                t_start, t_end, t_samp = self.domain['t_domain']
                t_domain = np.linspace(t_start, t_end, t_samp)
                test_domain = np.column_stack((x_domain.flatten(), t_domain.flatten()))
            else:
                test_domain = x_domain.reshape(x_samp, 1)
        return test_domain

# ----------------------------------------------------------------------------------------------------------------------
# losses section

    def inverse_loss(self, x_in, outputs):
        if self.u_model is not None:
            u = self.u_model(x_in)  # Interpolated u(x)
        else:
            raise ValueError('Keras Model NN for u(.) is None')

        if self.f_model is not None:
            f = self.u_model(x_in)  # Interpolated f(x)
        else:
            f = 0
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
        return res

    @staticmethod
    def boundary(x, on_boundary):
        return on_boundary

    @staticmethod
    def initial(x, on_initial):
        return on_initial

    def balance_loss_weights(self):
        current_losses = np.array(self.model.losshistory.loss_train[-1])
        L_res = np.abs(np.min(current_losses) - np.max(current_losses))
        L_mean = np.median(current_losses)
        if L_res > L_mean:
            print("Loss weights before balance:", self.loss_weights)
            non_zero_mask = current_losses != 0
            inverse_losses = np.zeros_like(current_losses)
            inverse_losses[non_zero_mask] = 1.0 / current_losses[non_zero_mask]
            normalized_weights = inverse_losses / np.sum(inverse_losses)
            for key, new_weight in zip(self.loss_weights.keys(), normalized_weights):
                self.loss_weights[key] = float(new_weight)
            print("Loss weights after balance:", self.loss_weights)

# ----------------------------------------------------------------------------------------------------------------------
# model section

    def prepare_model(self):
        # condition_losses = []
        # u_bc_loss = dde.icbc.DirichletBC(self.pinn_domain, self.u_dirichlet_bc, self.boundary, component=0)
        # condition_losses.append(u_bc_loss)
        # if self.u_initial_conditions is not None and self.time_dependent:
        #     ic_loss = dde.icbc.IC(self.pinn_domain, self.u_initial_conditions, self.initial, component=0)
        #     condition_losses.append(ic_loss)
        # if self.u_neumann_bc is not None:
        #     u_nbc_loss = dde.icbc.NeumannBC(self.pinn_domain, self.u_neumann_bc, self.boundary, component=0)
        #     condition_losses.append(u_nbc_loss)

        if self.time_dependent:
            data = dde.data.TimePDE(self.pinn_domain, self.inverse_loss, [],
                                    num_domain=self.num_domain, num_boundary=self.num_boundary,
                                    num_initial=self.num_initial, anchors=self.prepare_test_domain(),
                                    num_test=self.num_test)
        else:
            data = dde.data.PDE(self.pinn_domain, self.inverse_loss, [],
                                num_domain=self.num_domain, num_boundary=self.num_boundary,
                                anchors=self.prepare_test_domain(), num_test=self.num_test)

        a_net = dde.nn.FNN([1] + [10] + [20] + [10] + [1], "tanh", "Glorot normal")
        # a_net = dde.nn.FNN([2 if self.two_dim else 1] +
        #                    [self.nn_dims['num_neurons']] * self.nn_dims['num_layers'] +
        #                    [1], "tanh", "Glorot normal")
        composed_model = CompositeModel(a_net, self.time_dependent, self.two_dim)
        self.model = dde.Model(data, composed_model)

    def train(self, iterations=10000, u_iterations=5000, f_iterations=5000, display_results_every=1000, model_optimization=False,
              callback_path="callbacks", best_model_name="best_model.ckpt"):

        ################################################################################################################
        global global_step
        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        ################################################################################################################

        self.prepare_save_dir(callback_path)
        start_time = time.time()
        # First stage: interpolates u and f on measured data
        self.learn_u_and_f(u_iterations, f_iterations, callback_path)
        # Second stage: learning u
        self.learn_a(best_model_name, callback_path, display_results_every, iterations, model_optimization)

        self.total_iterations = self.train_state.step
        self.training_time = time.time() - start_time
        self.save_model_and_params(callback_path, best_model_name, prepare_dir=False)

    def learn_a(self, best_model_name, callback_path, display_results_every, iterations, model_optimization):
        # !!!MAYBE NEEDS TO EXTEND TRAINING ANCHORS!!! #
        self.prepare_model()
        callbacks = self.prepare_callbacks(best_model_name, callback_path)
        print('#####################################################')
        print('# Starts to learn a(.) ...                          #')
        print('#####################################################')
        self.model.compile("adam", lr=self.learning_rate, loss_weights=list(self.loss_weights.values()))
        self.loss_history, self.train_state = self.model.train(iterations=iterations,
                                                               display_every=display_results_every,
                                                               callbacks=callbacks)
        if model_optimization:
            self.model.compile("L-BFGS-B", loss_weights=list(self.loss_weights.values()))
            self.loss_history, self.train_state = self.model.train(callbacks=callbacks)

    def learn_u_and_f(self, u_iterations, f_iterations, callback_path):
        print('#####################################################')
        print('# Starts to learn u(.) from observed u-measurements #')
        print('#####################################################')
        self.u_model = Interpolator(self.dimension)
        self.u_model.fit(self.observed_domain, self.u_obs, iterations=u_iterations)
        self.u_model.model.save_weights(os.path.join(callback_path, str("u_model.weights.h5")))
        if self.f_obs is not None:
            print('#####################################################')
            print('# Starts to learn f(.) from observed f-measurements #')
            print('#####################################################')
            # Assumes that the source should be allways a positive value
            self.f_model = Interpolator(self.dimension, positive_output=True)
            self.f_model.fit(self.observed_domain, self.f_obs, iterations=f_iterations, early_stop=1e-8)
            self.f_model.model.save_weights(os.path.join(callback_path, str("f_model.weights.h5")))

    def prepare_callbacks(self, best_model_name, callback_path):
        pde_resampler = dde.callbacks.PDEPointResampler(period=100)
        checkpoint_cb = dde.callbacks.ModelCheckpoint(
            filepath=os.path.join(callback_path, best_model_name),
            verbose=1,  # Print messages when saving the model
            save_better_only=True,
            period=1,
            monitor='test loss'
        )
        early_stopping = dde.callbacks.EarlyStopping(baseline=1e-4, monitor='loss_train', patience=100)
        self.callbacks = [checkpoint_cb, pde_resampler, early_stopping]

    def predict(self, inp):
        u = self.u_model.predict(inp)
        f = self.f_model.predict(inp)
        a = self.model.predict(inp)#.numpy()
        return [u, f, a]

# ----------------------------------------------------------------------------------------------------------------------
# save/restore section

    def format_training_time(self):
        hours = int(self.training_time // 3600)
        minutes = int((self.training_time % 3600) // 60)
        seconds = int(self.training_time % 60)
        milliseconds = int((self.training_time % 1) * 1000)
        # Training time formated as HH:MM:SS.MMM
        return f"Training time: {hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    def save_model_and_params(self, save_dir, model_name_str="nn_model", prepare_dir=True):
        if prepare_dir and not self.prepare_save_dir(save_dir):
            print(f'failed to prepare save dir : {save_dir}')
            return

        self.model.save(os.path.join(save_dir, str(model_name_str)))
        hyper_params = {
            'domain': self.domain,
            'nn_dims': self.nn_dims,
            'loss_weights': self.loss_weights,
            'total_iterations': self.total_iterations,
            'learning_rate': self.learning_rate,
            'num_domain': self.num_domain,
            'num_boundary': self.num_boundary,
            'num_initial': self.num_initial,
            'num_test': self.num_test,
            'training_time': self.training_time,
            'dimension': self.dimension
        }
        with open(os.path.join(save_dir, str(model_name_str) + "_hyperparameters.json"), 'w') as f:
            json.dump(hyper_params, f, indent=4)
        if self.loss_history:
            with open(os.path.join(save_dir, str(model_name_str) + "_loss_history.pkl"), 'wb') as f:
                pickle.dump(self.loss_history, f)
            with open(os.path.join(save_dir, str(model_name_str) + "_train_state.pkl"), 'wb') as f:
                pickle.dump(self.train_state, f)

    @staticmethod
    def prepare_save_dir(save_dir):
        if os.path.exists(save_dir):
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
                    return False
            return True
        else:
            os.makedirs(save_dir)
            return True

    @staticmethod
    def restore_model_and_params(save_dir, model_name_str, obs_values,
                                 u_dbc, model_num=None, u_ic=None, u_nbc=None, f_right_side=None, precision=False):
        with open(os.path.join(save_dir, str(model_name_str) + "_hyperparameters.json"), 'r') as f:
            params = json.load(f)
        domain = params['domain']
        heat_solver = InverseHeatSolver(
            domain=domain,
            nn_dims=params['nn_dims'],
            obs_values=obs_values,
            u_dbc=u_dbc,
            u_ic=u_ic,
            u_nbc=u_nbc,
            loss_weights=params['loss_weights'],
            learning_rate=params['learning_rate'],
            num_domain=params['num_domain'],
            num_boundary=params['num_boundary'],
            num_initial=params['num_initial'],
            num_test=params['num_test']
        )
        heat_solver.total_iterations = params['total_iterations']
        if os.path.exists(os.path.join(save_dir, str(model_name_str) + "_loss_history.pkl")):
            with open(os.path.join(save_dir, str(model_name_str) + "_loss_history.pkl"), 'rb') as f:
                heat_solver.loss_history = pickle.load(f)
            with open(os.path.join(save_dir, str(model_name_str) + "_train_state.pkl"), 'rb') as f:
                heat_solver.train_state = pickle.load(f)
        heat_solver.prepare_model()
        heat_solver.model.compile("adam", lr=heat_solver.learning_rate, loss_weights=list(heat_solver.loss_weights.values()))
        test_domain = heat_solver.prepare_test_domain()
        heat_solver.model.predict(test_domain)
        if model_num is None:
            model_num = heat_solver.total_iterations
        heat_solver.model.restore(os.path.join(save_dir, str(model_name_str) + "-" + str(model_num) + ".weights.h5"))

        heat_solver.training_time = params['training_time']
        heat_solver.dimension = params['dimension']
        heat_solver.u_model = Interpolator(heat_solver.dimension)
        heat_solver.u_model.model.load_weights(os.path.join(save_dir, "u_model.weights.h5"))
        heat_solver.f_model = Interpolator(heat_solver.dimension, positive_output=True)
        heat_solver.f_model.model.load_weights(os.path.join(save_dir, "f_model.weights.h5"))
        heat_solver.prepare_model(heat_solver.u_model, heat_solver.f_model, dummy_init=True)

        return heat_solver

    @staticmethod
    def print_l2_error(exact_values, pred_values, exact_name, pred_name):
        diff = (exact_values[:, 0] - pred_values)
        print(f"l2 error for '{exact_name}' to '{pred_name}' : {np.sqrt(np.sum(diff ** 2))/diff.shape[0]:.4e}")


class CompositeModel(tf.keras.Model):
    def __init__(self, a_net, time_dependent=False, two_dim=False):
        super().__init__()
        self.time_dependent = time_dependent
        self.two_dim = two_dim
        self.regularizer = None
        self.a_net = a_net

    def call(self, inputs):
        if not self.two_dim and not self.time_dependent:
            a = self.a_net(inputs)
        elif not self.two_dim and self.time_dependent:
            x = inputs[:, :1]
            a = self.a_net(x)
        elif self.two_dim and not self.time_dependent:
            a = self.a_net(inputs)
        else:
            xy = inputs[:, :2]
            a = self.a_net(xy)

        return a


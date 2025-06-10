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
import json
import os
import shutil
import time

import numpy as np


from .Interpolator import Interpolator
from .PdeMinimizer import PdeMinimizer
from .PdeMinimizerDeepXde import PdeMinimizerDeepXde


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
# loss_weights   |   ['w_PDE_loss'  : _,               |    not None
#                |    'a_grad_loss'  : _ ]             |    can be None
# ======================================================================================================================
# INTERNAL VARIABLES STRUCTURE
# ----------------------------------------------------------------------------------------------------------------------
# u_model
# f_model
# a_model
# total_iterations
# training_time
# two_dim
# time_dependent
# u_obs
# f_obs
# observed_domain
# dimension
# ======================================================================================================================


class InverseHeatSolver:
    def __init__(self, domain, nn_dims, obs_values, loss_weights, learning_rate, use_deep_xde=False):
        self.history = None
        self.u_model = None
        self.f_model = None
        self.a_model = None

        self.total_iterations = 0
        self.training_time = 0

        self.domain = domain
        self.two_dim = True if self.domain['y_domain'] is not None else False
        self.time_dependent = True if self.domain['t_domain'] is not None else False

        self.nn_dims = nn_dims

        self.u_obs = obs_values['u_obs']
        self.f_obs = obs_values['f_obs']
        self.observed_domain = obs_values['dom_obs']
        self.dimension = self.observed_domain.shape[1]

        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.use_deep_xde = use_deep_xde
        # --------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# preparing section

    def prepare_train_domain(self):
        x_start, x_end, x_samp = self.domain['x_domain']
        x_domain = np.random.uniform(x_start, x_end, size=(x_samp, 1))
        if self.domain['y_domain'] is not None:
            y_start, y_end, y_samp = self.domain['y_domain']
            y_domain = np.random.uniform(y_start, y_end, size=(y_samp, 1))
            if self.domain['t_domain'] is not None:
                t_start, t_end, t_samp = self.domain['t_domain']
                t_domain = np.linspace(t_start, t_end, t_samp)
                X, Y, T = np.meshgrid(x_domain, y_domain, t_domain, indexing='ij')
                train_domain = np.column_stack((X.flatten(), Y.flatten(), T.flatten()))
            else:
                train_domain = np.column_stack((x_domain.flatten(), y_domain.flatten()))
        else:
            if self.domain['t_domain'] is not None:
                t_start, t_end, t_samp = self.domain['t_domain']
                t_domain = np.linspace(t_start, t_end, t_samp)
                X, T = np.meshgrid(x_domain, t_domain)
                train_domain = np.column_stack((X.flatten(), T.flatten()))
            else:
                train_domain = x_domain.reshape(x_samp, 1)
        return train_domain

# ----------------------------------------------------------------------------------------------------------------------
# model section

    def train(self, a_iterations=10000, u_iterations=5000, f_iterations=5000, display_results_every=100,
              save_path="callbacks"):
        self.prepare_save_dir(save_path)
        start_time = time.time()
        # First stage: interpolates u and f on measured data
        u_history, f_history = self.learn_u_and_f(u_iterations, f_iterations, display_results_every)
        # Second stage: learning a
        a_history = self.learn_a(a_iterations, display_results_every)

        self.total_iterations = a_history.steps
        self.training_time = time.time() - start_time
        self.prepare_history(a_history, f_history, u_history)
        self.save_model_and_params(save_path, prepare_dir=False)

        return self.history

    def prepare_history(self, a_history, f_history, u_history):
        self.history = {
            "u_history": {
                "loss": u_history[0],
                "steps": u_history[1]
            },
            "f_history": {
                "loss": f_history[0],
                "steps": f_history[1]
            },
            "a_history": {
                "loss": a_history.losses['loss'],
                "pde_loss": a_history.losses['pde_loss'],
                "a_grad_loss": a_history.losses['a_grad_loss'],
                "steps": a_history.steps
            }
        }

    def learn_a(self, a_iterations, display_results_every):
        print('#####################################################')
        print('# Starts to learn a(.) ...                          #')
        print('#####################################################')
        if self.use_deep_xde:
            self.a_model = PdeMinimizerDeepXde(self.domain, self.prepare_train_domain(), u_model=self.u_model,
                                               f_model=self.f_model, input_dim=self.dimension,
                                               nn_dims=self.nn_dims, lr=self.learning_rate,
                                               time_dependent=self.time_dependent, two_dim=self.two_dim)
            a_history = self.a_model.train(self.loss_weights, a_iterations, print_every=display_results_every,
                                           early_stop=1e-6)
        else:
            if self.time_dependent:
                dim = self.dimension - 1
            else:
                dim = self.dimension
            self.a_model = PdeMinimizer(u_model=self.u_model, f_model=self.f_model, input_dim=dim,
                                        nn_dims=self.nn_dims, lr=self.learning_rate, time_dependent=self.time_dependent,
                                        two_dim=self.two_dim)
            a_history = self.a_model.train(self.prepare_train_domain(), self.loss_weights, a_iterations,
                                           print_every=display_results_every)
        return a_history

    def learn_u_and_f(self, u_iterations, f_iterations, display_results_every):
        print('#####################################################')
        print('# Starts to learn u(.) from observed u-measurements #')
        print('#####################################################')
        self.u_model = Interpolator(self.dimension, lr=self.learning_rate)
        u_loss, u_steps = self.u_model.fit(self.observed_domain, self.u_obs, iterations=u_iterations,
                                           print_every=display_results_every)
        f_loss = None
        f_steps = None
        if self.f_obs is not None:
            print('#####################################################')
            print('# Starts to learn f(.) from observed f-measurements #')
            print('#####################################################')
            # Assumes that the source should always be a positive value
            self.f_model = Interpolator(self.dimension, lr=self.learning_rate, positive_output=True)
            f_loss, f_steps = self.f_model.fit(self.observed_domain, self.f_obs, iterations=f_iterations,
                                               print_every=display_results_every)
        return [u_loss, u_steps], [f_loss, f_steps]

    def predict(self, inp):
        u = self.u_model.predict(inp)
        f = self.f_model.predict(inp)
        a = self.a_model.predict(inp)
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

    def save_model_and_params(self, save_path, prepare_dir=True):
        if prepare_dir and not self.prepare_save_dir(save_path):
            print(f'failed to prepare save dir : {save_path}')
            return

        #self.a_model.a_model.save_weights(os.path.join(save_path, str("a_model.weights.h5")))
        self.a_model.save(save_path, "a_model.weights.h5")
        self.u_model.save(save_path, "u_model.weights.h5")
        #self.u_model.model.save_weights(os.path.join(save_path, str("u_model.weights.h5")))
        if self.f_model is not None:
            #self.f_model.model.save_weights(os.path.join(save_path, str("f_model.weights.h5")))
            self.f_model.save(save_path, str("f_model.weights.h5"))

        obs_values = {'dom_obs': self.observed_domain.tolist(),
                      'u_obs': self.u_obs.tolist(),
                      'f_obs': self.f_obs.tolist()}

        history = {
            "u_history": {
                "loss": self.history['u_history']['loss'],
                "steps": self.history['u_history']['steps']
            },
            "f_history": {
                "loss": self.history['f_history']['loss'],
                "steps": self.history['f_history']['steps']
            },
            "a_history": {
                "loss": self.history['a_history']['loss'],
                "pde_loss": self.history['a_history']['pde_loss'],
                "a_grad_loss": self.history['a_history']['a_grad_loss'],
                "steps": self.history['a_history']['steps']
            }
        }

        hyper_params = {
            'domain': self.domain,
            'nn_dims': self.nn_dims,
            'obs_values': obs_values,
            'loss_weights': self.loss_weights,
            'learning_rate': self.learning_rate,
            'training_time': self.training_time,
            'total_iterations': self.total_iterations,
            'history': history,
            'use_deep_xde': self.use_deep_xde
        }
        with open(os.path.join(save_path, "hyperparameters.json"), 'w') as f:
            #json.dump(self.convert_to_serializable(hyper_params), f, indent=4)
            json.dump(hyper_params, f, indent=4, default=lambda o: float(o))
            #json.dump(hyper_params, f, indent=4)

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: obj.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [obj.convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

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
    def restore_model_and_params(save_dir):
        with open(os.path.join(save_dir, "hyperparameters.json"), 'r') as f:
            params = json.load(f)

        obs_values = {'dom_obs': np.array(params['obs_values']['dom_obs']),
                      'u_obs': np.array(params['obs_values']['u_obs']),
                      'f_obs': np.array(params['obs_values']['f_obs'])}

        history = {
            "u_history": {
                "loss": params['history']['u_history']['loss'],
                "steps": params['history']['u_history']['steps']
            },
            "f_history": {
                "loss": params['history']['f_history']['loss'],
                "steps": params['history']['f_history']['steps']
            },
            "a_history": {
                "loss": params['history']['a_history']['loss'],
                "pde_loss": params['history']['a_history']['pde_loss'],
                "a_grad_loss": params['history']['a_history']['a_grad_loss'],
                "steps": params['history']['a_history']['steps']
            }
        }

        heat_solver = InverseHeatSolver(
            domain=params['domain'],
            nn_dims=params['nn_dims'],
            obs_values=obs_values,
            loss_weights=params['loss_weights'],
            learning_rate=params['learning_rate'],
            use_deep_xde=params['use_deep_xde']
        )
        heat_solver.training_time = params['training_time']
        heat_solver.total_iterations = params['total_iterations']
        heat_solver.history = history

        heat_solver.u_model = Interpolator(heat_solver.dimension, lr=heat_solver.learning_rate)
        heat_solver.u_model.restore(save_dir, "u_model.weights.h5")
        #heat_solver.u_model.model.load_weights(os.path.join(save_dir, "u_model.weights.h5"))

        heat_solver.f_model = Interpolator(heat_solver.dimension, lr=heat_solver.learning_rate, positive_output=True)
        heat_solver.f_model.restore(save_dir, "f_model.weights.h5")
        #heat_solver.f_model.model.load_weights(os.path.join(save_dir, "f_model.weights.h5"))

        if heat_solver.use_deep_xde:
            heat_solver.a_model = PdeMinimizerDeepXde(heat_solver.domain, heat_solver.prepare_train_domain(),
                                                      u_model=heat_solver.u_model, f_model=heat_solver.f_model,
                                                      input_dim=heat_solver.dimension, nn_dims=heat_solver.nn_dims,
                                                      lr=heat_solver.learning_rate,
                                                      time_dependent=heat_solver.time_dependent,
                                                      two_dim=heat_solver.two_dim)
        else:
            if heat_solver.time_dependent:
                dim = heat_solver.dimension - 1
            else:
                dim = heat_solver.dimension
            heat_solver.a_model = PdeMinimizer(u_model=heat_solver.u_model, f_model=heat_solver.f_model,
                                               input_dim=dim, nn_dims=heat_solver.nn_dims,
                                               lr=heat_solver.learning_rate, time_dependent=heat_solver.time_dependent,
                                               two_dim=heat_solver.two_dim)
        heat_solver.a_model.restore(save_dir, "a_model.weights.h5")
        #heat_solver.a_model.a_model.load_weights(os.path.join(save_dir, "a_model.weights.h5"))

        return heat_solver

    @staticmethod
    def print_l2_error(exact_values, pred_values, exact_name, pred_name):
        # diff = (exact_values[:, 0] - pred_values)
        diff = (exact_values - pred_values)
        print(f"l2 error for '{exact_name}' to '{pred_name}' : {np.sqrt(np.sum(diff ** 2))/diff.shape[0]:.4e}")


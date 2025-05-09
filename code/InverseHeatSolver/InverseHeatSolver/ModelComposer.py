import json
import os
import pickle
import shutil
import time

import numpy as np
import tensorflow as tf

from InverseHeatSolver.Interpolator import Interpolator
from InverseHeatSolver.PdeMinimizer import PdeMinimizer

from InverseHeatSolver.Visualizer import plot_1D


class ModelComposer:

    def __init__(self, domain, nn_dims, obs_values, loss_weights, learning_rate, u_dbc, u_ic=None, u_nbc=None,
                 num_domain=1000, num_boundary=100, num_initial=None, num_test=1000):
        self.total_iterations = 0
        self.dimension = 1
        self.callbacks = []
        self.u_model = None
        self.f_model = None
        self.a_model = None
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

    def prepare_observed_values(self, obs_values):
        self.u_obs = obs_values['u_obs']
        self.f_obs = obs_values['f_obs']
        self.observed_domain = obs_values['dom_obs']
        self.dimension = self.observed_domain.shape[1]

    def train(self, iterations=10000, u_iterations=5000, f_iterations=5000, display_results_every=1000,
              model_optimization=False,
              callback_path="callbacks", best_model_name="best_model.ckpt"):

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
        print('#####################################################')
        print('# Starts to learn a(.) ...                          #')
        print('#####################################################')

        self.a_model = PdeMinimizer(self.u_model, self.f_model, self.nn_dims, self.learning_rate, self.time_dependent,
                                    self.two_dim, self.domain)

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
            self.f_model.fit(self.observed_domain, self.f_obs, iterations=f_iterations)
            self.f_model.model.save_weights(os.path.join(callback_path, str("f_model.weights.h5")))

    # --------------------------------------------------------------------------------------------------------------
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
        heat_solver.a_model.compile("adam", lr=heat_solver.learning_rate,
                                    loss_weights=list(heat_solver.loss_weights.values()))
        test_domain = heat_solver.prepare_test_domain()
        heat_solver.a_model.predict(test_domain)
        if model_num is None:
            model_num = heat_solver.total_iterations
        heat_solver.a_model.restore(
            os.path.join(save_dir, str(model_name_str) + "-" + str(model_num) + ".weights.h5"))

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
        print(f"l2 error for '{exact_name}' to '{pred_name}' : {np.sqrt(np.sum(diff ** 2)) / diff.shape[0]:.4e}")
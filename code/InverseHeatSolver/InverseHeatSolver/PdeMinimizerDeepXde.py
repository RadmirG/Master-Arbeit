import os

import deepxde as dde
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from InverseHeatSolver.CompositeModel import CompositeModel
from InverseHeatSolver.History import History


class PdeMinimizerDeepXde:
    def __init__(self, domain, obs_domain, u_model, f_model=None, input_dim=1,
                 nn_dims={'num_layers': 2, 'num_neurons': 20}, lr=0.01,
                 time_dependent=False, two_dim=False):
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
        self.prepare_model(obs_domain)
        self.learning_rate = lr
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def prepare_domain(self, domain):
        x_start, x_end = domain['x_domain'][:2]
        geometry_domain = dde.geometry.Interval(x_start, x_end)
        if self.two_dim:
            y_start, y_end = domain['y_domain'][:2]
            geometry_domain = dde.geometry.Rectangle([x_start, y_start], [x_end, y_end])
        if self.time_dependent:
            t_start, t_end = domain['t_domain'][:2]
            time_domain = dde.geometry.TimeDomain(t_start, t_end)
            self.pinn_domain = dde.geometry.GeometryXTime(geometry_domain, time_domain)
        else:
            self.pinn_domain = geometry_domain

    def prepare_model(self, obs_domain):
        if self.time_dependent:
            data = dde.data.TimePDE(self.pinn_domain, self.pde_loss, [], num_domain=1000,
                                    num_boundary=1000, num_initial=1000, anchors=obs_domain, num_test=1000)
        else:
            data = dde.data.PDE(self.pinn_domain, self.pde_loss, [], num_domain=1000, num_boundary=1000,
                                anchors=obs_domain, num_test=1000)

        if self.time_dependent:
            dim = self.input_dim - 1
        else:
            dim = self.input_dim
        a_net = dde.nn.FNN([dim]
                           + [self.nn_dims['num_neurons']] * self.nn_dims['num_layers']
                           + [1], "tanh", "Glorot normal")
        composite_net = CompositeModel(a_net, self.time_dependent, self.two_dim)
        self.a_model = dde.Model(data, composite_net)

    def predict(self, inputs):
        if not self.two_dim and not self.time_dependent:
            # a = self.a_model.net(inputs)
            a = self.a_model.predict(inputs)
        elif not self.two_dim and self.time_dependent:
            x = inputs[:, :1]
            # a = self.a_model.net(x)
            a = self.a_model.predict(x)
        elif self.two_dim and not self.time_dependent:
            # a = self.a_model.net(inputs)
            a = self.a_model.predict(inputs)
        else:
            xy = inputs[:, :2]
            # a = self.a_model.net(xy)
            a = self.a_model.predict(xy)
        return a #.numpy()

    def get_network(self):
        return self.a_model.net

    def a_tf(self, x, sigma=0.05, mu=0.5):
        return 1 + tf.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))

    def pde_loss(self, x_in, outputs):
        if self.u_model is not None:
            u = self.u_model(x_in)  # Interpolated u(x)
        else:
            raise ValueError('Keras Model NN for u(.) is None')

        if self.f_model is not None:
            f = self.f_model(x_in)  # Interpolated f(x)
        else:
            f = 0.0

        a = outputs

        self.global_step.assign_add(1)
        # ------------------------------------------------------------------------------------------------------------------
        if tf.equal(self.global_step % 1000, 0):
            a_true = self.a_tf(x_in)
            tf.print('=================================================')
            tf.print("L2 norm distances on ", tf.cast(tf.shape(x_in)[0], tf.float32), " points")
            tf.print('-------------------------------------------------')
            tf.print('|a_dde - a_tf|       : ',
                     tf.sqrt(tf.reduce_sum(tf.square(a - a_true))) / tf.cast(tf.shape(a_true)[0], tf.float32))
        # ------------------------------------------------------------------------------------------------------------------

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

    def train(self, loss_weights, iterations=5000, print_every=100, early_stop=1e-6):
        pde_resampler = dde.callbacks.PDEPointResampler(period=print_every)
        #early_stopping = dde.callbacks.EarlyStopping(baseline=early_stop, monitor='loss_train', patience=1000)
        callbacks = [pde_resampler]

        self.a_model.compile("adam", lr=self.learning_rate, loss_weights=list(loss_weights.values())[0])
        a_history, a_state = self.a_model.train(iterations=iterations, callbacks=callbacks)

        self.history.losses['loss'] = np.array(a_history.loss_train).flatten().tolist()
        self.history.steps = a_history.steps
        return self.history

    def save(self, save_dir, name="a_model.weights.h5"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.a_model.net.save_weights(os.path.join(save_dir, name))

    def restore(self, save_dir, name):
        if os.path.exists(save_dir):
            self.a_model.compile("adam", lr=self.learning_rate, loss_weights=[1])
            self.a_model.net(self.obs_domain)
            if not self.a_model.net.built:
                input_shape = (None, self.input_dim)
                self.a_model.net.build(input_shape)

            self.a_model.net.load_weights(os.path.join(save_dir, name))

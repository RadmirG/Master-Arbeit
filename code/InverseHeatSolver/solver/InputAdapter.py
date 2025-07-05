# ======================================================================================================================
# A helper Object for PdeMinimizerDeepXde to map the inputs defined in InverseHeatSolver to the PINN input in case of
# time dependency.
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================

import tensorflow as tf
import deepxde as dde


class InputAdapter(tf.keras.Model):
    def __init__(self, a_net, time_dependent=False, two_dim=False):
        super().__init__()
        self.time_dependent = time_dependent
        self.two_dim = two_dim
        self.regularizer = None
        self.a_net = a_net

    def call(self, inputs):
        if not self.time_dependent:
            return self.a_net(inputs)
        else:
            return self.a_net(inputs[:, :2] if self.two_dim else inputs[:, :1])

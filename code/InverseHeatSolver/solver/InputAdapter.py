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
        #if not self.two_dim and not self.time_dependent:
        if not self.time_dependent:
            a = self.a_net(inputs)
        elif not self.two_dim and self.time_dependent:
            x = inputs[:, :1]
            a = self.a_net(x)
        #elif self.two_dim and not self.time_dependent:
        #    a = self.a_net(inputs)
        else:
            xy = inputs[:, :2]
            a = self.a_net(xy)

        return a

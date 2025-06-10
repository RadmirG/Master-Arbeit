import deepxde as dde
import numpy as np
import tensorflow as tf

import functions
from solver.InverseHeatSolver import InverseHeatSolver
from solver import Visualizer

# -----------------------
# Parameters
sigma = 0.1
mu_x = 0.5
# -----------------------
# Thermal diffusivity a(x,y)
a = functions.a_1D_ti
f = functions.f_1D_ti

def a(x, sigma=0.05, mu=0.5):
    return 1 + tf.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))


def f(x, sigma=0.05, mu=0.5):
    numerator = (2 * sigma ** 2 + (1 - 2 * x) * (x - mu))
    exponent = tf.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))
    factor = 1 / (sigma ** 2)
    f = 2 + numerator * exponent * factor
    return f

# -----------------------
# PDE Definition
def pde(x, u):
    grad_u = dde.grad.jacobian(u, x)
    flux_x = a(x) * grad_u
    div_flux_x = dde.grad.jacobian(flux_x, x)
    return - div_flux_x - f(x)

# -----------------------
# Geometry
domain = dde.geometry.Interval(0, 1)

def boundary_func(x, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(domain, lambda x: 0, boundary_func)

# -----------------------
# Dataset
data = dde.data.PDE(
    domain,
    pde,
    [bc],
    num_domain=1000,
    num_boundary=2,
    num_test=1000,
)

# -----------------------
# Neural Network
net = dde.nn.FNN([1] + [5] * 2 + [1], "tanh", "Glorot normal")

# -----------------------
# Model
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=3000, display_every=50)
losses = np.array(losshistory.loss_train)
history = {"a_history": {
            "loss": (losses[:, 0] + losses[:, 1]).tolist(),
            "pde_loss": losses[:, 0].tolist(),
            "a_grad_loss": losses[:, 1].tolist(),
            "steps": losshistory.steps,
            "label_a_grad_loss": r"$\mathcal{L}_{BC}$",
            "title": "Verlustfunktion der direkten Lösung der WLG"
            }
        }

import matplotlib
matplotlib.use('TkAgg')

Visualizer.plot_losses(history)

# # Optional L-BFGS refinement
# model.compile("L-BFGS")
# model.train()

# -----------------------
# Evaluate
x_test = np.linspace(0, 1, 100).reshape(100, 1)
u_pred = model.predict(x_test)

# exact functions
u_exact = functions.u_1D_ti(x_test)
a_exact = functions.a_1D_ti(x_test)
f_exact = functions.f_1D_ti(x_test)

InverseHeatSolver.print_l2_error(u_exact, u_pred, "u_exact", "u_pred")

Visualizer.plot_1D(x_test, u_pred, x_test, None, None, None, a_exact, None, u_exact, f_exact, None, title=r"Gelernte Lösung")

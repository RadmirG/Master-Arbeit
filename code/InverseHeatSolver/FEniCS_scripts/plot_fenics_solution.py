import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import functions
from solver import Visualizer


# Load the saved solution
data = np.load("sol_1D_td_homogen_55x100.npz")
u_sol = data["u_sol"]
X_test = data["X_test"]
T_test = data["T_test"]
XT = np.column_stack((X_test.flatten(), T_test.flatten()))

a_values = functions.a_1D_td(XT).reshape(X_test.shape)
f_values = functions.f_1D_td_pulse(XT).reshape(X_test.shape)

Visualizer.plot_3d(X_test, T_test, u_sol, f_values, a_values, is_time_plot=True)
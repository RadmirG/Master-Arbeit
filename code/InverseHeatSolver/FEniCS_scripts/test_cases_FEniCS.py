import time

import numpy as np
from fenics import *

from FEniCS_scripts.ForwardHeatSolver import ForwardHeatSolver
from solver import Visualizer
import functions # Defines all used functions


def u_bc(x):
    return 0

def ic(x):
    return x * (1 - x)

case_1 = True  # 1D, time independent
case_2 = False  # 1D, time dependent
case_3 = False  # 2D, time independent
case_4 = False  # 2D, time dependent

if case_1:
    domain_ti = {'x_domain': [0, 1, 1100], 'y_domain': None, 't_domain': None}
    start_time = time.time()
    solver = ForwardHeatSolver(domain_ti, functions.a_1D_ti, functions.f_1D_ti, u_bc)
    u_sol, x_test = solver.solve()
    approximation_time = time.time() - start_time
    diff = (functions.u_1D_ti(x_test).reshape(x_test.shape) - u_sol)
    print(f"l2 error for u_exct to u_f : {np.sqrt(np.sum(diff ** 2)) / diff.shape[0]:.4e}")
    print(f"Approximation time         : {approximation_time} s")

    # Prepare data for plotting
    a_values = functions.a_1D_ti(x_test)
    f_values = functions.f_1D_ti(x_test)
    u_exact = functions.u_1D_ti(x_test)

    import matplotlib
    matplotlib.use('Agg')
    #          plot_1D(x, u_pred, x_obs=None, a_pred=None, u_obs=None, f_obs=None, a_exact=None, f_pred=None, u_exact=None, f_exact=None, a_obs=None)
    Visualizer.plot_1D(x_test, u_sol, x_test, None, None, None, a_values, None, u_exact, f_values, None,  title=r"Approximierte Lösung $u_l(x)$")

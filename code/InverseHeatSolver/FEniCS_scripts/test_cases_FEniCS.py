import numpy as np

from fenics import *

from FEniCS_scripts.ForwardHeatSolver import ForwardHeatSolver
from InverseHeatSolver import Visualizer
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
    domain_ti = {'x_domain': [0, 1, 100], 'y_domain': None, 't_domain': None}
    solver = ForwardHeatSolver(domain_ti, functions.a_1D_ti, functions.f_1D_ti, u_bc)
    u_sol, x_test = solver.solve()

    # Prepare data for plotting
    a_values = functions.a_1D_ti(x_test)
    f_values = functions.f_1D_ti(x_test)
    u_exact = functions.u_1D_ti(x_test)

    #          plot_1D(x, u_pred, x_obs=None, a_pred=None, u_obs=None, f_obs=None, a_exact=None, f_pred=None, u_exact=None, f_exact=None, a_obs=None)
    Visualizer.plot_1D(x_test, u_sol, x_test, a_values, None, None, None, f_values, u_exact, None, None)

if case_2:
    # Inner parameter
    a = Expression("1 + exp(-((x - mu) ** 2 / (2 * sigma ** 2)))", sigma=0.05, mu=0.5, degree=2)
    f = Expression("exp(-t) * ((1/sigma**2) * ((1 - 2 * x) * (x - mu) + 2 * sigma ** 2) "
                   "* exp(-((x - mu)**2)/(2 * sigma**2)) - x * (1 - x) + 2)", t=0.0, sigma=0.05, mu=0.5, degree=2)

    domain_td = {'x_domain': [0, 1, 500], 'y_domain': None, 't_domain': [0, 6, 100]}
    solver = ForwardHeatSolver(domain_td, a, f, u_bc, uI=ic)
    u_sol, x_test = solver.solve()

    a_values = functions.a_1D_td(x_test)
    f_values = functions.f_1D_td(x_test)
    u_exact = functions.u_1D_td(x_test)
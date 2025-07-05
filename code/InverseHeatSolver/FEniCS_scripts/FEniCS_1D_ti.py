# ======================================================================================================================
# Solving the forward problem for the heat equation
#                               ∂u − ∇⋅(a∇u) = f
# where ∂u time derivative of u(⋅), also searched solution for heat equation, a(⋅) is (known) heat diffusivity,
# and f(⋅) is some initial system input.
# ----------------------------------------------------------------------------------------------------------------------
# Steps:
#
#     1. Definition of Geometry and Mesh: 2D rectangle.
#
#     2. Definition of Function Space: P2 elements for approximating u(x,y,t).
#
#     3. The Variational Formulation: ∫(∂t/∂u)vdx + ∫a(u)∇u⋅∇vdx = ∫fvdx, v is a test function.
#
#     4. Time discretization with backward Euler or Crank-Nicolson for time derivatives.
#
#     5. Solve in FEniCS (on Windows take a look to A_Dockerfile).
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================

import time

import numpy as np
from fenics import *

from ForwardHeatSolver import ForwardHeatSolver

import functions # Defines all used functions
from solver import Visualizer


def u_bc(x):
    return 0

domain_ti = {'x_domain': [0, 1, 1100], 'y_domain': None, 't_domain': None}
start_time = time.time()
#solver = ForwardHeatSolver(domain_ti, functions.a_1D_ti, u_bc, f_expr=functions.f_1D_ti)
solver = ForwardHeatSolver(domain_ti, functions.a_1D_ti, u_bc)
u_sol, x_test = solver.solve()
approximation_time = time.time() - start_time
diff = (functions.u_1D_ti(x_test).reshape(x_test.shape) - u_sol)
print(f"l2 error for u_exct to u_f : {np.sqrt(np.sum(diff ** 2)) / diff.shape[0]:.4e}")
print(f"Approximation time         : {approximation_time} s")

# Prepare data for plotting
a_values = functions.a_1D_ti(x_test)
f_values = functions.f_1D_ti(x_test)
x_obs = np.linspace(0,1,50)
u_obs = functions.u_1D_ti(x_obs)

Visualizer.plot_1D(x_test, u_sol, x_obs, None, u_obs, None, a_values,
                   None, None, f_values, None,  title=r"Approximierte Lösung $u_l(x)$")

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
# Radmir Gesler, 2025, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================


import time

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from ForwardHeatSolver import ForwardHeatSolver
from solver import Visualizer
import functions

# ======================================================================================================================
# DEFINITION OF THE FORWARD PROBLEM AND DOMAIN
# ======================================================================================================================
# STEP 1
# ======================================================================================================================
# Mesh and function space in 1D
start_time = time.time()

nx = 54
mesh = IntervalMesh(nx, 0, 1)  # 1D interval mesh [0, 1]
sigma = 0.05
mu = 0.5

# Time-stepping parameters
dt = 0.002
T_start = 0.2
T_end = 0.35

# ======================================================================================================================
# STEP 2
# ======================================================================================================================
# Approximation polynomials
V = FunctionSpace(mesh, "P", 2)

# Define boundary condition u(0,t) = u(1,t) = 0
def boundary(x, on_boundary):
    return on_boundary and (near(x[0], 0) or near(x[0], 1))
u_D = Constant(0.0)
bc = DirichletBC(V, u_D, boundary)

# Initial condition u(x, 0) for the previous time step
u_n = interpolate(Expression("x[0]*(1 - x[0])", degree=2), V)


a = Expression("1 + exp(-((x[0] - mu) * (x[0] - mu)) / (2 * sigma * sigma))",
               mu=mu, sigma=sigma, degree=2)

#f = Expression(
#    "(exp(-t) * ((1 / pow(sigma, 2)) * ((1 - 2 * x[0]) * (x[0] - mu) + 2 * pow(sigma, 2)) * "
#    "exp(-pow(x[0] - mu, 2) / (2 * pow(sigma, 2))) - x[0] * (1 - x[0]) + 2))",
#    mu=mu, sigma=sigma, t=0.0, degree=3)

f = Expression(
    "exp(-pow(t - t0, 2) / (2 * pow(tau, 2))) * "
    "((1 / pow(sigma, 2)) * ((1 - 2 * x[0]) * (x[0] - mu) + 2 * pow(sigma, 2)) * "
    "exp(-pow(x[0] - mu, 2) / (2 * pow(sigma, 2))) - x[0] * (1 - x[0]) + 2)",
    degree=3, sigma=0.14, mu=0.5, t0=0.0, tau=0.01, t=0.0
)

# ======================================================================================================================
# STEP 3
# ======================================================================================================================
# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Variational problem
u_t = (u - u_n) / dt  # Time derivative
F = u_t * v * dx + a * dot(grad(u), grad(v)) * dx - f * v * dx
# F = u_t * v * dx + a * dot(grad(u), grad(v)) * dx
a_form, L_form = lhs(F), rhs(F)

# Time-stepping
u_ = Function(V)
t = T_start

# ======================================================================================================================
# STEP 4
# ======================================================================================================================
# Containers for storing the solution at each time step
t_stp = []
u_td = []
# x = V.tabulate_dof_coordinates().flatten()  # 1D coordinate array of mesh nodes
while t < T_end:
    t += dt
    f.t = t  # Update the time in the source term
    solve(a_form == L_form, u_, bc)
    u_n.assign(u_)  # Update for the next time step
    t_stp.append(t)  # Store solution snapshot
    u_td.append(u_.compute_vertex_values(mesh))

approximation_time = time.time() - start_time

# Save the final solution in a file
#file = File("1D_td.pvd")
#file << u_

# ======================================================================================================================
# PREPARING FOR PLOTS AND SAVING APPROXIMATED DATA
# ======================================================================================================================
# Prepare data for plotting
space_dom = mesh.coordinates().flatten()  # Mesh vertex
time_dom = np.array(t_stp)
time_mask = (time_dom >= T_start) & (time_dom <= T_end)
time_interest_region = time_dom[time_mask]

sizeof_t = np.shape(time_dom)[0]
X_test, T_test = np.meshgrid(space_dom, time_interest_region)
XT = np.column_stack((X_test.flatten(), T_test.flatten()))

u_sol = np.array(u_td)
u_sol = u_sol[time_mask]

np.savez("sol_1D_td_homogen_55x100.npz", u_sol=u_sol, X_test=X_test, T_test=T_test)

diff = (functions.u_1D_td(XT).reshape(X_test.shape) - u_sol)
print(f"l2 error for u_exct to u_f : {np.sqrt(np.sum(diff ** 2))/diff.shape[0]:.4e}")
print(f"Approximation time         : {approximation_time} s")

# a_values = functions.a_1D_td(XT).reshape(X_test.shape)
# f_values = functions.f_1D_td_pulse(XT).reshape(X_test.shape)
# u_values = functions.u_1D_td(XT).reshape(X_test.shape)
#
# Visualizer.plot_1D(X_test[0,:], u_sol[0,:], X_test, None, None, None, a_values[0], None,
#                    u_values[0], f_values[0], None, title=r"Approximierte Lösung $u_l(x)$")
#
# Visualizer.plot_3d(X_test, T_test, u_sol, f_values, a_values, is_time_plot=True)


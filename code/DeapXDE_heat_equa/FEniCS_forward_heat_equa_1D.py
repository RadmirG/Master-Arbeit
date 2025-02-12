# ======================================================================================================================
# Solving the forward problem for the heat equation
#                               ∂u − ∇⋅(a∇u) = f
# where ∂u time derivative of u(x,y,t), also searched solution for heat equation, a(⋅) is (known) heat diffusivity,
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

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Mesh and function space in 1D
nx = 100
mesh = IntervalMesh(nx, 0, 1)  # 1D interval mesh [0, 1]

# Time-stepping parameters
dt = 0.01
T = 1.0

# Approximation polynomials
V = FunctionSpace(mesh, "P", 1)

# Boundary conditions
u_D = Constant(0.0)  # Dirichlet boundary condition
bc = DirichletBC(V, u_D, "on_boundary")

# Initial condition
u_n = interpolate(Constant(0.0), V)

# Inner parameter (a(x) = 1 + x)
a = Expression("1 + x[0]", degree=1)  # Linear function of x

# Source term f(x,t) = (1/t)*e^(-x^2/t) (time-dependent distribution)
# f = Expression("1.0/t * exp(-x[0]*x[0]/(2*t))", t=0.01, degree=2)
f = Expression("1+8*x[0]", degree=2)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Variational problem
u_t = (u - u_n) / dt  # Time derivative
F = u_t * v * dx + a * dot(grad(u), grad(v)) * dx - f * v * dx
a_form, L_form = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0.0

# Containers for storing the solution at each time step
time_steps = []
x = V.tabulate_dof_coordinates().flatten()  # 1D coordinate array of mesh nodes

while t < T:
    t += dt
    f.t = t  # Update the time in the source term
    solve(a_form == L_form, u, bc)
    u_n.assign(u)  # Update for the next time step
    time_steps.append(u.copy(deepcopy=True))  # Store solution snapshot

# Save the final solution in a file
# file = File("solution_1D.pvd")
# file << u

# Prepare data for plotting
x_coords = mesh.coordinates().flatten()  # Mesh vertex
u_values = u.compute_vertex_values(mesh)  # Solution values at mesh vertices
a_values = np.array([a(xi) for xi in x_coords])  # Evaluate a(x) at mesh vertices
f_values = np.array([f(xi) for xi in x_coords])  # Evaluate f(x, t=final)

# Plotting
plt.figure(figsize=(12, 6))

# Plot a(x)
plt.subplot(2, 2, 1)
plt.plot(x_coords, a_values, label="a(x) = 1 + x", color="blue")
plt.title("Heat diffusivity (a(x))")
plt.xlabel("x")
plt.ylabel("a(x)")
plt.grid()
plt.legend()

# Plot f(x, t_final)
plt.subplot(2, 2, 2)
plt.plot(x_coords, f_values, label=f"f(x, t={t:.2f})", color="orange")
plt.title(f"Source Term (f(x, t) at t={t:.2f})")
plt.xlabel("x")
plt.ylabel("f(x, t)")
plt.grid()
plt.legend()

# Plot the final solution u(x)
plt.subplot(2, 1, 2)
plt.plot(x_coords, u_values, label="u(x) Final solution", color="green")
plt.title("Final solution (u(x) at t=T)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
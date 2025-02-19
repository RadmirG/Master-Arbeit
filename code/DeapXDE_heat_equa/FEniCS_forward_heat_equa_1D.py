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

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


# ======================================================================================================================
# DEFINITION OF THE FORWARD PROBLEM AND DOMAIN
# ======================================================================================================================
# Mesh and function space in 1D
nx = 100
mesh = IntervalMesh(nx, 0, 1)  # 1D interval mesh [0, 1]

# Time-stepping parameters
dt = 0.01
T = 1.0

# Approximation polynomials
V = FunctionSpace(mesh, "P", 2)

# Boundary conditions
u_D = Constant(0.0)  # Dirichlet boundary condition
bc = DirichletBC(V, u_D, "on_boundary")

# Initial condition
u_n = interpolate(Constant(0.0), V)

# Inner parameter a(x) = 1 + x
a = Expression("1 + x[0]", degree=1)  # Linear function of x

# Source term f(x) = 1 + 4x (time-dependent distribution)
f = Expression("1 + 4*x[0]", degree=2)

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


# ======================================================================================================================
# PREPARING FOR PLOTS AND SAVING APPROXIMATED DATA
# ======================================================================================================================
# Prepare data for plotting
x_coords = mesh.coordinates().flatten()  # Mesh vertex
u_values = u.compute_vertex_values(mesh)  # Solution values at mesh vertices
a_values = np.array([a(xi) for xi in x_coords])  # Evaluate a(x) at mesh vertices
f_values = np.array([f(xi) for xi in x_coords])  # Evaluate f(x, t=final)
# Exact solution function u(x) = x(1-x)
u_exact = x_coords * (1 - x_coords)

# Create XML root element
root = ET.Element("Solution")

# Function to add properly indented text inside an element
def create_element_with_text(tag, values):
    elem = ET.SubElement(root, tag)
    formatted_text = "\n        " + " ".join(map(str, values)) + "\n    "
    elem.text = formatted_text
    return elem

# Add elements with formatted content
create_element_with_text("Point", x_coords)
create_element_with_text("PointData", u_values)

# Convert to pretty XML
xml_str = ET.tostring(root, encoding="utf-8")
pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")

# Save to file
with open("u_1D.xml", "w", encoding="utf-8") as f:
    f.write(pretty_xml)

# Plotting
plt.figure(figsize=(12, 6))

# Plot a(x)
plt.subplot(2, 2, 1)
plt.plot(x_coords, a_values, label=r"$a(x) = 1 + x$", color="blue")
plt.title(r"Wärmeleitungsfähigkeit")
plt.xlabel("x")
plt.ylabel(r"$a(x)$")
plt.grid()
plt.legend()

# Plot f(x)
plt.subplot(2, 2, 2)
plt.plot(x_coords, f_values, label=r"$f(x)$", color="orange")
plt.title(r"Source Term")
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x) = 1 + 4x$")
plt.grid()
plt.legend()

# Plot the final solution u(x) and u_exact
plt.subplot(2, 1, 2)
plt.plot(x_coords, u_values, label=r"$u_{a}(x)$", color="green")
plt.plot(x_coords, u_exact, label=r"$u_{e}(x) = x(1-x)$", color="red", linestyle="dashed")
plt.title(r"Approximierte Lösung und exakte Lösung der Wärmeleitungsgleichung")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x)$")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
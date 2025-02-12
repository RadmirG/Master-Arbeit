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
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof.Dr.Frank Haußer
# ======================================================================================================================

from fenics import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Mesh and function space
nx, ny = 50, 50
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)
# Time-stepping parameters
dt = 0.01
T = 1.0

# Approximation polynomials
V = FunctionSpace(mesh, "P", 2)

# Problem Conditions
# Boundary conditions
u_D = Constant(0.0)  # Dirichlet
bc = DirichletBC(V, u_D, "on_boundary")
# Initial condition
u_n = interpolate(Constant(0.0), V)

# Inner parameter
a = Expression("1 + u*u", u=u_n, degree=2)  # Example: a(u) = 1 + u^2
# Heat source
f = Expression("sin(pi*x[0])*sin(pi*x[1])*exp(-t)", t=0.0, degree=2)  # Example: f(x,y,t)

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

# List to save the solution for each time step
time_steps = []

while t < T:
    t += dt
    f.t = t  # Update time-dependent source term
    solve(a_form == L_form, u, bc)
    u_n.assign(u)  # Update for the next time step
    time_steps.append(u.copy(deepcopy=True))  # Save snapshot of u for animation

# Save the final solution
file = File("solution.pvd")
file << u


# ======================================================================================================================
# Animation with Matplotlib
# ======================================================================================================================

# Convert solution to NumPy for 3D plotting
def function_to_numpy(func, nx, ny):
    """Convert a FEniCS function to a NumPy 2D array for plotting."""
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    xv, yv = np.meshgrid(x, y)  # Create grid for plotting
    values = np.array([func(Point(x, y)) for x, y in zip(xv.flatten(), yv.flatten())])
    return xv, yv, values.reshape((nx + 1, ny + 1))


# Extract initial data for 3D animation
xv, yv, u_values = function_to_numpy(time_steps[0], nx, ny)

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 3D axis

# Initialize the surface plot
surf = ax.plot_surface(xv, yv, u_values, cmap="viridis", edgecolor="none")

# Set axis labels
ax.set_title(f"Heat Equation Solution u(x, y, t=0.00)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")

# Colorbar for surface plot
cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.set_label("u(x, y, t)")


# Update function for animation
def update_3d(frame):
    _, _, u_values = function_to_numpy(time_steps[frame], nx, ny)
    ax.clear()  # Clear previous surface
    # Update surface plot
    surf = ax.plot_surface(xv, yv, u_values, cmap="viridis", edgecolor="none")
    ax.set_title(f"Heat Equation Solution u(x, y, t={frame * dt:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    return (surf,)


# Create animation
anim = FuncAnimation(fig, update_3d, frames=len(time_steps), repeat=False)

# Show the animation
plt.show()

# ======================================================================================================================
# PLOTS
# ======================================================================================================================

# Plot diffusivity a
a_plot = interpolate(a, V)
plt.figure()
diffusivity_plot = plot(a_plot)  # Plot using FEniCS's function
plt.title("Heat Diffusivity: a(u) = 1 - u^3")
plt.colorbar(diffusivity_plot)  # Pass the mappable object (return value of plot) to colorbar
plt.show()

# Plot f(x, y, t) at t=0.5
f.t = 0.5
f_plot = interpolate(f, V)
plt.figure()
source_plot = plot(f_plot)  # Plot using FEniCS's function
plt.title("Heat Source Term: f(x, y, t=0.5)")
plt.colorbar(source_plot)  # Pass the mappable object to colorbar
plt.show()

# Visualize test function v
test_function_index = 0  # Index of the basis function to visualize
v_function = Function(V)  # Create a Function object in the same space as `v`
v_function.vector()[:] = 0  # Initialize all values to zero
v_function.vector()[test_function_index] = 1  # Set the value of a specific basis function

# Plot the basis function
plt.figure()
test_function_plot = plot(v_function)  # Create a plot for v as a basis function
plt.title(f"Test Function v: Basis Function {test_function_index}")
plt.colorbar(test_function_plot)
plt.show()



# ======================================================================================================================
# 1. Interactive 3D plot for the final solution u(x, y, T)
# ======================================================================================================================

# Convert solution to NumPy for 3D plotting
def function_to_numpy(func, nx, ny):
    """Convert a FEniCS function to a NumPy 2D array for plotting."""
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    xv, yv = np.meshgrid(x, y)  # Create grid for plotting
    values = np.array([func(Point(x, y)) for x, y in zip(xv.flatten(), yv.flatten())])
    return xv, yv, values.reshape((nx + 1, ny + 1))


# Interactive 3D plot for the final solution `u(x, y, T)` using Matplotlib
def plot_3d_solution(u, nx=50, ny=50, title="Solution u(x, y, T)", cmap="viridis"):
    xv, yv, u_values = function_to_numpy(u, nx=nx, ny=ny)

    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(xv, yv, u_values, cmap=cmap, edgecolor='none')

    # Set axis labels and title
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    # Add colorbar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label("u(x, y)")

    # Display plot
    plt.show()


# Example: Plot diffusivity `a(u)` using Matplotlib
def plot_3d_diffusivity(a, V, nx=50, ny=50):
    a_plot = interpolate(a, V)
    plot_3d_solution(
        a_plot,
        nx=nx,
        ny=ny,
        title="Heat Diffusivity: a(u)",
        cmap="plasma"
    )


# Example: Plot source term `f(x, y, t)` at t=0.5 using Matplotlib
def plot_3d_source_term(f, V, nx=50, ny=50, t=0.5):
    f.t = t  # Set time for the source term
    f_plot = interpolate(f, V)
    plot_3d_solution(
        f_plot,
        nx=nx,
        ny=ny,
        title=f"Heat Source Term: f(x, y, t={t})",
        cmap="cividis"
    )


# Example: Animate solution `u(x, y, t)` over time using Matplotlib
def animate_3d_solution(time_steps, nx=50, ny=50, dt=0.01, zlim=None):
    xv, yv, u_values = function_to_numpy(time_steps[0], nx=nx, ny=ny)

    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Solution u(x, y, t)")

    # Plot surface
    surf = ax.plot_surface(xv, yv, u_values, cmap="viridis", edgecolor='none')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    if zlim:
        ax.set_zlim(zlim)

    # Update function for animation
    def update(frame):
        _, _, u_values = function_to_numpy(time_steps[frame], nx=nx, ny=ny)
        ax.clear()
        ax.plot_surface(xv, yv, u_values, cmap="viridis", edgecolor='none')
        ax.set_title(f"Solution u(x, y, t={frame * dt:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        if zlim:
            ax.set_zlim(zlim)

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(time_steps), repeat=False)
    plt.show()


# ======================================================================================================================
# EXAMPLES OF USAGE
# ======================================================================================================================

# Problem setup remains unchanged...
# (Refer to the original code above for time-stepping loop and FEniCS calculations)

# Final solution plot
plot_3d_solution(u, nx=50, ny=50)

# Diffusivity plot
plot_3d_diffusivity(a, V, nx=50, ny=50)

# Source term plot at t = 0.5
plot_3d_source_term(f, V, nx=50, ny=50, t=0.5)

# Animation over time
animate_3d_solution(time_steps, nx=50, ny=50, dt=dt)

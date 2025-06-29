import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

# Load the saved solution
data = np.load("sol_1D_td_1000x1000.npz")
u_sol = data["u_sol"]
X_test = data["X_test"]
T_test = data["T_test"]

# Example plot: surface plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_test, T_test, u_sol, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
plt.title("Stored heat solution")
plt.show()

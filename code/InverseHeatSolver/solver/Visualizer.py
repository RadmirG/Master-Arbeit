import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('font', family='serif')
import shutil
if shutil.which("latex") is not None:
    plt.rc('text', usetex=True)
else:
    plt.rc('text', usetex=False)

# plt.locator_params(axis='x', nbins=4)
# plt.locator_params(axis='y', nbins=4)

plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "serif",
    "font.size": 16,              # Standardgröße
    "axes.titlesize": 16,         # Titel der Subplots
    "axes.labelsize": 16,         # Achsenbeschriftung
    "xtick.labelsize": 16,        # X-Tick Labels
    "ytick.labelsize": 16,        # Y-Tick Labels
    "legend.fontsize": 16,        # Legende
    "figure.titlesize": 16        # Gesamttitel
})

import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.widgets import Slider


def plot_losses(loss_history): # , loss_labels=None):
    # === Plot U-loss ===
    if "u_history" in loss_history:
        plt.figure()
        plt.semilogy(loss_history["u_history"]["steps"], loss_history["u_history"]["loss"], label=r"$\mathcal{L}_u$")
        plt.title(r"$\mathcal{L}_u$")
        plt.xlabel("Iterations")
        #plt.ylabel(r"$\mathcal{L}_u$")
        plt.grid()
        plt.legend()
        plt.show()

    # === Plot F-loss ===
    if "f_history" in loss_history and loss_history["f_history"]["loss"] is not None:
        plt.figure()
        plt.semilogy(loss_history["f_history"]["steps"], loss_history["f_history"]["loss"], label=r"$\mathcal{L}_f$")
        plt.title(r"$\mathcal{L}_f$")
        plt.xlabel("Iterations")
        # plt.ylabel(r"$\mathcal{L}_f$")
        plt.grid()
        plt.legend()
        plt.show()

    # === Plot A-losses ===
    if "a_history" in loss_history:
        a_hist = loss_history["a_history"]
        steps = a_hist["steps"]
        plt.figure()
        if "loss" in a_hist:
            plt.semilogy(steps, a_hist["loss"], label=r"$\mathcal{L}$")
            # num_samples = 50
            # indices = np.linspace(0, len(steps) - 1, num_samples, dtype=int)  # Gleichmäßig verteilte Indizes
            # x_sampled = np.array(steps)[indices]
            # u_exact_sampled = np.array(a_hist["loss"])[indices]
            # plt.scatter(x_sampled, u_exact_sampled, label=r"$\mathcal{L}$", color='r', s=5)
        if "pde_loss" in a_hist and len(a_hist["pde_loss"]) > 0:
            plt.semilogy(steps, a_hist["pde_loss"], label=r"$\mathcal{L}_{PDE}$")
        if "a_grad_loss" in a_hist and len(a_hist["a_grad_loss"]) > 0:
            plt.semilogy(steps, a_hist["a_grad_loss"],
                         label=a_hist["label_a_grad_loss"] if "label_a_grad_loss" in a_hist else r"$L_{|\nabla a|}$")
        if "gPINN_loss" in a_hist and len(a_hist["gPINN_loss"]) > 0:
            plt.semilogy(steps, a_hist["gPINN_loss"], label=r"$\mathcal{L}_{gPINN}$")
        plt.title(a_hist["title"] if "title" in a_hist else r"$\mathcal{L}_{PINN}$")
        plt.xlabel("Iterations")
        # plt.ylabel(r"$\mathcal{L}$")
        plt.grid()
        plt.legend()
        plt.show()


def plot_1D(x, u_pred, x_obs=None, a_pred=None, u_obs=None, f_obs=None, a_exact=None, f_pred=None, u_exact=None,
            f_exact=None, a_obs=None, title=None):
    f_exact_min, f_exact_max, f_pred_min, f_pred_max, f_obs_min, f_obs_max = [0] * 6
    a_exact_min, a_exact_max, a_pred_min, a_pred_max, a_obs_min, a_obs_max = [0] * 6
    plt.figure(figsize=(12, 6))
    # Plot a(x)
    plt.subplot(2, 2, 1)
    if a_exact is not None:
        plt.plot(x, a_exact, label=r"$a(x)$ : exact")
        a_exact_min, a_exact_max = np.min(a_exact), np.max(a_exact)
    if a_pred is not None:
        plt.plot(x, a_pred, label=r"$a_{l}(x)$ : learned", color='green')
        a_pred_min, a_pred_max = np.min(a_pred), np.max(a_pred)
    if a_obs is not None:
        plt.scatter(x_obs, a_obs, label=r"$a(x)$ : observed", color='r', s=5)
        a_obs_min, a_obs_max = np.min(a_obs), np.max(a_obs)
    plt.title(r"Wärmeleitfähigkeit")
    plt.xlabel("x")
    plt.ylabel(r"$a(x)$")
    plt.grid()
    if a_obs is not None or a_pred is not None or a_exact is not None:
        plt.legend()
        plt.ticklabel_format(useOffset=False)
        plt.ylim([np.min([a_exact_min, a_pred_min, a_obs_min])-1,
              np.max([a_exact_max, a_pred_max, a_obs_max])+1])
        #plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    # Plot f(x)
    plt.subplot(2, 2, 2)
    if f_exact is not None:
        plt.plot(x, f_exact, label=r"$f(x)$ : exact", color='orange')
        f_exact_min, f_exact_max = np.min(f_exact), np.max(f_exact)
    if f_pred is not None:
        plt.plot(x, f_pred, label=r"$f_{l}(x)$ : learned", color='green')
        f_pred_min, f_pred_max = np.min(f_pred), np.max(f_pred)
    if f_obs is not None:
        plt.scatter(x_obs, f_obs, label=r"$f(x)$ : observed", color='r', s=5)
        f_obs_min, f_obs_max = np.min(f_obs), np.max(f_obs)
    plt.title(r"Source Term")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.grid()
    if f_pred is not None or f_exact is not None or f_obs is not None:
        plt.legend()
        plt.ticklabel_format(useOffset=False)
        plt.ylim([np.min([f_exact_min, f_pred_min, f_obs_min])-1,
              np.max([f_exact_max, f_pred_max, f_obs_max])+1])
        #plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    # Plot u(x)
    plt.subplot(2, 1, 2)
    if u_exact is not None:
        plt.plot(x, u_exact, label=r"$u(x)$ : exact", color='orange')
        # num_samples = 50
        # indices = np.linspace(0, len(x) - 1, num_samples, dtype=int)  # Gleichmäßig verteilte Indizes
        # x_sampled = x[indices]
        # u_exact_sampled = u_exact[indices]
        # plt.scatter(x_sampled, u_exact_sampled, label=r"$u(x)$ : exact", color='r', s=5)
    plt.plot(x, u_pred, label=r"$u_{l}(x)$ : learned", color='green')
    if u_obs is not None:
        plt.scatter(x_obs, u_obs, label=r"$u(x)$ : observed", color='r', s=5)
    plt.title(title if title is not None else r"Gelernte Lösung und gemessene Werte")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_3d(x, y, u_pred, f_pred, a_pred=None, is_time_plot=False):
    # Ensure all arrays have the correct shape
    if f_pred.ndim == 1:
        f_pred = f_pred.reshape(x.shape)
    if u_pred.shape != x.shape:
        try:
            u_pred = u_pred.reshape(x.shape)
        except ValueError:
            print(f"Error: Cannot reshape u_pred with shape {u_pred.shape} to match x with shape {x.shape}")
            return
    if a_pred is not None and a_pred.ndim == 1:
        a_pred = a_pred.reshape(x.shape)

    if a_pred is None:
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2)
    else:
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # Plot u(x,t) or u(x,y)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot_surface(x, y, u_pred, cmap='viridis')
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$" if is_time_plot else r"$y$")
    ax1.set_title(r"$u(x,t)$" if is_time_plot else r"$u(x,y,0)$")

    # Plot f(x,t) or f(x,y)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot_surface(x, y, f_pred, cmap='plasma')
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$t$" if is_time_plot else r"$y$")
    ax2.set_title(r"$f(x,t)$" if is_time_plot else r"$f(x,y,0)$")

    # Plot a(x,y) if given
    if a_pred is not None:
        ax3 = fig.add_subplot(gs[1, :], projection='3d')
        ax3.plot_surface(x, y, a_pred, cmap='cividis')
        ax3.set_xlabel(r"$x$")
        ax3.set_ylabel(r"$t$" if is_time_plot else r"$y$")
        ax3.set_title(r"$a(x)$" if is_time_plot else r"$a(x,y)$")

    plt.tight_layout()
    plt.show()


def time_plot(X, range_t, sizeof_t, u_pred, f_pred=None, a_pred=None, u_exct=None, f_exct=None, a_exct=None):
    f_exct_min, f_exct_max, f_pred_min, f_pred_max = [0] * 4
    a_exct_min, a_exct_max, a_pred_min, a_pred_max = [0] * 4
    u = u_exct(np.column_stack((X, np.repeat(0, len(X)))))
    u_exct_min = np.min(u)
    u_exct_max = np.max(u)
    u_pred_min = np.min(u_pred)
    u_pred_max = np.max(u_pred)
    if f_exct is not None:
        f = f_exct(np.column_stack((X, np.repeat(0, len(X)))))
        f_exct_min = np.min(f)
        f_exct_max = np.max(f)
    if f_pred is not None:
        f_pred_min = np.min(f_pred)
        f_pred_max = np.max(f_pred)
    if a_exct is not None:
        a = a_exct(np.column_stack((X, np.repeat(0, len(X)))))
        a_exct_min = np.min(a)
        a_exct_max = np.max(a)
    if a_pred is not None:
        a_pred_min = np.min(a_pred)
        a_pred_max = np.max(a_pred)
    # Initial plots
    fig, (ax_u, ax_f, ax_a) = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    line_u, = ax_u.plot(X, u_pred[0, :], label=r"$u_{l}(x,t)$")
    if u_exct is not None:
        u_exact = u_exct(np.column_stack((X, np.repeat(0, len(X)))))
        line_u_exact, = ax_u.plot(X, u_exact, label=r"$u(x,t)$", linestyle=":")
    ax_u.set_title(r"$u(x,t)$")
    ax_u.set_xlabel("x")
    ax_u.grid()
    ax_u.set_ylim([min([u_pred_min, u_exct_min]) - 0.2,
                   max([u_pred_max, u_exct_max]) + 0.2])
    ax_u.legend()

    if f_pred is not None:
        line_f, = ax_f.plot(X, f_pred[0, :], label=r"$f_{l}(x,t)$")
    if f_exct is not None:
        f_exact = f_exct(np.column_stack((X, np.repeat(0, len(X)))))
        line_f_exact, = ax_f.plot(X, f_exact, label=r"$f(x,t)$", linestyle=":")
    ax_f.set_title("$f(x,t)$")
    ax_f.set_xlabel("x")
    ax_f.grid()
    if f_exct is not None:
        ax_f.set_ylim([min([f_pred_min, f_exct_min]) - 0.2,
                       max([f_pred_max, f_exct_max]) + 0.2])
    ax_f.legend()

    if a_pred is not None:
        line_a, = ax_a.plot(X, a_pred[0, :], label=r"$a_{l}(x)$")
    if a_exct is not None:
        a_exact = a_exct(np.column_stack((X, np.repeat(0, len(X)))))
        line_a_exact, = ax_a.plot(X, a_exact, label=r"$a(x)$", linestyle=":")
    ax_a.set_title("$a(x)$")
    ax_a.set_xlabel("x")
    ax_a.grid()
    if a_pred is not None:
        ax_a.set_ylim([min([a_pred_min, a_exct_min]) - 0.2,
                       max([a_pred_max, a_exct_max]) + 0.2])
    ax_a.legend()

    # Slider setup
    ax_slider = plt.axes([0.25, 0.03, 0.65, 0.03])
    t_slider = Slider(ax_slider, 'Time', 0, range_t, valinit=0)

    def update(val):
        t = t_slider.val
        t_idx = int(t / range_t * (sizeof_t-1))  # assuming t ranges from 0 to 6
        xt = np.column_stack((X, np.repeat(t, len(X))))
        line_u.set_ydata(u_pred[t_idx, :])
        if u_exct is not None:
            line_u_exact.set_ydata(u_exct(xt))

        if f_pred is not None:
            line_f.set_ydata(f_pred[t_idx, :])
        if f_exct is not None:
            line_f_exact.set_ydata(f_exct(xt))

        if a_pred is not None:
            line_a.set_ydata(a_pred[t_idx, :])
        if a_exct is not None:
            line_a_exact.set_ydata(a_exct(xt))

        fig.canvas.draw_idle()

    t_slider.on_changed(update)
    plt.show()


def animation_3d(x, y, t, range_t, u, f):
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)

    u_min, u_max = np.min(u), np.max(u)
    f_min, f_max = np.min(f), np.max(f)

    axes[0].plot_surface(x, y, u[:, :, 0], cmap="viridis")
    axes[0].set(title="Temperature $u(x,y,t)$", xlabel="x", ylabel="y", zlabel="u", zlim=(u_min, u_max))
    axes[1].plot_surface(x, y, f[:, :, 0], cmap="plasma")
    axes[1].set(title="Heat Source $f(x,y,t)$", xlabel="x", ylabel="y", zlabel="f", zlim=(f_min, f_max))

    # Slider setup
    slider_ax = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(slider_ax, 'Time t', 0, 6, valinit=0)

    def update(val):
        idx = int((val / range_t) * (len(t) - 1))
        axes[0].clear()
        axes[1].clear()

        axes[0].plot_surface(x, y, u[:, :, idx], cmap='viridis')
        axes[0].set(title=f"Temperature $u(x,y,t)$ at t={val:.2f}",
                    xlabel="x", ylabel="y", zlabel="u", zlim=(u_min, u_max))

        axes[1].plot_surface(x, y, f[:, :, idx], cmap='plasma')
        axes[1].set(title=f"Heat Source $f(x,y,t)$ at t={val:.2f}",
                    xlabel="x", ylabel="y", zlabel="f", zlim=(f_min, f_max))

        fig.canvas.draw_idle()

    slider.on_changed(update)

    def animate(i):
        val = (i % len(t)) * range_t / len(t)
        slider.set_val(val)

    FuncAnimation(fig, animate, frames=len(t), interval=100, repeat=True)

    plt.show()

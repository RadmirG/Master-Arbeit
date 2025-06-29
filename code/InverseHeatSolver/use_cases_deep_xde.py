import numpy as np

# --------------------------------------------------------------------------------------------------------------
# local packages
from solver.InverseHeatSolver import InverseHeatSolver
from solver import Visualizer
import functions # Defines all used functions

# --------------------------------------------------------------------------------------------------------------

# seed = 42  # Choose any integer seed
# np.random.seed(seed)


def add_noise_to_measurements(data, eps):
    return data + eps * np.random.randn(*data.shape)


# ======================================================================================================================
# Main script
if __name__ == "__main__":
    case_1 = False  # 1D, time independent
    case_2 = False  # 1D, time dependent
    case_3 = False  # 2D, time independent
    case_4 = False  # 2D, time dependent
    case_5 = True  # 1D, time dependent, f=0 => ∂u − ∇⋅(a∇u) = 0
    case_6 = False  # 2D, time dependent, f=0 => ∂u − ∇⋅(a∇u) = 0

    # ==================================================================================================================
    # USE CASE 1.
    # ------------------------------------------------------------------------------------------------------------------
    if case_1:

        # Evaluate functions
        num_points = 100
        obs_dom = np.linspace(0, 1, num_points).reshape(num_points, 1)

        #u_obs = functions.u_1D_ti(obs_dom)
        #f_obs = functions.f_1D_ti(obs_dom)
        u_obs = add_noise_to_measurements(functions.u_1D_ti(obs_dom), 0.005)
        f_obs = add_noise_to_measurements(functions.f_1D_ti(obs_dom), 0.01)
        # --------------------------------------------------------------------------------------------------------------
        # Solver parameters

        domain = {'x_domain': [0, 1, 1000], 'y_domain': None, 't_domain': None}
        nn_dims = {'num_layers': 2, 'num_neurons': 10}
        obs_values = {'dom_obs': obs_dom,
                      'u_obs': u_obs,
                      'f_obs': f_obs}
        loss_weights = {'w_PDE_loss': 1 / 5,
                        'a_grad_loss': 1e-3,
                        'gPINN_loss' : 1e-3
                        }
        learning_rate = 1e-2

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate solver

        use_solved_model = False
        inv_solver = None

        if use_solved_model:
            inv_solver = InverseHeatSolver.restore_model_and_params("models/1D_ti_noisy_u_0.005_f_0.01")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate, True)
            inv_solver.train(a_iterations=20000, u_iterations=10000, f_iterations=20000,
                             use_regularization=True, use_gPINN=True,
                             display_results_every=500, save_path="models/1D_ti_noisy_u_0.005_f_0.01")

        import matplotlib
        matplotlib.use('TkAgg')

        loss_labels = [
            r"$L_{PDE}$"
        ]
        Visualizer.plot_losses(inv_solver.history)


        num_test_points = 1000
        x_test = np.linspace(0, 1, num_test_points).reshape(num_test_points, 1)
        outputs = inv_solver.predict(x_test)

        u_pred = outputs[0]
        f_pred = outputs[1]
        a_pred = outputs[2]

        # exact functions
        u_exact = functions.u_1D_ti(x_test)
        a_exact = functions.a_1D_ti(x_test)
        f_exact = functions.f_1D_ti(x_test)

        Visualizer.plot_1D(x_test, u_pred, obs_dom, a_pred, u_obs, f_obs, a_exact, f_pred, u_exact, f_exact, None)
        #Visualizer.plot_1D(x_test, u_pred, None, a_pred, None, None, None,
        #                   None, None, None, None)

        inv_solver.print_l2_error(u_exact, u_pred, "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact, a_pred, "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, f_pred, "f_exact", "f_pred")

        print(inv_solver.format_training_time())

    if case_2:

        # Evaluate functions
        num_x_points = 55
        num_t_points = 100
        x = np.linspace(0, 1, num_x_points).reshape(num_x_points, 1)
        t = np.linspace(0, 6, num_t_points).reshape(num_t_points, 1)
        X, T = np.meshgrid(x, t)
        xt = np.column_stack((X.flatten(), T.flatten()))
        u_obs = functions.u_1D_td(xt)
        f_obs = functions.f_1D_td(xt)

        # --------------------------------------------------------------------------------------------------------------
        # Solver parameters

        domain = {'x_domain': [0, 1, 500], 'y_domain': None, 't_domain': [0, 6, 100]}
        nn_dims = {'num_layers': 3, 'num_neurons': 20}
        obs_values = {'dom_obs': xt,
                      'u_obs': u_obs,
                      'f_obs': f_obs}
        loss_weights = {'w_PDE_loss': 1 / 5,
                        'a_grad_loss': 1e-3,
                        'gPINN_loss' : 1e-3}
        learning_rate = 1e-2

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate solver

        use_solved_model = True
        inv_solver = None

        if use_solved_model:
            inv_solver = InverseHeatSolver.restore_model_and_params("models/1D_td_dde")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate, True)
            inv_solver.train(a_iterations=20000, u_iterations=20000, f_iterations=25000,
                             use_regularization=True, use_gPINN=True,
                             display_results_every=500, save_path="models/1D_td_dde")

        loss_labels = [
            r"$L_{PDE}$"
        ]

        import matplotlib
        matplotlib.use('TkAgg')

        Visualizer.plot_losses(inv_solver.history)

        sizeof_t = 100
        range_t = 6
        X = np.linspace(0, 1, 1000)
        T = np.linspace(0, range_t, sizeof_t)
        X_test, T_test = np.meshgrid(X, T)
        XT = np.column_stack((X_test.flatten(), T_test.flatten()))
        outputs = inv_solver.predict(XT)

        u_pred = outputs[0].reshape(X_test.shape)
        f_pred = outputs[1].reshape(X_test.shape)
        a_pred = outputs[2].reshape(X_test.shape)

        # exact functions
        u_exact = functions.u_1D_td(XT)
        a_exact = functions.a_1D_td(XT)
        f_exact = functions.f_1D_td(XT)

        inv_solver.print_l2_error(u_exact, outputs[0], "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact.reshape(X_test.shape)[0, :],
                                  a_pred[0, :], "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, outputs[1], "f_exact", "f_pred")
        print(inv_solver.format_training_time())

        Visualizer.plot_3d(X_test, T_test, u_pred, f_pred, a_pred, is_time_plot=True)
        Visualizer.time_plot(X, range_t, sizeof_t, u_pred, f_pred, a_pred,
                             functions.u_1D_td, functions.f_1D_td, functions.a_1D_td)

        import matplotlib.pyplot as plt
        from matplotlib import rc
        plt.rc('font', family='serif')
        import shutil
        if shutil.which("latex") is not None:
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
        plt.rcParams.update({
            # "text.usetex": True,
            "font.family": "serif",
            "font.size": 16,  # Standardgröße
            "axes.titlesize": 16,  # Titel der Subplots
            "axes.labelsize": 16,  # Achsenbeschriftung
            "xtick.labelsize": 16,  # X-Tick Labels
            "ytick.labelsize": 16,  # Y-Tick Labels
            "legend.fontsize": 16,  # Legende
            "figure.titlesize": 16  # Gesamttitel
        })
        fig, (ax_a) = plt.subplots(1, 1, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        if a_pred is not None:
            line_a, = ax_a.plot(X, a_pred[0, :], label=r"$a_{l}(x)$")
        if a_exact is not None:
            line_a_exact, = ax_a.plot(X, a_exact.reshape(X_test.shape)[0, :], label=r"$a(x)$", linestyle=":")
        ax_a.set_title(r"$a(x)$")
        ax_a.set_xlabel(r"$x$")
        ax_a.grid()
        ax_a.legend()
        plt.show()

    if case_3:

        # Evaluate functions
        num_x_points = 55
        num_y_points = 55
        # x_obs = np.random.rand(num_points).reshape(num_points, 1)
        x = np.linspace(0, 1, num_x_points).reshape(num_x_points, 1)
        y = np.linspace(0, 1, num_y_points).reshape(num_y_points, 1)
        X, Y = np.meshgrid(x, y)
        xy = np.column_stack((X.flatten(), Y.flatten()))
        u_obs = functions.u_2D_ti(xy)
        f_obs = functions.f_2D_ti(xy)

        # --------------------------------------------------------------------------------------------------------------
        # Solver parameters

        domain = {'x_domain': [0, 1, 500], 'y_domain': [0, 1, 500], 't_domain': None}
        nn_dims = {'num_layers': 3, 'num_neurons': 50}
        obs_values = {'dom_obs': xy,
                      'u_obs': u_obs,
                      'f_obs': f_obs}
        loss_weights = {'w_PDE_loss': 1 / 5,
                        'a_grad_loss': 0
                        }
        learning_rate = 1e-2

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate solver

        use_solved_model = True
        inv_solver = None

        if use_solved_model:
            inv_solver = InverseHeatSolver.restore_model_and_params("models/2D_ti_dde")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate, True)
            inv_solver.train(a_iterations=25000, u_iterations=25000, f_iterations=35000,
                             display_results_every=500, save_path="models/2D_ti_dde")

        loss_labels = [
            r"$L_{PDE}$"
        ]

        import matplotlib
        matplotlib.use('TkAgg')

        Visualizer.plot_losses(inv_solver.history)

        X = np.linspace(0, 1, 1000)
        Y = np.linspace(0, 1, 1000)
        X_test, Y_test = np.meshgrid(X, Y)
        XY = np.column_stack((X_test.flatten(), Y_test.flatten()))
        outputs = inv_solver.predict(XY)

        u_pred = outputs[0].reshape(X_test.shape)
        f_pred = outputs[1].reshape(X_test.shape)
        a_pred = outputs[2].reshape(X_test.shape)

        # exact functions
        u_exact = functions.u_2D_ti(XY).reshape(X_test.shape)
        a_exact = functions.a_2D_ti(XY).reshape(X_test.shape)
        f_exact = functions.f_2D_ti(XY).reshape(X_test.shape)

        inv_solver.print_l2_error(u_exact, u_pred, "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact, a_pred, "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, f_pred, "f_exact", "f_pred")
        print(inv_solver.format_training_time())

        Visualizer.plot_3d(X_test, Y_test, u_pred, f_pred, a_pred, is_time_plot=False)
        Visualizer.time_plot(X, 1, 100, u_pred, f_pred, a_pred,
                             functions.u_2D_ti, functions.f_2D_ti, functions.a_2D_ti)

    if case_4:

        # Evaluate functions
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        t = np.linspace(0, 6, 50)
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        xyt = np.column_stack((X.flatten(), Y.flatten(), T.flatten()))
        u_obs = functions.u_2D_td(xyt)
        f_obs = functions.f_2D_td(xyt)

        # --------------------------------------------------------------------------------------------------------------
        # Solver parameters

        domain = {'x_domain': [0, 1, 100], 'y_domain': [0, 1, 100], 't_domain': [0, 1, 5]}
        nn_dims = {'num_layers': 3, 'num_neurons': 20}
        obs_values = {'dom_obs': xyt,
                      'u_obs': u_obs,
                      'f_obs': f_obs}
        loss_weights = {'w_PDE_loss': 1 / 5,
                        'a_grad_loss': 0,
                        'gPINN_loss' : 1e-2
                        }
        learning_rate = 1e-2

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate solver

        use_solved_model = True
        inv_solver = None

        if use_solved_model:
            inv_solver = InverseHeatSolver.restore_model_and_params("models/2D_td_dde"

                                                                    ) #_100x100x5_reg_gPINN")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate, True)
            inv_solver.train(a_iterations=10000, u_iterations=35000, f_iterations=40000,
                             use_regularization=True, use_gPINN=True,
                             display_results_every=500, save_path="models/2D_td_dde") #_100x100x5_reg_gPINN")

        loss_labels = [
            r"$L_{PDE}$"
        ]

        import matplotlib
        matplotlib.use('TkAgg')

        Visualizer.plot_losses(inv_solver.history)

        x_num = 100
        y_num = 100
        t_num = 100
        X = np.linspace(0, 1, x_num)
        Y = np.linspace(0, 1, y_num)
        T = np.linspace(0, 6, t_num)

        X_test_2D, Y_test_2D = np.meshgrid(X, Y)
        XY = np.column_stack((X_test_2D.flatten(), Y_test_2D.flatten()))

        X_test, Y_test, T_test = np.meshgrid(X, Y, T)
        XYT = np.column_stack((X_test.flatten(), Y_test.flatten(), T_test.flatten()))
        outputs = inv_solver.predict(XYT)

        u_pred = outputs[0].reshape(x_num, y_num, t_num)
        f_pred = outputs[1].reshape(x_num, y_num, t_num)
        a_pred = outputs[2].reshape(x_num, y_num, t_num)

        # exact functions
        u_exact = functions.u_2D_td(XYT).reshape(x_num, y_num, t_num)
        # a_exact = functions.a_2D_td(XYT).reshape(x_num, y_num, t_num)
        a_exact = functions.a_2D_td(XY).reshape(x_num, y_num)
        f_exact = functions.f_2D_td(XYT).reshape(x_num, y_num, t_num)

        inv_solver.print_l2_error(u_exact, u_pred, "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact, a_pred[:,:,0], "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, f_pred, "f_exact", "f_pred")
        print(inv_solver.format_training_time())

        x_mesh, y_mesh = np.meshgrid(X, Y)
        Visualizer.plot_3d(x_mesh, y_mesh, u_pred[:, :, 0], f_pred[:, :, 0], a_pred[:, :, 99])
        Visualizer.animation_3d(x_mesh, y_mesh, T, 6, u_pred, f_pred)

    if case_5:

        # Load the saved solution from FEniCS
        data = np.load("FEniCS_scripts/sol_1D_td_100x100.npz")
        u_obs = data["u_sol"].reshape(-1, 1)
        X = data["X_test"]
        T = data["T_test"]
        xt = np.column_stack((X.flatten(), T.flatten()))
        # --------------------------------------------------------------------------------------------------------------
        # Solver parameters

        domain = {'x_domain': [0, 1, 1000], 'y_domain': None, 't_domain': [0, 0.5, 10]}
        nn_dims = {'num_layers': 10, 'num_neurons': 100}
        obs_values = {'dom_obs': xt,
                      'u_obs': u_obs,
                      'f_obs': None}
        loss_weights = {'w_PDE_loss': 1 / 5,
                        'a_grad_loss': 1e-3,
                        'gPINN_loss' : 1e-1}
        learning_rate = 1e-2

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate solver

        use_solved_model = False
        inv_solver = None



        if use_solved_model:
            inv_solver = InverseHeatSolver.restore_model_and_params("models/1D_td_dde_homogen_1000x1000")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate, True,
                                           load_prelearned_u_model=True, load_prelearned_a_model=False)
            inv_solver.train(a_iterations=4000, u_iterations=15000, f_iterations=0,
                             use_regularization=True, use_gPINN=True,
                             display_results_every=500, save_path="models/1D_td_dde_homogen_1000x1000")

        loss_labels = [
            r"$L_{PDE}$"
        ]

        import matplotlib
        matplotlib.use('TkAgg')

        Visualizer.plot_losses(inv_solver.history)

        sizeof_t = 100
        range_t = 0.5
        X = np.linspace(0, 1, 100)
        T = np.linspace(0, range_t, sizeof_t)
        X_test, T_test = np.meshgrid(X, T)
        XT = np.column_stack((X_test.flatten(), T_test.flatten()))
        outputs = inv_solver.predict(XT)

        u_pred = outputs[0].reshape(X_test.shape)
        f_pred = outputs[1]#.reshape(X_test.shape)
        a_pred = outputs[2].reshape(X_test.shape)

        # exact functions
        u_exact = functions.u_1D_td(XT)
        a_exact = functions.a_1D_td(XT)
        f_exact = functions.f_1D_td(XT)

        inv_solver.print_l2_error(u_exact, outputs[0], "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact.reshape(X_test.shape)[0, :],
                                  a_pred[0, :], "a_exact", "a_pred")
        print(inv_solver.format_training_time())

        Visualizer.plot_3d(X_test, T_test, u_pred, u_pred, a_pred, is_time_plot=True)
        Visualizer.time_plot(X, range_t, sizeof_t, u_pred, u_pred, a_pred,
                             functions.u_1D_td, functions.f_1D_td, functions.a_1D_td)

        import matplotlib.pyplot as plt
        from matplotlib import rc
        plt.rc('font', family='serif')
        import shutil
        if shutil.which("latex") is not None:
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
        plt.rcParams.update({
            # "text.usetex": True,
            "font.family": "serif",
            "font.size": 16,  # Standardgröße
            "axes.titlesize": 16,  # Titel der Subplots
            "axes.labelsize": 16,  # Achsenbeschriftung
            "xtick.labelsize": 16,  # X-Tick Labels
            "ytick.labelsize": 16,  # Y-Tick Labels
            "legend.fontsize": 16,  # Legende
            "figure.titlesize": 16  # Gesamttitel
        })
        fig, (ax_a) = plt.subplots(1, 1, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        if a_pred is not None:
            line_a, = ax_a.plot(X, a_pred[0, :], label=r"$a_{l}(x)$")
        if a_exact is not None:
            line_a_exact, = ax_a.plot(X, a_exact.reshape(X_test.shape)[0, :], label=r"$a(x)$", linestyle=":")
        ax_a.set_title(r"$a(x)$")
        ax_a.set_xlabel(r"$x$")
        ax_a.grid()
        ax_a.legend()
        plt.show()

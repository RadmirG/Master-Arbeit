import numpy as np

# --------------------------------------------------------------------------------------------------------------
# local packages
from InverseHeatSolver.InverseHeatSolver import InverseHeatSolver
from InverseHeatSolver import Visualizer
import functions # Defines all used functions

# --------------------------------------------------------------------------------------------------------------

seed = 42  # Choose any integer seed
np.random.seed(seed)

# ======================================================================================================================
# Main script
if __name__ == "__main__":
    case_1 = True    # 1D, time independent
    case_2 = True    # 1D, time dependent
    case_3 = True     # 2D, time independent
    case_4 = True    # 2D, time dependent

    # ==================================================================================================================
    # USE CASE 1.
    # ------------------------------------------------------------------------------------------------------------------
    if case_1:

        # Evaluate functions
        num_points = 25
        obs_dom = np.linspace(0, 1, num_points).reshape(num_points, 1)

        u_obs = functions.u_1D_ti(obs_dom)
        f_obs = functions.f_1D_ti(obs_dom)

        # --------------------------------------------------------------------------------------------------------------
        # Solver parameters

        domain = {'x_domain': [0, 1, 10000], 'y_domain': None, 't_domain': None}
        nn_dims = {'num_layers': 3, 'num_neurons': 40}
        obs_values = {'dom_obs': obs_dom,
                      'u_obs': u_obs,
                      'f_obs': f_obs}
        loss_weights = {'w_PDE_loss': 1/5,
                        'a_grad_loss' : 0
                        }
        learning_rate = 1e-2

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate solver

        use_solved_model = True
        inv_solver = None

        if use_solved_model:
            inv_solver = InverseHeatSolver.restore_model_and_params("models/1D_ti")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate)
            inv_solver.train(a_iterations=10000, u_iterations=10000, f_iterations=15000,
                             display_results_every=500, save_path="models/1D_ti")

        loss_labels = [
            r"$L_{PDE}$",
            # r"$L_{|\nabla a|}$"
        ]
        Visualizer.plot_losses(inv_solver.history)

        #import matplotlib
        #matplotlib.use('TkAgg')

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

        domain = {'x_domain': [0, 1, 1000], 'y_domain': None, 't_domain': [0, 6, 100]}
        nn_dims = {'num_layers': 3, 'num_neurons': 40}
        obs_values = {'dom_obs': xt,
                      'u_obs': u_obs,
                      'f_obs': f_obs}
        loss_weights = {'w_PDE_loss': 1/5,
                        'a_grad_loss' : 0
                        }
        learning_rate = 1e-2

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate solver

        use_solved_model = True
        inv_solver = None

        if use_solved_model:
            inv_solver = InverseHeatSolver.restore_model_and_params("models/1D_td")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate)
            inv_solver.train(a_iterations=20000, u_iterations=20000, f_iterations=25000,
                             display_results_every=500, save_path="models/1D_td")

        loss_labels = [
            r"$L_{PDE}$",
            # r"$L_{|\nabla a|}$"
        ]
        Visualizer.plot_losses(inv_solver.history)

        sizeof_t = 100
        range_t = 6
        X = np.linspace(0, 1, 100)
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
        inv_solver.print_l2_error(a_exact, outputs[2], "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, outputs[1], "f_exact", "f_pred")
        print(inv_solver.format_training_time())

        Visualizer.plot_3d(X_test, T_test, u_pred, f_pred, a_pred, is_time_plot=True)

        #import matplotlib
        #matplotlib.use('TkAgg')
        Visualizer.time_plot(X, range_t, sizeof_t, u_pred, f_pred, a_pred,
                             functions.u_1D_td, functions.f_1D_td, functions.a_1D_td)

    if case_3:

       # Evaluate functions
        num_x_points = 105
        num_y_points = 105
        # x_obs = np.random.rand(num_points).reshape(num_points, 1)
        x = np.linspace(0, 1, num_x_points).reshape(num_x_points, 1)
        y = np.linspace(0, 1, num_y_points).reshape(num_y_points, 1)
        X, Y = np.meshgrid(x, y)
        xy = np.column_stack((X.flatten(), Y.flatten()))
        u_obs = functions.u_2D_ti(xy)
        f_obs = functions.f_2D_ti(xy)

        # --------------------------------------------------------------------------------------------------------------
        # Solver parameters

        domain = {'x_domain': [0, 1, 1000], 'y_domain': [0, 1, 1000], 't_domain': None}
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
            inv_solver = InverseHeatSolver.restore_model_and_params("models/2D_ti")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate)
            inv_solver.train(a_iterations=25000, u_iterations=25000, f_iterations=30000,
                             display_results_every=500, save_path="models/2D_ti")

        loss_labels = [
            r"$L_{PDE}$",
            # r"$L_{|\nabla a|}$"
        ]
        Visualizer.plot_losses(inv_solver.history)

        X = np.linspace(0, 1, 100)
        Y = np.linspace(0, 1, 100)
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

        inv_solver.print_l2_error(u_exact, outputs[0], "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact, outputs[2], "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, outputs[1], "f_exact", "f_pred")
        print(inv_solver.format_training_time())

        Visualizer.plot_3d(X_test, Y_test, u_pred, f_pred, a_pred, is_time_plot=False)

        # import matplotlib
        # matplotlib.use('TkAgg')
        Visualizer.time_plot(X, 1, 100, u_pred, f_pred, a_pred,
                             functions.u_2D_ti, functions.f_2D_ti, functions.a_2D_ti)

    if case_4:

        # Evaluate functions
        x = np.linspace(0, 1, 25)
        y = np.linspace(0, 1, 25)
        t = np.linspace(0, 6, 55)
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        xyt = np.column_stack((X.flatten(), Y.flatten(), T.flatten()))
        u_obs = functions.u_2D_td(xyt)
        f_obs = functions.f_2D_td(xyt)

        # --------------------------------------------------------------------------------------------------------------
        # Solver parameters

        domain = {'x_domain': [0, 1, 50], 'y_domain': [0, 1, 50], 't_domain': [0, 6, 25]}
        nn_dims = {'num_layers': 3, 'num_neurons': 50}
        obs_values = {'dom_obs': xyt,
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
            inv_solver = InverseHeatSolver.restore_model_and_params("models/2D_td")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate)
            inv_solver.train(a_iterations=25000, u_iterations=25000, f_iterations=35000,
                             display_results_every=500, save_path="models/2D_td")

        loss_labels = [
            r"$L_{PDE}$",
            # r"$L_{|\nabla a|}$"
        ]
        Visualizer.plot_losses(inv_solver.history)

        x_num = 100
        y_num = 100
        t_num = 100
        X = np.linspace(0, 1, x_num)
        Y = np.linspace(0, 1, y_num)
        T = np.linspace(0, 6, t_num)
        X_test, Y_test, T_test = np.meshgrid(X, Y, T)
        XYT = np.column_stack((X_test.flatten(), Y_test.flatten(), T_test.flatten()))
        outputs = inv_solver.predict(XYT)

        u_pred = outputs[0].reshape(x_num, y_num, t_num)
        f_pred = outputs[1].reshape(x_num, y_num, t_num)
        a_pred = outputs[2].reshape(x_num, y_num, t_num)

        # exact functions
        u_exact = functions.u_2D_td(XYT).reshape(x_num, y_num, t_num)
        a_exact = functions.a_2D_td(XYT).reshape(x_num, y_num, t_num)
        f_exact = functions.f_2D_td(XYT).reshape(x_num, y_num, t_num)

        inv_solver.print_l2_error(u_exact, u_pred, "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact, a_pred, "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, f_pred, "f_exact", "f_pred")
        print(inv_solver.format_training_time())

        x_mesh, y_mesh = np.meshgrid(X, Y)
        Visualizer.plot_3d(x_mesh, y_mesh, u_pred[:, :, 0], f_pred[:, :, 0], a_pred[:, :, 0])

        import matplotlib
        matplotlib.use('TkAgg')
        Visualizer.animation_3d(x_mesh, y_mesh, T, 6, u_pred, f_pred)

        Visualizer.time_plot(X, 6, t_num,
                             u_pred[:, :, 0].reshape(x_num, t_num),
                             f_pred[:, :, 0].reshape(x_num, t_num),
                             a_pred[:, :, 0].reshape(x_num, t_num),
                             functions.u_2D_ti,
                             functions.f_2D_ti,
                             functions.a_2D_ti)

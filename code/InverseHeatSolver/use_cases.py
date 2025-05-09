import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec

# --------------------------------------------------------------------------------------------------------------
# local packages
from InverseHeatSolver.InverseHeatSolver_my_implementation import InverseHeatSolver
from InverseHeatSolver import Visualizer

# --------------------------------------------------------------------------------------------------------------

seed = 42  # Choose any integer seed
np.random.seed(seed)

# ======================================================================================================================
# Main script
if __name__ == "__main__":
    case_1 = True    # 1D, time independent
    case_2 = True    # 1D, time dependent
    case_3 = True     # 2D, time independent
    case_4 = False    # 2D, time dependent

    # ==================================================================================================================
    # USE CASE 1.
    # ------------------------------------------------------------------------------------------------------------------
    if case_1:
        # Functions
        def u(x):
            return x * (1 - x)


        def a(x, sigma=0.05, mu=0.5):
            return 1 + np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))


        def f(x, sigma=0.05, mu=0.5):
            numerator = (2 * sigma ** 2 + (1 - 2 * x) * (x - mu))
            exponent = np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))
            factor = 1 / (sigma ** 2)
            f = 2 + numerator * exponent * factor
            return f


        def u_bc(x):
            return 0

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate functions
        num_points = 25
        obs_dom = np.linspace(0, 1, num_points).reshape(num_points, 1)

        u_obs = u(obs_dom)
        f_obs = f(obs_dom)

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
            inv_solver = InverseHeatSolver.restore_model_and_params("1D_ti")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate)
            inv_solver.train(a_iterations=10000, u_iterations=10000, f_iterations=15000,
                             display_results_every=500, save_path="1D_ti")

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
        u_exact = u(x_test)
        a_exact = a(x_test)
        f_exact = f(x_test)

        Visualizer.plot_1D(x_test, u_pred, obs_dom, a_pred, u_obs, f_obs, a_exact, f_pred, u_exact, f_exact, None)

        inv_solver.print_l2_error(u_exact, u_pred, "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact, a_pred, "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, f_pred, "f_exact", "f_pred")

        print(inv_solver.format_training_time())

    if case_2:

        def u(xt):
            return np.exp(-xt[:, 1:]) * xt[:, 0:1] * (1 - xt[:, 0:1])


        def a(xt, sigma=0.05, mu=0.5):
            return 1 + np.exp(-((xt[:, 0:1] - mu) ** 2 / (2 * sigma ** 2)))


        def f(xt, sigma=0.05, mu=0.5):
            return (np.exp(-xt[:, 1:]) *
                    (1 / sigma ** 2 * ((1 - 2 * xt[:, 0:1]) * (xt[:, 0:1] - mu)
                                       + 2 * sigma ** 2) * np.exp(-((xt[:, 0:1] - mu) ** 2) / (2 * sigma ** 2))
                     - xt[:, 0:1] * (1 - xt[:, 0:1]) + 2))

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate functions
        num_x_points = 55
        num_t_points = 100
        x = np.linspace(0, 1, num_x_points).reshape(num_x_points, 1)
        t = np.linspace(0, 6, num_t_points).reshape(num_t_points, 1)
        X, T = np.meshgrid(x, t)
        xt = np.column_stack((X.flatten(), T.flatten()))
        u_obs = u(xt)
        f_obs = f(xt)

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
            inv_solver = InverseHeatSolver.restore_model_and_params("1D_td")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate)
            inv_solver.train(a_iterations=20000, u_iterations=20000, f_iterations=25000,
                                       display_results_every=500, save_path="1D_td")

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
        u_exact = u(XT)
        a_exact = a(XT)
        f_exact = f(XT)

        inv_solver.print_l2_error(u_exact, outputs[0], "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact, outputs[2], "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, outputs[1], "f_exact", "f_pred")
        print(inv_solver.format_training_time())

        Visualizer.plot_3d(X_test, T_test, u_pred, f_pred, a_pred, is_time_plot=True)

        Visualizer.time_plot(X, range_t, sizeof_t, u_pred, f_pred, a_pred, u, f, a)

        #import matplotlib
        #matplotlib.use('TkAgg')
        #Visualizer.time_plot(X, range_t, sizeof_t, u_pred, f_pred, a_pred, u, f, a)

    if case_3:

        def u(xy):
            x = xy[:, 0]
            y = xy[:, 1]
            return np.sin(np.pi * x) * np.sin(np.pi * y)


        def a(xy, mu_x=0.5, mu_y=0.5, sigma=0.1):
            x = xy[:, 0]
            y = xy[:, 1]
            return 1 + np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))


        def f(xy, mu_x=0.5, mu_y=0.5, sigma=0.1):
            x = xy[:, 0]
            y = xy[:, 1]

            exp_term = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))
            term1 = (x - mu_x) * np.cos(np.pi * x) * np.sin(np.pi * y)
            term2 = (y - mu_y) * np.sin(np.pi * x) * np.cos(np.pi * y)
            term3 = 2 * np.pi * sigma ** 2 * (np.sin(np.pi * x) * np.sin(np.pi * y))
            first_part = (np.pi / sigma ** 2) * (term1 + term2 + term3) * exp_term
            second_part = 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
            return first_part + second_part



        # --------------------------------------------------------------------------------------------------------------
        # Evaluate functions
        num_x_points = 105
        num_y_points = 105
        # x_obs = np.random.rand(num_points).reshape(num_points, 1)
        x = np.linspace(0, 1, num_x_points).reshape(num_x_points, 1)
        y = np.linspace(0, 1, num_y_points).reshape(num_y_points, 1)
        X, Y = np.meshgrid(x, y)
        xy = np.column_stack((X.flatten(), Y.flatten()))
        u_obs = u(xy)
        f_obs = f(xy)

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
            inv_solver = InverseHeatSolver.restore_model_and_params("2D_ti")
        else:
            inv_solver = InverseHeatSolver(domain, nn_dims, obs_values, loss_weights, learning_rate)
            inv_solver.train(a_iterations=25000, u_iterations=25000, f_iterations=30000,
                             display_results_every=500, save_path="2D_ti")

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
        u_exact = u(XY).reshape(X_test.shape)
        a_exact = a(XY).reshape(X_test.shape)
        f_exact = f(XY).reshape(X_test.shape)

        inv_solver.print_l2_error(u_exact, outputs[0], "u_exact", "u_pred")
        inv_solver.print_l2_error(a_exact, outputs[2], "a_exact", "a_pred")
        inv_solver.print_l2_error(f_exact, outputs[1], "f_exact", "f_pred")
        print(inv_solver.format_training_time())

        Visualizer.plot_3d(X_test, Y_test, u_pred, f_pred, a_pred, is_time_plot=False)

        import matplotlib
        matplotlib.use('TkAgg')
        Visualizer.time_plot(X, 1, 100, u_pred, f_pred, a_pred, u, f, a)

    if case_4:
        None

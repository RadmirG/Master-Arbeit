# ======================================================================================================================
# Solving the forward problem for the heat equation
#                               ∂u − ∇⋅(a∇u) = f
# where ∂u time derivative of u(⋅) is, also searched solution for the heat equation. a(⋅) is (known) heat diffusivity,
# and f(⋅) is some initial system input, also called right side.
# ----------------------------------------------------------------------------------------------------------------------
# The current file presents all possible input cases of functions for the above problem. Thus, are analytical continues
# functions for cases like:
#
#     1. One dimensional time independent : u(x), f(x) and a(x).
#     2. One dimensional time dependent : u(x,t), f(x,t) and a(x).
#     3. Two dimensional time independent : u(x,y), f(x,y) and a(x,y).
#     4. Two dimensional time dependent : u(x,y,t), f(x,y,t) and a(x,y).
#
# Remark: All cases are designed for a(.) as only spatial dependent function.
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================

import numpy as np

# ======================================================================================================================
# USE CASE 1.
# ----------------------------------------------------------------------------------------------------------------------

def u_1D_ti(x):
    return x * (1 - x)


def a_1D_ti(x, sigma=0.05, mu=0.5):
    return 1 + np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))


def f_1D_ti(x, sigma=0.05, mu=0.5):
    numerator = (2 * sigma ** 2 + (1 - 2 * x) * (x - mu))
    exponent = np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))
    factor = 1 / (sigma ** 2)
    f = 2 + numerator * exponent * factor
    return f

# ======================================================================================================================
# USE CASE 2.
# ----------------------------------------------------------------------------------------------------------------------

def u_1D_td(xt):
    return np.exp(-xt[:, 1:]) * xt[:, 0:1] * (1 - xt[:, 0:1])


def a_1D_td(xt, sigma=0.05, mu=0.5):
    return 1 + np.exp(-((xt[:, 0:1] - mu) ** 2 / (2 * sigma ** 2)))


def f_1D_td(xt, sigma=0.05, mu=0.5):
    return (np.exp(-xt[:, 1:]) *
            (1 / sigma ** 2 * ((1 - 2 * xt[:, 0:1]) * (xt[:, 0:1] - mu)
                               + 2 * sigma ** 2) * np.exp(-((xt[:, 0:1] - mu) ** 2) / (2 * sigma ** 2))
             - xt[:, 0:1] * (1 - xt[:, 0:1]) + 2))

# ======================================================================================================================
# USE CASE 3.
# ----------------------------------------------------------------------------------------------------------------------

def u_2D_ti(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def a_2D_ti(xy, mu_x=0.5, mu_y=0.5, sigma=0.1):
    x = xy[:, 0]
    y = xy[:, 1]
    return 1 + np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))


def f_2D_ti(xy, mu_x=0.5, mu_y=0.5, sigma=0.1):
    x = xy[:, 0]
    y = xy[:, 1]
    exp_term = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))
    term1 = (x - mu_x) * np.cos(np.pi * x) * np.sin(np.pi * y)
    term2 = (y - mu_y) * np.sin(np.pi * x) * np.cos(np.pi * y)
    term3 = 2 * np.pi * sigma ** 2 * (np.sin(np.pi * x) * np.sin(np.pi * y))
    first_part = (np.pi / sigma ** 2) * (term1 + term2 + term3) * exp_term
    second_part = 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    return first_part + second_part

# ======================================================================================================================
# USE CASE 4.
# ----------------------------------------------------------------------------------------------------------------------

def u_2D_td(xyt):
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def a_2D_td(xy, mu_x=0.5, mu_y=0.5, sigma=0.1):
    x = xy[:, 0]
    y = xy[:, 1]
    return 1 + np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))


def f_2D_td(xyt, mu_x=0.5, mu_y=0.5, sigma=0.1):
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    pi = np.pi
    gaussian_exponent = -((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2)
    gaussian_term = np.exp(gaussian_exponent)
    first_term = (x - mu_x) * np.cos(pi * x) * np.sin(pi * y)
    second_term = (y - mu_y) * np.sin(pi * x) * np.cos(pi * y)
    third_term = 2 * pi * sigma ** 2 * np.sin(pi * x) * np.sin(pi * y)
    combined_term = first_term + second_term + third_term
    exponential_decay = np.exp(-t)
    sinusoidal_term = (2 * pi ** 2 - 1) * np.sin(pi * x) * np.sin(pi * y)
    result = exponential_decay * ((pi / sigma ** 2) * combined_term * gaussian_term + sinusoidal_term)
    return result
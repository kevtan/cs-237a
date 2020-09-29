"""Quick scripts to test splrep and splev functions."""

import pdb

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splrep


def plot_functions(*functions):
    """Plots every |function| in |functions| where |function| is of the
    form: (function, start, end, resolution)"""
    for function, start, end, resolution in functions:
        num_points = round((end-start)/resolution)
        samples = np.linspace(start, end, num_points)
        output = np.vectorize(function)(samples)
        plt.plot(samples, output)
    plt.show()

# # EXAMPLE of a spline
# poly_pieces = [
#     (lambda x: -1 + 4*x - x**2, 0, 1, 0.1),
#     (lambda x: 2*x, 1, 2, 0.1),
#     (lambda x: 2 - x + x**2, 2, 3, 0.1)
# ]
# plot_functions(*poly_pieces)


x = np.arange(0, 7, 1)
y = np.array([0, 1, -1, 5, 1, 1, 5])

# t: 1D array of knots
# c: 1D array of B-spline coefficients
# k: degree of spline (default 3)
t, c, k = splrep(x, y, s=5)
inputs = np.linspace(0, 6, 1000)
outputs = splev(inputs, (t, c, k))
outputs_1_deriv = splev(inputs, (t, c, k), der=1)
outputs_2_deriv = splev(inputs, (t, c, k), der=2)
outputs_3_deriv = splev(inputs, (t, c, k), der=3)

# plot original data
plt.plot(x, y)
# plot cubic-interpolated data
plt.plot(inputs, outputs_1_deriv)
plt.plot(inputs, outputs_2_deriv)
plt.plot(inputs, outputs_3_deriv)
plt.show()

import math

import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy import linalg

import scikits.bvp_solver


def dzdx(x, z):
    """
    Callback function that returns the value of the system of ODEs given values
    for the variables and unknown parameters (in our case none).
    """
    return np.array([z[1], -np.abs(z[0])])


def bc_fun(za, zb):
    """
    Callback function that evaluates the difference between the boundary conditions
    and the given values for the variables and unknown parameters.
    """
    left_bc_residual = np.array([za[0]])
    right_bc_residual = np.array([zb[0] + 2])
    return left_bc_residual, right_bc_residual


# Define Problem
problem = scikits.bvp_solver.ProblemDefinition(
    num_ODE=2,  # Number of ODes
    num_parameters=0,  # Number of unknown parameters
    num_left_boundary_conditions=1,  # We only have 1, which explains the bc_fun
    boundary_points=(0, 4),  # Boundary points of independent coordinate
    function=dzdx,  # ODE function
    boundary_conditions=bc_fun,
)  # BC function

# Defne initial guess as a tuple (constant solution)
guess = (1.0, 0.0)

# Solve  - returns a solution structure
soln = scikits.bvp_solver.solve(problem, solution_guess=guess)

# Evaluate solution at choice of grid
x_grid = np.linspace(0, 4, 100)
z = soln(x_grid)  # solution components arranged row-wise

# Plot

plt.figure()
plt.plot(x_grid, z[0, :], "b-", linewidth=2)
plt.plot(x_grid, z[1, :], "r-", linewidth=2)
plt.grid("on")
plt.xlabel("X")
plt.ylabel("Z")
plt.legend(["Z_1", "Z_2"])

plt.show()

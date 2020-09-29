"""
Here is a summary of the time re-scaling trick.
(1) Notice that tf is not fixed for many robotics problems, so you can't
    directly use the 2P-BVP solver. You have to define a new time tau=t/tf,
    which measures the percentage of total time.
(2) You have to rewrite the entire system of ODEs in terms of the new time.
(3) tf becomes another variable that the BVP solver will find for you, so
    you need to specify an ODE that models it's derivative. This additional
    ODE is trivial (equals 0) because tf is a constant over our new time tau.
(4) Because we added a new differential equation, we also need to add a new
    boundary condition to solve for the extra constant of integration. This
    new BC comes from the "master equation."
"""

import numpy as np
import math
from scipy import linalg
import scikits.bvp_solver
import matplotlib.pyplot as plt


def TPBVPode(tau, y):
    # Return array containing RHS of ODEs
    # y = [x1 x2 p1 p2 r]

    # global A,B

    u = (-1 / b) * y[3]
    x_dot = y[4] * (np.dot(A, y[0:2]) + B * u)  # state odes
    p_dot = y[4] * (-np.dot(A.T, y[2:4]))  # co-state odes
    r_dot = 0  # dummy state ode

    return np.hstack((x_dot, p_dot, r_dot))


def TPBVPbc(ya, yb):
    # Return a tuple containing 2 arrays - left and right side BC residuals
    # Note: len(left BCs) + len(right BCs) = num of ODEs

    # global b,alp,x0

    # Left BCs
    BC_left = np.array([ya[0] - x0[0], ya[1] - x0[1]])

    # Free final time constraint
    H_f = -0.5 * (yb[3] ** 2.0) / b + alp * yb[4]

    # Right BCs
    BC_right = np.array([yb[0], yb[1], H_f])

    return (BC_left, BC_right)


def TPBVPinit():
    # global x0
    return (x0[0], x0[1], 1.0, 0.0, 1.0)


def TPBVP(p1, alpha):

    global alp, x0, b

    # Set problem parameters
    x0 = np.array([10, 0])
    b = p1
    alp = alpha

    print "b = %f, alp = %f" % (b, alp)

    # Define Problem
    problem = scikits.bvp_solver.ProblemDefinition(
        num_ODE=5,  # Number of ODes
        num_parameters=0,  # Number of parameters
        num_left_boundary_conditions=2,  # Number of left BCs
        boundary_points=(0, 1),  # Boundary points of independent coordinate
        function=TPBVPode,  # ODE function
        boundary_conditions=TPBVPbc,
    )  # BC function

    # Defne initial guess as a tuple (constant solution)
    guess = TPBVPinit()

    # Solve  - returns a
    soln = scikits.bvp_solver.solve(problem, solution_guess=guess)

    # Get terminal time
    y_0 = soln(0)
    t_f = y_0[-1]

    # Evaluate solution at choice of grid
    x_grid = np.linspace(0, t_f, 100)
    y = soln(x_grid / t_f)  # solution components arranged row-wise
    u = -(1 / b) * y[3, :]

    # Output structure
    import pdb

    pdb.set_trace()
    m = np.vstack((x_grid, y[0:4, :], u))

    return m


# Define global variables
global A, B

A = np.array([[0, 1], [0, 0]])
B = np.array([0, 1])

b_val = 0.1
alpha = np.logspace(-2, 2, 10)
t_f = np.zeros((10))
for al in range(10):
    m = TPBVP(b_val, alpha[al])
    t_f[al] = m[0, -1]


# Plot

plt.figure()
plt.semilogx(alpha, (1800 * b_val / alpha) ** 0.2, "b-", linewidth=2)
plt.semilogx(alpha, t_f, "rs", linewidth=2)
plt.grid("on")
plt.xlabel("X")
plt.ylabel("t_f")
plt.legend(["Analytic", "Numerical"])

plt.show()

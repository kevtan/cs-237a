import math

import matplotlib.pyplot as plt
import numpy as np
import scikits.bvp_solver

from utils import *

dt = 0.005


def ode_fun(tau, z):
    """
    This function computes the dz given tau and z. It is used in the bvp solver.
    Inputs:
        tau: the independent variable. This must be the first argument.
        z: the state vector. The first three states are [x, y, th, ...]
    Output:
        dz: the state derivative vector. Returns a numpy array.
    """
    _, _, th, p1, p2, p3, r = z
    v = -0.5 * (p1 * math.cos(th) + p2 * math.sin(th))
    return r * np.array([
        v * math.cos(th),
        v * math.sin(th),
        -0.5 * p3,
        0,
        0,
        p1 * v * math.sin(th) - p2 * v * math.cos(th),
        0
    ])


def bc_fun(za, zb):
    """
    This function computes boundary conditions. It is used in the bvp solver.
    Inputs:
        za: the state vector at the initial time
        zb: the state vector at the final time
        Note: both z-vectors have the format [x, y, th, p1, p2, p3, tf]
    Output:
        bca: tuple of boundary conditions at initial time
        bcb: tuple of boundary conditions at final time
    """
    # initial and final configurations
    x_ig, y_ig, th_ig = 0, 0, -np.pi/2.0
    x_fg, y_fg, th_fg = 5, 5, -np.pi/2.0
    # unpack the necessary input vectors
    x_i, y_i, th_i = za[:3]
    x_f, y_f, th_f, p1_f, p2_f, p3_f = zb[:6]
    # find the left and right BC residuals
    left_residual = np.array([
        x_i - x_ig,
        y_i - y_ig,
        th_i - th_ig
    ])
    v = -0.5 * (p1_f * math.cos(th_f) + p2_f * math.sin(th_f))
    om = -0.5 * p3_f
    LAMBDA = 0.25
    right_residual = np.array([
        x_f - x_fg,
        y_f - y_fg,
        th_f - th_fg,
        LAMBDA + v ** 2 + om ** 2 + p1_f * v *
        math.cos(th_f) + p2_f * v * math.sin(th_f) + p3_f * om
    ])
    return left_residual, right_residual


def solve_bvp(problem_inputs, initial_guess):
    """
    This function solves the bvp_problem.
    Inputs:
        problem_inputs: a dictionary of the arguments needs to define the problem
                        num_ODE, num_parameters, num_left_boundary_conditions,
                        boundary_points, function, boundary_conditions
        initial_guess: initial guess of the solution
    Output:
        z: a numpy array of the solution. It is of size [time, state_dim]

    Read this documentation -- https://pythonhosted.org/scikits.bvp_solver/tutorial.html
    """
    problem = scikits.bvp_solver.ProblemDefinition(**problem_inputs)
    soln = scikits.bvp_solver.solve(problem, solution_guess=initial_guess)

    # Test if time is reversed in bvp_solver solution
    flip, tf = check_flip(soln(0))
    t = np.arange(0, tf, dt)
    z = soln(t/tf)
    if flip:
        z[3:7, :] = -z[3:7, :]
    z = z.T  # solution arranged so that it is [time, state_dim]
    return z


def compute_controls(z):
    """
    This function computes the controls V, om, given the state z. It is used in main().
    Input:
        z: z is the state vector for multiple time instances. It has size [time, state_dim]
    Outputs:
        V: velocity control input
        om: angular rate control input
    """
    V = np.zeros(len(z))
    om = V.copy()
    for i in range(len(z)):
        _, _, th, p1, p2, p3, _ = z[i]
        V[i] = -0.5 * (p1 * math.cos(th) + p2 * math.sin(th))
        om[0] = -0.5 * p3
    return V, om


def main():
    """
    This function solves the specified bvp problem and returns the corresponding optimal contol sequence
    Outputs:
        V: optimal V control sequence 
        om: optimal om ccontrol sequence
    You are required to define the problem inputs, initial guess, and compute the controls

    Hint: The total time is between 15-25
    """
    problem_inputs = {
        'num_ODE': 7,
        'num_parameters': 0,
        'num_left_boundary_conditions': 3,
        'boundary_points': (0, 1),
        'function': ode_fun,
        'boundary_conditions': bc_fun
    }
    initial_guess = np.array([2.5, 2.5, -0.5, 2.0, 2.0, 0.0, 1.0])
    z = solve_bvp(problem_inputs, initial_guess)
    V, om = compute_controls(z)
    return z, V, om


if __name__ == '__main__':
    z, V, om = main()
    tf = z[0, -1]
    t = np.arange(0, tf, dt)
    x = z[:, 0]
    y = z[:, 1]
    th = z[:, 2]
    data = {'z': z, 'V': V, 'om': om}
    save_dict(data, 'data/optimal_control.pkl')
    maybe_makedirs('plots')

    # plotting
    # plt.rc('font', weight='bold', size=16)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'k-', linewidth=2)
    plt.quiver(x[1:-1:200], y[1:-1:200],
               np.cos(th[1:-1:200]), np.sin(th[1:-1:200]))
    plt.grid('on')
    plt.plot(0, 0, 'go', markerfacecolor='green', markersize=15)
    plt.plot(5, 5, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis([-1, 6, -1, 6])
    plt.title('Optimal Control Trajectory')

    plt.subplot(1, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid('on')
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
    plt.title('Optimal control sequence')
    plt.tight_layout()
    plt.savefig('plots/optimal_control.png')
    plt.show()

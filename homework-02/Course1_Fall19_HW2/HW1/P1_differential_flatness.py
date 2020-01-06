import numpy as np
import math
import itertools
from numpy import linalg
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from utils import *


class State:
    def __init__(self, x, y, V, th):
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self):
        return self.V * np.cos(self.th)

    @property
    def yd(self):
        return self.V * np.sin(self.th)


def compute_traj_coeffs(initial_state, final_state, tf):
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    coefficient_matrix = np.array(
        [
            [1, 0, 0, 0],
            [1, tf, tf ** 2, tf ** 3],
            [0, 1, 0, 0],
            [0, 1, 2 * tf, 3 * tf ** 2],
        ]
    )
    output_vector_x = np.array(
        [initial_state.x, final_state.x, initial_state.xd, final_state.xd]
    )
    x_coeffs = np.linalg.solve(coefficient_matrix, output_vector_x)
    output_vector_y = np.array(
        [initial_state.y, final_state.y, initial_state.yd, final_state.yd]
    )
    y_coeffs = np.linalg.solve(coefficient_matrix, output_vector_y)
    return np.concatenate((x_coeffs, y_coeffs))


def compute_traj(coeffs, tf, N):
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    x_coeffs, y_coeffs = coeffs[:4], coeffs[4:]
    times = np.linspace(0, tf, N)
    polynomial_function = lambda t, coeffs: np.dot(np.array([1, t, t**2, t**3]), coeffs)
    first_derivative = lambda t, coeffs: np.dot(np.array([0, 1, 2 * t, 3 * t**2]), coeffs)
    second_derivative = lambda t, coeffs: np.dot(np.array([0, 0, 2, 6 * t]), coeffs)
    traj = np.zeros((N, 7))
    for i, time in enumerate(times):
        traj[i][0] = polynomial_function(time, x_coeffs)
        traj[i][1] = polynomial_function(time, y_coeffs)
        traj[i][3] = first_derivative(time, x_coeffs)
        traj[i][4] = first_derivative(time, y_coeffs)
        traj[i][2] = math.atan2(traj[i][4], traj[i][3])
        traj[i][5] = second_derivative(time, x_coeffs)
        traj[i][6] = second_derivative(time, y_coeffs)
    return times, traj


def compute_controls(traj):
    """Takes in N trajectory records and returns a velocity and heading angle
    history (both of length N) that the robot has to execute in order to achieve
    the target trajectory.
    
        Format of each record:
        (x, y, theta, x', y', x", y")

    Parameters:
        traj (np.array shape (N, 7))

    Returns:
        V (np.array shape (N,)) velocity history
        om (np.array shape (N,)) heading angle history
    """
    V = np.zeros(len(traj))
    om = V.copy()
    for i, record in enumerate(traj):
        theta, v_x, v_y = record[2:5]
        V[i] = (v_x ** 2 + v_y ** 2) ** 0.5
        J = np.array([
            [math.cos(theta), -V[i] * math.sin(theta)],
            [math.sin(theta), V[i] * math.cos(theta)]
        ])
        linear_acceleration_vector = record[-2:]
        om[i] = np.linalg.solve(J, linear_acceleration_vector)[-1]
    return V, om


def compute_arc_length(V, t):
    """
    This function computes arc-length s as a function of t.
    Inputs:
        V: a vector of velocities of length T
        t: a vector of time of length T
    Output:
        s: the arc-length as a function of time. s[i] is the arc-length at time
            t[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    return np.concatenate((np.zeros(1), cumtrapz(V, t)))


def rescale_V(V, om, V_max, om_max):
    """
    This function computes V_tilde, given the unconstrained solution V, and om.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained,
            differential flatness problem.
        om: vector of angular velocities of length T. Solution from the
            unconstrained, differential flatness problem.
    Output:
        V_tilde: Rescaled velocity that satisfies the control constraints.

    Hint: At each timestep V_tilde should be computed as a minimum of the
    original value V, and values required to ensure _both_ constraints are
    satisfied.
    Hint: This should only take one or two lines.
    """
    return np.minimum(np.minimum(V, V_max), (om_max / np.absolute(om)) * V)


def compute_tau(V_tilde, s):
    """
    This function computes the new time history tau as a function of s.
    Inputs:
        V_tilde: a sequence of scaled velocities of length T.
        s: a sequence of arc-length of length T.
    Output:
        tau: the new time history for the sequence. tau[i] is the time at s[i]. This has length T.

    Hint: Use the function cumtrapz. This should take one line.
    """
    return np.concatenate((np.zeros(1), cumtrapz(1 / V_tilde, s)))


def rescale_om(V, om, V_tilde):
    """
    This function computes the rescaled om control.
    Inputs:
        V: vector of velocities of length T. Solution from the unconstrained, differential flatness problem.
        om:  vector of angular velocities of length T. Solution from the unconstrained, differential flatness problem.
        V_tilde: vector of scaled velocities of length T.
    Output:
        om_tilde: vector of scaled angular velocities

    Hint: This should take one line.
    """
    return (V_tilde / V) * om


def compute_traj_with_limits(z_0, z_f, tf, N, V_max, om_max):
    coeffs = compute_traj_coeffs(initial_state=z_0, final_state=z_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V, om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    return traj, tau, V_tilde, om_tilde


def interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f):
    """
    Inputs:
        traj (np.array [N,7]) original unscaled trajectory
        tau (np.array [N]) rescaled time at orignal traj points
        V_tilde (np.array [N]) new velocities to use
        om_tilde (np.array [N]) new rotational velocities to use
        dt (float) timestep for interpolation

    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    """
    # Get new final time
    tf_new = tau[-1]

    # Generate new uniform time grid
    N_new = int(tf_new / dt)
    t_new = dt * np.array(range(N_new + 1))

    # Interpolate for state trajectory
    traj_scaled = np.zeros((N_new + 1, 7))
    traj_scaled[:, 0] = np.interp(t_new, tau, traj[:, 0])  # x
    traj_scaled[:, 1] = np.interp(t_new, tau, traj[:, 1])  # y
    traj_scaled[:, 2] = np.interp(t_new, tau, traj[:, 2])  # th
    # Interpolate for scaled velocities
    V_scaled = np.interp(t_new, tau, V_tilde)  # V
    om_scaled = np.interp(t_new, tau, om_tilde)  # om
    # Compute xy velocities
    traj_scaled[:, 3] = V_scaled * np.cos(traj_scaled[:, 2])  # xd
    traj_scaled[:, 4] = V_scaled * np.sin(traj_scaled[:, 2])  # yd
    # Compute xy acclerations
    traj_scaled[:, 5] = np.append(
        np.diff(traj_scaled[:, 3]) / dt, -s_f.V * om_scaled[-1] * np.sin(s_f.th)
    )  # xdd
    traj_scaled[:, 6] = np.append(
        np.diff(traj_scaled[:, 4]) / dt, s_f.V * om_scaled[-1] * np.cos(s_f.th)
    )  # ydd

    return t_new, V_scaled, om_scaled, traj_scaled


if __name__ == "__main__":
    # traj, V, om = differential_flatness_trajectory()
    # Constants
    tf = 15.0
    V_max = 0.5
    om_max = 1

    # time
    dt = 0.005
    N = int(tf / dt) + 1
    t = dt * np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=V_max, th=-np.pi / 2)

    # Final conditions
    s_f = State(x=5, y=5, V=V_max, th=-np.pi / 2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V, om = compute_controls(traj=traj)

    part_b_complete = False
    s = compute_arc_length(V, t)
    if s is not None:
        part_b_complete = True
        V_tilde = rescale_V(V, om, V_max, om_max)
        tau = compute_tau(V_tilde, s)
        om_tilde = rescale_om(V, om, V_tilde)

        t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(
            traj, tau, V_tilde, om_tilde, dt, s_f
        )

        # Save trajectory data
        data = {"z": traj_scaled, "V": V_scaled, "om": om_scaled}
        save_dict(data, "data/differential_flatness.pkl")

    maybe_makedirs("plots")

    # Plots
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.plot(traj[:, 0], traj[:, 1], "k-", linewidth=2)
    plt.grid("on")
    plt.plot(s_0.x, s_0.y, "go", markerfacecolor="green", markersize=15)
    plt.plot(s_f.x, s_f.y, "ro", markerfacecolor="red", markersize=15)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(2, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid("on")
    plt.xlabel("Time [s]")
    plt.legend(["V [m/s]", "$\omega$ [rad/s]"], loc="best")
    plt.title("Original Control Input")
    plt.tight_layout()

    plt.subplot(2, 2, 4, sharex=ax)
    if part_b_complete:
        plt.plot(t_new, V_scaled, linewidth=2)
        plt.plot(t_new, om_scaled, linewidth=2)
        plt.legend(["V [m/s]", "$\omega$ [rad/s]"], loc="best")
        plt.grid("on")
    else:
        plt.text(
            0.5,
            0.5,
            "[Problem iv not completed]",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
    plt.xlabel("Time [s]")
    plt.title("Scaled Control Input")
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    if part_b_complete:
        h, = plt.plot(t, s, "b-", linewidth=2)
        handles = [h]
        labels = ["Original"]
        h, = plt.plot(tau, s, "r-", linewidth=2)
        handles.append(h)
        labels.append("Scaled")
        plt.legend(handles, labels, loc="best")
    else:
        plt.text(
            0.5,
            0.5,
            "[Problem iv not completed]",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
    plt.grid("on")
    plt.xlabel("Time [s]")
    plt.ylabel("Arc-length [m]")
    plt.title("Original and scaled arc-length")
    plt.tight_layout()
    plt.savefig("plots/differential_flatness.png")
    plt.show()

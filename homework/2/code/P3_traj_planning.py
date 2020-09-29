import matplotlib.pyplot as plt
import numpy as np
from P1_astar import AStar, DetOccupancyGrid2D
from scipy.interpolate import splev, splrep

import HW1.P1_differential_flatness as P1
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *
from P2_rrt import *


class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """

    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        t_final = self.traj_controller.traj_times[-1]
        if t < t_final - self.t_before_switch:
            return self.traj_controller.compute_control(x, y, th, t)
        else:
            return self.pose_controller.compute_control(x, y, th, t)


def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    distances = np.zeros(len(path)-1)
    for i in range(len(path)-1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        distances[i] = ((x2-x1)**2 + (y2-y1)**2)**0.5
    timesteps = map(lambda distance: round(
        (distance / V_des) + 0.5), distances)
    intervals = np.zeros(len(path))
    for i in range(1, len(intervals)):
        intervals[i] = intervals[i-1] + timesteps[i-1] * dt
    # interpolate the x and y values
    old_x = np.array([tup[0] for tup in path])
    old_y = np.array([tup[1] for tup in path])
    tck_x = splrep(intervals, old_x, s=alpha)
    tck_y = splrep(intervals, old_y, s=alpha)
    total_time = intervals[-1]
    times = np.linspace(0, total_time, total_time/dt)
    smooth_traj = np.zeros((len(times), 7))
    smooth_traj[:, 0], smooth_traj[:, 1] = splev(
        times, tck_x), splev(times, tck_y)
    smooth_traj[:, 3], smooth_traj[:, 4] = splev(
        times, tck_x, der=1), splev(times, tck_y, der=1)
    smooth_traj[:, 2] = np.vectorize(math.atan2)(
        smooth_traj[:, 4], smooth_traj[:, 3])
    smooth_traj[:, 5], smooth_traj[:, 6] = splev(
        times, tck_x, der=2), splev(times, tck_y, der=2)
    return smooth_traj, times


def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajectory
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    velocity, omega = P1.compute_controls(traj)
    s = P1.compute_arc_length(velocity, t)
    V_tilde = P1.rescale_V(velocity, omega, V_max, om_max)
    tau = P1.compute_tau(V_tilde, s)
    om_tilde = P1.rescale_om(velocity, omega, V_tilde)
    x, y, th, _, _, _, _ = traj[-1]
    s_f = P1.State(x, y, V_tilde[-1], th)
    return P1.interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

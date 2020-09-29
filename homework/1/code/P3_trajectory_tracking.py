import math

import numpy as np
from numpy import linalg

V_PREV_THRES = 0.0001


class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """

    def __init__(self, kpx, kpy, kdx, kdy, V_max=0.5, om_max=1):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        # Polynomial coefficients for x(t) and y(t) as
        self.coeffs = np.zeros(8)
        # returned by the differential flatness code

    def reset(self):
        self.V_prev = 0
        self.om_prev = 0
        self.t_prev = 0

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def get_desired_state(self, t):
        """
        Input:
            t: Current time
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at time t according to self.coeffs
        """
        x_d = np.interp(t, self.traj_times, self.traj[:, 0])
        y_d = np.interp(t, self.traj_times, self.traj[:, 1])
        xd_d = np.interp(t, self.traj_times, self.traj[:, 3])
        yd_d = np.interp(t, self.traj_times, self.traj[:, 4])
        xdd_d = np.interp(t, self.traj_times, self.traj[:, 5])
        ydd_d = np.interp(t, self.traj_times, self.traj[:, 6])

        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs: 
            V, om: Control actions
        """
        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)
        # compute velocities
        x_velo = self.V_prev * math.cos(th)
        y_velo = self.V_prev * math.sin(th)
        # compute virtual control inputs
        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - x_velo)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - y_velo)
        # compute real control inputs
        if math.fabs(self.V_prev) < V_PREV_THRES:
            self.V_prev = (xd_d ** 2 + xd_d ** 2) ** 0.5
        coefficients = np.array([
            [math.cos(th), -self.V_prev * math.sin(th)],
            [math.sin(th), self.V_prev * math.cos(th)]
        ])
        outputs = np.array([u1, u2])
        dvdt, om = np.linalg.solve(coefficients, outputs)
        V = self.V_prev + dvdt * dt
        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)
        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om

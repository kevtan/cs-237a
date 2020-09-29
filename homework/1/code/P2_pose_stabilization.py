import math

import numpy as np

from utils import wrapToPi

# command zero velocities once we are this close to the goal
THRESHOLD_RHO = 0.05
THRESHOLD_ALPHA = 0.1
THRESHOLD_DELTA = 0.1


class PoseController:
    """ Pose stabilization controller """

    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.V_max = V_max
        self.om_max = om_max

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs:
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        # compute new configuration variables
        angle_to_goal = math.atan2(self.y_g - y, self.x_g - x)
        alpha = wrapToPi(angle_to_goal - th)
        rho = ((self.x_g - x) ** 2 + (self.y_g - y) ** 2) ** 0.5
        delta = wrapToPi(angle_to_goal - self.th_g)
        if (
            math.fabs(alpha) < THRESHOLD_ALPHA
            and rho < THRESHOLD_RHO
            and math.fabs(delta) < THRESHOLD_DELTA
        ):
            return 0, 0
        # compute control inputs
        V = self.k1 * rho * math.cos(alpha)
        om = self.k2 * alpha + self.k1 * (
            (math.sin(alpha) * math.cos(alpha)) / alpha
        ) * (alpha + self.k3 * delta)
        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)
        return V, om

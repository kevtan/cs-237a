import math
import random

import dubins
import matplotlib.pyplot as plt
import numpy as np

from utils import line_line_intersection, plot_line_segments


class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        # state space lower bound (e.g., [-5, -5])
        self.statespace_lo = np.array(statespace_lo)
        # state space upper bound (e.g., [5, 5])
        self.statespace_hi = np.array(statespace_hi)
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        # obstacle set (line segments)
        self.obstacles = obstacles
        self.path = None        # the final path as a list of states

    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError(
            "is_free_motion must be overriden by a subclass of RRT")

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        raise NotImplementedError(
            "find_nearest must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError(
            "steer_towards must be overriden by a subclass of RRT")

    def random_state(self):
        """Returns a random state in the free space."""
        return [
            random.uniform(self.statespace_lo[dim], self.statespace_hi[dim])
            for dim in range(len(self.statespace_lo))
        ]

    def reconstruct_path(self, V, P, n):
        path = []
        curr = n - 1
        while curr != -1:
            path.append(V[curr])
            curr = P[curr]
        path.reverse()
        return path

    def solve(self, eps, max_iters=1000, goal_bias=0.05, shortcut=False):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randomly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """
        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        V = np.zeros((max_iters, state_dim))
        V[0, :] = self.x_init    # RRT is rooted at self.x_init
        # the current size of the RRT (states accessible as V[range(n),:])
        n = 1

        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root, in general 0 <= P[i] < i for all i < n
        P = -np.ones(max_iters, dtype=int)

        success = False

        # Intermediate Outputs
        # You must update and/or populate:
        #    - V, P, n: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of states (tree nodes)
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.

        # Hints:
        #   - use the helper functions find_nearest, steer_towards, and is_free_motion
        #   - remember that V and P always contain max_iters elements, but only the first n
        #     are meaningful! keep this in mind when using the helper functions!

        ########## Code starts here ##########
        for _ in range(max_iters - 1):
            # randomly sample a point (or goal)
            z = random.uniform(0, 1)
            if z < goal_bias:
                sample = self.x_goal
            else:
                sample = self.random_state()
            # find the nearest point and steer it towards random sample
            nearest_index = self.find_nearest(V[range(n), :], sample)
            nearest = V[nearest_index]
            new = self.steer_towards(nearest, sample, eps)
            if self.is_free_motion(self.obstacles, nearest, new):
                V[n, :], P[n], n = new, nearest_index, n+1
                if np.array_equal(new, self.x_goal):
                    self.path = self.reconstruct_path(V, P, n)
                    success = True
                    break
        ########## Code ends here ##########

        plt.figure()
        self.plot_problem()
        self.plot_tree(V, P, color="blue", linewidth=.5,
                       label="RRT tree", alpha=0.5)
        if success:
            if shortcut:
                self.plot_path(color="purple", linewidth=2,
                               label="Original solution path")
                self.shortcut_path()
                self.plot_path(color="green", linewidth=2,
                               label="Shortcut solution path")
            else:
                self.plot_path(color="green", linewidth=2,
                               label="Solution path")
            plt.legend(loc='upper center', bbox_to_anchor=(
                0.5, -0.03), fancybox=True, ncol=3)
            plt.scatter(V[:n, 0], V[:n, 1])
        else:
            print "Solution not found!"

        return success

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red",
                           linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [
                    self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(
            0.5, -0.03), fancybox=True, ncol=3)
        plt.axis('scaled')

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        Input:
            None
        Output:
            None, but should modify self.path
        """
        if len(self.path) <= 2:
            # path cannot be shortened
            return
        for i in range(len(self.path)-2, 0, -1):
            if self.is_free_motion(self.obstacles, self.path[i-1], self.path[i+1]):
                self.path.pop(i)


class GeometricRRT(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric).
    """

    def find_nearest(self, V, x):
        """Returns the index of the nearest point in V to x."""
        min_index, min_value = 0, float('inf')
        for index, value in enumerate(V):
            distance = np.linalg.norm(value - x)
            if distance < min_value:
                min_index = index
                min_value = distance
        return min_index

    def steer_towards(self, x1, x2, eps):
        angle = math.atan2(x2[1] - x1[1], x2[0] - x1[0])
        return x2 if np.linalg.norm(x2 - x1) < eps else np.array([x1[0] + eps * math.cos(angle), x1[1] + eps * math.sin(angle)])

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i], :], V[i, :])
                            for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], **kwargs)


class DubinsRRT(RRT):
    """
    Represents a planning problem for the Dubins car, a model of a simple
    car that moves at a constant speed forward and has a limited turning
    radius. We will use this v0.9.2 of the package at
    https://github.com/AndrewWalker/pydubins/blob/0.9.2/dubins/dubins.pyx
    to compute steering distances and steering trajectories. In particular,
    note the functions dubins.path_length and dubins.path_sample (read
    their documentation at the link above). See
    http://planning.cs.uiuc.edu/node821.html
    for more details on how these steering trajectories are derived.
    """

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles, turning_radius):
        self.turning_radius = turning_radius
        super(self.__class__, self).__init__(
            statespace_lo, statespace_hi, x_init, x_goal, obstacles)

    def find_nearest(self, V, x):
        """Returns the index of the nearest point in V to x."""
        min_index, min_value = 0, float('inf')
        for index, value in enumerate(V):
            distance = dubins.path_length(value, x, self.turning_radius)
            if distance < min_value:
                min_index = index
                min_value = distance
        return min_index

    def steer_towards(self, x1, x2, eps):
        """
        A subtle issue: if you use dubins.path_sample to return the point
        at distance eps along the path from x to y, use a turning radius
        slightly larger than self.turning_radius
        (i.e., 1.001*self.turning_radius). Without this hack,
        dubins.path_sample might return a point that can't quite get to in
        distance eps (using self.turning_radius) due to numerical precision
        issues.
        """
        if dubins.path_length(x1, x2, self.turning_radius) < eps:
            return x2
        else:
            return dubins.path_sample(x1, x2, 1.001*self.turning_radius, eps)[0][1]

    def is_free_motion(self, obstacles, x1, x2, resolution=np.pi/6):
        from dubins import path_sample
        pts = path_sample(x1, x2, self.turning_radius,
                          self.turning_radius*resolution)[0]
        pts.append(x2)
        for i in range(len(pts) - 1):
            for line in obstacles:
                if line_line_intersection([pts[i][:2], pts[i+1][:2]], line):
                    return False
        return True

    def plot_tree(self, V, P, resolution=np.pi/24, **kwargs):
        from dubins import path_sample
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(
                    V[P[i], :], V[i, :], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[i, :])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_path(self, resolution=np.pi/24, **kwargs):
        from dubins import path_sample
        pts = []
        path = np.array(self.path)
        for i in range(path.shape[0] - 1):
            pts.extend(path_sample(
                path[i], path[i+1], self.turning_radius, self.turning_radius*resolution)[0])
        plt.plot([x for x, y, th in pts], [y for x, y, th in pts], **kwargs)

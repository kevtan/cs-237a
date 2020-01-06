import numpy as np
import random
import math
import matplotlib.pyplot as plt
from dubins import path_length, path_sample
from utils import plot_line_segments, line_line_intersection

# Represents a motion planning problem to be solved using the RRT algorithm
class RRTConnect(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
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
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRTConnect")

    def find_nearest_forward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering forward from x
        """
        raise NotImplementedError("find_nearest_forward must be overriden by a subclass of RRTConnect")

    def find_nearest_backward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from x to V[i] is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering backward from x
        """
        raise NotImplementedError("find_nearest_backward must be overriden by a subclass of RRTConnect")

    def steer_towards_forward(self, x1, x2, eps):
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
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRTConnect")

    def steer_towards_backward(self, x1, x2, eps):
        """
        Steers backward from x2 towards x1 along the shortest path (subject
        to robot dynamics). Returns x1 if the length of this shortest path is
        less than eps, otherwise returns the point at distance eps along the
        path backward from x2 to x1.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards_backward must be overriden by a subclass of RRTConnect")

    def random_state(self):
        """Returns a random state in the free space."""
        return [
            random.uniform(self.statespace_lo[dim], self.statespace_hi[dim])
            for dim in range(len(self.statespace_lo))
        ]

    def reconstruct_path(self, fw_tree, bw_tree, intersection):
        """Reconstructs the bidirectional RRT given the forward and backward trees."""
        V_fw, P_fw, n_fw = fw_tree
        V_bw, P_bw, n_bw = bw_tree
        path_fw, path_bw = [], []
        # find the subpath in the forward tree
        curr_fw = self.find_nearest_forward(V_fw[range(n_fw), :], intersection)
        while curr_fw != -1:
            path_fw.append(V_fw[curr_fw])
            curr_fw = P_fw[curr_fw]
        path_fw.reverse()
        # find the subpath in the backward tree
        curr_bw = self.find_nearest_backward(V_bw[range(n_bw), :], intersection)
        while curr_bw != -1:
            path_bw.append(V_bw[curr_bw])
            curr_bw = P_bw[curr_bw]
        return path_fw + path_bw

    def add_to_tree(self, tree, node, parent_index):
        """Adds a new node to a tree, given its parent's index in the tree."""
        n = tree[2]
        tree[0][n,:], tree[1][n] = node, parent_index
        tree[2] += 1

    def grow_fw_tree(self, fw_tree, bw_tree, eps):
        """Samples a random point in the state space, grows the forward tree towards
        the random point, and then tries best to connect the backward tree to the state
        newly added to the forward tree.
        """
        V_fw, _, n_fw = fw_tree
        V_bw, _, n_bw = bw_tree
        # sample point (x_rand) and steer towards it (x_near)
        x_rand = self.random_state()
        x_near_index = self.find_nearest_forward(V_fw[range(n_fw),:], x_rand)
        x_near = V_fw[x_near_index]
        x_new = self.steer_towards_forward(x_near, x_rand, eps)
        # check if new path violates state space constraints
        if self.is_free_motion(self.obstacles, x_near, x_new):
            # add vertex and associated edge
            self.add_to_tree(fw_tree, x_new, x_near_index)
            # find nearest point in backward tree (x_connect) to the (x_new)
            x_connect_index = self.find_nearest_backward(V_bw[range(n_bw),:], x_new)
            x_connect = V_bw[x_connect_index]
            while True:
                # repeatedly try to extend (x_connect) towards (x_new)
                x_newconnect = self.steer_towards_backward(x_new, x_connect, eps)
                if self.is_free_motion(self.obstacles, x_newconnect, x_connect):
                    self.add_to_tree(bw_tree, x_newconnect, x_connect_index)
                    if np.array_equal(x_newconnect, x_new):
                        self.path = self.reconstruct_path(fw_tree, bw_tree, x_newconnect)
                        return True
                    x_connect = x_newconnect
                else:
                    break
        # new path violates state space constraints or unable to connect backward tree
        return False

    def grow_bw_tree(self, fw_tree, bw_tree, eps):
        """Samples a random point in the state space, grows the backward tree towards
        the random point, and then tries best to connect the forward tree to the state
        newly added to the backward tree.
        """
        V_bw, _, n_bw = bw_tree
        V_fw, _, n_fw = fw_tree
        # sample point (x_rand) and steer towards it (x_near)
        x_rand = self.random_state()
        x_near_index = self.find_nearest_backward(V_bw[range(n_bw),:], x_rand)
        x_near = V_bw[x_near_index]
        x_new = self.steer_towards_backward(x_rand, x_near, eps)
        # check if new path violates state space constraints
        if self.is_free_motion(self.obstacles, x_new, x_near):
            # add vertex and associated edge
            self.add_to_tree(bw_tree, x_new, x_near_index)
            # find nearest point in backward tree (x_connect) to the (x_new)
            x_connect_index = self.find_nearest_forward(V_fw[range(n_fw),:], x_new)
            x_connect = V_fw[x_connect_index]
            while True:
                # repeatedly try to extend (x_connect) towards (x_new)
                x_newconnect = self.steer_towards_forward(x_connect, x_new, eps)
                if self.is_free_motion(self.obstacles, x_connect, x_newconnect):
                    self.add_to_tree(fw_tree, x_newconnect, x_connect_index)
                    if np.array_equal(x_newconnect, x_new):
                        self.path = self.reconstruct_path(bw_tree, fw_tree, x_newconnect)
                        return True
                    x_connect = x_newconnect
                else:
                    break
        # new path violates state space constraints or unable to connect backward tree
        return False

    def solve(self, eps, max_iters = 1000):
        """
        Uses RRT-Connect to perform bidirectional RRT, with a forward tree
        rooted at self.x_init and a backward tree rooted at self.x_goal, with
        the aim of producing a dynamically-feasible and obstacle-free trajectory
        from self.x_init to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
                
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """
        
        state_dim = len(self.x_init)

        # represent the forward tree
        V_fw = np.zeros((max_iters, state_dim))     # nodes
        V_fw[0,:] = self.x_init
        n_fw = 1                                    # number of nodes
        P_fw = -np.ones(max_iters, dtype=int)       # nodal relationships
        fw_tree = [V_fw, P_fw, n_fw]
        

        # represent the backward tree
        V_bw = np.zeros((max_iters, state_dim))     # nodes
        V_bw[0,:] = self.x_goal
        n_bw = 1                                    # number of nodes
        P_bw = -np.ones(max_iters, dtype=int)       # nodal relationships
        bw_tree = [V_bw, P_bw, n_bw]

        # whether we were able to find a collision-free path
        success = False

        for _ in range(max_iters - 1):
            if not success:
                success = self.grow_fw_tree(fw_tree, bw_tree, eps)
            if not success:
                success = self.grow_bw_tree(fw_tree, bw_tree, eps)

        # update n_fw and n_bw (the grow_*_tree methods do not)
        n_fw = fw_tree[2]
        n_bw = bw_tree[2]

        plt.figure()
        self.plot_problem()
        self.plot_tree(V_fw, P_fw, color="blue", linewidth=.5, label="RRTConnect forward tree")
        self.plot_tree_backward(V_bw, P_bw, color="purple", linewidth=.5, label="RRTConnect backward tree")
        
        if success:
            self.plot_path(color="green", linewidth=2, label="solution path")
            plt.scatter(V_fw[:n_fw,0], V_fw[:n_fw,1], color="blue")
            plt.scatter(V_bw[:n_bw,0], V_bw[:n_bw,1], color="purple")
        plt.scatter(V_fw[:n_fw,0], V_fw[:n_fw,1], color="blue")
        plt.scatter(V_bw[:n_bw,0], V_bw[:n_bw,1], color="purple")

        plt.show()

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

class GeometricRRTConnect(RRTConnect):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest_forward(self, V, x):
        min_index, min_value = 0, float('inf')
        for index, value in enumerate(V):
            distance = np.linalg.norm(value - x)
            if distance < min_value:
                min_index = index
                min_value = distance
        return min_index

    def find_nearest_backward(self, V, x):
        return self.find_nearest_forward(V, x)

    def steer_towards_forward(self, x1, x2, eps):
        angle = math.atan2(x2[1] - x1[1], x2[0] - x1[0])
        return x2 if np.linalg.norm(x2 - x1) < eps else np.array([x1[0] + eps * math.cos(angle), x1[1] + eps * math.sin(angle)])

    def steer_towards_backward(self, x1, x2, eps):
        return self.steer_towards_forward(x2, x1, eps)

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_tree_backward(self, V, P, **kwargs):
        self.plot_tree(V, P, **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)

class DubinsRRTConnect(RRTConnect):
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
        super(self.__class__, self).__init__(statespace_lo, statespace_hi, x_init, x_goal, obstacles)

    def reverse_heading(self, x):
        """
        Reverses the heading of a given pose.
        Input: x (np.array [3]): Dubins car pose
        Output: x (np.array [3]): Pose with reversed heading
        """
        theta = x[2]
        if theta < np.pi:
            theta_new = theta + np.pi
        else:
            theta_new = theta - np.pi
        return np.array((x[0], x[1], theta_new))

    def find_nearest_forward(self, V, x):
        """Returns the index of the nearest point in V to x."""
        min_index, min_value = 0, float('inf')
        for index, value in enumerate(V):
            distance = path_length(value, x, self.turning_radius)
            if distance < min_value:
                min_index = index
                min_value = distance
        return min_index

    def find_nearest_backward(self, V, x):
        """Returns the index of the nearest point in V to x."""
        min_index, min_value = 0, float('inf')
        for index, value in enumerate(V):
            distance = path_length(x, value, self.turning_radius)
            if distance < min_value:
                min_index = index
                min_value = distance
        return min_index

    def steer_towards_forward(self, x1, x2, eps):
        if path_length(x1, x2, self.turning_radius) < eps:
            return x2
        else:
            return path_sample(x1, x2, 1.001*self.turning_radius, eps)[0][1]

    def steer_towards_backward(self, x1, x2, eps):
        if path_length(x1, x2, self.turning_radius) < eps:
            return x1
        else:
            return path_sample(x1, x2, 1.001*self.turning_radius, eps)[0][-1]

    def is_free_motion(self, obstacles, x1, x2, resolution = np.pi/6):
        pts = path_sample(x1, x2, self.turning_radius, self.turning_radius*resolution)[0]
        pts.append(x2)
        for i in range(len(pts) - 1):
            for line in obstacles:
                if line_line_intersection([pts[i][:2], pts[i+1][:2]], line):
                    return False
        return True

    def plot_tree(self, V, P, resolution = np.pi/24, **kwargs):
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(V[P[i],:], V[i,:], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[i,:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_tree_backward(self, V, P, resolution = np.pi/24, **kwargs):
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(V[i,:], V[P[i],:], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[P[i],:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_path(self, resolution = np.pi/24, **kwargs):
        pts = []
        path = np.array(self.path)
        for i in range(path.shape[0] - 1):
            pts.extend(path_sample(path[i], path[i+1], self.turning_radius, self.turning_radius*resolution)[0])
        plt.plot([x for x, y, th in pts], [y for x, y, th in pts], **kwargs)

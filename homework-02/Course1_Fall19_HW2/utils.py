import numpy as np
import matplotlib.pyplot as plt
from numpy import cross

def plot_line_segments(segments, **kwargs):
    plt.plot([x for tup in [(p1[0], p2[0], None) for (p1, p2) in segments] for x in tup],
             [y for tup in [(p1[1], p2[1], None) for (p1, p2) in segments] for y in tup], **kwargs)

def generate_planning_problem(width, height, num_obs, min_size, max_size):
    from P1_astar import DetOccupancyGrid2D
    x_margin = round(width*0.1)
    y_margin = round(height*0.1)
    obs_corners_x = np.random.randint(-x_margin, width+x_margin, num_obs)
    obs_corners_y = np.random.randint(-y_margin, height+y_margin, num_obs)
    obs_lower_corners = np.vstack([obs_corners_x, obs_corners_y]).T
    obs_sizes = np.random.randint(min_size, max_size, (num_obs,2))
    obs_upper_corners = obs_lower_corners + obs_sizes
    obstacles = zip(obs_lower_corners, obs_upper_corners)
    occupancy = DetOccupancyGrid2D(width, height, obstacles)

    x_init = tuple(np.random.randint(0, width-x_margin, 2).tolist())
    while not occupancy.is_free(x_init):
        x_init = tuple(np.random.randint(0, width-x_margin, 2).tolist())
    x_goal = x_init
    while (not occupancy.is_free(x_goal)) or (np.linalg.norm(np.array(x_goal)-np.array(x_init)) < np.sqrt(width**2 + height**2)*0.4):
        x_goal = tuple(np.random.randint(0, width-x_margin, 2).tolist())

    return occupancy, x_init, x_goal

## Magical black-box collition checking
def ccw(A, B, C):
    return np.cross(B - A, C - A) > 0

def line_line_intersection(l1, l2):
    A, B = np.array(l1)
    C, D = np.array(l2)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

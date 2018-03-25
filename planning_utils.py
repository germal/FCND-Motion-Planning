#__________________________________________________________________________80->|
# planning_utils.py
# Engineer: James W. Dunn
# This module provides support routines to assist with motion planning

from enum import Enum
from queue import PriorityQueue
import numpy as np
import math
from udacidrone.frame_utils import global_to_local

# Diagonal distance costs
SQRT2 = math.sqrt(2)

# Cost of altitude gain
CLIMB_COST = 100.0

class Grid:
    def __init__(self, data):
        """
        Creates a 2D grid representation of a 2D configuration space
        """

        # minimum and maximum north coordinates
        self.north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
        self.north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

        # minimum and maximum east coordinates
        self.east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
        self.east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

        # given the minimum and maximum coordinates we can
        # calculate the size of the grid.
        self.north_size = int(np.ceil((self.north_max - self.north_min + 1)))
        self.east_size = int(np.ceil((self.east_max - self.east_min + 1)))

        # Initialize empty nav grid
        self._grid = np.zeros((self.north_size, self.east_size))
        self._landing_alts = np.zeros((self.north_size, self.east_size))
        self._data = data

    def get_offset(self):
        return int(self.north_min), int(self.east_min)

    def get_altitude(self, x, y):
        return self._grid[x, y]

    def get_landing_altitude(self, x, y):
        return self._landing_alts[x, y]

    def update_obstacles(self, safety_distance):
        """
        Updates nav grid with obstacle heights given safety distance.
        """
        print("Safety distance:",safety_distance)
        self._grid.fill(0)
        # Populate the grid with obstacles
        for i in range(self._data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = self._data[i, :]
            obstacle = [ # for navigation
                int(np.clip(north - d_north - safety_distance - self.north_min, 0, self.north_size-1)),
                int(np.clip(north + d_north + safety_distance - self.north_min, 0, self.north_size-1)),
                int(np.clip(east - d_east - safety_distance - self.east_min, 0, self.east_size-1)),
                int(np.clip(east + d_east + safety_distance - self.east_min, 0, self.east_size-1))
            ]
            # Results in a broken grid: self._grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = int(alt + d_alt)
            area = self._grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1]
            np.maximum(area, np.ceil(alt+d_alt), area)

            obstacle = [ # for landing
                int(np.clip(north - d_north - self.north_min, 0, self.north_size-1)),
                int(np.clip(north + d_north - self.north_min, 0, self.north_size-1)),
                int(np.clip(east - d_east - self.east_min, 0, self.east_size-1)),
                int(np.clip(east + d_east - self.east_min, 0, self.east_size-1))
            ]
            area = self._landing_alts[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1]
            np.maximum(area, np.ceil(alt+d_alt), area)

    def define_edges(self, margin):
        # Populate a grid with obstacle edges
        grid = np.zeros((self.north_size, self.east_size), dtype=np.uint8)
        for i in range(self._data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = self._data[i, :]
            if alt + d_alt > 0:
                obstacle = [
                    int(np.clip(north - d_north - margin - self.north_min, 0, self.north_size-1)),
                    int(np.clip(north + d_north + margin - self.north_min, 0, self.north_size-1)),
                    int(np.clip(east - d_east - margin - self.east_min, 0, self.east_size-1)),
                    int(np.clip(east + d_east + margin - self.east_min, 0, self.east_size-1)),
                ]
                grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 255
        return grid

class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third value
    is the cost of performing the action.
    """

    NORTHEAST = (-1,  1, SQRT2)
    SOUTHEAST = ( 1,  1, SQRT2)
    SOUTHWEST = ( 1, -1, SQRT2)
    NORTHWEST = (-1, -1, SQRT2)

    NORTH = (-1,  0, 1.0)
    EAST =  ( 0,  1, 1.0)
    SOUTH = ( 1,  0, 1.0)
    WEST =  ( 0, -1, 1.0)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])

def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid
    if x - 1 < 0:
        valid_actions.remove(Action.NORTH)
        valid_actions.remove(Action.NORTHEAST)
        valid_actions.remove(Action.NORTHWEST)
    if x + 1 > n:
        valid_actions.remove(Action.SOUTH)
        valid_actions.remove(Action.SOUTHEAST)
        valid_actions.remove(Action.SOUTHWEST)
    if y - 1 < 0:
        valid_actions.remove(Action.WEST)
        if Action.NORTHWEST in valid_actions: valid_actions.remove(Action.NORTHWEST)
        if Action.SOUTHWEST in valid_actions: valid_actions.remove(Action.SOUTHWEST)
    if y + 1 > m:
        valid_actions.remove(Action.EAST)
        if Action.NORTHEAST in valid_actions: valid_actions.remove(Action.NORTHEAST)
        if Action.SOUTHEAST in valid_actions: valid_actions.remove(Action.SOUTHEAST)

    return valid_actions


def a_star(grid, h, start, goal, cur_alt):
    """
    Given a 2.5D grid, heuristic function, start, goal, 
    and current altitude, this function returns the lowest cost path.
    """

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start, cur_alt))
    visited = set(start)
    cycles = 0

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        current_alt = item[2]
        cycles += 1
        if cycles > 1000000:
            print('WARNING: path not found within 1000000 iterations')
            found = False
            break

        if cycles%100000 == 0:
            print('Hold on, still planning...', cycles)

        if current_node == goal:
            print('Found a path on the grid. Cycles:', cycles)
            found = True
            break
        else:
            # Get the new vertexes connected to the current vertex
            for a in valid_actions(grid, current_node):
                next_node = (current_node[0] + a.delta[0], current_node[1] + a.delta[1])
                # get the altitude differential
                climb = grid[next_node[0],next_node[1]] - current_alt
                if climb < 0.0:
                    climb = 0.0
                # going around a building is preferred, thus climbing is penalized
                new_cost = current_cost + a.cost + climb*CLIMB_COST + h(next_node, goal)
                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node, current_alt+climb))
                    branch[next_node] = (new_cost, current_node, a)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

#******************************************************************************
# Heuristic functions

# Tuned to reduce magnitude estimation error
def octile(dx, dy):
    if dx < dy:
        return 0.4142135623731 * dx + dy - 0.060662851
    else:
        return 0.4142135623731 * dy + dx - 0.060662851

def manhattan(dx, dy):
    return dx + dy

def chebyshev(dx, dy):
    return max(dx, dy)

def euclidean(dx, dy):
    return math.sqrt(dx * dx + dy * dy)

def heuristic(position, goal_position):
    dx = abs(goal_position[0]-position[0])
    dy = abs(goal_position[1]-position[1])
    #return octile(dx, dy)
    #return euclidean(dx, dy)
    #return chebyshev(dx, dy)
    return manhattan(dx, dy)

# #############################################################################
# Pruning functions

def colinear(p1, p2, p3): 
    colinear = False
    # Calculate the determinant of the matrix using integer arithmetic 
    det = p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])
    # Set colinear to True if the determinant is equal to zero
    if det == 0:
        colinear = True
    return colinear

def prune(path, grid, current_alt):
    prunedpath = []
    if len(path) > 0:
        prunedpath.append(path[0])
        for i in range(len(path)):
            if i > 1:
                climb = grid.get_altitude(path[i][0], path[i][1]) - current_alt
                if climb < 0.0:
                    climb = 0.0
                else:
                    current_alt += climb
                if climb > 0.0:
                    prunedpath.append(path[i-1])
                    prunedpath.append(path[i])
                elif not colinear(path[i-2], path[i-1], path[i]):
                    prunedpath.append(path[i-1])
        prunedpath.append(path[i])
    return prunedpath


# #############################################################################
# Graph functions

def find_closest_nodes(graph, current_position_local, global_home):
    # use brute force method (only 50 nodes in the graph)
    lowest_dist = 99999
    closest = 0
    next_closest = 0
    for inode in range(len(graph)):
        position1 = global_to_local(np.array([graph[inode,2], graph[inode,1], 0.0]), global_home)
        dist = np.linalg.norm(np.array(position1) - np.array([current_position_local[0], current_position_local[1], 0.0]))
        if dist < lowest_dist:
            next_closest = closest # also report the next closest
            closest = inode
            lowest_dist = dist
    return closest+1, next_closest+1 # node ids are 1 -> 50


def compute_cost(graph, current_node, target_node, global_home):
    position1 = global_to_local(np.array([graph[current_node-1,2], graph[current_node-1,1], 0.0]), global_home)
    position2 = global_to_local(np.array([graph[target_node-1,2], graph[target_node-1,1], 0.0]), global_home)
    return np.linalg.norm(np.array(position1) - np.array(position2))

def get_local_from_graph(graph, node_number, global_home):
    return global_to_local(np.array([graph[node_number-1,2], graph[node_number-1,1], 0.0]), global_home)

 # return actions with costs
def valid_actionsg(graph, current_node, global_home):
    next_node = int(graph[current_node-1,3])
    cost = compute_cost(graph, current_node, next_node, global_home)
    valid = [(next_node, cost)] # at least one path is guaranteed
    # iterate until end or 0 is encountered
    for i in range(4,8):
        next_node = int(graph[current_node-1, i])
        if next_node == 0:
            return valid
        else:
            cost = compute_cost(graph, current_node, next_node, global_home)
            valid.append((next_node, cost))
    return valid
 
def a_starg(graph, h, start, goal, global_home):
    """
    Given a graph and heuristic function returns
    the lowest cost path from start to goal.
    """

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set([start])
    cycles = 0

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        cycles += 1

        if current_node == goal:
            print('Found a path on the graph. Cycles:', cycles)
            found = True
            break
        else:
            # Get the new vertexes connected to the current vertex
            for a in valid_actionsg(graph, current_node, global_home):
                next_node = a[0]
                new_cost = current_cost + a[1] + h(graph, next_node, goal, global_home)
                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))
                    branch[next_node] = (new_cost, current_node) #,a
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

def heuristic2(graph, position, goal_position, global_home):
    return compute_cost(graph, position, goal_position, global_home)
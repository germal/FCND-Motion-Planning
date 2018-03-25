#__________________________________________________________________________80->|
# motion_planning.py
# Engineer: James W. Dunn
# This module plans pathways through a virtual city

import argparse
from time import sleep, gmtime, strftime
import msgpack
import csv
from enum import Enum, auto
from random import randint

import numpy as np

from planning_utils import a_star, heuristic, find_closest_nodes, a_starg, heuristic2, Grid, prune, get_local_from_graph
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

GRAPH_THRESHOLD = 222.0

class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        #**********************************************************************
        # Control option 1
        # IMPORTANT: Set to None to use the itinerary
        self.goal = None
        
        #self.goal = [37.7919133, -122.4010902]
        #self.goal = [37.7912622,-122.3999394]

        # A goal must be defined in global [latitude, longitude] format:
        # for example: [37.793618, -122.396565]

        # Random location:
        # Some colliders do not align properly to the buildings in the simulator;
        # all goals are snapped to the nearest predetermined safe landing area.
        # See option 7 below to override this action.
        #self.goal = [37.7896596+randint(0,82483)/1e7, -122.402427+randint(0,10310)/1e6]

        # Building courtyards: 
        #   [37.7912813, -122.4012908] #the deep one on the west side
        #   [37.791807, -122.395814] #the small one to the south of origin
        #   [37.792555, -122.39626] #the big one to the south of origin

        # Other points of interest:
        #self.goal = [37.7936, -122.396628] # top of Wells Fargo bank
        #self.goal = [37.7936202, -122.3949171] # plaza
        #self.goal = [37.79733, -122.402224] # location in original rubric
        #self.goal = [37.79279, -122.40127] # north mid-level of tall tower
        #self.goal = [37.795619, -122.39572] # The Tulip by John C. Portman, Jr.

        #**********************************************************************
        # Control option 2
        # Use a city map graph overlay: each node services an area of the city
        self.graph_mode = True

        #**********************************************************************
        # Control option 3
        # Fly at a lower altitude when traversing graph edges
        self.fly_low = False

        #**********************************************************************
        # Control option 4
        # Randomize itinerary
        self.randomize_itin = True

        #**********************************************************************
        # Control option 5
        # Visit this itinerary item first, even if randomized
        # Set to None or 0 through 221.
        self.visit_first = None

        #**********************************************************************
        # Control option 6
        # Enables pruning of co-linear points on grid paths
        self.cull = True

        #**********************************************************************
        # Control option 7
        # Enables snapping of specified goals to safe landing areas.
        # WARNING: due to the collider alignment issue, setting this option 
        # to False may result in unsafe landings.
        self.snap = True


        #**********************************************************************
        # Internals
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.leg = 0
        self.hold_count = 0
        self.last_id = None
        self.last_TARGET_ALTITUDE = None
        self.target_position = [0, 0, 0]

        # Initial state
        self.flight_state = States.MANUAL

        # Register callbacks
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if abs(self.local_position[2] + self.target_position[2]) < 1.0:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            # Deadband of 5 meters (also check height)
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 5.0 \
                and abs(self.local_position[2] + self.target_position[2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    # Last waypoint needs lower deadband of 50 centimeters
                    if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 0.5 \
                        and np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            # If drone is at or lower than landing position
            if self.global_position[2] <= self.landing_position[2]:
                # and is no longer moving in the Z dimension
                if abs(self.local_velocity[2]) < 0.01:
                    if abs(self.global_position[2] - self.landing_position[2]) > 0.5:
                        print("Note: potential collider calibration error at this location.")
                    if self.hold_count > 3: # breathe and count to 3
                        if self.leg < self.number_of_stops-1:
                            self.hold_count = 0
                            self.leg += 1
                            self.plan_path(self.leg)
                        else:
                            self.disarming_transition()
                    else:
                        self.hold_count += 1
                        sleep(1)
                        if self.hold_count < 3:
                            print("Delivering...", self.hold_count)
                        else:
                            print("Loading...", self.hold_count)
#            else:
#                print(abs(self.global_position[2] - self.landing_position[2]))

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                if self.global_position[0] == 0.0 and self.global_position[1] == 0.0: 
                    print("no global position data, waiting...")
                    return
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path(self.leg)
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition, seeking altitude:", self.landing_position[2])
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        if len(self.waypoints) > 0:
            print("Sending waypoints to simulator ...")
            data = msgpack.dumps(self.waypoints)
            self.connection._master.write(data)
        else:
            print("No waypoints")

    def compute_heading(self, start_point, end_point):
        x = end_point[0]-start_point[0] # north
        y = end_point[1]-start_point[1] # east
        return np.arctan2(y,x)

    def snap_to_closest(self, goal):
        closest_dist = 99999
        closest_idx = 0
        print("Original goal:", goal)
        for i in range(len(self.itinerary)):
            dist = np.linalg.norm(np.array(self.itinerary[i]) - np.array(goal))
            if dist < closest_dist:
                closest_idx = i
                closest_dist = dist
        print("Adjusted to safe goal:", self.itinerary[closest_idx])
        return np.array([self.itinerary[closest_idx]])

    def plan_simple_path(self, target_global, altitude, safety):
        print("Departing from:", self.local_position)
        # Use grid to determine path from start -> goal
        grid_start = (int(self.local_position[0]-self.north_offset), int(self.local_position[1]-self.east_offset))

        # Set the target
        self.landing_position = global_to_local(np.array(target_global), self.global_home)
        print("Landing at:", self.landing_position)
        grid_goal = (int(self.landing_position[0]-self.north_offset), int(self.landing_position[1]-self.east_offset))

        path, cost = a_star(self.grid._grid, heuristic, grid_start, grid_goal, -self.local_position[2])
        print('cur->goal: ', grid_start, grid_goal, cost)
        # prune path
        if self.cull: path = prune(path, self.grid, -self.local_position[2])
        # and convert to waypoints, climbing if necessary
        waypoints = []
        last_alt = int(-self.local_position[2]) + safety # current altitude
        prevpoint = None
        for p in path:
            climb = (self.grid.get_altitude(p[0], p[1]) + safety) - last_alt
            if climb < 0: climb = 0
            altitude = last_alt + climb
            last_alt = altitude
            if prevpoint is not None:
                heading = self.compute_heading(prevpoint, (p[0],p[1]))
            else:
                heading = 0.0
            prevpoint = (p[0],p[1])
            waypoints.append([p[0] + self.north_offset, p[1] + self.east_offset, int(altitude), heading])
        return waypoints

    def plan_compound_path(self, target_global, altitude, safety):
        # Plan summary
        print("Departing from:", self.local_position)
        # Set the target
        self.landing_position = global_to_local(np.array(target_global), self.global_home)
        print("Landing at:", self.landing_position)

        # Obtain closest graph nodes to current position (call it "A")
        node_A, node_Aalt = find_closest_nodes(self.graph, self.local_position, self.global_home)
        print("Closest node_A:", node_A)

        # and get the local coordinates
        local_A = get_local_from_graph(self.graph, node_A, self.global_home)
        #local_Aalt = get_local_from_graph(self.graph, node_Aalt, self.global_home)

        # Use grid to determine path from start -> A
        grid_start = (int(self.local_position[0]-self.north_offset), int(self.local_position[1]-self.east_offset))
        grid_goal = (int(local_A[0]-self.north_offset), int(local_A[1]-self.east_offset))

        path, cost = a_star(self.grid._grid, heuristic, grid_start, grid_goal, -self.local_position[2])
        print('cur->A: ', grid_start, grid_goal, cost)

        # prune path for leg1
        if self.cull: path = prune(path, self.grid, -self.local_position[2])

        # and convert to waypoints using previous altitude to takeoff from the current landing position
        waypoints = []
        last_alt = int(-self.local_position[2]) + safety
        prevpoint = None
        for p in path:
            climb = (self.grid.get_altitude(p[0], p[1]) + safety) - last_alt
            if climb < 0: climb = 0
            altitude = last_alt + climb
            last_alt = altitude
            if prevpoint is not None:
                heading = self.compute_heading(prevpoint, (p[0],p[1]))
            else:
                heading = 0.0
            prevpoint = (p[0],p[1])
            waypoints.append([p[0] + self.north_offset, p[1] + self.east_offset, int(altitude), heading])

        # Obtain closest graph node to landing position (call it "B")
        node_B, node_Balt = find_closest_nodes(self.graph, self.landing_position, self.global_home)
        print("Closest node_B:", node_B)
        # and get the local coordinates
        local_B = get_local_from_graph(self.graph, node_B, self.global_home)

        # Use graph to determine path from A to B
        if node_A != node_B:
            npath, cost = a_starg(self.graph, heuristic2, node_A, node_B, self.global_home)
            lpath = [get_local_from_graph(self.graph, node, self.global_home) for node in npath]
            print("node_A-> node_B:", node_A, node_B, cost)
            # and convert to leg2 waypoints at minimum altitude of 11 meters
            alt = 11
            if not self.fly_low and altitude > 11:
                alt = altitude

            leg2 = []
            prevpoint = None
            for p in lpath:
                if prevpoint is not None:
                    heading = self.compute_heading(prevpoint, (p[0],p[1]))
                else:
                    heading = 0.0
                prevpoint = (p[0],p[1])
                leg2.append([int(p[0]), int(p[1]), int(alt), heading])
        else:
            leg2 = [] # leg of length zero
            # reroute instead from start to goal
            print("Replanning with direct path...")
            waypoints = [] # void the start -> A plan
            local_B = self.local_position
            alt = -self.local_position[2]

        # Finally, use grid to determine path from B -> goal
        grid_start = (int(local_B[0])-self.north_offset, int(local_B[1])-self.east_offset)
        grid_goal = (int(self.landing_position[0]-self.north_offset), int(self.landing_position[1]-self.east_offset))
        path, cost = a_star(self.grid._grid, heuristic, grid_start, grid_goal, alt)
        print('B->goal: ', grid_start, grid_goal, cost)
        # prune path
        last_alt = int(alt) # last altitude from node B
        if self.cull: path = prune(path, self.grid, last_alt)
        # and convert to leg3 waypoints
        #leg3 = [[p[0] + self.north_offset, p[1] + self.east_offset, altitude, 0] for p in path]

        leg3 = []
        prevpoint = None
        for p in path:
            climb = (self.grid.get_altitude(p[0], p[1]) + safety) - last_alt
            if climb < 0: climb = 0
            altitude = last_alt + climb
            last_alt = altitude
            if prevpoint is not None:
                heading = self.compute_heading(prevpoint, (p[0],p[1]))
            else:
                heading = 0.0
            prevpoint = (p[0],p[1])
            leg3.append([p[0] + self.north_offset, p[1] + self.east_offset, int(altitude), heading])

        # Summarize
        print("cur->A waypoints:", waypoints)
        print("A->B waypoints:", leg2)
        print("B->goal waypoints:", leg3)

        # Concatatenate leg2 and leg3
        if self.fly_low:
            waypoints.extend(leg2)
            waypoints.extend(leg3)
        else: # or fly at last departure height
            waypoints.extend(leg2[1:]) # skip the first waypoint
            waypoints.extend(leg3[1:])
        return waypoints

    def plan_path(self, id):
        self.flight_state = States.PLANNING
        print("Planning...")
        TARGET_ALTITUDE = 6
        SAFETY_DISTANCE = 7.0

        if id == 0:
            # Initial
            # Read lat0, lon0 from colliders into floating point values
            with open('colliders.csv', newline='') as file:
                line = csv.reader(file)
                (lat0, lon0) = next(line)
                lat0, lon0 = lat0[5:], lon0[5:]

            # Set home position to (lat0, lon0, 0)
            print("Setting home position:", lat0, lon0, 0.)
            self.set_home_position(float(lon0), float(lat0), 0.) # NOTE THE ORDER!!!

            # Retrieve current global position
            print("Current global position:", self.global_position)
            print("Verify:", self._longitude, self._latitude, self._altitude)

            print('global home:', self.global_home)
            print('global position:', self.global_position)
            print('local position:', self.local_position)

            if self.last_TARGET_ALTITUDE is None:
                if self.local_position[2] < 0.0:
                    self.last_TARGET_ALTITUDE = int(-self.local_position[2] + SAFETY_DISTANCE + 1)
                else:
                    self.last_TARGET_ALTITUDE = TARGET_ALTITUDE

            # Read in obstacle map
            self.data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

            # Create a 2D grid object
            self.grid = Grid(self.data)
            self.north_offset, self.east_offset = self.grid.get_offset()
            self.grid.update_obstacles(SAFETY_DISTANCE)
            print("Offset north: {0}, east: {1}".format(self.north_offset, self.east_offset))

            # Read in graph (map of city intersections)
            self.graph = np.loadtxt('graph.csv', delimiter=',', dtype='Float64', skiprows=0)

            # Read in itinerary (list of points to visit)
            self.itinerary = np.loadtxt('itinerary.csv', delimiter=',', dtype='Float64', skiprows=1)
            self.number_of_stops = len(self.itinerary)

            # Note: to set a specific goal, visit the top of this class
            if self.goal is not None:
                # Itinerary override: targeting specified goal instead
                # For safety, snap to closest itinerary item
                if self.snap:
                    self.itinerary = self.snap_to_closest(self.goal)
                else:
                    self.itinerary = np.array([[self.goal[0], self.goal[1]]])
                self.number_of_stops = 1
                self.visit_first = 0
            elif self.visit_first is None and not self.randomize_itin:
                # no priority on itin item
                self.last_id = -1

        # Set flight altitude to be current height + TARGET_ALTITUDE
        # in case drone is on top of a building
        self.target_position[2] = int(-self.local_position[2] + TARGET_ALTITUDE)

        print("Take off target altitude:", self.target_position[2])
        if id == 0 and self.visit_first is not None:
            next_id = self.visit_first
            self.last_id = next_id
        else:
            if self.randomize_itin:
                next_id = randint(0,len(self.itinerary)-1) # get another itinerary destination
                while next_id == self.last_id:
                    next_id = randint(0,len(self.itinerary)-1) # oops, same one, try again
                self.last_id = next_id
            else:
                next_id = self.last_id + 1
                if next_id > self.number_of_stops-1: # loop back to first item
                    next_id = 0
                self.last_id = next_id

        print(" ")
        print("###############################################################################")
        print("Time stamp:", strftime("%Y%m%d.%H%M%S", gmtime()))
        print("Stop number: {0} of {1}".format(id+1, self.number_of_stops))
        print("Itinerary index number:", next_id)
        # set up the target
        local_target = global_to_local(np.array([self.itinerary[next_id,1], self.itinerary[next_id,0], 0.0]), self.global_home)
        landing_altitude = self.grid.get_landing_altitude(int(local_target[0])-self.north_offset, int(local_target[1])-self.east_offset)
        target = [self.itinerary[next_id,1], self.itinerary[next_id,0], -landing_altitude]
        print("Global target:", target)
        target_distance = heuristic(self.local_position, local_target)
        print("Distance to target:", target_distance)
        if self.graph_mode and target_distance>GRAPH_THRESHOLD:
            print("Attempting compound path...")
            waypoints = self.plan_compound_path(target, TARGET_ALTITUDE, SAFETY_DISTANCE)
            self.last_TARGET_ALTITUDE = TARGET_ALTITUDE
        else:
            print("Planning direct path...")
            waypoints = self.plan_simple_path(target, TARGET_ALTITUDE, SAFETY_DISTANCE)

        # Set self.waypoints for a queue of points to follow toward goal
        self.waypoints = waypoints
        # Send waypoints to sim for visualization
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if threaded
        # while self.in_mission:
        #    pass

        self.stop_log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    sleep(2)
    drone.start()

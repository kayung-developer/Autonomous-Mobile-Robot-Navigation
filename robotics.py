import tkinter
import customtkinter as ctk
from tkinter import colorchooser, messagebox, simpledialog
import asyncio
import threading
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import math
import heapq
import random
import time
import json
import queue
import websockets  # For GUI's websocket client
import requests  # For sync requests from GUI to backend
from contextlib import asynccontextmanager

# --- CONFIGURATION PARAMETERS ---
CONFIG = {
    "map_width": 30,
    "map_height": 20,
    "cell_size_gui": 25,
    "simulation_tick_rate": 0.1,  # Target seconds per simulation tick
    "robot_radius_grid": 0.5,
    "robot_wheel_base": 0.8,  # For differential drive kinematics (if fully implemented)
    "robot_max_linear_speed": 2.0,  # cells per second
    "robot_max_angular_speed": math.pi / 1.5,  # radians per second (increased slightly)
    "robot_linear_acceleration": 3.0,  # cells per second^2 (increased slightly)
    "robot_angular_acceleration": math.pi * 1.5,  # rad per second^2 (increased slightly)
    "lidar_range_grid": 8,
    "lidar_rays": 60,
    "lidar_fov_deg": 180,
    "lidar_distance_noise_std_dev": 0.05,  # Noise added to LIDAR distances
    "odometry_pos_noise_std_dev_per_meter": 0.02,  # Proportional to distance travelled
    "odometry_angle_noise_std_dev_per_radian": 0.01,  # Proportional to rotation
    "camera_fov_deg": 70,  # Slightly wider FOV
    "camera_range_grid": 10,
    "battery_initial": 100.0,
    "battery_drain_idle": 0.001,  # per tick
    "battery_drain_move_linear": 0.05,  # per grid unit moved
    "battery_drain_move_angular": 0.02,  # per radian turned
    "battery_drain_lidar": 0.005,  # per scan
    "battery_drain_camera": 0.003,  # per frame processing
    "dynamic_obstacle_speed_grid": 0.5,  # cells per second
    "path_following_Kp_angular": 3.0,  # Proportional gain for angular error
    "path_following_Kp_linear": 1.8,  # Proportional gain for linear speed
    "path_angle_threshold_rad": math.radians(12),  # Angle tolerance for moving forward
    "path_arrival_radius_grid": 0.5 * 1.5,  # When to consider waypoint reached
    "lidar_emergency_stop_distance_factor": 2.0,  # Multiplier of robot_radius_grid
    "lidar_emergency_stop_fov_rad": math.radians(20),  # +/- this angle from front
    "float_button_linear_speed": 1.5,
    "float_button_angular_speed": math.pi / 4,
}


# --- SHARED UTILITIES & DATA STRUCTURES ---
class Point(BaseModel):
    x: float
    y: float


class Goal(BaseModel):
    target_position: Point


class ManualControlCommand(BaseModel):
    action: str
    linear_speed: float = 0.0
    angular_speed: float = 0.0


class ObstacleData(BaseModel):
    x: int
    y: int
    width: int = 1
    height: int = 1
    obstacle_type: str = "static"
    terrain_cost: float = 1.0


class DetectedObject(BaseModel):
    id: str
    label: str
    confidence: float
    bbox: list[float]
    map_coords: list[Point]


# --- BACKEND (FastAPI & ROBOT SIMULATION) ---
class Robot:
    def __init__(self, x, y, orientation_rad):
        self.x = x
        self.y = y
        self.orientation_rad = orientation_rad

        self.wheel_base = CONFIG["robot_wheel_base"]
        self.current_linear_speed = 0.0
        self.current_angular_speed = 0.0
        self.target_linear_speed = 0.0
        self.target_angular_speed = 0.0

        self.odometry_x = x
        self.odometry_y = y
        self.odometry_orientation_rad = orientation_rad

        self.fused_x = x
        self.fused_y = y
        self.fused_orientation_rad = orientation_rad

        self.battery = CONFIG["battery_initial"]
        self.status_message = "Idle"
        self.current_path = []
        self.local_path_segment = []

    def update_speeds(self, dt):
        accel_linear = np.clip(self.target_linear_speed - self.current_linear_speed,
                               -CONFIG["robot_linear_acceleration"] * dt,
                               CONFIG["robot_linear_acceleration"] * dt)
        self.current_linear_speed += accel_linear

        accel_angular = np.clip(self.target_angular_speed - self.current_angular_speed,
                                -CONFIG["robot_angular_acceleration"] * dt,
                                CONFIG["robot_angular_acceleration"] * dt)
        self.current_angular_speed += accel_angular

        self.current_linear_speed = np.clip(self.current_linear_speed, -CONFIG["robot_max_linear_speed"],
                                            CONFIG["robot_max_linear_speed"])
        self.current_angular_speed = np.clip(self.current_angular_speed, -CONFIG["robot_max_angular_speed"],
                                             CONFIG["robot_max_angular_speed"])

    def move(self, dt, environment):
        self.update_speeds(dt)

        delta_s = self.current_linear_speed * dt
        delta_theta = self.current_angular_speed * dt

        new_x = self.x
        new_y = self.y
        new_orientation_rad = (self.orientation_rad + delta_theta) % (2 * math.pi)

        # Simplified kinematic model: apply linear motion along current orientation, then rotate
        # For more accuracy, especially with high angular speeds, a curved path model (e.g., arc) would be used.
        if abs(self.current_angular_speed) < 1e-4:  # Primarily straight motion
            new_x += delta_s * math.cos(self.orientation_rad)
            new_y += delta_s * math.sin(self.orientation_rad)
        elif abs(self.current_linear_speed) < 1e-4:  # Primarily in-place rotation
            # Orientation already updated, x and y don't change much
            pass
        else:  # Combined motion (approximated as arc)
            # Using midpoint velocity for arc approximation can be more stable
            avg_orientation = self.orientation_rad + delta_theta / 2.0
            new_x += delta_s * math.cos(avg_orientation)
            new_y += delta_s * math.sin(avg_orientation)

        if not environment.is_occupied(new_x, new_y, robot_radius_grid=CONFIG["robot_radius_grid"]):
            self.x = new_x
            self.y = new_y
            self.orientation_rad = new_orientation_rad

            pos_noise_std = CONFIG["odometry_pos_noise_std_dev_per_meter"] * abs(delta_s)
            angle_noise_std = CONFIG["odometry_angle_noise_std_dev_per_radian"] * abs(delta_theta)

            noisy_delta_s = delta_s + random.gauss(0, pos_noise_std)
            noisy_delta_theta = delta_theta + random.gauss(0, angle_noise_std)

            prev_odom_orientation = self.odometry_orientation_rad  # Store before updating
            self.odometry_orientation_rad = (self.odometry_orientation_rad + noisy_delta_theta) % (2 * math.pi)

            # Apply translation based on the orientation *before* this step's rotation for odometry
            # This is a common approach for discrete odometry updates
            self.odometry_x += noisy_delta_s * math.cos(prev_odom_orientation)
            self.odometry_y += noisy_delta_s * math.sin(prev_odom_orientation)

            # Simulate a basic sensor fusion result (e.g., Kalman filter output proxy)
            # For simplicity, it's a slightly noisy version of true pose.
            self.fused_x = self.x + random.gauss(0, 0.03)  # Reduced noise for "fused"
            self.fused_y = self.y + random.gauss(0, 0.03)
            self.fused_orientation_rad = (self.orientation_rad + random.gauss(0, 0.005)) % (2 * math.pi)

            self.battery -= CONFIG["battery_drain_move_linear"] * abs(delta_s)
            self.battery -= CONFIG["battery_drain_move_angular"] * abs(delta_theta)

        else:
            self.status_message = "Collision! Stopped."
            self.current_linear_speed = 0
            self.current_angular_speed = 0
            self.target_linear_speed = 0
            self.target_angular_speed = 0
            global is_navigating_autonomously
            if is_navigating_autonomously:
                is_navigating_autonomously = False
                self.current_path = []  # Clear path on collision

        self.battery -= CONFIG["battery_drain_idle"]
        self.battery = max(0, self.battery)
        if self.battery <= 1e-6:  # Effectively zero
            self.status_message = "Battery depleted! System shutdown."
            self.current_linear_speed = 0
            self.current_angular_speed = 0
            self.target_linear_speed = 0
            self.target_angular_speed = 0
            is_navigating_autonomously = False


class DynamicObstacleSim:
    def __init__(self, x, y, width, height, path_coords: list = None, speed=1.0):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.path_coords = [(float(px), float(py)) for px, py in path_coords] if path_coords else []
        self.current_target_idx = 0
        self.speed = float(speed)
        self.obstacle_type = "dynamic"

    def move(self, dt, environment):
        if not self.path_coords:
            return

        target_x, target_y = self.path_coords[self.current_target_idx]

        dist_to_target = math.sqrt((target_x - self.x) ** 2 + (target_y - self.y) ** 2)

        if dist_to_target < self.speed * dt * 1.5:  # Approximation for reaching waypoint
            self.x, self.y = target_x, target_y
            self.current_target_idx = (self.current_target_idx + 1) % len(self.path_coords)
        else:
            angle_to_target = math.atan2(target_y - self.y, target_x - self.x)
            move_dist = self.speed * dt

            new_x = self.x + move_dist * math.cos(angle_to_target)
            new_y = self.y + move_dist * math.sin(angle_to_target)

            # Simplified collision check for dynamic obstacles (check center against static map)
            # This prevents dynamic obstacles from walking into static ones.
            can_move = True
            # Check approximate center of dynamic obstacle's new position against static grid.
            check_pos_x, check_pos_y = new_x + self.width / 2.0, new_y + self.height / 2.0
            if environment.is_occupied(check_pos_x, check_pos_y, robot_radius_grid=0, check_dynamic=False):
                can_move = False

            if can_move:
                self.x = new_x
                self.y = new_y

    def get_bounds(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def model_dump(self):
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height,
                "obstacle_type": self.obstacle_type}


class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)  # 0: free, 1: static obstacle
        self.terrain_costs = np.ones((height, width), dtype=float)  # Cost multiplier for path planning
        self.dynamic_obstacles_sim = []  # List of DynamicObstacleSim objects

    def add_obstacle(self, x, y, w=1, h=1, obstacle_type="static", terrain_cost=10000.0, dynamic_path=None):
        if obstacle_type == "dynamic":
            dyn_obs_speed = CONFIG["dynamic_obstacle_speed_grid"]
            self.dynamic_obstacles_sim.append(DynamicObstacleSim(x, y, w, h, dynamic_path, dyn_obs_speed))
        else:  # static
            for r_idx in range(int(y), min(int(y) + int(h), self.height)):
                for c_idx in range(int(x), min(int(x) + int(w), self.width)):
                    if 0 <= r_idx < self.height and 0 <= c_idx < self.width:
                        self.grid[r_idx, c_idx] = 1  # Mark as obstacle
                        self.terrain_costs[r_idx, c_idx] = terrain_cost  # High cost for static obstacles

    def set_terrain_cost(self, x, y, cost):
        x_int, y_int = int(x), int(y)
        if 0 <= y_int < self.height and 0 <= x_int < self.width and self.grid[
            y_int, x_int] == 0:  # Can only set cost for non-obstacle cells
            self.terrain_costs[y_int, x_int] = cost

    def is_occupied(self, x_check, y_check, robot_radius_grid=0.0, check_dynamic=True):
        # Check map boundaries first
        if not (robot_radius_grid <= x_check < self.width - robot_radius_grid and \
                robot_radius_grid <= y_check < self.height - robot_radius_grid):
            return True  # Out of bounds is considered occupied

        # Check static obstacles considering robot radius (inflation by checking surrounding cells)
        min_ix_flt, max_ix_flt = x_check - robot_radius_grid, x_check + robot_radius_grid
        min_iy_flt, max_iy_flt = y_check - robot_radius_grid, y_check + robot_radius_grid

        # Iterate over cells whose area *might* overlap with the robot's circular footprint
        for iy_cell in range(int(math.floor(min_iy_flt)), int(math.ceil(max_iy_flt))):
            for ix_cell in range(int(math.floor(min_ix_flt)), int(math.ceil(max_ix_flt))):
                if not (0 <= iy_cell < self.height and 0 <= ix_cell < self.width):
                    continue  # Cell out of bounds

                # If this cell is a static obstacle, perform a more precise circle-rectangle collision
                if self.grid[iy_cell, ix_cell] == 1:
                    # Closest point in cell (rectangle [ix_cell, iy_cell] to [ix_cell+1, iy_cell+1]) to circle center (x_check, y_check)
                    closest_x = np.clip(x_check, float(ix_cell), float(ix_cell) + 1.0)
                    closest_y = np.clip(y_check, float(iy_cell), float(iy_cell) + 1.0)

                    dist_sq = (x_check - closest_x) ** 2 + (y_check - closest_y) ** 2
                    if dist_sq < robot_radius_grid ** 2 - 1e-9:  # Subtract epsilon for float safety
                        return True  # Collision with static obstacle

        # Check dynamic obstacles (axis-aligned bounding box collision with robot's circular footprint approximated as a square for simplicity here)
        if check_dynamic:
            robot_min_x, robot_min_y = x_check - robot_radius_grid, y_check - robot_radius_grid
            robot_max_x, robot_max_y = x_check + robot_radius_grid, y_check + robot_radius_grid
            for obs in self.dynamic_obstacles_sim:
                obs_min_x, obs_min_y, obs_max_x, obs_max_y = obs.get_bounds()
                # Check for overlap (AABB vs AABB)
                if not (robot_max_x <= obs_min_x or robot_min_x >= obs_max_x or \
                        robot_max_y <= obs_min_y or robot_min_y >= obs_max_y):
                    return True  # Collision with dynamic obstacle
        return False  # No collision detected

    def update_dynamic_obstacles(self, dt):
        for obs in self.dynamic_obstacles_sim:
            obs.move(dt, self)


class AStarNode:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to end
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):  # For heapq comparison
        return self.f < other.f

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):  # For adding to sets
        return hash(self.position)


class PathPlanner:
    def __init__(self, environment):
        self.environment = environment

    def heuristic(self, a, b):  # Euclidean distance heuristic
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_neighbors(self, node_pos):  # node_pos is (int_x, int_y)
        neighbors = []
        # 8-directional movement
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            x, y = node_pos[0] + dx, node_pos[1] + dy
            # Check if neighbor is within map bounds and traversable (checking center of cell)
            if 0 <= x < self.environment.width and \
                    0 <= y < self.environment.height and \
                    not self.environment.is_occupied(x + 0.5, y + 0.5, robot_radius_grid=CONFIG["robot_radius_grid"]):
                cost_multiplier = self.environment.terrain_costs[y, x]
                move_cost = math.sqrt(dx ** 2 + dy ** 2) * cost_multiplier  # Cost is distance * terrain_factor
                neighbors.append(((x, y), move_cost))
        return neighbors

    def find_path(self, start_pos_float, end_pos_float):  # start/end are float (robot's actual position)
        # Snap float positions to grid cell integers for planning
        start_pos = (int(round(start_pos_float[0])), int(round(start_pos_float[1])))
        end_pos = (int(round(end_pos_float[0])), int(round(end_pos_float[1])))

        # Validate start/end points are not inside an obstacle (using robot's actual float position for this check)
        if self.environment.is_occupied(start_pos_float[0], start_pos_float[1],
                                        robot_radius_grid=CONFIG["robot_radius_grid"]):
            print(f"A* Planner: Start position ({start_pos_float[0]:.2f}, {start_pos_float[1]:.2f}) is occupied.")
            return None
        if self.environment.is_occupied(end_pos_float[0], end_pos_float[1],
                                        robot_radius_grid=CONFIG["robot_radius_grid"]):
            print(f"A* Planner: End position ({end_pos_float[0]:.2f}, {end_pos_float[1]:.2f}) is occupied.")
            return None

        start_node = AStarNode(start_pos)
        end_node = AStarNode(end_pos)

        open_list = []  # Priority queue (min-heap)
        closed_set = set()  # Set of visited node positions
        heapq.heappush(open_list, start_node)

        # Store g-costs to reach nodes, to update if a shorter path is found
        g_costs = {start_node.position: 0}

        while open_list:
            current_node = heapq.heappop(open_list)

            if current_node.position in closed_set:
                continue  # Already processed this node via a shorter or equal path

            closed_set.add(current_node.position)

            if current_node.position == end_node.position:  # Goal reached
                path = []
                temp_current = current_node
                while temp_current:
                    # Path waypoints are cell centers
                    path.append((float(temp_current.position[0]) + 0.5, float(temp_current.position[1]) + 0.5))
                    temp_current = temp_current.parent
                return self.smooth_path(path[::-1])  # Return reversed and smoothed path

            neighbors_info = self.get_neighbors(current_node.position)
            for neighbor_pos, move_cost in neighbors_info:
                if neighbor_pos in closed_set:
                    continue  # Skip already processed neighbors

                tentative_g_score = current_node.g + move_cost

                if tentative_g_score < g_costs.get(neighbor_pos, float('inf')):
                    g_costs[neighbor_pos] = tentative_g_score
                    neighbor_node = AStarNode(neighbor_pos, current_node)
                    neighbor_node.g = tentative_g_score
                    neighbor_node.h = self.heuristic(neighbor_node.position, end_node.position)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    heapq.heappush(open_list, neighbor_node)

        print(f"A* Planner: Path not found from {start_pos} to {end_pos}")
        return None  # Path not found

    def smooth_path(self, path, iterations=50, pull_strength=0.2, straighten_strength=0.1):  # Reduced iterations
        if not path or len(path) < 3:
            return path

        # Work with a list of mutable lists or Point objects if preferred, tuples are fine too
        smoothed_path_tuples = list(path)

        # Iterative smoothing (like Laplacian smoothing)
        for _ in range(iterations):
            if len(smoothed_path_tuples) < 3: break
            new_path_tuples_iter = list(smoothed_path_tuples)

            for i in range(1, len(smoothed_path_tuples) - 1):
                p_prev_x, p_prev_y = smoothed_path_tuples[i - 1]
                p_curr_x, p_curr_y = smoothed_path_tuples[i]
                p_next_x, p_next_y = smoothed_path_tuples[i + 1]

                # Pull current point towards the midpoint of its neighbors
                mid_x = (p_prev_x + p_next_x) / 2.0
                mid_y = (p_prev_y + p_next_y) / 2.0

                candidate_x = p_curr_x + pull_strength * (mid_x - p_curr_x)
                candidate_y = p_curr_y + pull_strength * (mid_y - p_curr_y)

                # Basic collision check for the new smoothed point - check with a smaller radius for the point itself
                # This prevents smoothing into an obstacle too aggressively.
                if not self.environment.is_occupied(candidate_x, candidate_y,
                                                    robot_radius_grid=CONFIG["robot_radius_grid"] * 0.3):
                    new_path_tuples_iter[i] = (candidate_x, candidate_y)

            smoothed_path_tuples = new_path_tuples_iter

        # Final pass: try to shortcut segments (Greedy algorithm to remove redundant waypoints)
        if len(smoothed_path_tuples) < 2: return smoothed_path_tuples  # Guard against very short paths

        final_smoothed_path = [smoothed_path_tuples[0]]
        current_idx = 0
        while current_idx < len(smoothed_path_tuples) - 1:
            best_next_idx = current_idx + 1  # Default to next point if no shortcut found
            # Try to connect current_idx to points further down the path
            for test_idx in range(len(smoothed_path_tuples) - 1, current_idx + 1, -1):  # Iterate backwards
                p1_x, p1_y = smoothed_path_tuples[current_idx]
                p2_x, p2_y = smoothed_path_tuples[test_idx]

                # Check line-of-sight between p1 and p2
                # Ray casting check with steps based on robot radius for safety
                dist_segment = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
                num_steps = max(2, int(dist_segment / (CONFIG["robot_radius_grid"] * 0.4)))  # Smaller steps for safety

                is_clear = True
                for i_step in range(1, num_steps):  # Exclude endpoints already known/assumed clear
                    t = i_step / float(num_steps)
                    check_x = p1_x * (1.0 - t) + p2_x * t
                    check_y = p1_y * (1.0 - t) + p2_y * t
                    if self.environment.is_occupied(check_x, check_y, robot_radius_grid=CONFIG["robot_radius_grid"]):
                        is_clear = False
                        break
                if is_clear:
                    best_next_idx = test_idx  # Found a valid shortcut
                    break
            final_smoothed_path.append(smoothed_path_tuples[best_next_idx])
            current_idx = best_next_idx

        return final_smoothed_path


robot = Robot(x=3.0, y=3.0, orientation_rad=0.0)
environment = Environment(CONFIG["map_width"], CONFIG["map_height"])
path_planner = PathPlanner(environment)
is_navigating_autonomously = False
navigation_goal_pos = None
current_lidar_data = {"angles": [], "distances": [], "robot_pose_at_scan": Point(x=robot.fused_x, y=robot.fused_y)}
current_camera_data = {"detected_objects": []}

environment.add_obstacle(10, 5, 3, 3)
environment.add_obstacle(15, 10, 2, 5)
environment.add_obstacle(5, 15, 4, 2)
environment.add_obstacle(x=20, y=3, w=1, h=1, obstacle_type="dynamic",
                         dynamic_path=[(20, 3), (20, 17), (25, 17), (25, 3)])
environment.add_obstacle(x=1, y=10, w=1, h=1, obstacle_type="dynamic",
                         dynamic_path=[(1, 10), (7, 10), (7, 1), (1, 1)])

simulation_task_handle = None


async def robot_simulation_loop():
    global is_navigating_autonomously, navigation_goal_pos, robot, simulation_task_handle
    last_path_check_time = time.time()
    dt = CONFIG["simulation_tick_rate"]

    while True:
        try:
            current_time = time.time()
            loop_start_time = current_time  # For precise sleep calculation later

            environment.update_dynamic_obstacles(dt)

            # LIDAR-based Emergency Stop
            emergency_stopped_this_tick = False
            if is_navigating_autonomously and robot.current_linear_speed > 0.05:  # If moving forward
                min_front_dist = CONFIG["lidar_range_grid"]
                if current_lidar_data and current_lidar_data.get("angles"):
                    for angle_rad, dist in zip(current_lidar_data["angles"], current_lidar_data["distances"]):
                        relative_angle = (angle_rad - robot.fused_orientation_rad + math.pi) % (2 * math.pi) - math.pi
                        if abs(relative_angle) < CONFIG["lidar_emergency_stop_fov_rad"]:
                            min_front_dist = min(min_front_dist, dist)

                if min_front_dist < CONFIG["robot_radius_grid"] * CONFIG["lidar_emergency_stop_distance_factor"]:
                    robot.status_message = "LIDAR Emergency Stop! Replanning."
                    robot.target_linear_speed = 0
                    robot.target_angular_speed = 0
                    robot.current_linear_speed = 0  # Hard stop
                    robot.current_angular_speed = 0
                    robot.current_path = []  # Force replan
                    emergency_stopped_this_tick = True
                    print(f"LIDAR Emergency Stop triggered. Min front dist: {min_front_dist:.2f}")

            if robot.battery <= 1e-6:  # Effectively zero
                robot.status_message = "Battery critical. System halted."
                robot.target_linear_speed = 0;
                robot.target_angular_speed = 0
                is_navigating_autonomously = False
            elif is_navigating_autonomously and not emergency_stopped_this_tick:
                robot.status_message = "Navigating Autonomously"

                path_needs_replan = False
                # Dynamic path obstruction check (more robust: check lookahead distance)
                if robot.current_path and (current_time - last_path_check_time > 0.75 or len(robot.current_path) < 3):
                    last_path_check_time = current_time
                    lookahead_dist_check = 3.0  # Grid units
                    cumulative_dist = 0.0
                    for i in range(len(robot.current_path)):
                        wp_x, wp_y = robot.current_path[i]
                        if i > 0:
                            prev_wp_x, prev_wp_y = robot.current_path[i - 1]
                            cumulative_dist += math.sqrt((wp_x - prev_wp_x) ** 2 + (wp_y - prev_wp_y) ** 2)

                        if environment.is_occupied(wp_x, wp_y, robot_radius_grid=CONFIG["robot_radius_grid"]):
                            path_needs_replan = True;
                            break
                        if cumulative_dist > lookahead_dist_check and i > 0:  # Checked enough waypoints within lookahead
                            break
                    if path_needs_replan:
                        robot.status_message = "Path obstructed! Replanning..."
                        robot.current_path = []

                        # Path Following Controller
                if robot.current_path and not path_needs_replan:
                    next_wp_x, next_wp_y = robot.current_path[0]
                    dist_to_wp = math.sqrt((next_wp_x - robot.fused_x) ** 2 + (next_wp_y - robot.fused_y) ** 2)
                    angle_to_wp_rad = math.atan2(next_wp_y - robot.fused_y, next_wp_x - robot.fused_x)
                    required_turn_rad = (angle_to_wp_rad - robot.fused_orientation_rad + math.pi) % (
                                2 * math.pi) - math.pi

                    # Angular speed control
                    robot.target_angular_speed = np.clip(CONFIG["path_following_Kp_angular"] * required_turn_rad,
                                                         -CONFIG["robot_max_angular_speed"],
                                                         CONFIG["robot_max_angular_speed"])

                    # Linear speed control
                    if abs(required_turn_rad) < CONFIG["path_angle_threshold_rad"]:
                        target_lin_speed = CONFIG["path_following_Kp_linear"] * dist_to_wp
                        # Slow down if it's the last waypoint (final approach)
                        if len(robot.current_path) == 1:
                            target_lin_speed = min(target_lin_speed, CONFIG[
                                "robot_max_linear_speed"] * 0.6)  # Slower max for final approach

                        # Reduce speed if still turning significantly, even if within angle threshold
                        if abs(robot.target_angular_speed) > CONFIG["robot_max_angular_speed"] * 0.5:
                            target_lin_speed *= 0.6

                        robot.target_linear_speed = np.clip(target_lin_speed, 0, CONFIG["robot_max_linear_speed"])
                    else:  # Prioritize turning
                        robot.target_linear_speed = 0.0

                        # Waypoint arrival
                    if dist_to_wp < CONFIG["path_arrival_radius_grid"]:
                        robot.current_path.pop(0)
                        if not robot.current_path:  # Final goal reached
                            robot.status_message = "Goal Reached!"
                            is_navigating_autonomously = False;
                            navigation_goal_pos = None
                            robot.target_linear_speed = 0;
                            robot.target_angular_speed = 0

                elif navigation_goal_pos:  # No current path, but a goal exists -> Plan
                    start_plan_pos = (robot.fused_x, robot.fused_y)
                    end_plan_pos = navigation_goal_pos

                    robot.status_message = f"Planning path to ({end_plan_pos[0]:.1f}, {end_plan_pos[1]:.1f})..."
                    await manager.broadcast(json.dumps(get_current_state(include_map=False)))

                    new_path = await asyncio.to_thread(path_planner.find_path, start_plan_pos, end_plan_pos)

                    if new_path:
                        robot.current_path = new_path
                        robot.status_message = "Path Planned. Navigating."
                    else:
                        robot.status_message = "Path to goal not found. Stopping."
                        is_navigating_autonomously = False;
                        navigation_goal_pos = None
                        robot.target_linear_speed = 0;
                        robot.target_angular_speed = 0
                elif is_navigating_autonomously:  # In autonomous mode, but no goal and no path
                    robot.status_message = "Idle (Autonomous Mode - No Goal)"
                    robot.target_linear_speed = 0;
                    robot.target_angular_speed = 0

            robot.move(dt, environment)
            simulate_lidar()  # Simulate sensors after movement
            simulate_camera_cv_ml()

            current_state_json = json.dumps(get_current_state(include_map=False))
            await manager.broadcast(current_state_json)

            # Precise sleep to maintain tick rate
            elapsed_time = time.time() - loop_start_time
            sleep_duration = max(0, dt - elapsed_time)
            await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            print("Robot simulation loop was cancelled.")
            break
        except Exception as e:
            print(f"Error in robot_simulation_loop: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(dt)  # Fallback sleep on error


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global simulation_task_handle
    print("Lifespan: Startup sequence initiated.")
    simulation_task_handle = asyncio.create_task(robot_simulation_loop())
    print("Lifespan: Robot simulation loop task created.")
    yield
    print("Lifespan: Shutdown sequence initiated.")
    if simulation_task_handle and not simulation_task_handle.done():
        print("Lifespan: Cancelling simulation loop task...")
        simulation_task_handle.cancel()
        try:
            await simulation_task_handle
        except asyncio.CancelledError:
            print("Lifespan: Simulation loop task successfully cancelled.")
        except Exception as e:
            print(f"Lifespan: Exception during simulation task shutdown: {e}")
    print("Lifespan: Shutdown complete.")


app = FastAPI(title="Advanced Autonomous Mobile Robot Navigation API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except (
                    WebSocketDisconnect, websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK,
                    RuntimeError):  # Added RuntimeError for abrupt closes
                self.disconnect(connection)
            except Exception as e:
                print(f"Error broadcasting to a websocket client: {type(e).__name__} - {e}. Removing.")
                self.disconnect(connection)


manager = ConnectionManager()


def simulate_lidar():
    global current_lidar_data, robot
    angles = []
    distances = []
    current_lidar_data["robot_pose_at_scan"] = Point(x=robot.fused_x, y=robot.fused_y)

    start_angle_rad = robot.fused_orientation_rad - math.radians(CONFIG["lidar_fov_deg"] / 2)
    angle_increment_rad = math.radians(CONFIG["lidar_fov_deg"]) / (
        CONFIG["lidar_rays"] - 1 if CONFIG["lidar_rays"] > 1 else 1)

    for i in range(CONFIG["lidar_rays"]):
        angle_rad = start_angle_rad + i * angle_increment_rad
        angles.append(angle_rad)

        dist = 0.01  # Start closer to robot
        hit = False
        # More precise ray casting steps
        ray_step_size = 0.1
        max_steps = int(CONFIG["lidar_range_grid"] / ray_step_size)

        for _ in range(max_steps):
            check_x = robot.fused_x + dist * math.cos(angle_rad)
            check_y = robot.fused_y + dist * math.sin(angle_rad)

            if not (0 <= check_x < environment.width and 0 <= check_y < environment.height):
                dist = CONFIG["lidar_range_grid"]  # Hit boundary
                hit = True;
                break

            ix, iy = int(check_x), int(check_y)
            if environment.grid[iy, ix] == 1:  # Static obstacle
                hit = True;
                break

            for dyn_obs in environment.dynamic_obstacles_sim:  # Dynamic obstacles
                o_x1, o_y1, o_x2, o_y2 = dyn_obs.get_bounds()
                if o_x1 <= check_x <= o_x2 and o_y1 <= check_y <= o_y2:
                    hit = True;
                    break
            if hit: break
            dist += ray_step_size
            if dist >= CONFIG["lidar_range_grid"]:  # Exceeded range
                dist = CONFIG["lidar_range_grid"];
                hit = True;
                break

        noisy_dist = max(0, dist + random.gauss(0, CONFIG["lidar_distance_noise_std_dev"]))
        distances.append(min(noisy_dist, CONFIG["lidar_range_grid"]))

    current_lidar_data["angles"] = angles
    current_lidar_data["distances"] = distances
    robot.battery -= CONFIG["battery_drain_lidar"]


def simulate_camera_cv_ml():
    global current_camera_data, robot
    detected_objects_list = []

    cam_fov_rad = math.radians(CONFIG["camera_fov_deg"])
    robot_angle = robot.fused_orientation_rad

    for i, obs in enumerate(environment.dynamic_obstacles_sim):
        obs_center_x = obs.x + obs.width / 2.0
        obs_center_y = obs.y + obs.height / 2.0

        angle_to_obs_rad = math.atan2(obs_center_y - robot.fused_y, obs_center_x - robot.fused_x)
        dist_to_obs = math.sqrt((obs_center_x - robot.fused_x) ** 2 + (obs_center_y - robot.fused_y) ** 2)

        angle_diff = (angle_to_obs_rad - robot_angle + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) < cam_fov_rad / 2.0 and dist_to_obs < CONFIG["camera_range_grid"]:
            # Refined apparent size calculation
            # Project obstacle width/height to a plane perpendicular to view direction at object's distance
            apparent_angular_width = 2 * math.atan2(obs.width / 2.0, dist_to_obs) if dist_to_obs > 0 else math.pi
            apparent_angular_height = 2 * math.atan2(obs.height / 2.0, dist_to_obs) if dist_to_obs > 0 else math.pi

            # Convert angular size to normalized bbox size relative to camera FOV
            bbox_width_norm = np.clip(apparent_angular_width / cam_fov_rad, 0.01, 0.9)
            bbox_height_norm = np.clip(apparent_angular_height / cam_fov_rad, 0.01,
                                       0.9)  # Assume square pixels for FOV aspect ratio

            norm_x_in_fov = angle_diff / (cam_fov_rad / 2.0)
            img_center_x = (norm_x_in_fov + 1.0) / 2.0
            img_center_y = 0.5  # Assume vertically centered for simplicity

            x_min_norm = np.clip(img_center_x - bbox_width_norm / 2.0, 0.0, 1.0)
            y_min_norm = np.clip(img_center_y - bbox_height_norm / 2.0, 0.0, 1.0)
            x_max_norm = np.clip(img_center_x + bbox_width_norm / 2.0, 0.0, 1.0)
            y_max_norm = np.clip(img_center_y + bbox_height_norm / 2.0, 0.0, 1.0)

            if x_max_norm > x_min_norm + 1e-3 and y_max_norm > y_min_norm + 1e-3:
                map_pt = Point(x=obs_center_x, y=obs_center_y)
                detected_obj = DetectedObject(
                    id=f"dyn_obs_{i}", label=random.choice(["Box", "Cart", "DynamicObj"]),
                    confidence=random.uniform(0.65, 0.98), bbox=[x_min_norm, y_min_norm, x_max_norm, y_max_norm],
                    map_coords=[map_pt]
                )
                detected_objects_list.append(detected_obj)

    current_camera_data["detected_objects"] = [obj.model_dump() for obj in detected_objects_list]
    robot.battery -= CONFIG["battery_drain_camera"]


def get_current_state(include_map=True):
    lidar_data_serializable = {
        "angles": current_lidar_data.get("angles", []),
        "distances": current_lidar_data.get("distances", []),
        "robot_pose_at_scan": current_lidar_data.get("robot_pose_at_scan").model_dump()
        if isinstance(current_lidar_data.get("robot_pose_at_scan"), Point) else None
    }

    camera_data_serializable = current_camera_data

    state = {
        "robot_status": {
            "x_true": robot.x, "y_true": robot.y, "orientation_true_rad": robot.orientation_rad,
            "x_odom": robot.odometry_x, "y_odom": robot.odometry_y,
            "orientation_odom_rad": robot.odometry_orientation_rad,
            "x_fused": robot.fused_x, "y_fused": robot.fused_y, "orientation_fused_rad": robot.fused_orientation_rad,
            "linear_speed": robot.current_linear_speed, "angular_speed": robot.current_angular_speed,
            "target_linear_speed": robot.target_linear_speed, "target_angular_speed": robot.target_angular_speed,
            "battery": round(robot.battery, 2), "message": robot.status_message,
            "is_navigating": is_navigating_autonomously
        },
        "lidar_data": lidar_data_serializable,
        "camera_data": camera_data_serializable,
        "current_path": robot.current_path,
        "dynamic_obstacles": [obs.model_dump() for obs in environment.dynamic_obstacles_sim]
    }
    if include_map:
        state["map_info"] = {
            "width": environment.width, "height": environment.height,
            "grid": environment.grid.tolist(),
            "terrain_costs": environment.terrain_costs.tolist()
        }
    return state


@app.websocket("/ws/robot_updates")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        initial_state = get_current_state(include_map=True)
        await websocket.send_text(json.dumps(initial_state))
        while True:
            await asyncio.sleep(0.05)  # Shorter sleep to be more responsive to disconnects
            try:
                # Periodically expect a message or check connection state
                # This also helps keep the connection alive if there are proxies/load balancers
                _ = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
            except asyncio.TimeoutError:
                pass  # No message from client, which is fine.
            except (WebSocketDisconnect, websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK) as e:
                print(f"WebSocket client disconnected during receive check: {type(e).__name__}")
                break
            except RuntimeError as e:
                if "WebSocket is closed" in str(e) or "receive_bytes" in str(e).lower() or "send_bytes" in str(
                        e).lower():
                    print(f"WebSocket RuntimeError (likely closed): {e}")
                    break
                raise  # Re-raise if not a known close-related RuntimeError
    except (
            WebSocketDisconnect, websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosedOK) as e:
        print(f"WebSocket client disconnected: {type(e).__name__}")
    except Exception as e:
        print(f"WebSocket error in endpoint: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.disconnect(websocket)


@app.post("/set_goal")
async def set_goal_endpoint(goal: Goal):
    global navigation_goal_pos
    target_x, target_y = goal.target_position.x, goal.target_position.y

    if not (0 <= target_x < CONFIG["map_width"] and 0 <= target_y < CONFIG["map_height"]):
        raise HTTPException(status_code=400, detail="Goal position out of bounds.")
    if environment.is_occupied(target_x, target_y, robot_radius_grid=CONFIG["robot_radius_grid"]):
        raise HTTPException(status_code=400, detail="Goal position is obstructed.")

    navigation_goal_pos = (target_x, target_y)
    robot.current_path = []  # Clear previous path, force replan
    robot.status_message = f"New goal set: ({target_x:.1f}, {target_y:.1f}). Ready to navigate."
    # Do not automatically start navigation; user should press "Start Autonomous"
    return {"message": f"Goal set to ({target_x:.1f}, {target_y:.1f})."}


@app.post("/toggle_navigation")
async def toggle_navigation_endpoint(start: bool):
    global is_navigating_autonomously
    if start:
        if navigation_goal_pos is None and not robot.current_path:  # Path might exist if goal was reached and user wants to restart to same goal
            # If no explicit new goal and no old path, then error
            active_goal = navigation_goal_pos if navigation_goal_pos else (
                robot.current_path[-1] if robot.current_path else None)
            if not active_goal:
                raise HTTPException(status_code=400, detail="No goal set or previous path to resume.")

        is_navigating_autonomously = True
        robot.status_message = "Autonomous navigation started."
        # If path is empty but navigation_goal_pos exists, the loop will plan.
        # If path exists, it will resume.
    else:
        is_navigating_autonomously = False
        robot.target_linear_speed = 0.0  # Command robot to stop
        robot.target_angular_speed = 0.0
        robot.status_message = "Autonomous navigation stopped by user."
    return {"message": f"Autonomous navigation {'started' if start else 'stopped'}."}


@app.post("/manual_control")
async def manual_control_endpoint(command: ManualControlCommand):
    global is_navigating_autonomously, robot
    if is_navigating_autonomously:  # If user takes manual control, stop autonomous mode
        is_navigating_autonomously = False
        robot.status_message = "Manual override: Autonomous navigation stopped."

    robot.current_path = []  # Clear any planned path

    if command.action == "set_target_speeds":
        robot.target_linear_speed = np.clip(command.linear_speed, -CONFIG["robot_max_linear_speed"],
                                            CONFIG["robot_max_linear_speed"])
        robot.target_angular_speed = np.clip(command.angular_speed, -CONFIG["robot_max_angular_speed"],
                                             CONFIG["robot_max_angular_speed"])
        if abs(command.linear_speed) < 1e-3 and abs(command.angular_speed) < 1e-3:
            robot.status_message = "Manual Stop Command."
        else:
            robot.status_message = f"Manual Control: V={robot.target_linear_speed:.1f}, W={robot.target_angular_speed:.1f}"
    else:
        raise HTTPException(status_code=400, detail="Invalid manual control command action.")
    return {"message": f"Manual command '{command.action}' executed."}


@app.post("/modify_map")
async def modify_map_endpoint(obstacle_data: ObstacleData):
    msg = ""
    obstacle_x_int, obstacle_y_int = int(obstacle_data.x), int(obstacle_data.y)
    width, height = int(obstacle_data.width), int(obstacle_data.height)

    if obstacle_data.obstacle_type == "add_static":
        environment.add_obstacle(obstacle_x_int, obstacle_y_int, width, height,
                                 obstacle_type="static", terrain_cost=10000.0)
        msg = f"Static obstacle added at ({obstacle_x_int}, {obstacle_y_int})."
    elif obstacle_data.obstacle_type == "remove_static":
        if 0 <= obstacle_y_int < environment.height and 0 <= obstacle_x_int < environment.width:
            for r_idx in range(obstacle_y_int, min(obstacle_y_int + height, environment.height)):
                for c_idx in range(obstacle_x_int, min(obstacle_x_int + width, environment.width)):
                    if 0 <= r_idx < environment.height and 0 <= c_idx < environment.width:
                        environment.grid[r_idx, c_idx] = 0
                        environment.terrain_costs[r_idx, c_idx] = 1.0
            msg = f"Static obstacle removed / terrain reset at ({obstacle_x_int}, {obstacle_y_int})."
        else:
            msg = "Cannot remove: out of bounds."
    elif obstacle_data.obstacle_type == "set_terrain":
        if 0 <= obstacle_y_int < environment.height and 0 <= obstacle_x_int < environment.width:
            environment.set_terrain_cost(obstacle_x_int, obstacle_y_int, obstacle_data.terrain_cost)
            msg = f"Terrain cost at ({obstacle_x_int}, {obstacle_y_int}) set to {obstacle_data.terrain_cost}."
        else:
            msg = "Cannot set terrain: out of bounds."
    else:
        raise HTTPException(status_code=400, detail="Invalid map modification type.")

    if is_navigating_autonomously and robot.current_path:
        robot.status_message = "Map changed. Replanning highly recommended / may occur."
        # Optionally force replan: robot.current_path = []

    map_state = {
        "map_info": {
            "width": environment.width, "height": environment.height,
            "grid": environment.grid.tolist(),
            "terrain_costs": environment.terrain_costs.tolist()
        }
    }
    await manager.broadcast(json.dumps(map_state))
    return {"message": msg}


@app.get("/get_full_map")
async def get_full_map_data():
    current_state_data = get_current_state(include_map=True)
    return current_state_data.get("map_info", {})


class RobotNavigationApp(ctk.CTk):
    def __init__(self, backend_url="http://localhost:8000", ws_url="ws://localhost:8000/ws/robot_updates"):
        super().__init__()
        self.backend_url = backend_url.rstrip('/')
        self.ws_url = ws_url
        self.ws_message_queue = queue.Queue()
        self.ws_thread = None
        self.ws_client_running = True

        # Store some state from backend for GUI logic
        self.robot_is_navigating_autonomously = False
        self.robot_battery_level = 100.0
        self.has_goal_been_set = False  # Track if user has set a goal via GUI click

        self.title("Advanced Autonomous Mobile Robot Navigation")
        self.geometry("1300x900")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_panel.grid_rowconfigure(0, weight=3)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        self.map_canvas_frame = ctk.CTkFrame(self.left_panel, fg_color="gray20")
        self.map_canvas_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.map_canvas_frame.grid_propagate(False)
        self.map_canvas = tkinter.Canvas(self.map_canvas_frame, bg="gray10", highlightthickness=0)
        self.map_canvas.pack(fill=tkinter.BOTH, expand=True)
        self.map_canvas.bind("<Configure>", self.on_canvas_resize)
        self.map_canvas.bind("<Button-1>", self.on_map_click_primary)
        self.map_canvas.bind("<Button-3>", self.on_map_click_secondary)
        self.map_canvas.bind("<Button-2>", self.on_map_click_secondary)

        self.cell_pixel_width = float(CONFIG["cell_size_gui"])
        self.cell_pixel_height = float(CONFIG["cell_size_gui"])
        self.map_info = {"width": 0, "height": 0, "grid": [], "terrain_costs": []}
        self.dynamic_obstacles_display = []
        self.robot_display_info = {
            "x_fused": 0.0, "y_fused": 0.0, "orientation_fused_rad": 0.0,
            "x_odom": 0.0, "y_odom": 0.0, "orientation_odom_rad": 0.0,
            "x_true": 0.0, "y_true": 0.0, "orientation_true_rad": 0.0,
            "path": []
        }
        self.camera_fov_display_poly = []

        self.sensor_display_panel = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.sensor_display_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.sensor_display_panel.grid_columnconfigure(0, weight=1)
        self.sensor_display_panel.grid_columnconfigure(1, weight=1)
        self.sensor_display_panel.grid_rowconfigure(0, weight=1)

        self.lidar_frame = ctk.CTkFrame(self.sensor_display_panel, fg_color="gray20")
        self.lidar_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        ctk.CTkLabel(self.lidar_frame, text="LIDAR Scan", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(5, 0))
        self.lidar_canvas = tkinter.Canvas(self.lidar_frame, bg="gray15", highlightthickness=0)
        self.lidar_canvas.pack(pady=5, fill=tkinter.BOTH, expand=True)

        self.camera_frame = ctk.CTkFrame(self.sensor_display_panel, fg_color="gray20")
        self.camera_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        ctk.CTkLabel(self.camera_frame, text="Simulated Camera View", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(5, 0))
        self.camera_canvas = tkinter.Canvas(self.camera_frame, bg="gray15", highlightthickness=0)
        self.camera_canvas.pack(pady=5, fill=tkinter.BOTH, expand=True)
        self.camera_detected_objects_display = []

        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.right_panel.grid_rowconfigure(0, weight=2)
        self.right_panel.grid_rowconfigure(1, weight=2)
        self.right_panel.grid_rowconfigure(2, weight=1)
        self.right_panel.grid_rowconfigure(3, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        self.control_frame = ctk.CTkFrame(self.right_panel)
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(self.control_frame, text="Robot Control", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        self.start_nav_button = ctk.CTkButton(self.control_frame, text="Start Autonomous",
                                              command=lambda: self.toggle_navigation_api(True))
        self.start_nav_button.pack(fill="x", padx=10, pady=3)
        self.stop_nav_button = ctk.CTkButton(self.control_frame, text="Stop Autonomous",
                                             command=lambda: self.toggle_navigation_api(False))
        self.stop_nav_button.pack(fill="x", padx=10, pady=3)

        ctk.CTkLabel(self.control_frame, text="Manual Target Speeds:").pack(pady=(10, 0))
        self.manual_linear_speed_slider = ctk.CTkSlider(self.control_frame, from_=-CONFIG["robot_max_linear_speed"],
                                                        to=CONFIG["robot_max_linear_speed"], number_of_steps=40,
                                                        command=self.update_manual_speeds_display)
        self.manual_linear_speed_slider.set(0)
        self.manual_linear_speed_slider.pack(fill="x", padx=10, pady=2)
        self.manual_linear_speed_label = ctk.CTkLabel(self.control_frame, text="Linear: 0.00 m/s")
        self.manual_linear_speed_label.pack()

        self.manual_angular_speed_slider = ctk.CTkSlider(self.control_frame, from_=-CONFIG["robot_max_angular_speed"],
                                                         to=CONFIG["robot_max_angular_speed"], number_of_steps=40,
                                                         command=self.update_manual_speeds_display)
        self.manual_angular_speed_slider.set(0)
        self.manual_angular_speed_slider.pack(fill="x", padx=10, pady=2)
        self.manual_angular_speed_label = ctk.CTkLabel(self.control_frame, text="Angular: 0.00 rad/s")
        self.manual_angular_speed_label.pack()

        self.send_manual_speeds_button = ctk.CTkButton(self.control_frame, text="Send Manual Speeds",
                                                       command=self.send_manual_control_api)
        self.send_manual_speeds_button.pack(fill="x", padx=10, pady=5)
        self.manual_stop_button = ctk.CTkButton(self.control_frame, text="STOP (Manual)", command=self.manual_stop_api,
                                                fg_color="red")
        self.manual_stop_button.pack(fill="x", padx=10, pady=3)

        self.status_frame = ctk.CTkFrame(self.right_panel)
        self.status_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(self.status_frame, text="Robot Status", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.true_pos_label = ctk.CTkLabel(self.status_frame, text="True Pos: (N/A, N/A)")
        self.true_pos_label.pack(anchor="w", padx=10)
        self.odom_pos_label = ctk.CTkLabel(self.status_frame, text="Odom Pos: (N/A, N/A)")
        self.odom_pos_label.pack(anchor="w", padx=10)
        self.fused_pos_label = ctk.CTkLabel(self.status_frame, text="Fused Pos: (N/A, N/A) | Orient: N/A°")
        self.fused_pos_label.pack(anchor="w", padx=10)
        self.speed_label = ctk.CTkLabel(self.status_frame, text="Speeds (Lin/Ang): N/A m/s, N/A rad/s")
        self.speed_label.pack(anchor="w", padx=10)
        self.battery_label = ctk.CTkLabel(self.status_frame, text="Battery: N/A %")
        self.battery_label.pack(anchor="w", padx=10)
        self.message_label = ctk.CTkLabel(self.status_frame, text="Message: Connecting...", wraplength=280)
        self.message_label.pack(anchor="w", padx=10, pady=5)

        self.cv_ml_frame = ctk.CTkFrame(self.right_panel)
        self.cv_ml_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(self.cv_ml_frame, text="Camera Detections", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.cv_ml_objects_label = ctk.CTkLabel(self.cv_ml_frame, text="Detected: None", wraplength=280, justify="left")
        self.cv_ml_objects_label.pack(anchor="w", padx=10, fill="x", expand=True)

        self.map_edit_frame = ctk.CTkFrame(self.right_panel)
        self.map_edit_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(self.map_edit_frame, text="Map Editor", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        self.map_edit_mode_var = tkinter.StringVar(value="set_goal")
        ctk.CTkLabel(self.map_edit_frame, text="Left Click Mode:").pack(anchor="w", padx=10)
        ctk.CTkRadioButton(self.map_edit_frame, text="Set Goal", variable=self.map_edit_mode_var,
                           value="set_goal", command=self.update_gui_button_states).pack(anchor="w", padx=20)
        ctk.CTkRadioButton(self.map_edit_frame, text="Add Obstacle (1x1)", variable=self.map_edit_mode_var,
                           value="add_obstacle", command=self.update_gui_button_states).pack(anchor="w", padx=20)
        ctk.CTkRadioButton(self.map_edit_frame, text="Remove Obstacle/Reset Terrain (1x1)",
                           variable=self.map_edit_mode_var, value="remove_obstacle",
                           command=self.update_gui_button_states).pack(anchor="w", padx=20)

        ctk.CTkLabel(self.map_edit_frame, text="Right Click: Set Terrain Cost (1-10):").pack(anchor="w", padx=10,
                                                                                             pady=(5, 0))

        FLOAT_BTN_SIZE = 40
        FLOAT_FONT_SIZE = 18
        self.floating_controls_frame = ctk.CTkFrame(self.map_canvas_frame, fg_color=("gray70", "gray25"),
                                                    corner_radius=10)
        self.floating_controls_frame.place(relx=0.97, rely=0.97, anchor="se")

        self.float_btn_up = ctk.CTkButton(self.floating_controls_frame, text="▲", width=FLOAT_BTN_SIZE,
                                          height=FLOAT_BTN_SIZE, font=("Arial", FLOAT_FONT_SIZE),
                                          command=lambda: self.set_floating_control_speeds(
                                              CONFIG["float_button_linear_speed"], 0))
        self.float_btn_up.grid(row=0, column=1, padx=2, pady=2)

        self.float_btn_left = ctk.CTkButton(self.floating_controls_frame, text="◄", width=FLOAT_BTN_SIZE,
                                            height=FLOAT_BTN_SIZE, font=("Arial", FLOAT_FONT_SIZE),
                                            command=lambda: self.set_floating_control_speeds(0, CONFIG[
                                                "float_button_angular_speed"]))
        self.float_btn_left.grid(row=1, column=0, padx=2, pady=2)

        self.float_btn_stop = ctk.CTkButton(self.floating_controls_frame, text="●", width=FLOAT_BTN_SIZE,
                                            height=FLOAT_BTN_SIZE, font=("Arial", FLOAT_FONT_SIZE),
                                            fg_color="firebrick", hover_color="darkred",
                                            command=lambda: self.set_floating_control_speeds(0, 0))
        self.float_btn_stop.grid(row=1, column=1, padx=2, pady=2)

        self.float_btn_right = ctk.CTkButton(self.floating_controls_frame, text="►", width=FLOAT_BTN_SIZE,
                                             height=FLOAT_BTN_SIZE, font=("Arial", FLOAT_FONT_SIZE),
                                             command=lambda: self.set_floating_control_speeds(0, -CONFIG[
                                                 "float_button_angular_speed"]))
        self.float_btn_right.grid(row=1, column=2, padx=2, pady=2)

        self.float_btn_down = ctk.CTkButton(self.floating_controls_frame, text="▼", width=FLOAT_BTN_SIZE,
                                            height=FLOAT_BTN_SIZE, font=("Arial", FLOAT_FONT_SIZE),
                                            command=lambda: self.set_floating_control_speeds(
                                                -CONFIG["float_button_linear_speed"], 0))
        self.float_btn_down.grid(row=2, column=1, padx=2, pady=2)

        self.after(100, self.process_ws_queue)
        self.start_ws_client()
        self.after(250, self.request_full_map_from_server)
        self.update_gui_button_states()  # Initial state update for buttons

    def set_floating_control_speeds(self, lin_speed, ang_speed):
        # These buttons imply immediate action, not setting sliders for later "Send"
        self.manual_linear_speed_slider.set(lin_speed)
        self.manual_angular_speed_slider.set(ang_speed)
        self.update_manual_speeds_display()  # Update labels
        self.send_manual_control_api()  # Send to backend immediately

    def update_manual_speeds_display(self, _=None):
        lin_speed = self.manual_linear_speed_slider.get()
        ang_speed = self.manual_angular_speed_slider.get()
        self.manual_linear_speed_label.configure(text=f"Linear: {lin_speed:.2f} m/s")
        self.manual_angular_speed_label.configure(text=f"Angular: {ang_speed:.2f} rad/s")

    def send_manual_control_api(self):
        lin_speed = self.manual_linear_speed_slider.get()
        ang_speed = self.manual_angular_speed_slider.get()
        try:
            payload = {"action": "set_target_speeds", "linear_speed": lin_speed, "angular_speed": ang_speed}
            response = requests.post(f"{self.backend_url}/manual_control", json=payload, timeout=2.0)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            err_detail = "Unknown error"
            if e.response is not None and e.response.content:
                try:
                    err_detail = e.response.json().get('detail', str(e.response.content))
                except json.JSONDecodeError:
                    err_detail = str(e.response.content)
            else:
                err_detail = str(e)
            messagebox.showerror("Error", f"Manual control failed: {err_detail}")

    def manual_stop_api(self):
        self.manual_linear_speed_slider.set(0)
        self.manual_angular_speed_slider.set(0)
        self.update_manual_speeds_display()
        self.send_manual_control_api()

    def on_canvas_resize(self, event):
        self.after(50, self.redraw_map_canvas_full)

    def redraw_map_canvas_full(self):
        self.redraw_map_canvas()
        global current_lidar_data, current_camera_data  # Access globals for initial draw
        self.redraw_lidar_canvas(current_lidar_data)
        self.redraw_camera_canvas(current_camera_data)

    def redraw_map_canvas(self):
        self.map_canvas.delete("all")
        canvas_width = self.map_canvas.winfo_width()
        canvas_height = self.map_canvas.winfo_height()

        if not self.map_info or not self.map_info.get("width", 0) or canvas_width <= 1 or canvas_height <= 1:
            if canvas_width > 1 and canvas_height > 1:
                self.map_canvas.create_text(canvas_width / 2, canvas_height / 2, text="Loading Map...", fill="white",
                                            font=("Arial", 16))
            return

        grid_w = self.map_info["width"]
        grid_h = self.map_info["height"]

        self.cell_pixel_width = float(canvas_width) / grid_w
        self.cell_pixel_height = float(canvas_height) / grid_h

        if self.map_info.get("terrain_costs"):
            tc = self.map_info["terrain_costs"]
            for r in range(grid_h):
                for c in range(grid_w):
                    cost = tc[r][c]
                    color = "gray12"
                    if cost > 1.1 and cost < 100:
                        color = "#202830"  # Medium cost terrain
                    elif cost >= 1000:
                        color = "black"  # Obstacle likely

                    x0, y0 = c * self.cell_pixel_width, r * self.cell_pixel_height
                    x1, y1 = (c + 1) * self.cell_pixel_width, (r + 1) * self.cell_pixel_height
                    self.map_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray25", width=0.5)

        if self.map_info.get("grid"):
            grid_data = self.map_info["grid"]
            for r in range(grid_h):
                for c in range(grid_w):
                    if grid_data[r][c] == 1:
                        x0, y0 = c * self.cell_pixel_width, r * self.cell_pixel_height
                        x1, y1 = (c + 1) * self.cell_pixel_width, (r + 1) * self.cell_pixel_height
                        self.map_canvas.create_rectangle(x0, y0, x1, y1, fill="slate gray", outline="gray50")

        for obs in self.dynamic_obstacles_display:
            x0 = obs['x'] * self.cell_pixel_width
            y0 = obs['y'] * self.cell_pixel_height
            x1 = (obs['x'] + obs['width']) * self.cell_pixel_width
            y1 = (obs['y'] + obs['height']) * self.cell_pixel_height
            self.map_canvas.create_rectangle(x0, y0, x1, y1, fill="purple4", outline="magenta")

        if self.robot_display_info["path"]:
            path_pixels = []
            for i, (p_x, p_y) in enumerate(self.robot_display_info["path"]):
                center_x = p_x * self.cell_pixel_width
                center_y = p_y * self.cell_pixel_height
                path_pixels.extend([center_x, center_y])
                if i == 0:  # Highlight next waypoint
                    self.map_canvas.create_oval(center_x - 5, center_y - 5, center_x + 5, center_y + 5,
                                                fill="cyan", outline="blue", width=2)

            if len(path_pixels) > 3:  # Need at least 2 points (4 coords) for a line
                self.map_canvas.create_line(path_pixels, fill="spring green", width=2.5, arrow=tkinter.LAST,
                                            arrowshape=(10, 12, 5))  # Slightly larger arrow

        # Draw overall goal if path exists (last point of path)
        if self.robot_display_info["path"] and len(self.robot_display_info["path"]) > 0:
            goal_x_map, goal_y_map = self.robot_display_info["path"][-1]
            gx = goal_x_map * self.cell_pixel_width
            gy = goal_y_map * self.cell_pixel_height
            self.map_canvas.create_polygon(gx, gy - 8, gx - 8, gy + 4, gx + 8, gy + 4, fill="gold",
                                           outline="yellow", width=1.5)  # Larger goal marker

        robot_pixel_radius_map = CONFIG["robot_radius_grid"] * min(self.cell_pixel_width, self.cell_pixel_height)

        # True Pose (semi-transparent or smaller)
        rtx, rty = self.robot_display_info["x_true"] * self.cell_pixel_width, self.robot_display_info[
            "y_true"] * self.cell_pixel_height
        self.map_canvas.create_oval(rtx - robot_pixel_radius_map * 0.7, rty - robot_pixel_radius_map * 0.7,
                                    rtx + robot_pixel_radius_map * 0.7, rty + robot_pixel_radius_map * 0.7,
                                    fill="lightcoral", outline="red", stipple="gray25", width=1)

        # Odometry Pose
        rox, roy = self.robot_display_info["x_odom"] * self.cell_pixel_width, self.robot_display_info[
            "y_odom"] * self.cell_pixel_height
        self.map_canvas.create_oval(rox - robot_pixel_radius_map * 0.9, roy - robot_pixel_radius_map * 0.9,
                                    rox + robot_pixel_radius_map * 0.9, roy + robot_pixel_radius_map * 0.9,
                                    fill="orange", outline="darkorange", width=1.5)
        orient_len_odom = robot_pixel_radius_map * 1.1
        self.map_canvas.create_line(rox, roy,
                                    rox + orient_len_odom * math.cos(self.robot_display_info["orientation_odom_rad"]),
                                    roy + orient_len_odom * math.sin(self.robot_display_info["orientation_odom_rad"]),
                                    fill="darkorange", width=2)

        # Fused Pose (main representation)
        rfx, rfy = self.robot_display_info["x_fused"] * self.cell_pixel_width, self.robot_display_info[
            "y_fused"] * self.cell_pixel_height
        self.map_canvas.create_oval(rfx - robot_pixel_radius_map, rfy - robot_pixel_radius_map,
                                    rfx + robot_pixel_radius_map, rfy + robot_pixel_radius_map,
                                    fill="deep sky blue", outline="cyan", width=2)
        orient_len_fused = robot_pixel_radius_map * 1.3
        self.map_canvas.create_line(rfx, rfy,
                                    rfx + orient_len_fused * math.cos(self.robot_display_info["orientation_fused_rad"]),
                                    rfy + orient_len_fused * math.sin(self.robot_display_info["orientation_fused_rad"]),
                                    fill="cyan", width=3, arrow=tkinter.LAST, arrowshape=(10, 12, 5))

        # Camera FOV cone
        cam_fov_rad = math.radians(CONFIG["camera_fov_deg"])
        cam_range_pixels = CONFIG["camera_range_grid"] * min(self.cell_pixel_width, self.cell_pixel_height)
        robot_angle = self.robot_display_info["orientation_fused_rad"]

        p1_fov = (rfx + cam_range_pixels * math.cos(robot_angle - cam_fov_rad / 2),
                  rfy + cam_range_pixels * math.sin(robot_angle - cam_fov_rad / 2))
        p2_fov = (rfx + cam_range_pixels * math.cos(robot_angle + cam_fov_rad / 2),
                  rfy + cam_range_pixels * math.sin(robot_angle + cam_fov_rad / 2))
        self.camera_fov_display_poly = [rfx, rfy, p1_fov[0], p1_fov[1], p2_fov[0], p2_fov[1]]
        self.map_canvas.create_polygon(self.camera_fov_display_poly, fill="lightyellow", outline="khaki",
                                       stipple="gray12", width=1)

    def redraw_lidar_canvas(self, lidar_data_dict):
        self.lidar_canvas.delete("all")
        canvas_width = self.lidar_canvas.winfo_width()
        canvas_height = self.lidar_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return

        center_x, center_y = canvas_width / 2.0, canvas_height * 0.8

        if not lidar_data_dict or not lidar_data_dict.get("angles"):
            self.lidar_canvas.create_text(canvas_width / 2, canvas_height / 2, text="No LIDAR data", fill="gray60")
            return

        self.lidar_canvas.create_oval(center_x - 5, center_y - 5, center_x + 5, center_y + 5, fill="blue")
        max_vis_range = float(CONFIG["lidar_range_grid"])
        # Ensure scale is not zero if max_vis_range is zero, though CONFIG should prevent this.
        scale = min(canvas_width / (2.2 * max_vis_range if max_vis_range > 0 else 1),
                    canvas_height / (1.3 * max_vis_range if max_vis_range > 0 else 1))

        robot_orientation_for_plot = self.robot_display_info["orientation_fused_rad"]

        for angle_rad_world, dist in zip(lidar_data_dict.get("angles", []), lidar_data_dict.get("distances", [])):
            relative_angle = angle_rad_world - robot_orientation_for_plot
            plot_angle = relative_angle - math.pi / 2.0

            end_x = center_x + dist * scale * math.cos(plot_angle)
            end_y = center_y + dist * scale * math.sin(plot_angle)

            color = "PaleGreen1"
            if dist < 1.0:
                color = "tomato"
            elif dist < 3.0:
                color = "orange"
            self.lidar_canvas.create_line(center_x, center_y, end_x, end_y, fill=color, width=1)
            self.lidar_canvas.create_oval(end_x - 1.5, end_y - 1.5, end_x + 1.5, end_y + 1.5, fill=color, outline=color)

        for r_s_factor in [0.25, 0.5, 0.75, 1.0]:
            r_s = CONFIG["lidar_range_grid"] * r_s_factor
            if r_s * scale > 1:  # Only draw if visible
                self.lidar_canvas.create_oval(center_x - r_s * scale, center_y - r_s * scale,
                                              center_x + r_s * scale, center_y + r_s * scale,
                                              outline="gray40", dash=(2, 2))
            if r_s * scale > 10:
                self.lidar_canvas.create_text(center_x, center_y - r_s * scale - 8, text=f"{r_s:.1f}m", fill="gray60",
                                              font=("Arial", 7))

    def redraw_camera_canvas(self, camera_data_dict):
        self.camera_canvas.delete("all")
        canvas_width = self.camera_canvas.winfo_width()
        canvas_height = self.camera_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return

        self.camera_canvas.create_text(canvas_width / 2, 10, text="Normalized View (CV Detections)", fill="gray70",
                                       font=("Arial", 8))

        if not camera_data_dict or not camera_data_dict.get("detected_objects"):
            self.camera_canvas.create_text(canvas_width / 2, canvas_height / 2, text="No Detections", fill="gray60")
            return

        for obj in camera_data_dict.get("detected_objects", []):
            bbox_norm = obj["bbox"]

            x0 = bbox_norm[0] * canvas_width
            y0 = bbox_norm[1] * canvas_height
            x1 = bbox_norm[2] * canvas_width
            y1 = bbox_norm[3] * canvas_height

            self.camera_canvas.create_rectangle(x0, y0, x1, y1, outline="cyan", width=2)
            label_text = f"{obj['label']} ({obj.get('confidence', 0.0):.2f})"
            self.camera_canvas.create_text(x0 + 5, y0 + 10, text=label_text, fill="cyan", anchor="nw",
                                           font=("Arial", 9))

    def on_map_click_actions(self, event, button_type):
        if not self.map_info or not self.map_info.get("width",
                                                      0) or self.cell_pixel_width == 0 or self.cell_pixel_height == 0: return

        grid_x_float = event.x / self.cell_pixel_width
        grid_y_float = event.y / self.cell_pixel_height
        grid_x_int = int(grid_x_float)
        grid_y_int = int(grid_y_float)

        if not (0 <= grid_x_int < self.map_info["width"] and 0 <= grid_y_int < self.map_info["height"]):
            messagebox.showwarning("Map Click", "Clicked outside map boundaries.")
            return

        mode = self.map_edit_mode_var.get() if button_type == "primary" else "set_terrain_cost_prompt"

        if mode == "set_goal":
            try:
                response = requests.post(f"{self.backend_url}/set_goal",
                                         json={"target_position": {"x": grid_x_float, "y": grid_y_float}})
                response.raise_for_status()
                self.has_goal_been_set = True  # Mark that a goal has been set by user
                self.update_gui_button_states()
            except requests.exceptions.RequestException as e:
                err_detail = "Unknown error"
                if e.response is not None and e.response.content:
                    try:
                        err_detail = e.response.json().get('detail', str(e.response.content))
                    except json.JSONDecodeError:
                        err_detail = str(e.response.content)
                else:
                    err_detail = str(e)
                messagebox.showerror("Error", f"Failed to set goal: {err_detail}")

        elif mode == "add_obstacle":
            self.modify_map_api(grid_x_int, grid_y_int, "add_static")
        elif mode == "remove_obstacle":
            self.modify_map_api(grid_x_int, grid_y_int, "remove_static")
        elif mode == "set_terrain_cost_prompt":
            cost_str = simpledialog.askstring("Set Terrain Cost",
                                              f"Enter cost for cell ({grid_x_int},{grid_y_int}) (e.g., 1-10, 1 is normal):",
                                              parent=self)
            if cost_str:
                try:
                    cost = float(cost_str)
                    if cost > 0:
                        self.modify_map_api(grid_x_int, grid_y_int, "set_terrain", terrain_cost_val=cost)
                    else:
                        messagebox.showwarning("Invalid Cost", "Terrain cost must be positive.")
                except ValueError:
                    messagebox.showwarning("Invalid Cost", "Please enter a numeric value for cost.")

    def on_map_click_primary(self, event):
        self.on_map_click_actions(event, "primary")

    def on_map_click_secondary(self, event):
        self.on_map_click_actions(event, "secondary")

    def modify_map_api(self, x, y, mod_type, terrain_cost_val=1.0):
        payload = {"x": x, "y": y, "width": 1, "height": 1, "obstacle_type": mod_type}
        if mod_type == "set_terrain":
            payload["terrain_cost"] = terrain_cost_val
        try:
            response = requests.post(f"{self.backend_url}/modify_map", json=payload, timeout=2.0)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            err_detail = "Unknown error"
            if e.response is not None and e.response.content:
                try:
                    err_detail = e.response.json().get('detail', str(e.response.content))
                except json.JSONDecodeError:
                    err_detail = str(e.response.content)
            else:
                err_detail = str(e)
            messagebox.showerror("Error", f"Map modification failed: {err_detail}")

    def toggle_navigation_api(self, start_flag: bool):
        try:
            url = f"{self.backend_url}/toggle_navigation"
            params = {"start": start_flag}
            response = requests.post(url, params=params, timeout=2.0)
            response.raise_for_status()
            # Backend will set is_navigating_autonomously, GUI will update via WebSocket
            # If stopping, explicitly mark that no goal is "active" from GUI perspective until a new one is set.
            if not start_flag:
                self.has_goal_been_set = False
            self.update_gui_button_states()  # Proactively update GUI button state
        except requests.exceptions.RequestException as e:
            err_detail = "Unknown error"
            if e.response is not None:
                if e.response.content:
                    try:
                        err_detail = e.response.json().get('detail', f"Status {e.response.status_code}")
                    except json.JSONDecodeError:
                        err_detail = f"Status {e.response.status_code}: {e.response.text[:100]}"
                else:
                    err_detail = f"Status {e.response.status_code} (No content)"
            else:
                err_detail = str(e)
            messagebox.showerror("Error", f"Failed to toggle navigation: {err_detail}")

    def request_full_map_from_server(self):
        try:
            response = requests.get(f"{self.backend_url}/get_full_map", timeout=3.0)
            response.raise_for_status()
            map_data = response.json()
            self.map_info = map_data
            self.after(10, self.redraw_map_canvas_full)
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch initial map: {e}")
            self.message_label.configure(text="Error: Failed to load map from server.")

    def update_gui_button_states(self):
        """Dynamically enable/disable navigation control buttons."""
        # Start navigation button: enabled if not currently navigating, a goal has been set, and battery is sufficient
        can_start = not self.robot_is_navigating_autonomously and self.has_goal_been_set and self.robot_battery_level > 5.0
        self.start_nav_button.configure(state=tkinter.NORMAL if can_start else tkinter.DISABLED)

        # Stop navigation button: enabled if currently navigating
        self.stop_nav_button.configure(
            state=tkinter.NORMAL if self.robot_is_navigating_autonomously else tkinter.DISABLED)

    def update_gui_from_state(self, state_data):
        full_redraw_map = False
        if "robot_status" in state_data:
            rs = state_data["robot_status"]
            self.robot_is_navigating_autonomously = rs.get('is_navigating', False)
            self.robot_battery_level = rs.get('battery', 0.0)

            self.robot_display_info.update({
                "x_fused": rs.get("x_fused", 0.0), "y_fused": rs.get("y_fused", 0.0),
                "orientation_fused_rad": rs.get("orientation_fused_rad", 0.0),
                "x_odom": rs.get("x_odom", 0.0), "y_odom": rs.get("y_odom", 0.0),
                "orientation_odom_rad": rs.get("orientation_odom_rad", 0.0),
                "x_true": rs.get("x_true", 0.0), "y_true": rs.get("y_true", 0.0),
                "orientation_true_rad": rs.get("orientation_true_rad", 0.0),
            })

            self.true_pos_label.configure(text=f"True Pos: ({rs.get('x_true', 0.0):.2f}, {rs.get('y_true', 0.0):.2f})")
            self.odom_pos_label.configure(text=f"Odom Pos: ({rs.get('x_odom', 0.0):.2f}, {rs.get('y_odom', 0.0):.2f})")
            self.fused_pos_label.configure(
                text=f"Fused Pos: ({rs.get('x_fused', 0.0):.2f}, {rs.get('y_fused', 0.0):.2f}) | Orient: {math.degrees(rs.get('orientation_fused_rad', 0.0)):.1f}°")
            self.speed_label.configure(
                text=f"Speeds (L/A): {rs.get('linear_speed', 0.0):.2f} m/s, {rs.get('angular_speed', 0.0):.2f} rad/s")
            self.battery_label.configure(text=f"Battery: {rs.get('battery', 0.0):.2f} %")
            self.message_label.configure(text=f"Message: {rs.get('message', 'N/A')}")
            full_redraw_map = True

            # If navigation stopped (e.g. goal reached, collision, manual stop), reset has_goal_been_set
            if not self.robot_is_navigating_autonomously and rs.get('message', '').lower() in ["goal reached!",
                                                                                               "collision! stopped.",
                                                                                               "manual override: autonomous navigation stopped."]:
                self.has_goal_been_set = False

            self.update_gui_button_states()

        if "current_path" in state_data:
            self.robot_display_info["path"] = state_data["current_path"]
            # If path becomes empty while navigating, it means goal was reached or replan failed.
            if not self.robot_display_info["path"] and self.robot_is_navigating_autonomously:
                self.has_goal_been_set = False  # Ready for a new goal
                self.update_gui_button_states()
            full_redraw_map = True

        if "map_info" in state_data:
            self.map_info = state_data["map_info"]
            full_redraw_map = True

        if "dynamic_obstacles" in state_data:
            self.dynamic_obstacles_display = state_data["dynamic_obstacles"]
            full_redraw_map = True

        if full_redraw_map:
            self.redraw_map_canvas()

        if "lidar_data" in state_data:
            self.redraw_lidar_canvas(state_data["lidar_data"])

        if "camera_data" in state_data:
            self.camera_detected_objects_display = state_data["camera_data"].get("detected_objects", [])
            self.redraw_camera_canvas(state_data["camera_data"])

            if self.camera_detected_objects_display:
                obj_texts = [f"- {obj['label']} ({obj.get('confidence', 0.0):.2f})" for obj in
                             self.camera_detected_objects_display]
                self.cv_ml_objects_label.configure(text="Detected:\n" + "\n".join(obj_texts))
            else:
                self.cv_ml_objects_label.configure(text="Detected: None")

    def _ws_client_thread_func(self):
        async def connect_and_listen():
            print("GUI: WebSocket client thread started.")
            while self.ws_client_running:
                try:
                    async with websockets.connect(self.ws_url, ping_interval=10, ping_timeout=10,
                                                  close_timeout=5) as websocket:
                        print(f"GUI: Connected to WebSocket at {self.ws_url}")
                        self.ws_message_queue.put({"type": "connection_status", "status": "connected"})
                        while self.ws_client_running:
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                                self.ws_message_queue.put({"type": "data", "payload": message})
                            except asyncio.TimeoutError:
                                continue
                            except websockets.exceptions.ConnectionClosed:
                                if self.ws_client_running:
                                    print("GUI: WebSocket connection closed by server during recv.")
                                    self.ws_message_queue.put(
                                        {"type": "connection_status", "status": "disconnected_by_server"})
                                break
                except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError,
                        websockets.exceptions.InvalidHandshake, websockets.exceptions.PayloadTooBig) as e:
                    if self.ws_client_running:
                        status_msg = "refused" if isinstance(e, ConnectionRefusedError) else "disconnected"
                        print(f"GUI: WebSocket {status_msg} ({type(e).__name__}: {e}). Retrying in 3s...")
                        self.ws_message_queue.put({"type": "connection_status", "status": status_msg})
                except Exception as e:
                    if self.ws_client_running:
                        print(f"GUI: WebSocket unexpected error: {e} ({type(e).__name__}). Retrying in 3s...")
                        self.ws_message_queue.put({"type": "connection_status", "status": "error"})

                if self.ws_client_running:
                    await asyncio.sleep(3)
            print("GUI: WebSocket client thread gracefully stopped.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(connect_and_listen())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def start_ws_client(self):
        if self.ws_thread is not None and self.ws_thread.is_alive():
            return
        self.ws_client_running = True
        self.ws_thread = threading.Thread(target=self._ws_client_thread_func, daemon=True)
        self.ws_thread.start()

    def stop_ws_client(self):
        self.ws_client_running = False
        print("GUI: WebSocket client stop requested. Thread will exit on next check.")

    def process_ws_queue(self):
        try:
            while True:
                msg_item = self.ws_message_queue.get_nowait()
                if msg_item["type"] == "connection_status":
                    status = msg_item["status"]
                    if status == "connected":
                        self.message_label.configure(text="Message: Connected to server.")
                    else:
                        self.message_label.configure(text=f"Message: WS {status}. Retrying...")
                    self.update_gui_button_states()  # Update buttons on connection change
                elif msg_item["type"] == "data":
                    try:
                        data_payload = json.loads(msg_item["payload"])
                        self.update_gui_from_state(data_payload)
                    except json.JSONDecodeError:
                        print(f"GUI: Invalid JSON: {msg_item['payload'][:100]}...")
                    except Exception as e:
                        print(f"GUI: Error processing WS message: {e}");
                        import traceback;
                        traceback.print_exc()
        except queue.Empty:
            pass
        finally:
            self.after(50, self.process_ws_queue)

    def on_closing(self):
        print("Closing application...")
        self.stop_ws_client()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=0.5)
        self.destroy()


def run_fastapi_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    fastapi_thread.start()
    print("FastAPI server starting in a background thread...")
    time.sleep(2.5)  # Slightly longer to ensure server is fully up

    gui_app = RobotNavigationApp()
    gui_app.protocol("WM_DELETE_WINDOW", gui_app.on_closing)
    gui_app.mainloop()

    print("GUI application closed.")
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.logging import LoggingSeverity

from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Point, Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import String, ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

import numpy as np
import math
from scipy.spatial.transform import Rotation
import time
import threading
from abc import ABC, abstractmethod
import traceback
import heapq
import logging
import copy

# --- Конфигурация движения ---
class MotionConfig:
    """Конфигурация базовых параметров движения."""
    def __init__(self):
        self.robot_radius: float = 0.5
        self.max_xy_speed: float = 15.0
        self.max_z_speed: float = 5.0
        self.max_yaw_rate: float = 1.8
        self.xy_control_gain: float = 0.9
        self.z_control_gain: float = 0.9
        self.yaw_control_gain: float = 1.5
        self.target_reach_tolerance_xy: float = 0.5
        self.target_reach_tolerance_z: float = 0.3
        self.dt: float = 0.05

# --- Конфигурация планировщика A* ---
class AStarPlannerConfig:
    def __init__(self):
        self.grid_resolution: float = 7.0
        self.robot_radius_astar: float = 0.5
        self.grid_padding_cells: int = 5
        self.allow_diagonal_movement: bool = True
        self.heuristic_weight: float = 1.0

# --- Узел A* ---
class NodeAstar:
    def __init__(self, position: tuple[int, int, int], parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    def __lt__(self, other):
        return self.f < other.f
    def __hash__(self):
        return hash(self.position)

# --- Планировщик A* (3D) ---
class AStarPlanner:
    def __init__(self, config: AStarPlannerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def _heuristic(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

    def _get_neighbors(self, current_node_pos: tuple[int, int, int], grid_shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        neighbors = []
        moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if not self.config.allow_diagonal_movement and (abs(dx) + abs(dy) + abs(dz) > 1):
                        continue
                    moves.append((dx, dy, dz))

        for move in moves:
            neighbor_pos = (current_node_pos[0] + move[0],
                            current_node_pos[1] + move[1],
                            current_node_pos[2] + move[2])
            if (0 <= neighbor_pos[0] < grid_shape[0] and
                0 <= neighbor_pos[1] < grid_shape[1] and
                0 <= neighbor_pos[2] < grid_shape[2]):
                neighbors.append(neighbor_pos)
        return neighbors

    def _reconstruct_path(self, current_node: NodeAstar, grid_origin_world: np.ndarray, resolution: float) -> list[np.ndarray]:
        path_grid = []
        current = current_node
        while current is not None:
            path_grid.append(current.position)
            current = current.parent
        path_grid = path_grid[::-1]

        path_world = []
        for gx, gy, gz in path_grid:
            wx = grid_origin_world[0] + (gx + 0.5) * resolution
            wy = grid_origin_world[1] + (gy + 0.5) * resolution
            wz = grid_origin_world[2] + (gz + 0.5) * resolution
            path_world.append(np.array([wx, wy, wz]))
        return path_world

    def plan(self, start_xyz_world: np.ndarray, goal_xyz_world: np.ndarray,
             obstacles_world_3d: list[dict]) -> list[np.ndarray] | None:
        resolution = self.config.grid_resolution
        robot_r_astar = self.config.robot_radius_astar
        padding_dist = self.config.grid_padding_cells * resolution

        all_points_x = [start_xyz_world[0], goal_xyz_world[0]]
        all_points_y = [start_xyz_world[1], goal_xyz_world[1]]
        all_points_z = [start_xyz_world[2], goal_xyz_world[2]]

        for obs in obstacles_world_3d:
            all_points_x.extend([obs['pos_center'][0] - obs['radius_xy'], obs['pos_center'][0] + obs['radius_xy']])
            all_points_y.extend([obs['pos_center'][1] - obs['radius_xy'], obs['pos_center'][1] + obs['radius_xy']])
            all_points_z.extend([obs['pos_center'][2] - obs['half_height_z'], obs['pos_center'][2] + obs['half_height_z']])

        min_x_w = min(all_points_x) - padding_dist
        max_x_w = max(all_points_x) + padding_dist
        min_y_w = min(all_points_y) - padding_dist
        max_y_w = max(all_points_y) + padding_dist
        min_z_w = min(all_points_z) - padding_dist
        max_z_w = max(all_points_z) + padding_dist

        grid_origin_world = np.array([min_x_w, min_y_w, min_z_w])
        grid_size_x = int(math.ceil((max_x_w - min_x_w) / resolution))
        grid_size_y = int(math.ceil((max_y_w - min_y_w) / resolution))
        grid_size_z = int(math.ceil((max_z_w - min_z_w) / resolution))

        if grid_size_x <= 0 or grid_size_y <= 0 or grid_size_z <= 0:
            self.logger.error(f"A* Planner: Invalid grid dimensions ({grid_size_x}, {grid_size_y}, {grid_size_z}). Cannot plan.")
            return None

        self.logger.info(f"A* Planner: Grid Size (cells): X={grid_size_x}, Y={grid_size_y}, Z={grid_size_z}. Grid Origin (world): {vector_to_str(grid_origin_world)}")

        grid = np.zeros((grid_size_x, grid_size_y, grid_size_z), dtype=np.uint8) # 0: пуст, 1: препятсвие

        def world_to_grid_xyz(wx, wy, wz):
            gx = int(math.floor((wx - grid_origin_world[0]) / resolution))
            gy = int(math.floor((wy - grid_origin_world[1]) / resolution))
            gz = int(math.floor((wz - grid_origin_world[2]) / resolution))
            return max(0, min(gx, grid_size_x - 1)), \
                   max(0, min(gy, grid_size_y - 1)), \
                   max(0, min(gz, grid_size_z - 1))

        # Voxelize obstacles
        for obs in obstacles_world_3d:
            obs_center_w = obs['pos_center']
            obs_r_xy = obs['radius_xy']
            obs_h_half_z = obs['half_height_z']

            inflated_r_xy = obs_r_xy + robot_r_astar
            inflated_obs_z_min_w = obs_center_w[2] - obs_h_half_z - robot_r_astar 
            inflated_obs_z_max_w = obs_center_w[2] + obs_h_half_z + robot_r_astar

            min_gx_obs, min_gy_obs, min_gz_obs_approx = world_to_grid_xyz(
                obs_center_w[0] - inflated_r_xy,
                obs_center_w[1] - inflated_r_xy,
                inflated_obs_z_min_w
            )
            max_gx_obs, max_gy_obs, max_gz_obs_approx = world_to_grid_xyz(
                obs_center_w[0] + inflated_r_xy,
                obs_center_w[1] + inflated_r_xy,
                inflated_obs_z_max_w
            )

            for igx in range(min_gx_obs, max_gx_obs + 1):
                for igy in range(min_gy_obs, max_gy_obs + 1):
                    cell_center_xy_w_x = grid_origin_world[0] + (igx + 0.5) * resolution
                    cell_center_xy_w_y = grid_origin_world[1] + (igy + 0.5) * resolution

                    dist_sq_xy_to_obs_axis = (cell_center_xy_w_x - obs_center_w[0])**2 + \
                                             (cell_center_xy_w_y - obs_center_w[1])**2

                    if dist_sq_xy_to_obs_axis <= inflated_r_xy**2:
                        for igz in range(min_gz_obs_approx, max_gz_obs_approx + 1):
                            cell_center_z_w = grid_origin_world[2] + (igz + 0.5) * resolution
                            if inflated_obs_z_min_w <= cell_center_z_w <= inflated_obs_z_max_w:
                                if 0 <= igx < grid_size_x and 0 <= igy < grid_size_y and 0 <= igz < grid_size_z:
                                    grid[igx, igy, igz] = 1

        start_gx, start_gy, start_gz = world_to_grid_xyz(*start_xyz_world)
        goal_gx, goal_gy, goal_gz = world_to_grid_xyz(*goal_xyz_world)

        start_node = NodeAstar((start_gx, start_gy, start_gz))
        goal_node_pos = (goal_gx, goal_gy, goal_gz)

        if grid[start_node.position] == 1:
            self.logger.warn(f"A* Planner: Start position {vector_to_str(start_xyz_world)} (grid {start_node.position}) is inside an obstacle. Trying to find a nearby free cell...")
            found_new_start = False
            for r_search in range(1, 6):
                for dx_s in range(-r_search, r_search + 1):
                    for dy_s in range(-r_search, r_search + 1):
                        for dz_s in range(-r_search, r_search+1):
                            if dx_s == 0 and dy_s == 0 and dz_s == 0: continue
                            alt_pos = (start_node.position[0] + dx_s, start_node.position[1] + dy_s, start_node.position[2] + dz_s)
                            if 0 <= alt_pos[0] < grid_size_x and \
                               0 <= alt_pos[1] < grid_size_y and \
                               0 <= alt_pos[2] < grid_size_z and \
                               grid[alt_pos] == 0:
                                start_node = NodeAstar(alt_pos)
                                self.logger.info(f"A* Planner: Found alternative start cell: {start_node.position}.")
                                found_new_start = True
                                break
                        if found_new_start: break
                    if found_new_start: break
                if found_new_start: break
            if not found_new_start:
                self.logger.error("A* Planner: Start is in obstacle, and no free nearby cell found. Planning failed.")
                return None

        if grid[goal_node_pos] == 1:
            self.logger.warn(f"A* Planner: Goal position {vector_to_str(goal_xyz_world)} (grid {goal_node_pos}) is inside an obstacle. Path may lead to edge of obstacle.")

        open_list = [] 
        heapq.heappush(open_list, start_node)
        closed_set = set() 
        g_costs = {start_node.position: 0}

        while open_list:
            current_node = heapq.heappop(open_list)

            if current_node.position == goal_node_pos:
                self.logger.info(f"A* Planner: Path found to grid cell {goal_node_pos}.")
                return self._reconstruct_path(current_node, grid_origin_world, resolution)

            if current_node.position in closed_set:
                continue
            closed_set.add(current_node.position)

            for neighbor_pos_grid in self._get_neighbors(current_node.position, (grid_size_x, grid_size_y, grid_size_z)):
                if grid[neighbor_pos_grid[0], neighbor_pos_grid[1], neighbor_pos_grid[2]] == 1:
                    continue

                move_cost = self._heuristic(current_node.position, neighbor_pos_grid)
                new_g_cost = g_costs[current_node.position] + move_cost

                if neighbor_pos_grid not in g_costs or new_g_cost < g_costs[neighbor_pos_grid]:
                    g_costs[neighbor_pos_grid] = new_g_cost
                    h_cost = self._heuristic(neighbor_pos_grid, goal_node_pos) * self.config.heuristic_weight
                    f_cost = new_g_cost + h_cost

                    neighbor_node = NodeAstar(neighbor_pos_grid, current_node)
                    neighbor_node.g = new_g_cost
                    neighbor_node.h = h_cost
                    neighbor_node.f = f_cost
                    heapq.heappush(open_list, neighbor_node)

        self.logger.warn("A* Planner: Path not found.")
        return None

# --- Абстрактные классы Контроллера и Шага, Утилиты ---
class Controller(ABC):
    @property
    @abstractmethod
    def position(self) -> np.ndarray | None: pass
    @property
    @abstractmethod
    def orientation(self) -> np.ndarray | None: pass
    @property
    @abstractmethod
    def velocity_publisher(self) -> rclpy.publisher.Publisher: pass
    @property
    @abstractmethod
    def viz_publisher(self) -> rclpy.publisher.Publisher | None: pass
    @property
    @abstractmethod
    def logger(self) -> logging.Logger: pass
    @property
    @abstractmethod
    def motion_config(self) -> MotionConfig: pass
    @property
    @abstractmethod
    def origin_pos_abs(self) -> np.ndarray: pass
    @property
    @abstractmethod
    def astar_planner(self) -> AStarPlanner: pass
    @property
    @abstractmethod
    def new_obstacles_for_replan(self) -> bool: pass
    @new_obstacles_for_replan.setter
    @abstractmethod
    def new_obstacles_for_replan(self, value: bool) -> None: pass


class Step(ABC):
    def init(self, controller: Controller) -> None: pass
    @abstractmethod
    def update(self, controller: Controller) -> bool: pass
    @property
    def uses_velocity_control(self) -> bool: return False
    def __str__(self) -> str: return self.__class__.__name__

def vector_to_str(vector: np.ndarray | None, precision: int = 2) -> str:
    if vector is None: return "None"
    fmt = f"%.{precision}f"
    return np.array2string(vector, formatter={'float_kind': lambda x: fmt % x})[1:-1]

def normalize_angle(angle: float) -> float:
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle


# --- Контроллер SimpleController ---
class SimpleController(Controller):
    def __init__(self, node: Node, steps: list[Step], origin_pos_abs: np.ndarray, control_frequency: float = 20.0):
        self.node = node
        self.steps = steps
        self.step: Step | None = None
        self._pose: Pose | None = None
        self._velocity_body_twist: Twist | None = None
        self._detected_obstacles_global: list = []
        self._last_known_obstacles_for_replan_check: list = []
        self.current_state = State()
        
        self._motion_config = MotionConfig()
        self._motion_config.dt = 1.0 / control_frequency

        self.arming_req_sent = False
        self.offboard_req_sent = False
        self.last_state_check_time = node.get_clock().now()
        self._state_lock = threading.Lock()
        self.state_callback_group = ReentrantCallbackGroup()
        self.services_ready = False
        self._origin_pos_abs = origin_pos_abs

        self.astar_planner_config = AStarPlannerConfig()
        self.astar_planner_config.robot_radius_astar = self._motion_config.robot_radius + 0.3
        self._astar_planner = AStarPlanner(self.astar_planner_config, self.node.get_logger())
        
        self._path_marker_id_counter = 0
        self._new_obstacles_for_replan: bool = False

        self.node.get_logger().info("SimpleController (A* only) инициализирован.")
        
        mavros_ns = '/mavros'
        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self._velocity_publisher = node.create_publisher(TwistStamped, f'{mavros_ns}/setpoint_velocity/cmd_vel', qos_cmd)
        
        qos_viz = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._viz_publisher = node.create_publisher(MarkerArray, '~/astar_path_viz', qos_viz)

        qos_odom = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.odom_subscription = node.create_subscription(Odometry, f'{mavros_ns}/local_position/odom', self.odom_callback, qos_odom)
        qos_state_sub = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_subscription = node.create_subscription(State, f'{mavros_ns}/state', self.state_callback, qos_state_sub, callback_group=self.state_callback_group)
        qos_obstacles = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.obstacle_subscription = node.create_subscription(String, '/obstacles', self.obstacles_callback, qos_obstacles)
        
        self.set_mode_client = self.create_client_and_wait(SetMode, f'{mavros_ns}/set_mode')
        self.arming_client = self.create_client_and_wait(CommandBool, f'{mavros_ns}/cmd/arming')

        timer_period = 1.0 / control_frequency
        self.timer = node.create_timer(timer_period, self.timer_callback)
        self.node.get_logger().info(f"Частота управления: {control_frequency} Гц (dt={self._motion_config.dt:.3f}с).")

    def create_client_and_wait(self, srv_type, srv_name):
        client = self.node.create_client(srv_type, srv_name)
        self.node.get_logger().info(f'Ожидание сервиса {srv_name}...')
        while not client.wait_for_service(timeout_sec=1.0) and rclpy.ok():
            self.node.get_logger().info(f'Сервис {srv_name} недоступен, ожидание...')
        if not rclpy.ok(): raise SystemExit("Запрос на выключение во время ожидания сервиса")
        self.node.get_logger().info(f'Сервис {srv_name} доступен.')
        return client

    def odom_callback(self, msg: Odometry):
        self._pose = msg.pose.pose
        self._velocity_body_twist = msg.twist.twist

    def obstacles_callback(self, msg: String):
        old_obstacle_count = len(self._detected_obstacles_global)
        old_obstacle_summary = str(self._detected_obstacles_global)

        new_obstacles_parsed = []
        parts = msg.data.split()
        
        if msg.data.strip() == "":
            if self._detected_obstacles_global:
                self.node.get_logger().info("Obstacles cleared by empty message.")
                self._detected_obstacles_global = []
                self.new_obstacles_for_replan = True
            return

        if len(parts) > 0 and len(parts) % 7 == 0:
            num_obs_in_msg = len(parts) // 7
            for i in range(num_obs_in_msg):
                obs_chunk = parts[i*7 : (i+1)*7]
                try:
                    obs_name = obs_chunk[0]
                    x_abs, y_abs, z_abs = float(obs_chunk[1]), float(obs_chunk[2]), float(obs_chunk[3])
                    sx, sy, sz_full = float(obs_chunk[4]), float(obs_chunk[5]), float(obs_chunk[6])
                    obs_radius_xy = max(sx, sy) / 2.0
                    new_obstacles_parsed.append({
                        'name': obs_name,
                        'pos_center_abs': np.array([x_abs, y_abs, z_abs]),
                        'radius_xy': obs_radius_xy,
                        'sz_full': sz_full
                    })
                except ValueError as e:
                    self.node.get_logger().error(f"Error parsing obstacle data chunk '{' '.join(obs_chunk)}': {e}")
                    if self._detected_obstacles_global:
                        self.node.get_logger().warn("Clearing existing obstacles due to parsing error.")
                        self._detected_obstacles_global = []
                        self.new_obstacles_for_replan = True
                    return 
            
            current_obstacle_summary = str(new_obstacles_parsed)
            if current_obstacle_summary != old_obstacle_summary:
                self._detected_obstacles_global = new_obstacles_parsed
                self.new_obstacles_for_replan = True
                YELLOW = '\033[93m'
                RESET = '\033[0m'
                num_obs = len(self._detected_obstacles_global)
                obs_summary_list = [f"'{o.get('name', 'N/A')}' @ GLOB [{vector_to_str(o['pos_center_abs'])}]" for o in self._detected_obstacles_global[:3]]
                if num_obs > 3: obs_summary_list.append("...")
                self.node.get_logger().warn(f"{YELLOW}OBSTACLES UPDATED! Count: {num_obs}. Details: {'; '.join(obs_summary_list)}{RESET}. Replanning will be triggered if in A* Nav.")
            # else:
                # self.node.get_logger().debug("Obstacle data received, but no change from current list.")

        elif msg.data.strip():
            self.node.get_logger().warn(f"Invalid obstacle message format: '{msg.data}'. Expected multiples of 7 items. Clearing obstacles.")
            if self._detected_obstacles_global:
                self._detected_obstacles_global = []
                self.new_obstacles_for_replan = True

    def state_callback(self, msg: State):
        with self._state_lock:
            prev_mode = self.current_state.mode
            prev_armed = self.current_state.armed
            self.current_state = msg
            if prev_mode != msg.mode: self.node.get_logger().info(f"MAVROS Mode changed: {prev_mode} -> {msg.mode}")
            if prev_armed != msg.armed: self.node.get_logger().info(f"MAVROS Armed state changed: {prev_armed} -> {msg.armed}")

            if not self.services_ready:
                if self.set_mode_client.service_is_ready() and self.arming_client.service_is_ready(): self.services_ready = True
                else: return

            now = self.node.get_clock().now()
            if (now - self.last_state_check_time) < Duration(seconds=1.0): return
            self.last_state_check_time = now

            if not msg.connected:
                self.node.get_logger().warn("MAVROS disconnected from FCU!", throttle_duration_sec=5)
                self.arming_req_sent = False
                self.offboard_req_sent = False
                return

            target_mode = 'OFFBOARD'
            if self.step is not None and self.step.uses_velocity_control and (msg.mode != target_mode or not msg.armed):
                 self.node.get_logger().warn(f"FCU state changed during active step ({self.step})! Mode: {msg.mode}, Armed: {msg.armed}. Controller will not intervene.", throttle_duration_sec=5)
                 self.arming_req_sent = False
                 self.offboard_req_sent = False
                 return

            if msg.mode != target_mode:
                if not self.offboard_req_sent:
                    self.node.get_logger().info(f"Requesting {target_mode} mode...")
                    self.set_mode_client.call_async(SetMode.Request(custom_mode=target_mode))
                    self.offboard_req_sent = True
            elif not msg.armed:
                if not self.arming_req_sent:
                    self.node.get_logger().info("OFFBOARD mode set. Requesting ARM...")
                    self.arming_client.call_async(CommandBool.Request(value=True))
                    self.arming_req_sent = True
                self.offboard_req_sent = False 
            else: # Armed and OFFBOARD
                if self.arming_req_sent or self.offboard_req_sent:
                     self.node.get_logger().info("State: ARMED and OFFBOARD. Ready for velocity commands.")
                self.arming_req_sent = False
                self.offboard_req_sent = False

    def timer_callback(self):
        if self.position is None or self.orientation is None:
            self.node.get_logger().warn("Waiting for odometry data...", throttle_duration_sec=5.0)
            if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                ts = TwistStamped()
                ts.header.stamp = self.node.get_clock().now().to_msg()
                ts.header.frame_id = 'odom'
                self._velocity_publisher.publish(ts)
            return

        with self._state_lock:
            is_ready_to_fly = self.current_state.connected and self.current_state.armed and self.current_state.mode == 'OFFBOARD'
        
        current_step_requires_fcu_ready = self.step.uses_velocity_control if self.step else False

        if current_step_requires_fcu_ready and not is_ready_to_fly:
            self.node.get_logger().warn(f"Not ready to fly for current step ({self.step}). State: Connected={self.current_state.connected}, Armed={self.current_state.armed}, Mode='{self.current_state.mode}'. Holding...", throttle_duration_sec=5.0)
            if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                 ts = TwistStamped()
                 ts.header.stamp = self.node.get_clock().now().to_msg()
                 ts.header.frame_id = 'odom'
                 self._velocity_publisher.publish(ts)
            return
        elif not self.step and not is_ready_to_fly and self.steps:
             self.node.get_logger().warn(f"Waiting to start next step, FCU not ready. State: Connected={self.current_state.connected}, Armed={self.current_state.armed}, Mode='{self.current_state.mode}'. Holding...", throttle_duration_sec=5.0)
             if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                 ts = TwistStamped()
                 ts.header.stamp = self.node.get_clock().now().to_msg()
                 ts.header.frame_id = 'odom'
                 self._velocity_publisher.publish(ts)
             return

        if self.step is None:
            if not self.steps:
                self.node.get_logger().info("Mission completed. Holding position.")
                if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                    ts = TwistStamped()
                    ts.header.stamp = self.node.get_clock().now().to_msg()
                    ts.header.frame_id = 'odom'
                    self._velocity_publisher.publish(ts)
                return
            self.step = self.steps.pop(0)
            self.node.get_logger().info(f"--- Starting Step: {self.step} ---")
            try:
                 self.step.init(self)
            except Exception as e:
                 self.logger.error(f"Error initializing step '{self.step}': {e}", exc_info=True)
                 self.step = None
                 return

        step_completed = False
        if self.step:
            try:
                if not self.step.uses_velocity_control:
                    step_completed = self.step.update(self)
                elif is_ready_to_fly :
                    step_completed = self.step.update(self)
                else:
                    self.node.get_logger().warn(f"Step {self.step} requires velocity control, but FCU not ready. Skipping update.", throttle_duration_sec=2.0)
                    ts = TwistStamped()
                    ts.header.stamp = self.node.get_clock().now().to_msg()
                    ts.header.frame_id = 'odom'
                    self.velocity_publisher.publish(ts)
            except Exception as e:
                self.logger.error(f"Error executing step '{self.step}': {e}", exc_info=True)
                self.step = None
                if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                    ts = TwistStamped()
                    ts.header.stamp = self.node.get_clock().now().to_msg()
                    ts.header.frame_id = 'odom'
                    self._velocity_publisher.publish(ts)
                return

        if step_completed:
            self.node.get_logger().info(f"--- Completed Step: {self.step} ---")
            if self.step and self.step.uses_velocity_control:
                if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                    ts = TwistStamped()
                    ts.header.stamp = self.node.get_clock().now().to_msg()
                    ts.header.frame_id = 'odom'
                    self._velocity_publisher.publish(ts)
            self.step = None
        
    @property
    def position(self) -> np.ndarray | None:
        if self._pose: return np.array([self._pose.position.x, self._pose.position.y, self._pose.position.z])
        return None
    @property
    def orientation(self) -> np.ndarray | None:
        if self._pose: return np.array([self._pose.orientation.x, self._pose.orientation.y, self._pose.orientation.z, self._pose.orientation.w])
        return None
    @property
    def detected_obstacles_global(self) -> list: return self._detected_obstacles_global
    @property
    def velocity_publisher(self) -> rclpy.publisher.Publisher: return self._velocity_publisher
    @property
    def viz_publisher(self) -> rclpy.publisher.Publisher | None: return self._viz_publisher
    @property
    def logger(self) -> logging.Logger: return self.node.get_logger()
    @property
    def motion_config(self) -> MotionConfig: return self._motion_config
    @property
    def origin_pos_abs(self) -> np.ndarray: return self._origin_pos_abs
    @property
    def astar_planner(self) -> AStarPlanner: return self._astar_planner


    @property
    def new_obstacles_for_replan(self) -> bool:
        return self._new_obstacles_for_replan
    @new_obstacles_for_replan.setter
    def new_obstacles_for_replan(self, value: bool) -> None:
        self._new_obstacles_for_replan = value


    def calculate_direct_velocity_to_target(self, target_pos_local_odom: np.ndarray) -> TwistStamped:
        cfg = self.motion_config
        ts = TwistStamped()
        ts.header.stamp = self.node.get_clock().now().to_msg()
        ts.header.frame_id = 'odom'

        current_pos = self.position
        current_orient_q = self.orientation

        if current_pos is None or current_orient_q is None:
            self.logger.warn("CalculateDirectVel: Odometry data missing.", throttle_duration_sec=2.0)
            return ts

        delta_to_target = target_pos_local_odom - current_pos
        dist_to_target_xy = np.linalg.norm(delta_to_target[:2])
        dist_to_target_z = delta_to_target[2]

        if dist_to_target_xy > 1e-3:
            dir_to_target_xy = delta_to_target[:2] / dist_to_target_xy
            speed_xy_magnitude = min(cfg.max_xy_speed, cfg.xy_control_gain * dist_to_target_xy)
            ts.twist.linear.x = dir_to_target_xy[0] * speed_xy_magnitude
            ts.twist.linear.y = dir_to_target_xy[1] * speed_xy_magnitude
        else:
            ts.twist.linear.x = 0.0
            ts.twist.linear.y = 0.0

        ts.twist.linear.z = np.clip(cfg.z_control_gain * dist_to_target_z, -cfg.max_z_speed, cfg.max_z_speed)
        
        if dist_to_target_xy > 0.1: 
            current_yaw = Rotation.from_quat(current_orient_q).as_euler('xyz', degrees=False)[2]
            target_yaw = math.atan2(delta_to_target[1], delta_to_target[0])
            yaw_error = normalize_angle(target_yaw - current_yaw)
            ts.twist.angular.z = np.clip(cfg.yaw_control_gain * yaw_error, -cfg.max_yaw_rate, cfg.max_yaw_rate)
        else:
            ts.twist.angular.z = 0.0
        
        return ts

    def publish_astar_path_markers(self, path_local_odom: list[np.ndarray] | None):
        if self.viz_publisher is None or path_local_odom is None or not path_local_odom:
            return

        marker_array = MarkerArray()
        
        line_marker = Marker()
        line_marker.header.frame_id = "odom"
        line_marker.header.stamp = self.node.get_clock().now().to_msg()
        line_marker.ns = "astar_path_line" 
        line_marker.id = self._path_marker_id_counter
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.1
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        
        for wp_local_odom in path_local_odom:
            p = Point(x=wp_local_odom[0], y=wp_local_odom[1], z=wp_local_odom[2])
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)


        sphere_id_offset = self._path_marker_id_counter + 1
        for i, wp_local_odom in enumerate(path_local_odom):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "odom"
            sphere_marker.header.stamp = self.node.get_clock().now().to_msg()
            sphere_marker.ns = "astar_waypoints_spheres"
            sphere_marker.id = sphere_id_offset + i
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position = Point(x=wp_local_odom[0], y=wp_local_odom[1], z=wp_local_odom[2])
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale = Vector3(x=0.3, y=0.3, z=0.3)
            sphere_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.7)
            marker_array.markers.append(sphere_marker)

        if marker_array.markers:
            self.viz_publisher.publish(marker_array)
            self._path_marker_id_counter = sphere_id_offset + len(path_local_odom)


    def clear_astar_path_markers(self):
        if self.viz_publisher is None:
            return
        
        marker_array = MarkerArray()
        delete_line_marker = Marker()
        delete_line_marker.header.frame_id = "odom"
        delete_line_marker.header.stamp = self.node.get_clock().now().to_msg()
        delete_line_marker.ns = "astar_path_line"
        delete_line_marker.id = 0
        delete_line_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_line_marker)

        delete_spheres_marker = Marker()
        delete_spheres_marker.header.frame_id = "odom"
        delete_spheres_marker.header.stamp = self.node.get_clock().now().to_msg()
        delete_spheres_marker.ns = "astar_waypoints_spheres"
        delete_spheres_marker.id = 0
        delete_spheres_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_spheres_marker)
        
        self.viz_publisher.publish(marker_array)
        self._path_marker_id_counter = 0


# --- Шаг: Взлет (Takeoff) ---
class Takeoff(Step):
    def __init__(self, relative_altitude: float, tolerance: float = 0.3):
        self.relative_altitude = abs(relative_altitude)
        self.tolerance = tolerance
        self._initial_pos_z_local_odom: float | None = None
        self._target_z_local_odom: float | None = None

    @property
    def uses_velocity_control(self) -> bool: return True

    def init(self, controller: Controller) -> None:
        current_pos_local_odom = controller.position
        if current_pos_local_odom is None: raise RuntimeError("Takeoff init: current position unknown")
        self._initial_pos_z_local_odom = current_pos_local_odom[2]
        self._target_z_local_odom = self._initial_pos_z_local_odom + self.relative_altitude
        controller.logger.info(f"Takeoff Init: Start LocalOdom Z={self._initial_pos_z_local_odom:.2f}, Rel.Climb={self.relative_altitude:.2f}, Target LocalOdom Z={self._target_z_local_odom:.2f}")
        controller.clear_astar_path_markers()

    def update(self, controller: Controller) -> bool:
        current_pos_local_odom = controller.position
        if current_pos_local_odom is None or self._target_z_local_odom is None:
            controller.logger.warn("Takeoff: Waiting for odom data.", throttle_duration_sec=2.0)
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return False

        current_z_local_odom = current_pos_local_odom[2]
        height_reached = abs(current_z_local_odom - self._target_z_local_odom) < self.tolerance
        
        target_pos_for_takeoff = np.array([current_pos_local_odom[0], current_pos_local_odom[1], self._target_z_local_odom])
        
        twist_cmd = controller.calculate_direct_velocity_to_target(target_pos_for_takeoff)
        twist_cmd.twist.linear.x = 0.0
        twist_cmd.twist.linear.y = 0.0
        twist_cmd.twist.angular.z = 0.0

        if height_reached:
            controller.logger.info(f"Takeoff complete: Current LocalOdom Z={current_z_local_odom:.2f}")
            stop_ts = TwistStamped()
            stop_ts.header.stamp = controller.node.get_clock().now().to_msg()
            stop_ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(stop_ts)
            return True
        else:
            controller.velocity_publisher.publish(twist_cmd)
            return False
    def __str__(self) -> str: return f"Takeoff (Relative) +{self.relative_altitude:.2f}m"


# --- Шаг: Навигация A* к глобальной цели (AStarNavToGlobalGoal) ---
class AStarNavToGlobalGoal(Step):
    def __init__(self, target_x_rel_mission: float, target_y_rel_mission: float, target_z_rel_mission: float,
                 final_tolerance_xy: float = 0.5, final_tolerance_z: float = 0.3):
        self.final_target_pos_local_odom = np.array([target_x_rel_mission, target_y_rel_mission, target_z_rel_mission])
        self.final_tolerance_xy = final_tolerance_xy
        self.final_tolerance_z = final_tolerance_z
        
        self._astar_path_local_odom: list[np.ndarray] | None = None
        self._current_astar_waypoint_idx: int = -1
        self._current_sub_goal_local_odom: np.ndarray | None = None
        
        self._sub_goal_reach_tolerance_xy: float = 0.5 
        self._sub_goal_reach_tolerance_z: float = 0.3
        self._last_replan_time: rclpy.time.Time | None = None
        self._replan_interval_seconds: float = 2.0

    @property
    def uses_velocity_control(self) -> bool: return True

    def _plan_path(self, controller: Controller, start_pos_local_odom: np.ndarray) -> bool:
        """Helper function to plan or replan the path. Returns True if new path found."""
        astar_planner_instance = controller.astar_planner
        
        astar_obstacles_local_odom_3d = []
        for obs_global in controller.detected_obstacles_global:
            obs_pos_center_local_odom = obs_global['pos_center_abs'] - controller.origin_pos_abs
            astar_obstacles_local_odom_3d.append({
                'pos_center': obs_pos_center_local_odom,
                'radius_xy': obs_global['radius_xy'],
                'half_height_z': obs_global['sz_full'] / 2.0
            })
        
        if astar_obstacles_local_odom_3d:
            controller.logger.info(f"AStarNav: Passing {len(astar_obstacles_local_odom_3d)} obstacles to A* planner.")
        else:
            controller.logger.info(f"AStarNav: No obstacles detected, planning in free space.")

        controller.logger.info(f"AStarNav: Planning from LocalOdom {vector_to_str(start_pos_local_odom)} to FinalLocalOdom {vector_to_str(self.final_target_pos_local_odom)}")

        new_path = astar_planner_instance.plan(
            start_pos_local_odom,
            self.final_target_pos_local_odom,
            astar_obstacles_local_odom_3d 
        )

        if not new_path:
            controller.logger.error("AStarNav: A* planning FAILED!")
            return False
        else:
            controller.logger.info(f"AStarNav: A* plan SUCCESSFUL, {len(new_path)} 3D waypoints.")
            self._astar_path_local_odom = new_path
            controller.clear_astar_path_markers()
            if controller.viz_publisher:
                 controller.publish_astar_path_markers(self._astar_path_local_odom)
            self._current_astar_waypoint_idx = 0
            self._set_next_sub_goal(controller)
            self._last_replan_time = controller.node.get_clock().now()
            return True


    def init(self, controller: Controller):
        motion_cfg = controller.motion_config
        self._sub_goal_reach_tolerance_xy = motion_cfg.target_reach_tolerance_xy
        self._sub_goal_reach_tolerance_z = motion_cfg.target_reach_tolerance_z
        
        controller.clear_astar_path_markers()

        current_pos_local_odom = controller.position
        if current_pos_local_odom is None:
            raise RuntimeError("AStarNavToGlobalGoal init: current position unknown")

        if not self._plan_path(controller, current_pos_local_odom):
            controller.logger.error("AStarNav: Initial planning failed. Step might not execute correctly.")
            self._current_sub_goal_local_odom = None
        
        controller.new_obstacles_for_replan = False

    def _set_next_sub_goal(self, controller: Controller):
        if self._astar_path_local_odom and self._current_astar_waypoint_idx < len(self._astar_path_local_odom):
            self._current_sub_goal_local_odom = self._astar_path_local_odom[self._current_astar_waypoint_idx]
            controller.logger.info(f"AStarNav: Set A* sub-goal {self._current_astar_waypoint_idx + 1}/{len(self._astar_path_local_odom)}: {vector_to_str(self._current_sub_goal_local_odom)}")
        else:
            self._current_sub_goal_local_odom = None
            controller.logger.info("AStarNav: All A* path waypoints processed.")

    def update(self, controller: Controller) -> bool:
        current_pos_local_odom = controller.position
        if current_pos_local_odom is None:
            controller.logger.warn("AStarNav: Waiting for position data.", throttle_duration_sec=2.0)
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return False

        if controller.new_obstacles_for_replan:
            now = controller.node.get_clock().now()
            if self._last_replan_time is None or (now - self._last_replan_time).nanoseconds / 1e9 > self._replan_interval_seconds:
                controller.logger.warn("AStarNav: New obstacles detected! Attempting to REPLAN...")
                if self._plan_path(controller, current_pos_local_odom):
                    controller.logger.info("AStarNav: REPLAN successful. Following new path.")
                else:
                    controller.logger.error("AStarNav: REPLAN FAILED. Continuing on old path if available, or moving to final target.")
                controller.new_obstacles_for_replan = False
            else:
                controller.logger.debug(f"AStarNav: New obstacles detected, but replanning is throttled. Last replan: {self._last_replan_time}, Now: {now}")

        if self._current_sub_goal_local_odom is None:
            controller.logger.info(f"AStarNav: No A* sub-goal. Moving directly towards final target: {vector_to_str(self.final_target_pos_local_odom)}")
            
            delta_final = self.final_target_pos_local_odom - current_pos_local_odom
            dist_xy_final = np.linalg.norm(delta_final[:2])
            dist_z_final = abs(delta_final[2])

            if dist_xy_final < self.final_tolerance_xy and dist_z_final < self.final_tolerance_z:
                controller.logger.info(f"AStarNav: Final target {vector_to_str(self.final_target_pos_local_odom)} REACHED!")
                ts = TwistStamped()
                ts.header.stamp = controller.node.get_clock().now().to_msg()
                ts.header.frame_id = 'odom'
                controller.velocity_publisher.publish(ts)
                if controller.viz_publisher: controller.clear_astar_path_markers()
                return True
            else:
                twist_cmd = controller.calculate_direct_velocity_to_target(self.final_target_pos_local_odom)
                controller.velocity_publisher.publish(twist_cmd)
                return False

        delta_sub_goal = self._current_sub_goal_local_odom - current_pos_local_odom
        dist_xy_sub_goal = np.linalg.norm(delta_sub_goal[:2])
        dist_z_sub_goal = abs(delta_sub_goal[2])

        if dist_xy_sub_goal < self._sub_goal_reach_tolerance_xy and \
           dist_z_sub_goal < self._sub_goal_reach_tolerance_z:
            controller.logger.info(f"AStarNav: Reached A* sub-goal {self._current_astar_waypoint_idx + 1} ({vector_to_str(self._current_sub_goal_local_odom)}).")
            self._current_astar_waypoint_idx += 1
            self._set_next_sub_goal(controller)
            ts_hold = TwistStamped()
            ts_hold.header.stamp = controller.node.get_clock().now().to_msg()
            ts_hold.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts_hold)
            return False
        
        twist_cmd = controller.calculate_direct_velocity_to_target(self._current_sub_goal_local_odom)
        controller.velocity_publisher.publish(twist_cmd)
        return False

    def __str__(self) -> str:
        return f"AStarNavToGlobalGoal -> FinalLocalOdom {vector_to_str(self.final_target_pos_local_odom)}"


# --- Шаг: Посадка (Land) ---
class Land(Step):
    def __init__(self, target_z_local_odom_for_landing: float = 0.1, tolerance: float = 0.15):
        self.target_z_local_odom_for_landing = target_z_local_odom_for_landing
        self.tolerance = tolerance
        self._target_xy_local_odom_hold: np.ndarray | None = None

    @property
    def uses_velocity_control(self) -> bool: return True

    def init(self, controller: Controller) -> None:
        current_pos_local_odom = controller.position
        if current_pos_local_odom is None:
            raise RuntimeError("Land init: current position unknown")
        self._target_xy_local_odom_hold = current_pos_local_odom[:2]
        controller.logger.info(f"Land Init: Target LocalOdom Z for landing={self.target_z_local_odom_for_landing:.2f}. Holding XY at {vector_to_str(self._target_xy_local_odom_hold)}")
        if controller.viz_publisher: controller.clear_astar_path_markers()

    def update(self, controller: Controller) -> bool:
        current_pos_local_odom = controller.position
        if current_pos_local_odom is None or self._target_xy_local_odom_hold is None:
            controller.logger.warn("Land: Waiting for odom data.", throttle_duration_sec=2.0)
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return False

        current_z_local_odom = current_pos_local_odom[2]
        landed_at_target_z = abs(current_z_local_odom - self.target_z_local_odom_for_landing) < self.tolerance
        
        target_pos_for_landing = np.array([self._target_xy_local_odom_hold[0], 
                                           self._target_xy_local_odom_hold[1], 
                                           self.target_z_local_odom_for_landing])
        twist_cmd = controller.calculate_direct_velocity_to_target(target_pos_for_landing)
        
        if np.linalg.norm(current_pos_local_odom[:2] - self._target_xy_local_odom_hold) < controller.motion_config.target_reach_tolerance_xy / 2.0 :
            twist_cmd.twist.linear.x = 0.0
            twist_cmd.twist.linear.y = 0.0

        twist_cmd.twist.angular.z = 0.0

        if landed_at_target_z:
            controller.logger.info(f"Land complete: Current LocalOdom Z={current_z_local_odom:.2f}")
            stop_ts = TwistStamped()
            stop_ts.header.stamp = controller.node.get_clock().now().to_msg()
            stop_ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(stop_ts)
            return True
        else:
            controller.velocity_publisher.publish(twist_cmd)
            return False

    def __str__(self) -> str: return f"Land at LocalOdom Z={self.target_z_local_odom_for_landing:.2f}m"


# --- Шаг: Дизарм (Disarm) ---
class Disarm(Step):
    def __init__(self, attempts=5, delay_between_attempts=1.0):
        self.attempts_left = attempts
        self.delay_between_attempts = Duration(seconds=delay_between_attempts)
        self.last_attempt_time: rclpy.time.Time | None = None
        self.disarm_request_active = False

    @property
    def uses_velocity_control(self) -> bool: return False

    def init(self, controller: Controller) -> None:
        controller.logger.info(f"Disarm Init: Preparing to send disarm command ({self.attempts_left} attempts).")
        self.last_attempt_time = None
        self.disarm_request_active = False
        if controller.viz_publisher: controller.clear_astar_path_markers()

    def _disarm_response_callback(self, future, controller: Controller):
        self.disarm_request_active = False
        try:
            response = future.result()
            if response.success:
                controller.logger.info(f"Disarm: Command acknowledged successfully: {response.success}, result: {response.result}")
            else:
                controller.logger.warn(f"Disarm: Command acknowledged, but FAILED: {response.success}, result: {response.result}. Will retry if attempts remain.")
        except Exception as e:
            controller.logger.error(f"Disarm: Service call failed: {e}")

    def update(self, controller: Controller) -> bool:
        now = controller.node.get_clock().now()
        
        with controller._state_lock:
            is_fcu_disarmed = not controller.current_state.armed
            is_fcu_connected = controller.current_state.connected

        if is_fcu_disarmed:
            controller.logger.info("Disarm: FCU confirmed DISARMED via MAVROS state.")
            return True
        
        if not is_fcu_connected:
            controller.logger.warn("Disarm: MAVROS not connected to FCU. Cannot disarm.", throttle_duration_sec=5)
            return False
        
        if self.attempts_left <= 0:
            controller.logger.error("Disarm: Failed to disarm after all attempts (FCU still armed). Step terminating.")
            return True

        if not self.disarm_request_active and \
           (self.last_attempt_time is None or (now - self.last_attempt_time) >= self.delay_between_attempts):
            
            if controller.arming_client.service_is_ready():
                controller.logger.info(f"Disarm: Sending disarm command (attempt {self.attempts_left})...")
                future = controller.arming_client.call_async(CommandBool.Request(value=False))
                future.add_done_callback(lambda f: self._disarm_response_callback(f, controller))
                self.disarm_request_active = True
                self.last_attempt_time = now
                self.attempts_left -= 1
            else:
                controller.logger.warn("Disarm: Arming service not ready. Will retry.")
        
        return False
    def __str__(self) -> str: return "Disarm Engines"


# --- Основная функция main ---
def main(args=None):
    rclpy.init(args=args)
    log_level_str = "INFO" # заменить на "DEBUG" для детального логгирования
    log_level = LoggingSeverity.INFO if log_level_str == "INFO" else LoggingSeverity.DEBUG

    node = Node('astar_3d_controller_only')
    node.get_logger().set_level(log_level)
    node.get_logger().info(f"Logging level set to {log_level.name}")

    original_waypoints_abs = [
        [ 549.75,     207.8,      98.56658], # 0 Старт
        [ 549.75,     207.8,     125.0    ], # 1
        [ 580.0,      366.0,     125.0    ], # 2
        [ 624.55501,  598.40931,  70.0    ], # 3
        [ 928.84280,  947.73771,  95.0    ], # 4
        [ 965.0,     1238.0,     103.5    ], # 5
        [1108.28750, 1443.92409, 103.0    ], # 6
        [ 933.8,     1701.4,     108.0],  # 8
        [ 933.8,     1701.4,     140.0    ], # 7
        [ 933.8,     1701.4,     108.0]  # 8
    ]

    if len(original_waypoints_abs) > 1:
        last_nav_wp_abs = original_waypoints_abs[-1]
        cruise_altitude_abs = original_waypoints_abs[1][2]
        landing_approach_wp_abs = [last_nav_wp_abs[0], last_nav_wp_abs[1], cruise_altitude_abs]
        original_waypoints_abs.append(landing_approach_wp_abs)
    else:
        original_waypoints_abs.append(original_waypoints_abs[0])
    node.get_logger().info("Odometry provides LOCAL coordinates (relative to MAVROS local origin).")
    node.get_logger().info("Mission waypoints are ABSOLUTE (e.g., Gazebo world), will be converted to LOCAL targets.")

    mission_origin_abs = np.array(original_waypoints_abs[0])
    node.get_logger().info(f"Absolute Mission Origin (Global Coords, e.g. Gazebo): {vector_to_str(mission_origin_abs)}")
    node.get_logger().info("Controller will operate in a frame LOCAL to this mission origin (local_odom).")

    target_waypoints_local_odom = []
    for i in range(len(original_waypoints_abs)):
        wp_abs = np.array(original_waypoints_abs[i])
        target_local_odom = wp_abs - mission_origin_abs
        target_waypoints_local_odom.append(target_local_odom)
        node.get_logger().info(f"Target {i} (LocalOdom, Rel.ToMissionOrigin): {vector_to_str(target_local_odom)}")

    astar_nav_final_tolerance_xy = 0.5 
    astar_nav_final_tolerance_z = 1.5
    takeoff_tolerance = 0.3

    steps: list[Step] = []

    relative_takeoff_altitude = target_waypoints_local_odom[1][2] - target_waypoints_local_odom[0][2]
    node.get_logger().info(f"Step 1: Takeoff. Relative altitude: {relative_takeoff_altitude:.2f}m (Target LocalOdom Z: {target_waypoints_local_odom[1][2]:.2f})")
    steps.append(Takeoff(relative_altitude=relative_takeoff_altitude, tolerance=takeoff_tolerance))

    num_nav_goals = len(target_waypoints_local_odom)

    for i in range(1, num_nav_goals):
        target_nav_local_odom = target_waypoints_local_odom[i]
        node.get_logger().info(f"Step {len(steps)+1}: AStarNavTo Target LocalOdom {vector_to_str(target_nav_local_odom)}")
        steps.append(AStarNavToGlobalGoal(
            target_x_rel_mission=target_nav_local_odom[0],
            target_y_rel_mission=target_nav_local_odom[1],
            target_z_rel_mission=target_nav_local_odom[2],
            final_tolerance_xy=astar_nav_final_tolerance_xy,
            final_tolerance_z=astar_nav_final_tolerance_z
        ))

    landing_target_z_local_odom = target_waypoints_local_odom[0][2] + 0.1
    landing_tolerance = 0.15
    node.get_logger().info(f"Step {len(steps)+1}: Land at LocalOdom Z={landing_target_z_local_odom:.2f}m")
    steps.append(Land(target_z_local_odom_for_landing=landing_target_z_local_odom, tolerance=landing_tolerance))

    node.get_logger().info(f"Step {len(steps)+1}: Disarm Engines")
    steps.append(Disarm(attempts=5, delay_between_attempts=1.0))

    try:
        simple_controller = SimpleController(node, steps,
                                             origin_pos_abs=mission_origin_abs,
                                             control_frequency=20.0)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        node.get_logger().info("Starting A*(3D) Controller Node...")
        executor.spin()
    except KeyboardInterrupt: node.get_logger().info("Node stopped via KeyboardInterrupt.")
    except SystemExit as e: node.get_logger().info(f"Node stopped via SystemExit: {e}")
    except Exception as e:
        node.get_logger().fatal(f"Unhandled exception in main: {e}")
        node.get_logger().error(traceback.format_exc())
    finally:
        node.get_logger().info("Shutting down A*(3D) Controller Node...")
        if 'simple_controller' in locals() and hasattr(simple_controller, 'velocity_publisher') \
           and simple_controller.velocity_publisher and rclpy.ok() and node.context.ok():
            try:
                ts = TwistStamped()
                ts.header.stamp = node.get_clock().now().to_msg()
                ts.header.frame_id = 'odom'
                for _ in range(3):
                    simple_controller.velocity_publisher.publish(ts)
                    time.sleep(0.05)
                node.get_logger().info("Zero velocity published on shutdown.")
            except Exception as e_pub: node.get_logger().error(f"Error publishing zero velocity on shutdown: {e_pub}")
        
        if 'simple_controller' in locals() and hasattr(simple_controller, 'clear_astar_path_markers') and simple_controller.viz_publisher:
             simple_controller.clear_astar_path_markers()

        if 'executor' in locals() and executor:
            executor.shutdown()
            node.get_logger().info("Executor shut down.")
        if 'node' in locals() and node and rclpy.ok() and node.context.ok():
            node.destroy_node()
            print("ROS Node destroyed.")
        if rclpy.ok():
            rclpy.shutdown()
            print("RCLPY shut down successfully.")

if __name__ == '__main__':
    main()
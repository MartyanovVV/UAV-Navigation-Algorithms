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
from copy import copy
import traceback

# --- VFH Config ---
class VfhConfig:
    """Конфигурация параметров VFH."""
    def __init__(self):
        # --- Параметры Дрона ---
        self.robot_radius: float = 0.2
        # Физический радиус дрона в метрах.

        self.max_speed: float = 10.0
        # Максимальная желаемая горизонтальная скорость дрона в локальной системе координат мира (м/с).

        self.min_speed: float = 0.1
        # Минимальная скорость, с которой дрон будет двигаться вперед 
        
        self.max_z_velocity: float = 5.0
        # Максимальная вертикальная скорость (вверх или вниз) в м/с.

        self.max_yaw_rate: float = 1.5
        # Максимальная скорость рыскания (поворота вокруг вертикальной оси) дрона (рад/с).

        self.max_accel: float = 5.0
        # Максимальное желаемое линейное ускорение (м/с²).

        self.max_dyaw_rate: float = 3.0
        # Максимальное угловое ускорение рыскания (рад/с²).

        # --- Параметры Алгоритма VFH ---
        self.dt: float = 0.05
        # Временной шаг между циклами управления (секунды).

        self.predict_time: float = 1.5
        # Время предсказания (секунды).

        self.goal_tolerance: float = 1.5
        # Допуск по XY (метры) для завершения шага навигации VfhNavTo.
        # Определяет, насколько близко дрон должен подойти к целевой XY-координате,

        self.num_angle_bins: int = 180
        # Количество секторов (бинов) в полярной гистограмме VFH.
        # Определяет угловое разрешение. 180 бинов = 2 градуса на сектор.

        self.hist_threshold: float = 5.0
        # Пороговое значение "заполненности" для бина в полярной гистограмме.
        # Если значение в бине превышает этот порог, соответствующее направление
        # считается заблокированным препятствием. Критически важный параметр.

        self.obstacle_influence_factor: float = 55.0
        # Коэффициент, определяющий, насколько "сильно" (какое значение)
        # препятствие добавляет в бины гистограммы в зависимости от расстояния.
        # Большее значение = более сильное влияние от препятствий.

        self.obstacle_max_distance: float = 30.0
        # Максимальное расстояние (в метрах) по горизонтали (в локальной СК мира),
        # на котором препятствия учитываются при построении гистограммы VFH.
        # Препятствия дальше этой дистанции игнорируются горизонтальным VFH.

        self.safety_margin_factor: float = 1.2
        # Множитель, применяемый к сумме радиуса робота и радиуса препятствия.

        self.valley_min_width: int = 4
        # Минимальная ширина (в количестве смежных свободных бинов)
        # непрерывного свободного сектора, чтобы он считался валидной "долиной" (путем) для пролета.
        # Большее значение заставляет VFH искать более широкие проходы.

        # --- Параметры Управления ---
        self.speed_control_gain: float = 0.8
        # Коэффициент усиления P-регулятора для расчета желаемой скорости движения к цели.
        # speed_goal = gain * dist_to_goal_xy.

        self.obstacle_slowdown_distance: float = 3.0
        # Расстояние (в метрах) до препятствия (в выбранном VFH направлении),
        # при котором дрон начинает линейно снижать скорость.

        self.yaw_control_gain: float = 1.8
        # Коэффициент усиления P-регулятора для расчета желаемой скорости рыскания.
        # target_angular_rate_body = gain * angular_error_for_yaw.

        self.z_control_gain: float = 0.80
        # Коэффициент усиления P-регулятора для расчета вертикальной скорости.
        # vz = gain * error_z. Влияет на скорость достижения целевой высоты Z.
        
        
        
        
        self.vertical_evasion_max_dist_xy: float = 35.0
        # Максимальное горизонтальное расстояние до препятствия, при котором рассматривается вертикальное уклонение.

        self.vertical_evasion_cone_angle_rad: float = math.radians(25.0) # было 20.0
        # Угловой конус (в радианах) перед дроном, в котором проверяются препятствия для вертикального уклонения.
        # Если выбранное VFH направление близко к направлению на препятствие внутри этого конуса,
        # и есть пересечение по высоте, активируется вертикальное уклонение.

        self.vertical_evasion_clearance_margin: float = 1.5 # было 1.0 + robot_radius
        # Дополнительный безопасный зазор (в метрах) сверху/снизу препятствия при вертикальном уклонении.
        # Итоговый зазор будет robot_radius + vertical_evasion_clearance_margin.

        self.vertical_evasion_speed_factor: float = 0.8 # от 0.0 до 1.0
        # Коэффициент, на который умножается cfg.max_z_velocity для определения скорости вертикального уклонения.
        # Позволяет сделать уклонение менее агрессивным, чем максимально возможная вертикальная скорость.
        # evasion_speed = cfg.max_z_velocity * vertical_evasion_speed_factor

        self.min_vertical_evasion_duration: float = 1.0 # секунды
        # Минимальное время, в течение которого будет поддерживаться команда вертикального уклонения
        # после ее активации, даже если условие для уклонения исчезнет раньше.
        # Помогает избежать слишком частых переключений режима уклонения.

        self.vertical_evasion_z_filter_threshold: float = 7.0
        # Порог по Z (в метрах) для фильтрации препятствий, которые слишком высоко или низко относительно дрона,
        # чтобы их вообще рассматривать для VFH и вертикального уклонения.


# --- Controller ---
class Controller(ABC):
    @property
    @abstractmethod
    def position(self) -> np.ndarray | None: pass
    @property
    @abstractmethod
    def orientation(self) -> np.ndarray | None: pass
    @property
    @abstractmethod
    def velocity(self) -> np.ndarray | None: pass
    @property
    @abstractmethod
    def detected_obstacles_global(self) -> list: pass
    @property
    @abstractmethod
    def velocity_publisher(self) -> rclpy.publisher.Publisher: pass
    @property
    @abstractmethod
    def viz_publisher(self) -> rclpy.publisher.Publisher: pass
    @property
    @abstractmethod
    def logger(self): pass
    @property
    @abstractmethod
    def vfh_config(self) -> VfhConfig: pass

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

# --- SimpleController ---
class SimpleController(Controller):
    def __init__(self, node: Node, steps: list[Step], origin_pos_abs: np.ndarray, control_frequency: float = 20.0):
        self.node = node
        self.steps = steps
        self.step: Step | None = None
        self._pose: Pose | None = None
        self._velocity_body: Twist | None = None
        self._detected_obstacles_global: list = []
        self.current_state = State()
        self._vfh_config = VfhConfig()
        self._vfh_config.dt = 1.0 / control_frequency
        self.arming_req_sent = False
        self.offboard_req_sent = False
        self.last_state_check_time = node.get_clock().now()
        self._state_lock = threading.Lock()
        self.state_callback_group = ReentrantCallbackGroup()
        self.services_ready = False
        self._current_markers = []
        self.last_twist_cmd = TwistStamped()
        
        
        self.vfh_histogram = np.zeros(self.vfh_config.num_angle_bins)
        self.vfh_target_angle_global: float = 0.0
        self.vfh_chosen_angle_global: float | None = None

        self.is_actively_evading_vertically = True

        self.vertical_evasion_active_until_time = self.node.get_clock().now()

        self.last_commanded_evasion_vz = 0.0

        self.node.get_logger().info("SimpleController Initialized.")
        
        
        self.origin_pos_abs = origin_pos_abs

        self.node.get_logger().info("Starting SimpleController...")
        mavros_ns = '/mavros'

        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self._velocity_publisher = node.create_publisher(TwistStamped, f'{mavros_ns}/setpoint_velocity/cmd_vel', qos_cmd)
        self.node.get_logger().info(f"Using publisher: {self._velocity_publisher.topic_name}")

        qos_viz = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._viz_publisher = node.create_publisher(MarkerArray, '~/vfh_viz', qos_viz)

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
        self.node.get_logger().info(f"Control frequency set to {control_frequency} Hz (dt={self._vfh_config.dt:.3f}s).")

        self.vfh_histogram = np.zeros(self.vfh_config.num_angle_bins)
        self.vfh_target_angle_global: float = 0.0
        self.vfh_chosen_angle_global: float | None = None
        self.node.get_logger().info("SimpleController Initialized.")

    def create_client_and_wait(self, srv_type, srv_name):
        client = self.node.create_client(srv_type, srv_name)
        while not client.wait_for_service(timeout_sec=1.0) and rclpy.ok():
            self.node.get_logger().info(f'{srv_name} service not available, waiting...')
        if not rclpy.ok(): raise SystemExit("Shutdown requested during service wait")
        self.node.get_logger().info(f'{srv_name} service available.')
        return client

    def odom_callback(self, msg: Odometry):
        self._pose = msg.pose.pose
        self._velocity_body = msg.twist.twist

    def obstacles_callback(self, msg: String):
        """Обрабатывает сообщения о препятствиях и выводит предупреждение."""
        self.node.get_logger().debug(f"Obstacles raw data received: '{msg.data}'")
        new_obstacles = []
        parts = msg.data.split()
        was_obstacle_before = bool(self._detected_obstacles_global) 

        if len(parts) == 7:
            try:
                obs_name = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                sx, sy, sz = float(parts[4]), float(parts[5]), float(parts[6])
                obs_radius = max(sx, sy) / 2.0
                new_obstacles.append({'name': obs_name, 'pos': np.array([x, y, z]), 'radius': obs_radius, 'sz': sz  })
            except ValueError as e:
                self.node.get_logger().error(f"Error parsing obstacle data '{msg.data}': {e}")
                if was_obstacle_before:
                     self.node.get_logger().info("Obstacles cleared due to parsing error.")
                self._detected_obstacles_global = []
                return

        if new_obstacles:
            self._detected_obstacles_global = new_obstacles
            YELLOW = '\033[93m'
            RESET = '\033[0m'  
            num_obs = len(self._detected_obstacles_global)
            obs_summary = []
            for i, obs in enumerate(self._detected_obstacles_global):
                coord_str = vector_to_str(obs['pos'])
                obs_summary.append(f"'{obs.get('name', 'N/A')}' at [{coord_str}]")
                if i >= 2 and num_obs > 3:
                    obs_summary.append("...")
                    break
            self.node.get_logger().warn(f"{YELLOW}OBSTACLE DETECTED! Count: {num_obs}. Details: {'; '.join(obs_summary)}{RESET}")
        else:
            if msg.data.strip() and len(parts) != 7:
                self.node.get_logger().warn(f"Invalid obstacle msg format: '{msg.data}'. Expected 7 parts. Clearing obstacles.")
            if was_obstacle_before:
                 self.node.get_logger().info("Obstacles cleared (no valid obstacles in message).")
                 self._detected_obstacles_global = []

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
            if self.step is not None and (msg.mode != target_mode or not msg.armed):
                 self.node.get_logger().warn(f"FCU state changed during step execution! Mode: {msg.mode}, Armed: {msg.armed}. Controller will not intervene.", throttle_duration_sec=5)
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
                    self.node.get_logger().info("Mode is OFFBOARD. Requesting ARM...")
                    self.arming_client.call_async(CommandBool.Request(value=True))
                    self.arming_req_sent = True
                self.offboard_req_sent = False
            else:
                if self.arming_req_sent or self.offboard_req_sent:
                     self.node.get_logger().info("State: ARMED and OFFBOARD. Ready to fly.")
                self.arming_req_sent = False
                self.offboard_req_sent = False

    def timer_callback(self):
        if self.position is None or self.orientation is None or self.velocity is None:
            self.node.get_logger().warn("Waiting for Odometry data...", throttle_duration_sec=5.0)
            if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                 ts = TwistStamped()
                 ts.header.stamp = self.node.get_clock().now().to_msg()
                 ts.header.frame_id = 'odom' 
                 self._velocity_publisher.publish(ts)
            return
        with self._state_lock:
            is_ready_to_fly = self.current_state.connected and self.current_state.armed and self.current_state.mode == 'OFFBOARD'

        if not is_ready_to_fly:
            self.node.get_logger().warn(f"Not ready to fly. State: Connected={self.current_state.connected}, Armed={self.current_state.armed}, Mode='{self.current_state.mode}'. Holding...", throttle_duration_sec=5.0)

            if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                 ts = TwistStamped()
                 ts.header.stamp = self.node.get_clock().now().to_msg()
                 ts.header.frame_id = 'odom'
                 self._velocity_publisher.publish(ts)
                 self.last_twist_cmd = ts
            return

        if self.step is None:
            if not self.steps:
                self.node.get_logger().info("Mission complete. Holding position.")
                if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                    ts = TwistStamped()
                    ts.header.stamp = self.node.get_clock().now().to_msg()
                    ts.header.frame_id = 'odom'
                    self._velocity_publisher.publish(ts)
                    self.last_twist_cmd = ts
                return
            self.step = self.steps.pop(0)
            self.node.get_logger().info(f"--- Starting step: {self.step} ---")
            try:
                 self.step.init(self)
            except Exception as e:
                 self.logger.error(f"Error during step '{self.step}' initialization: {e}", exc_info=True)
                 self.step = None
                 return

        step_completed = False
        if self.step:
            try:
                step_completed = self.step.update(self)
            except Exception as e:
                self.logger.error(f"Error during step '{self.step}' update: {e}", exc_info=True)
                self.step = None
                if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                    ts = TwistStamped()
                    ts.header.stamp = self.node.get_clock().now().to_msg()
                    ts.header.frame_id = 'odom'
                    self._velocity_publisher.publish(ts)
                    self.last_twist_cmd = ts
                return

        if step_completed:
            self.node.get_logger().info(f"--- Completed step: {self.step} ---")
            if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                 ts = TwistStamped()
                 ts.header.stamp = self.node.get_clock().now().to_msg()
                 ts.header.frame_id = 'odom'
                 self._velocity_publisher.publish(ts)
                 self.last_twist_cmd = ts
            self.step = None

        self.publish_vfh_visualization()

    @property
    def position(self) -> np.ndarray | None:
        if self._pose: return np.array([self._pose.position.x, self._pose.position.y, self._pose.position.z])
        return None
    @property
    def orientation(self) -> np.ndarray | None:
        if self._pose: return np.array([self._pose.orientation.x, self._pose.orientation.y, self._pose.orientation.z, self._pose.orientation.w])
        return None
    @property
    def velocity(self) -> np.ndarray | None:
        if self._velocity_body: return np.array([self._velocity_body.linear.x, self._velocity_body.linear.y, self._velocity_body.linear.z, self._velocity_body.angular.x, self._velocity_body.angular.y, self._velocity_body.angular.z])
        return None
    @property
    def detected_obstacles_global(self) -> list: return self._detected_obstacles_global
    @property
    def velocity_publisher(self) -> rclpy.publisher.Publisher: return self._velocity_publisher
    @property
    def viz_publisher(self) -> rclpy.publisher.Publisher: return self._viz_publisher
    @property
    def logger(self): return self.node.get_logger()
    @property
    def vfh_config(self) -> VfhConfig: return self._vfh_config

    def calculate_vfh_velocity(self, target_pos_relative: np.ndarray) -> TwistStamped | None:
        cfg = self.vfh_config
        self._current_markers = []

        current_pos_local = self.position
        current_orient_quat = self.orientation
        current_vel_body = self.velocity
        now_time = self.node.get_clock().now()

        if current_pos_local is None or current_orient_quat is None or current_vel_body is None:
            self.logger.warn("VFH calc: Missing odom.", throttle_duration_sec=2.0)
            return None

        try:
            current_rot = Rotation.from_quat(current_orient_quat)
            current_yaw_local = current_rot.as_euler('xyz', degrees=False)[2]
            rot_local_world_to_base = current_rot.inv()
        except ValueError:
             self.logger.warn("VFH calc: Invalid orientation.", throttle_duration_sec=2.0)
             return None

        target_pos_xy_rel = target_pos_relative[:2]
        current_pos_xy_local = current_pos_local[:2]
        goal_vector_local_frame = target_pos_xy_rel - current_pos_xy_local
        dist_to_goal_xy = np.linalg.norm(goal_vector_local_frame)
        
        self.logger.info(f"  Dist Calc: Target={vector_to_str(target_pos_relative[:2])}, Current={vector_to_str(current_pos_xy_local)}, Dist={dist_to_goal_xy:.2f}, Tolerance={cfg.goal_tolerance:.2f}")

        obstacles_local_to_base = self._get_obstacles_local_assuming_local_odom(current_pos_local, rot_local_world_to_base)

        self.vfh_histogram = self._build_polar_histogram(obstacles_local_to_base)
        self.logger.debug(f"Histogram built. Max value: {np.max(self.vfh_histogram):.2f}, Threshold: {cfg.hist_threshold}")

        valleys = self._find_valleys(self.vfh_histogram, cfg.hist_threshold)
        if valleys: self.logger.debug(f"Found {len(valleys)} valleys.")
        else: self.logger.debug("No valleys found.")

        target_angle_in_local_frame = math.atan2(goal_vector_local_frame[1], goal_vector_local_frame[0])
        target_angle_error = normalize_angle(target_angle_in_local_frame - current_yaw_local)
        
        chosen_valley = self._select_best_valley(valleys, target_angle_error)

        if not valleys or chosen_valley is None:
            self.logger.error("VFH: No safe valleys found or could not select valley! Stopping XY motion.", throttle_duration_sec=1.0)

            final_vz_local_world = 0.0
            if self.is_actively_evading_vertically and now_time < self.vertical_evasion_active_until_time:
                self.logger.warn(f"VFH: No horizontal path, but continuing previous vertical evasion (vz={self.last_commanded_evasion_vz:.2f}).")
                final_vz_local_world = self.last_commanded_evasion_vz
            else:
                self.is_actively_evading_vertically = False
                final_vz_local_world = self.calculate_z_velocity(target_pos_relative[2])

            ts = TwistStamped()
            ts.header.stamp = now_time.to_msg()
            ts.header.frame_id = 'odom'
            ts.twist.linear.z = final_vz_local_world
            ts.twist.linear.x = 0.0
            ts.twist.linear.y = 0.0
            ts.twist.angular.z = 0.0
            self.last_twist_cmd = ts
            return ts
        else:
             self.logger.debug(f"Selected valley: {chosen_valley['start']}..{chosen_valley['end']}")

        chosen_angle_local_to_base = self._get_direction_in_valley(chosen_valley, target_angle_error)
        self.logger.debug(f"VFH Calc: TargetAngleLocal={math.degrees(target_angle_in_local_frame):.1f}, CurrentYawLocal={math.degrees(current_yaw_local):.1f}, TargetError={math.degrees(target_angle_error):.1f}, ChosenAngleToBase={math.degrees(chosen_angle_local_to_base):.1f}", throttle_duration_sec=1.0)

        target_speed_forward = self._calculate_target_speed(chosen_angle_local_to_base, obstacles_local_to_base, dist_to_goal_xy)
        angular_error_for_yaw = normalize_angle(chosen_angle_local_to_base)
        target_angular_rate_body = cfg.yaw_control_gain * angular_error_for_yaw
        current_wz_body = current_vel_body[5]
        wz_max_accel = current_wz_body + cfg.max_dyaw_rate * cfg.dt
        wz_min_accel = current_wz_body - cfg.max_dyaw_rate * cfg.dt
        wz_body_limited_by_accel = np.clip(target_angular_rate_body, wz_min_accel, wz_max_accel)
        wz_body_final = np.clip(wz_body_limited_by_accel, -cfg.max_yaw_rate, cfg.max_yaw_rate)

        vx_local_world = target_speed_forward * math.cos(current_yaw_local + chosen_angle_local_to_base) 
        vy_local_world = target_speed_forward * math.sin(current_yaw_local + chosen_angle_local_to_base)
        
        chosen_direction_in_local_world = normalize_angle(current_yaw_local + chosen_angle_local_to_base)
    
        vx_local_world = target_speed_forward * math.cos(chosen_direction_in_local_world)
        vy_local_world = target_speed_forward * math.sin(chosen_direction_in_local_world)

        new_vertical_threat_detected_this_cycle = False
        calculated_evasion_vz_this_cycle = 0.0

        if obstacles_local_to_base:
            for obs in obstacles_local_to_base:
                obs_pos_base = obs['pos_local']
                obs_sz = obs.get('sz', 0.0) / 2.0
                obs_name_debug = obs.get('name', 'N/A')
                dist_xy_base = math.hypot(obs_pos_base[0], obs_pos_base[1])

                if dist_xy_base < cfg.vertical_evasion_max_dist_xy:
                    obs_angle_base = math.atan2(obs_pos_base[1], obs_pos_base[0])
                    if abs(normalize_angle(obs_angle_base - chosen_angle_local_to_base)) < cfg.vertical_evasion_cone_angle_rad:
                        
                        obstacle_top_z_rel_drone = obs_pos_base[2] + obs_sz
                        obstacle_bottom_z_rel_drone = obs_pos_base[2] - obs_sz
                        
                        effective_clearance = cfg.robot_radius + cfg.vertical_evasion_clearance_margin

                        self.logger.warn(f"VERTICAL EVASION CHECK: Obstacle '{obs_name_debug}' close (distBaseXY={dist_xy_base:.1f}m) and near chosen path (angle diff {math.degrees(abs(normalize_angle(obs_angle_base - chosen_angle_local_to_base))):.1f} deg).")
                        self.logger.warn(f"  Obs '{obs_name_debug}': CenterZ_base={obs_pos_base[2]:.1f}, HalfHeight_sz={obs_sz:.1f}")
                        self.logger.warn(f"  Obs '{obs_name_debug}': TopZ_base={obstacle_top_z_rel_drone:.1f}, BottomZ_base={obstacle_bottom_z_rel_drone:.1f}, EffClearance={effective_clearance:.2f}")

                        if obstacle_bottom_z_rel_drone < effective_clearance and \
                           obstacle_top_z_rel_drone > -effective_clearance :
                            
                            new_vertical_threat_detected_this_cycle = True
                            evasion_speed_magnitude = cfg.max_z_velocity * cfg.vertical_evasion_speed_factor

                            if obs_pos_base[2] >= 0 :

                                calculated_evasion_vz_this_cycle = evasion_speed_magnitude
                                self.logger.warn(f"  -> THREAT: EVADING UP! Target obs_top={obstacle_top_z_rel_drone:.1f}. Commanded_vz={calculated_evasion_vz_this_cycle:.2f}")
                            else:
                                calculated_evasion_vz_this_cycle = -evasion_speed_magnitude
                                self.logger.warn(f"  -> THREAT: EVADING DOWN! Target obs_bottom={obstacle_bottom_z_rel_drone:.1f}. Commanded_vz={calculated_evasion_vz_this_cycle:.2f}")
                            break
        
        final_vz_local_world = 0.0

        if new_vertical_threat_detected_this_cycle:
            self.is_actively_evading_vertically = True
            self.last_commanded_evasion_vz = calculated_evasion_vz_this_cycle
            self.vertical_evasion_active_until_time = now_time + Duration(seconds=cfg.min_vertical_evasion_duration)
            final_vz_local_world = self.last_commanded_evasion_vz
            self.logger.warn(f"VERTICAL EVASION TRIGGERED/UPDATED: Vz={final_vz_local_world:.2f}, active until {self.vertical_evasion_active_until_time.nanoseconds / 1e9:.2f}")
        elif self.is_actively_evading_vertically and now_time < self.vertical_evasion_active_until_time:
            final_vz_local_world = self.last_commanded_evasion_vz
            self.logger.warn(f"CONTINUING PREVIOUS VERTICAL EVASION: Vz={final_vz_local_world:.2f}, active until {self.vertical_evasion_active_until_time.nanoseconds / 1e9:.2f}")
        else:
            if self.is_actively_evading_vertically:
                self.logger.info("Vertical evasion period ended. Switching to nominal Z control.")
            self.is_actively_evading_vertically = False
            self.last_commanded_evasion_vz = 0.0
            final_vz_local_world = self.calculate_z_velocity(target_pos_relative[2])

        ts = TwistStamped()
        ts.header.stamp = now_time.to_msg()
        ts.header.frame_id = 'odom' 

        ts.twist.linear.x = vx_local_world
        ts.twist.linear.y = vy_local_world
        ts.twist.linear.z = final_vz_local_world 

        ts.twist.angular.z = wz_body_final

        self.logger.info(f"Publishing LocalWorld Vel Twist: Lin=[X:{ts.twist.linear.x:.2f}, Y:{ts.twist.linear.y:.2f}, Z:{ts.twist.linear.z:.2f}], AngZ(body)={ts.twist.angular.z:.2f}", throttle_duration_sec=0.5)
        self.last_twist_cmd = ts
        return ts

    def calculate_z_velocity(self, target_z_relative: float) -> float:
        """Рассчитывает вертикальную скорость vz (+вверх) для достижения ОТНОСИТЕЛЬНОЙ target_z."""
        cfg = self.vfh_config
        current_pos_local = self.position
        if current_pos_local is None: return 0.0
        error_z = target_z_relative - current_pos_local[2]
        vz = cfg.z_control_gain * error_z
        vz = np.clip(vz, -cfg.max_z_velocity, cfg.max_z_velocity)
        return vz

    def _get_obstacles_local_assuming_local_odom(self, current_pos_local: np.ndarray, rot_local_world_to_base: Rotation) -> list:
        """
        Обрабатывает ГЛОБАЛЬНЫЕ препятствия для использования с ЛОКАЛЬНОЙ одометрией.
        Возвращает список препятствий в СК дрона ('base_link').
        """
        if self.origin_pos_abs is None:
            self.logger.error("Absolute origin position is not set! Cannot process global obstacles.")
            return []
        if not self.detected_obstacles_global:
            return []

        local_obstacles_in_base = []
        cfg = self.vfh_config

        for obs_global_data in self.detected_obstacles_global:
            obs_pos_global = obs_global_data['pos']
            obs_radius = obs_global_data['radius']
            obs_pos_local_world = obs_pos_global - self.origin_pos_abs
            delta_local_world = obs_pos_local_world - current_pos_local
            dist_xy_sq = delta_local_world[0]**2 + delta_local_world[1]**2
            if dist_xy_sq > cfg.obstacle_max_distance**2:
                self.logger.debug(f"Obstacle {obs_global_data.get('name','N/A')} ignored (too far in local XY plane)")
                continue
            pos_local_base = rot_local_world_to_base.apply(delta_local_world)
            z_threshold = self.vfh_config.vertical_evasion_z_filter_threshold
                
            if abs(pos_local_base[2]) > z_threshold:
                self.logger.debug(f"Obstacle {obs_global_data.get('name','N/A')} ignored (Z difference |{pos_local_base[2]:.1f}| > {z_threshold})")
                continue
            dist_xy = math.sqrt(dist_xy_sq)
            local_obstacles_in_base.append({
                'pos_local': pos_local_base,
                'radius': obs_radius,
                'dist_xy': dist_xy,
                'sz': obs_global_data['sz'] 
            })
            self.logger.debug(f"Obstacle {obs_global_data.get('name','N/A')} added for VFH: pos_base={vector_to_str(pos_local_base)}")

        return local_obstacles_in_base

    def _build_polar_histogram(self, local_obstacles_in_base: list) -> np.ndarray:
        cfg = self.vfh_config
        histogram = np.zeros(cfg.num_angle_bins)
        bin_width_rad = 2.0 * math.pi / cfg.num_angle_bins
        for obs in local_obstacles_in_base:
            obs_x = obs['pos_local'][0]
            obs_y = obs['pos_local'][1]
            obs_r = obs['radius']
            dist_xy = math.hypot(obs_x, obs_y)
            if dist_xy < cfg.robot_radius: continue
            eff_r = (obs_r + cfg.robot_radius) * cfg.safety_margin_factor
            try:
                phi_arg = eff_r / dist_xy
                if phi_arg >= 1.0: phi = math.pi / 2.0
                elif phi_arg <= -1.0: phi = math.pi / 2.0
                else: phi = math.asin(phi_arg)
            except ValueError: 
                phi = math.pi / 2.0
            beta = math.atan2(obs_y, obs_x)
            start_a = normalize_angle(beta - phi)
            end_a = normalize_angle(beta + phi)
            mag_raw = max(0.0, (cfg.obstacle_max_distance - dist_xy) / cfg.obstacle_max_distance)
            mag = cfg.obstacle_influence_factor * (mag_raw**2)
            start_a_0_2pi = (start_a + 2 * math.pi) % (2 * math.pi)
            end_a_0_2pi = (end_a + 2 * math.pi) % (2 * math.pi)
            start_bin = int(start_a_0_2pi / bin_width_rad) % cfg.num_angle_bins
            end_bin = int(end_a_0_2pi / bin_width_rad) % cfg.num_angle_bins
            current_bin = start_bin
            while True:
                histogram[current_bin] += mag
                if current_bin == end_bin: break
                current_bin = (current_bin + 1) % cfg.num_angle_bins
        return histogram

    def _find_valleys(self, histogram: np.ndarray, threshold: float) -> list:
        valleys = []
        num_bins = len(histogram)
        hist_extended = np.append(histogram, histogram[0])
        start_index = -1
        for i in range(num_bins + 1):
            is_below_threshold = hist_extended[i] < threshold
            if is_below_threshold and start_index == -1: start_index = i % num_bins
            elif not is_below_threshold and start_index != -1:
                end_index = (i - 1 + num_bins) % num_bins
                width = (end_index - start_index + 1 + num_bins) % num_bins if end_index < start_index else end_index - start_index + 1
                if width >= self.vfh_config.valley_min_width: valleys.append({'start': start_index, 'end': end_index, 'width': width})
                start_index = -1
        if start_index != -1 and not valleys:
             end_index = (start_index - 1 + num_bins) % num_bins
             width = num_bins
             if width >= self.vfh_config.valley_min_width: valleys.append({'start': start_index, 'end': end_index, 'width': width})
        return valleys

    def _get_angle_from_bin(self, bin_index: int) -> float:
        num_bins = self.vfh_config.num_angle_bins
        bin_width = 2.0 * math.pi / num_bins
        angle = (bin_index + 0.5) * bin_width - math.pi
        return normalize_angle(angle)

    def _get_bin_from_angle(self, angle: float) -> int:
        num_bins = self.vfh_config.num_angle_bins
        bin_width = 2.0 * math.pi / num_bins
        angle_0_2pi = (normalize_angle(angle) + 2 * math.pi) % (2*math.pi)
        bin_index = int(angle_0_2pi / bin_width)
        return max(0, min(num_bins - 1, bin_index))

    def _get_angle_diff(self, angle1: float, angle2: float) -> float: return normalize_angle(angle1 - angle2)

    def _select_best_valley(self, valleys: list, target_angle_local_error: float) -> dict | None:
        if not valleys: return None
        best_valley = None
        min_diff = float('inf')
        target_bin = self._get_bin_from_angle(target_angle_local_error)
        num_bins = self.vfh_config.num_angle_bins
        for valley in valleys:
            start_idx = valley['start']
            end_idx = valley['end']
            is_inside = (start_idx <= target_bin <= end_idx) if end_idx >= start_idx else (start_idx <= target_bin < num_bins or 0 <= target_bin <= end_idx)
            if is_inside: 
                best_valley = valley
                break
            else:
                start_angle = self._get_angle_from_bin(start_idx)
                end_angle = self._get_angle_from_bin(end_idx)
                diff = min(abs(self._get_angle_diff(target_angle_local_error, start_angle)), abs(self._get_angle_diff(target_angle_local_error, end_angle)))
                if diff < min_diff: 
                    min_diff = diff
                    best_valley = valley
        return best_valley

    def _get_direction_in_valley(self, valley: dict, target_angle_local_error: float) -> float:
        start_idx = valley['start']
        end_idx = valley['end']
        num_bins = self.vfh_config.num_angle_bins
        target_bin = self._get_bin_from_angle(target_angle_local_error)
        is_inside = (start_idx <= target_bin <= end_idx) if end_idx >= start_idx else (start_idx <= target_bin < num_bins or 0 <= target_bin <= end_idx)
        if is_inside: chosen_angle = target_angle_local_error
        else:
            start_angle = self._get_angle_from_bin(start_idx)
            end_angle = self._get_angle_from_bin(end_idx)
            diff_start = abs(self._get_angle_diff(target_angle_local_error, start_angle))
            diff_end = abs(self._get_angle_diff(target_angle_local_error, end_angle))
            chosen_angle = start_angle if diff_start <= diff_end else end_angle
        return chosen_angle
    def _calculate_target_speed(self, chosen_angle_local_to_base: float, local_obstacles_in_base: list, dist_to_goal_xy: float) -> float:
        cfg = self.vfh_config
        speed_goal = np.clip(cfg.speed_control_gain * dist_to_goal_xy, 0, cfg.max_speed)
        min_clearance_in_direction = float('inf')
        angle_tolerance = math.radians(15.0)
        relevant_obstacle_found = False
        for obs in local_obstacles_in_base:
            obs_angle_local = math.atan2(obs['pos_local'][1], obs['pos_local'][0])
            angle_diff_to_chosen = abs(self._get_angle_diff(obs_angle_local, chosen_angle_local_to_base))
            if angle_diff_to_chosen < angle_tolerance:
                relevant_obstacle_found = True
                eff_r = (obs['radius'] + cfg.robot_radius) * cfg.safety_margin_factor
                dist_to_center = math.hypot(obs['pos_local'][0], obs['pos_local'][1])
                clearance = max(0.0, dist_to_center - eff_r)
                min_clearance_in_direction = min(min_clearance_in_direction, clearance)
        speed_obstacle = cfg.max_speed
        if relevant_obstacle_found and min_clearance_in_direction < cfg.obstacle_slowdown_distance:
            speed_obstacle = max(0.0, cfg.max_speed * (min_clearance_in_direction / cfg.obstacle_slowdown_distance))
        target_speed = max(cfg.min_speed, min(speed_goal, speed_obstacle))
        return target_speed

    def publish_vfh_visualization(self):
        if self._current_markers:
            marker_array = MarkerArray(markers=self._current_markers)
            try: self.viz_publisher.publish(marker_array)
            except Exception as e: self.logger.error(f"Failed to publish viz markers: {e}")
            self._current_markers = []

    def _create_marker(self, ns: str, id: int, type: int, scale: Vector3, color: ColorRGBA, pose: Pose | None = None, points: list[Point] | None = None, text: str = "", frame_id: str = "odom", duration_sec: float = 0.2) -> Marker:
        marker = Marker(header=Header(stamp=self.node.get_clock().now().to_msg(), frame_id=frame_id), ns=ns, id=id, type=type, action=Marker.ADD, scale=scale, color=color, lifetime=Duration(seconds=duration_sec).to_msg())
        if pose: marker.pose = pose
        if points: marker.points = points
        if text: marker.text = text
        return marker

    def add_obstacle_markers(self, marker_list: list, local_obstacles_in_base: list, robot_radius: float): pass 

    def add_histogram_markers(self, marker_list: list, histogram: np.ndarray, robot_pos_local: np.ndarray, robot_yaw_local: float): pass 

    def add_direction_markers(self, marker_list: list, robot_pos_local: np.ndarray, robot_yaw_local: float, target_angle_local: float | None, chosen_angle_world: float | None): pass


# --- Takeoff ---
class Takeoff(Step):
    def __init__(self, relative_altitude: float, tolerance: float = 0.5, climb_speed: float = 1.0):
        self.relative_altitude = abs(relative_altitude)
        self.tolerance = tolerance
        self.climb_speed = abs(climb_speed) if climb_speed != 0 else 0.5
        self._initial_pos_z: float | None = None
        self._absolute_target_z: float | None = None

    @property
    def uses_velocity_control(self) -> bool: return True

    def init(self, controller: Controller) -> None:
        current_pos_local = controller.position
        if current_pos_local is None: raise RuntimeError("Takeoff init failed: unknown position")
        self._initial_pos_z = current_pos_local[2]
        self._absolute_target_z = self._initial_pos_z + self.relative_altitude
        controller.logger.info(f"Takeoff Init (Relative): Start Local Z={self._initial_pos_z:.2f}, Relative Climb={self.relative_altitude:.2f}, Target Local Z={self._absolute_target_z:.2f}, Climb Speed={self.climb_speed:.2f} m/s")

    def update(self, controller: Controller) -> bool:
        current_pos_local = controller.position
        if current_pos_local is None or self._absolute_target_z is None:
            controller.logger.warn("Takeoff (Relative): Waiting for data.", throttle_duration_sec=2.0)
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return False

        current_z_local = current_pos_local[2]
        height_reached = abs(current_z_local - self._absolute_target_z) < self.tolerance

        error_z = self._absolute_target_z - current_z_local
        vz_desired = controller.vfh_config.z_control_gain * error_z
        max_climb = controller.vfh_config.max_z_velocity
        max_descend = -controller.vfh_config.max_z_velocity
        vz_cmd = np.clip(vz_desired, max_descend, max_climb)

        controller.logger.info(f"Takeoff Update: CurrentLocalZ={current_z_local:.2f}, TargetLocalZ={self._absolute_target_z:.2f}, Reached={height_reached}, vz_cmd(Global)={vz_cmd:.2f}", throttle_duration_sec=0.5)

        ts = TwistStamped()
        ts.header.stamp = controller.node.get_clock().now().to_msg()
        ts.header.frame_id = 'odom'
        ts.twist.linear.z = vz_cmd

        if height_reached:
            controller.logger.info(f"Takeoff complete: Current Local Z={current_z_local:.2f}")
            ts.twist.linear.z = 0.0
            controller.velocity_publisher.publish(ts)
            return True
        else:
            controller.velocity_publisher.publish(ts)
            return False

    def __str__(self) -> str: return f"Takeoff (Relative) +{self.relative_altitude:.2f}m"


# --- VfhNavTo ---
class VfhNavTo(Step):
    def __init__(self, x_rel: float, y_rel: float, z_rel: float, tolerance_xy: float = 1.5, tolerance_z: float = 0.8):
        self.target_pos_relative = np.array([x_rel, y_rel, z_rel])
        self.tolerance_xy = tolerance_xy
        self.tolerance_z = tolerance_z

    @property
    def uses_velocity_control(self) -> bool: return True

    def init(self, controller: Controller) -> None:
        controller.logger.info(f"VfhNavTo: Init Target Relative: {vector_to_str(self.target_pos_relative)}")

    def update(self, controller: Controller) -> bool:
        current_pos_local = controller.position
        if current_pos_local is None:
            controller.logger.warn("VfhNavTo: Waiting for position.", throttle_duration_sec=2.0)
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return False

        delta_pos = self.target_pos_relative - current_pos_local
        dist_xy = np.linalg.norm(delta_pos[:2])
        dist_z = abs(delta_pos[2])

        if dist_xy < self.tolerance_xy and dist_z < self.tolerance_z:
            controller.logger.info(f"VfhNavTo: Target Relative {vector_to_str(self.target_pos_relative)} reached! (Dist XY: {dist_xy:.2f}m, Dist Z: {dist_z:.2f}m)")
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return True

        twist_stamped_cmd = controller.calculate_vfh_velocity(self.target_pos_relative)

        if twist_stamped_cmd is not None:
            controller.velocity_publisher.publish(twist_stamped_cmd)
        else:
            controller.logger.warn("VfhNavTo: VFH failed. Sending zero global XY velocity.", throttle_duration_sec=1.0)
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            ts.twist.linear.z = controller.calculate_z_velocity(self.target_pos_relative[2])
            controller.velocity_publisher.publish(ts)
        return False

    def __str__(self) -> str: return f"VfhNavTo -> Relative {vector_to_str(self.target_pos_relative)}"




# --- Land ---
class Land(Step):
    """Шаг для посадки дрона на заданную ОТНОСИТЕЛЬНУЮ высоту (обычно близкую к 0)."""
    def __init__(self, target_relative_z: float = 0.1, tolerance: float = 0.15, descend_speed: float = 3.0):
        """
        :param target_relative_z: Целевая относительная высота Z над точкой старта для посадки.
        :param tolerance: Допуск достижения высоты посадки.
        :param descend_speed: Скорость снижения (положительное значение).
        """
        self.target_relative_z = target_relative_z
        self.tolerance = tolerance
        self.descend_speed = abs(descend_speed)
        self._initial_pos_z_local: float | None = None
        self._target_z_local_for_landing: float | None = None

    @property
    def uses_velocity_control(self) -> bool:
        return True

    def init(self, controller: Controller) -> None:
        current_pos_local = controller.position
        if current_pos_local is None:
            controller.logger.error("Land: Cannot init, position unknown.")
            raise RuntimeError("Land step failed to initialize due to unknown position.")

        self._initial_pos_z_local = current_pos_local[2]
        self._target_z_local_for_landing = self.target_relative_z

        controller.logger.info(f"Land Init: Current Local Z={self._initial_pos_z_local:.2f}, Target Relative Z for Landing={self.target_relative_z:.2f}, Descend Speed={self.descend_speed:.2f} m/s")
        self._target_xy_local = current_pos_local[:2]


    def update(self, controller: Controller) -> bool:
        current_pos_local = controller.position
        if current_pos_local is None:
            controller.logger.warn("Land: Waiting for position data.", throttle_duration_sec=2.0)
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return False

        current_z_local = current_pos_local[2]
        height_reached = abs(current_z_local - self.target_relative_z) < self.tolerance

        ts = TwistStamped()
        ts.header.stamp = controller.node.get_clock().now().to_msg()
        ts.header.frame_id = 'odom'

        if height_reached:
            controller.logger.info(f"Land complete: Current Local Z={current_z_local:.2f} (Target Rel Z Land={self.target_relative_z:.2f}, Tol={self.tolerance:.2f})")
            ts.twist.linear.x = 0.0
            ts.twist.linear.y = 0.0
            ts.twist.linear.z = 0.0 
            ts.twist.angular.z = 0.0
            controller.velocity_publisher.publish(ts)
            return True
        else:
            error_z = self.target_relative_z - current_z_local
            vz_desired_global = controller.vfh_config.z_control_gain * error_z
            vz_cmd_global = np.clip(vz_desired_global, -self.descend_speed, self.descend_speed / 2)

            ts.twist.linear.z = vz_cmd_global

            ts.twist.linear.x = 0.0
            ts.twist.linear.y = 0.0
            ts.twist.angular.z = 0.0

            controller.logger.info(f"Land Update: CurrentLocalZ={current_z_local:.2f}, TargetRelZLand={self.target_relative_z:.2f}, vz_cmd(Global)={vz_cmd_global:.2f}", throttle_duration_sec=0.5)
            controller.velocity_publisher.publish(ts)
            return False

    def __str__(self) -> str:
        return f"Land to Relative Z={self.target_relative_z:.2f}m"



# --- Disarm ---
class Disarm(Step):
    """Шаг для отправки команды дизарма."""
    def __init__(self, attempts=5, delay_between_attempts=1.0):
        self.attempts_left = attempts
        self.delay_between_attempts = Duration(seconds=delay_between_attempts)
        self.last_attempt_time: rclpy.time.Time | None = None
        self.disarm_sent_successfully = False

    @property
    def uses_velocity_control(self) -> bool:
        return False

    def init(self, controller: Controller) -> None:
        controller.logger.info(f"Disarm Init: Preparing to send disarm command ({self.attempts_left} attempts).")
        self.last_attempt_time = None
        self.disarm_sent_successfully = False

    def update(self, controller: Controller) -> bool:
        now = controller.node.get_clock().now()
        if not controller.current_state.armed and self.disarm_sent_successfully:
            controller.logger.info("Disarm: Confirmed disarmed by MAVROS state.")
            return True

        if self.attempts_left <= 0 and not controller.current_state.armed:
            controller.logger.error("Disarm: Failed to confirm disarm after all attempts, but FCU reports disarmed.")
            return True
        if self.attempts_left <= 0 and controller.current_state.armed:
            controller.logger.error("Disarm: Failed to disarm after all attempts!")
            return True

        if self.last_attempt_time is None or (now - self.last_attempt_time) >= self.delay_between_attempts:
            controller.logger.info(f"Disarm: Sending disarm command (attempt {self.attempts_left})...")
            if controller.arming_client.service_is_ready():
                future = controller.arming_client.call_async(CommandBool.Request(value=False))
                self.disarm_sent_successfully = True 
            else:
                controller.logger.warn("Disarm: Arming service not ready. Skipping attempt.")
                self.disarm_sent_successfully = False

            self.attempts_left -= 1
            self.last_attempt_time = now

        return False

    def __str__(self) -> str:
        return "Disarm Motors"

# --- Основная функция main ---
def main(args=None):
    rclpy.init(args=args)
    log_level = LoggingSeverity.DEBUG
    node = Node('vfh_obstacle_controller')
    node.get_logger().set_level(log_level)
    node.get_logger().info(f"Logging level set to {log_level.name}")

    original_waypoints = [
        [ 549.75,     207.8,      98.56658], # 0: 
        [ 549.75,     207.8,     125.0    ], # 1
        [ 580.0,      366.0,     125.0    ], # 2
        [ 624.55501,  598.40931,  70.0    ], # 3
        [ 928.84280,  947.73771,  95.0    ], # 4
        [ 965.0,     1238.0,     100.0    ], # 5
        [1108.28750, 1443.92409, 103.0    ], # 6
        [ 933.8,     1701.4,     140.0    ], # 7
        [ 933.8,     1701.4,     107.66052]  # 8
    ]

    node.get_logger().info("Assuming Odometry provides LOCAL coordinates relative to start.")
    node.get_logger().info("Converting Absolute Waypoints to Relative Targets for VFHNavTo.")

    origin_pos_abs = np.array(original_waypoints[0])
    relative_step_targets = []
    for i in range(1, len(original_waypoints)):
        wp_abs = np.array(original_waypoints[i])
        relative_target = wp_abs - origin_pos_abs
        relative_step_targets.append(relative_target)
        node.get_logger().info(f"Calculated Relative Target {i}: {vector_to_str(relative_target)}")

    nav_tolerance_xy = 2.5
    nav_tolerance_z = 4.5
    takeoff_tolerance = 0.5
    climb_rate = 1.0

    steps: list[Step] = []

    relative_takeoff_alt = relative_step_targets[0][2]
    node.get_logger().info(f"Desired relative takeoff altitude: {relative_takeoff_alt:.2f}m")
    steps.append(Takeoff(relative_altitude=relative_takeoff_alt, tolerance=takeoff_tolerance, climb_speed=climb_rate))

    for i, target_rel in enumerate(relative_step_targets):
        target_x = target_rel[0]
        target_y = target_rel[1]
        target_z = target_rel[2]
        node.get_logger().info(f"Adding VFH Nav step {i+1} to Relative Target: [{target_x:.2f}, {target_y:.2f}, {target_z:.2f}]")
        steps.append(VfhNavTo(target_x, target_y, target_z, tolerance_xy=nav_tolerance_xy, tolerance_z=nav_tolerance_z))


    landing_target_relative_z = 0.1
    landing_tolerance = 0.15
    descend_rate = 0.5  
    node.get_logger().info(f"Step {len(steps)+1}: Land to Relative Z={landing_target_relative_z:.2f}m")
    steps.append(Land(target_relative_z=landing_target_relative_z,
                      tolerance=landing_tolerance,
                      descend_speed=descend_rate))

    node.get_logger().info(f"Step {len(steps)+1}: Disarm Motors")
    steps.append(Disarm(attempts=5, delay_between_attempts=1.0))

    try:
        simple_controller = SimpleController(node, steps, control_frequency=20.0, origin_pos_abs=origin_pos_abs)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        node.get_logger().info("Spinning VFH Controller node...")
        executor.spin()
    except KeyboardInterrupt: node.get_logger().info("Node stopped cleanly.")
    except SystemExit as e: node.get_logger().info(f"Node stopped by SystemExit: {e}")
    except Exception as e:
        node.get_logger().fatal(f"Unhandled exception: {e}")
        node.get_logger().error(traceback.format_exc())
    finally:
        node.get_logger().info("Shutting down...")
        if 'simple_controller' in locals() and hasattr(simple_controller, 'velocity_publisher') and simple_controller.velocity_publisher and rclpy.ok():
            try:
                from geometry_msgs.msg import TwistStamped
                ts = TwistStamped()
                ts.header.stamp = node.get_clock().now().to_msg()
                ts.header.frame_id = 'odom'
                for _ in range(3): simple_controller.velocity_publisher.publish(ts); time.sleep(0.05)
                node.get_logger().info("Zero velocity sent.")
            except Exception as e_pub: node.get_logger().error(f"Error sending zero velocity: {e_pub}")
        if 'executor' in locals() and executor: 
            executor.shutdown()
            node.get_logger().info("Executor shut down.")
        if 'node' in locals() and node and rclpy.ok() and node.context.ok(): 
            node.destroy_node()
            print("Node destroyed.")
        if rclpy.ok(): rclpy.shutdown()
        print("ROS Shutdown complete.")


if __name__ == '__main__':
    main()
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


import heapq
import sys
import cv2

import os
from cv2.aruco import ArucoDetector
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist
from mavros.base import SENSOR_QOS, STATE_QOS
from sensor_msgs.msg import Image

import glob
from ultralytics import YOLO
import random


def get_bbox_metrics(image_path, model, show=False):
    """
    Возвращает метрики для каждого обнаруженного объекта:
    - center_x, center_y: координаты центра bbox (пиксели)
    - width, height: размеры bbox (пиксели)
    - angle_rad: угол от центра камеры до центра bbox (радианы)
    
    Параметры:
        image_path (str): путь к изображению (640x480)
        model_path (str): путь к модели YOLOv8n (.pt файл)
    
    Возвращает:
        list[dict]: список словарей с метриками для каждого bbox
    """

    image_path = cv2.cvtColor(image_path, cv2.COLOR_BGRA2BGR)
    
    results = model.predict(image_path, conf=0.5, verbose=False)
    if show:
        annotated_image = results[0].plot()

    image_center_x = 320  # 640 / 2
    image_center_y = 240  # 480 / 2
    focal_length = 380  # фокусное расстояние в пикселях
    
    image_height = 720
    camera_vertical_fov = math.radians(86.8)
    camera_f = image_height / (2 * math.tan(camera_vertical_fov / 2))
    w_real = 1.5
    
    bbox_metrics = []

    for result in results:
        for box in result.boxes:
            # Координаты bbox (x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box.cls[0]
        
            # Центр bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
        
            # Ширина и высота
            width = x2 - x1
            height = y2 - y1
        
            # Угол от центра камеры (в радианах)
            dx = center_x - image_center_x
            dy = center_y - image_center_y
            angle_x = np.arctan2(dx, focal_length)  # угол по горизонтали
            angle_y = np.arctan2(dy, focal_length)  # угол по вертикали
            distance = 585.2 * height / width / width
            distance = (w_real * camera_f) / (width)
            # d = (camera_f * h_real * height) / width ** 2
            
            bbox_metrics.append({
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'distance': distance,
                'angle_x_rad': angle_x,
                'angle_y_rad': angle_y,
                'class': "tree"
            })
    if show:
        return annotated_image, bbox_metrics
    return bbox_metrics

model = YOLO('./classes_freezed_1-10.pt')

# --- VFH Config ---
class VfhConfig:
    """Конфигурация параметров VFH."""
    def __init__(self):
        self.robot_radius: float = 0.2
        self.max_speed: float = 10.0
        self.min_speed: float = 0.1
        self.max_yaw_rate: float = 1.5
        self.max_accel: float = 5.0 # Линейное ускорение
        self.max_dyaw_rate: float = 3.0 # Угловое ускорение рыскания
        self.dt: float = 0.05
        self.predict_time: float = 1.5
        self.goal_tolerance: float = 0.5
        self.num_angle_bins: int = 180
        self.hist_threshold: float = 5.0 # Порог для гистограммы
        self.obstacle_influence_factor: float = 35.0
        self.obstacle_max_distance: float = 5.0
        self.safety_margin_factor: float = 1.2
        self.valley_min_width: int = 8
        self.speed_control_gain: float = 0.8 # Усиление для скорости к цели
        self.obstacle_slowdown_distance: float = 3.0
        self.yaw_control_gain: float = 2.5 # Уменьшено усиление рыскания
        self.z_control_gain: float = 0.80
        self.max_z_velocity: float = 5.0

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
        self._image: np.ndarray | None = None
        self._image_pencil: np.ndarray | None = None
        
        self.camera_matrix_pencil: np.ndarray | None = None
        self.optical_to_camera_pencil: np.ndarray | None = None
        self.camera_to_base_pencil: np.ndarray | None = None
        
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
        qos_obstacles = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=20)
        self.obstacle_subscription = node.create_subscription(String, '/obstacles', self.obstacles_callback, qos_obstacles)
        self.set_mode_client = self.create_client_and_wait(SetMode, f'{mavros_ns}/set_mode')
        self.arming_client = self.create_client_and_wait(CommandBool, f'{mavros_ns}/cmd/arming')
        
        
        self.bridge = CvBridge()
        camera_topic = '/camera'
        camera_history_depth = 10
        self.camera_subscription = node.create_subscription(Image, camera_topic, self.camera_subscription_callback, camera_history_depth)

        self._display_image = None
        self._display_lock = threading.Lock()
        self._gui_thread = threading.Thread(target=self._gui_loop, daemon=True)
        self._gui_thread.start()
        


        timer_period = 1.0 / control_frequency
        self.timer = node.create_timer(timer_period, self.timer_callback)
        self.timer_img = node.create_timer(timer_period, self.timer_callback_img)

        self.node.get_logger().info(f"Control frequency set to {control_frequency} Hz (dt={self._vfh_config.dt:.3f}s).")

        self.vfh_histogram = np.zeros(self.vfh_config.num_angle_bins)
        self.vfh_target_angle_global: float = 0.0
        self.vfh_chosen_angle_global: float | None = None
        self.node.get_logger().info("SimpleController Initialized.")
        
        self.init_camera_params_for_yolo()
        
    def camera_subscription_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._display_lock:
                self._display_image = cv_image.copy()
        except CvBridgeError as e:
            print(e)

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
                new_obstacles.append({'name': obs_name, 'pos': np.array([x, y, z]), 'radius': obs_radius})
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

            if prev_mode != msg.mode:
                self.node.get_logger().info(f"MAVROS Mode changed: {prev_mode} -> {msg.mode}")
            if prev_armed != msg.armed:
                self.node.get_logger().info(f"MAVROS Armed state changed: {prev_armed} -> {msg.armed}")

            if not self.services_ready:
                if self.set_mode_client.service_is_ready() and self.arming_client.service_is_ready():
                    self.services_ready = True
                else:
                    return

            now = self.node.get_clock().now()
            if (now - self.last_state_check_time) < Duration(seconds=1.0):
                return
            self.last_state_check_time = now

            if not msg.connected:
                self.node.get_logger().warn("MAVROS disconnected from FCU!", throttle_duration_sec=5)
                self.arming_req_sent = False
                self.offboard_req_sent = False
                return

            target_mode = 'OFFBOARD'
            if self.step is not None and \
            (self.current_state.mode != target_mode or not self.current_state.armed):
                self.node.get_logger().warn(
                    f"FCU state changed during step execution! Mode: {self.current_state.mode}, Armed: {self.current_state.armed}. "
                    "Controller will not intervene during active step. Manual intervention may be required.",
                    throttle_duration_sec=5)
                self.arming_req_sent = False
                self.offboard_req_sent = False
                return

            if self.current_state.mode != target_mode:

                if not self.offboard_req_sent:
                    self.node.get_logger().info(f"Requesting {target_mode} mode...")
                self.set_mode_client.call_async(SetMode.Request(custom_mode=target_mode))
                self.offboard_req_sent = True
                self.arming_req_sent = False 

            elif not self.current_state.armed:
                if not self.arming_req_sent: 
                    self.node.get_logger().info(f"Mode is {target_mode}. Requesting ARM...")
                self.arming_client.call_async(CommandBool.Request(value=True))
                self.arming_req_sent = True 
                self.offboard_req_sent = False

            else: 
                if self.arming_req_sent or self.offboard_req_sent:
                    self.node.get_logger().info(f"State: ARMED and {target_mode}. Ready to fly.")
                self.arming_req_sent = False
                self.offboard_req_sent = False


    def init_camera_params_for_yolo(self):
        """Метод для инициализации параметров камеры из шага PassGatePencil."""
        
        optical_to_camera_rot = np.array([
             [ 0,  0,  1],
             [-1,  0,  0],
             [ 0, -1,  0]
        ])
        
        camera_width = 640
        camera_height = 480
        optical_to_camera = np.eye(4)
        optical_to_camera[:3, :3] = optical_to_camera_rot
        
        camera_to_base = np.eye(4)
        pitch_rad = math.radians(-20.0)
        camera_to_base[:3, :3] = Rotation.from_euler('y', pitch_rad).as_matrix()
        camera_to_base[:3, 3] = np.array([0.107, 0.0, 0.0])
        
        
        
        camera_vertical_fov_rad = math.radians(86.8)
        camera_f = camera_height / (2.0 * math.tan(camera_vertical_fov_rad / 2.0))
        camera_c_x = camera_width / 2.0
        camera_c_y = camera_height / 2.0
        
        camera_matrix = np.array([
            [camera_f, 0, camera_c_x],
            [0, camera_f, camera_c_y],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.camera_matrix_pencil = camera_matrix
        self.optical_to_camera_pencil = optical_to_camera
        self.camera_to_base_pencil = camera_to_base
        self.logger.info("Camera parameters for YOLO processing initialized.")

    def process_yolo_detections(self, yolo_bboxes):
        """
        Преобразует обнаружения YOLO в ГЛОБАЛЬНЫЕ координаты и обновляет
        self._detected_obstacles_global.
        yolo_bboxes: список из new_bboxes, каждый элемент:
            [center_x_px, center_y_px, width_px, height_px, distance_m (прямое), d_unused,
            angle_x_rad (отн. центра камеры), angle_y_rad (отн. центра камеры), class_name]
        """
        if not yolo_bboxes:
            if self._detected_obstacles_global: 
                self.logger.info("YOLO: No detections, clearing YOLO-detected obstacles.")
                self._detected_obstacles_global = []
            return

        if self.camera_matrix_pencil is None or self.optical_to_camera_pencil is None or self.camera_to_base_pencil is None or self.origin_pos_abs is None:
            self.logger.warn("YOLO Processor: Camera parameters or origin_pos_abs not initialized. Cannot process detections.")
            return

        new_yolo_obstacles_global = []
        current_pos_local = self.position
        current_orient_quat = self.orientation

        if current_pos_local is None or current_orient_quat is None:
            self.logger.warn("YOLO Processor: Missing drone odometry.")
            return

        try:
            R_local_world_to_base = Rotation.from_quat(current_orient_quat)
            R_base_to_local_world = R_local_world_to_base.inv()
            T_local_world_base = np.eye(4)
            T_local_world_base[:3, :3] = R_base_to_local_world.as_matrix()
            T_local_world_base[:3, 3] = current_pos_local
        except ValueError:
            self.logger.warn("YOLO Processor: Invalid drone orientation for transformations.")
            return

        full_transform_optical_to_local_world = T_local_world_base @ self.camera_to_base_pencil @ self.optical_to_camera_pencil

        fx = self.camera_matrix_pencil[0, 0]
        fy = self.camera_matrix_pencil[1, 1]
        cx = self.camera_matrix_pencil[0, 2]
        cy = self.camera_matrix_pencil[1, 2]

        for bbox_idx, bbox_data in enumerate(yolo_bboxes):
            try:
                center_x_pencilnet = float(bbox_data[0])
                center_y_pencilnet = float(bbox_data[1])
                width_pencilnet_px = float(bbox_data[2])
                height_pencilnet_px = float(bbox_data[3])
                depth_Z_optical = float(bbox_data[4])
                class_name = str(bbox_data[7])

                if depth_Z_optical <= 0.1 or depth_Z_optical > (self.vfh_config.obstacle_max_distance):
                    self.logger.debug(f"YOLO: Obstacle '{class_name}' ignored, depth {depth_Z_optical:.2f}m out of range.")
                    continue


                gate_X_optical = (center_x_pencilnet - cx) * depth_Z_optical / fx
                gate_Y_optical = (center_y_pencilnet - cy) * depth_Z_optical / fy
                gate_Z_optical = depth_Z_optical
                gate_pos_optical = np.array([gate_X_optical, gate_Y_optical, gate_Z_optical])

                self.logger.debug(f"  YOLO Obstacle '{class_name}': Depth={depth_Z_optical:.2f}, PxCoords=[{center_x_pencilnet:.1f},{center_y_pencilnet:.1f}] -> OpticalFrame={vector_to_str(gate_pos_optical)}")

                gate_pos_local_world_h = full_transform_optical_to_local_world @ np.append(gate_pos_optical, 1)
                gate_pos_local_world = gate_pos_local_world_h[:3]
                self.logger.debug(f"  -> LocalWorld (Odom) Frame: {vector_to_str(gate_pos_local_world)}")

                sx_m = (width_pencilnet_px / fx) * depth_Z_optical if fx > 0 else 1.0 
                real_height_m_optical_y = 2 * (height_pencilnet_px / fy) * depth_Z_optical if fy > 0 else 1.0

                final_sxy_m = 2 * 2 * sx_m
                final_sz_m = real_height_m_optical_y
                self.logger.debug(f"  Estimated size for '{class_name}': sx(opt)={sx_m:.2f}, sy(opt)_height={real_height_m_optical_y:.2f}")

                obs_radius_m_xy = final_sxy_m / 4.0

                obs_pos_global_calculated = gate_pos_local_world + self.origin_pos_abs

                new_yolo_obstacles_global.append({
                    'name': class_name,
                    'pos': obs_pos_global_calculated, 
                    'radius': obs_radius_m_xy,       
                    'sz': final_sz_m,                 
                    'source': 'yolo'                  
                })
                
                GREEN = '\033[92m'
                self.logger.info(f"{GREEN}YOLO Obstacle Processed: '{class_name}' GlobalPos={vector_to_str(obs_pos_global_calculated)} RadiusXY={obs_radius_m_xy:.2f} SZ={final_sz_m:.2f}")
                self.logger.info(f"{GREEN} GlobalPos={vector_to_str(current_pos_local)}")
                

            except Exception as e:
                self.logger.error(f"Error processing YOLO bbox {bbox_idx} data {bbox_data}: {e}", exc_info=True)
                continue 
        current_non_yolo_obstacles = [obs for obs in self._detected_obstacles_global if obs.get('source') != 'yolo']
        self._detected_obstacles_global = current_non_yolo_obstacles + new_yolo_obstacles_global

        if new_yolo_obstacles_global:
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            self.logger.warn(f"{YELLOW}YOLO UPDATED {len(new_yolo_obstacles_global)} OBSTACLES! Total: {len(self._detected_obstacles_global)}{RESET}")
        elif not new_yolo_obstacles_global and any(obs.get('source') == 'yolo' for obs in self._detected_obstacles_global):

            self.logger.info("YOLO: No new detections, previously YOLO-detected obstacles cleared.")

    def timer_callback_img(self):
        
        img_to_process = None
        with self._display_lock:
            if self._display_image is not None:
                img_to_process = self._display_image.copy()

        if img_to_process is not None:
            annotated_image, bboxes = get_bbox_metrics(cv2.resize(img_to_process, (640, 480)), model, True)
            annotated_image = cv2.resize(annotated_image, (1280, 720))
            
            new_bboxes = []
            for box in bboxes:
                new_bboxes.append([box["center_x"], box["center_y"], box["width"], box["height"], box["distance"], box["angle_x_rad"], box["angle_y_rad"], box['class']])
            bboxes = new_bboxes
            self.bboxes = bboxes
            self.process_yolo_detections(bboxes)
            if bboxes:
                for i in range(len(bboxes)):
                    cv2.circle(annotated_image, (int(bboxes[i][0] / 640 * 1280), int(bboxes[i][1] / 480 * 720)), 10, (0,0,255), 10)
            with self._display_lock:
                self._display_image = annotated_image

    def _gui_loop(self):
        """Цикл для отображения изображений в отдельном потоке."""
        while rclpy.ok():
            img_to_show = None
            with self._display_lock:
                if self._display_image is not None:
                    img_to_show = self._display_image.copy()

            if img_to_show is not None:
                cv2.imshow("SimpleController GUI", img_to_show)
            key = cv2.waitKey(20)
            if key == 27:
                 break
        cv2.destroyAllWindows()
        self.logger.info("GUI thread finished.")


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
        """
        Рассчитывает команду TwistStamped с linear в СК 'odom' и angular.z в СК 'base_link'.
        Работает с ЛОКАЛЬНОЙ одометрией и ОТНОСИТЕЛЬНОЙ целью.
        Включает логику VFH для обхода препятствий и простую вертикальную реакцию.
        """
        cfg = self.vfh_config
        self._current_markers = []

        current_pos_local = self.position 
        current_orient_quat = self.orientation 
        current_vel_body = self.velocity

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
        if valleys: 
            self.logger.debug(f"Found {len(valleys)} valleys.")
        else: 
            self.logger.debug("No valleys found.")

        if not valleys:
            self.logger.error("VFH: No safe valleys found! Stopping XY motion.", throttle_duration_sec=1.0)
            ts = TwistStamped()
            ts.header.stamp = self.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            ts.twist.linear.z = self.calculate_z_velocity(target_pos_relative[2])

            ts.twist.linear.x = 0.0
            ts.twist.linear.y = 0.0
            ts.twist.angular.z = 0.0
            self.last_twist_cmd = ts 
            return ts

        target_angle_in_local_frame = math.atan2(goal_vector_local_frame[1], goal_vector_local_frame[0])
        target_angle_error = normalize_angle(target_angle_in_local_frame - current_yaw_local)

        chosen_valley = self._select_best_valley(valleys, target_angle_error)
        if chosen_valley is None: 
            self.logger.error("VFH: Could not select a valley (should not happen if valleys exist)! Stopping XY.", throttle_duration_sec=1.0)
            ts = TwistStamped()
            ts.header.stamp = self.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            ts.twist.linear.z = self.calculate_z_velocity(target_pos_relative[2])
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

        vx_local_world = target_speed_forward * math.cos(current_yaw_local)
        vy_local_world = target_speed_forward * math.sin(current_yaw_local)

        vz_local_world_nominal = self.calculate_z_velocity(target_pos_relative[2])

        force_vertical_evasion = False
        evasion_vz = 0.0
        vertical_evasion_dist_xy = 25.0
        vertical_evasion_angle_thresh = math.radians(20.0) 

        if obstacles_local_to_base:
            for obs in obstacles_local_to_base:
                obs_pos_base = obs['pos_local']
                dist_xy_base = math.hypot(obs_pos_base[0], obs_pos_base[1]) 

                if dist_xy_base < vertical_evasion_dist_xy:
                    obs_angle_base = math.atan2(obs_pos_base[1], obs_pos_base[0])
                    if abs(normalize_angle(obs_angle_base - chosen_angle_local_to_base)) < vertical_evasion_angle_thresh:
                        self.logger.warn(f"VERTICAL EVASION CHECK: Obstacle '{obs.get('name', 'N/A')}' close (distBaseXY={dist_xy_base:.1f}m) and near chosen path!")

                        if obs_pos_base[2] >= -3.0:
                            target_vz = cfg.max_z_velocity
                            evasion_vz = max(evasion_vz, target_vz) if not force_vertical_evasion else max(evasion_vz, target_vz)
                            self.logger.warn(f"  -> EVADING UP! (Obstacle Z_base: {obs_pos_base[2]:.1f} >= -1.0). New evasion_vz={evasion_vz:.2f}")
                        else:
                            target_vz = -cfg.max_z_velocity 
                            evasion_vz = min(evasion_vz, target_vz) if not force_vertical_evasion else min(evasion_vz, target_vz)
                            self.logger.warn(f"  -> EVADING DOWN! (Obstacle Z_base: {obs_pos_base[2]:.1f} < -1.0). New evasion_vz={evasion_vz:.2f}")
                        force_vertical_evasion = True

        if force_vertical_evasion:
            self.logger.warn(f"VERTICAL EVASION ACTIVE: Overriding Z velocity to {evasion_vz:.2f}")
            vz_local_world = evasion_vz
        else:
            vz_local_world = vz_local_world_nominal

        ts = TwistStamped()
        ts.header.stamp = self.node.get_clock().now().to_msg()
        ts.header.frame_id = 'odom'

        ts.twist.linear.x = vx_local_world
        ts.twist.linear.y = vy_local_world
        ts.twist.linear.z = vz_local_world 

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

            z_threshold = 5.0 
            if abs(pos_local_base[2]) > z_threshold:
                self.logger.debug(f"Obstacle {obs_global_data.get('name','N/A')} ignored (Z difference |{pos_local_base[2]:.1f}| > {z_threshold})")
                continue

            dist_xy = math.sqrt(dist_xy_sq)
            local_obstacles_in_base.append({
                'pos_local': pos_local_base,
                'radius': obs_radius,
                'dist_xy': dist_xy
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
            if dist_xy < cfg.robot_radius: 
                continue
            eff_r = (obs_r + cfg.robot_radius) * cfg.safety_margin_factor
            try:
                phi_arg = eff_r / dist_xy
                if phi_arg >= 1.0: phi = math.pi / 2.0
                elif phi_arg <= -1.0: phi = math.pi / 2.0
                else: phi = math.asin(phi_arg)
            except ValueError: phi = math.pi / 2.0
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
            try: 
                self.viz_publisher.publish(marker_array)
            except Exception as e: 
                self.logger.error(f"Failed to publish viz markers: {e}")
            self._current_markers = []

    def _create_marker(self, ns: str, id: int, type: int, scale: Vector3, color: ColorRGBA, pose: Pose | None = None, points: list[Point] | None = None, text: str = "", frame_id: str = "odom", duration_sec: float = 0.2) -> Marker: 
        marker = Marker(header=Header(stamp=self.node.get_clock().now().to_msg(), frame_id=frame_id), ns=ns, id=id, type=type, action=Marker.ADD, scale=scale, color=color, lifetime=Duration(seconds=duration_sec).to_msg())
        if pose: 
            marker.pose = pose
        if points: 
            marker.points = points
        if text: 
            marker.text = text
        return marker

    def add_obstacle_markers(self, marker_list: list, local_obstacles_in_base: list, robot_radius: float): pass 

    def add_histogram_markers(self, marker_list: list, histogram: np.ndarray, robot_pos_local: np.ndarray, robot_yaw_local: float): pass 

    def add_direction_markers(self, marker_list: list, robot_pos_local: np.ndarray, robot_yaw_local: float, target_angle_local: float | None, chosen_angle_world: float | None): pass


# --- Takeoff Step ---
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
    log_level = LoggingSeverity.INFO
    node = Node('vfh_obstacle_controller')
    node.get_logger().set_level(log_level)
    node.get_logger().info(f"Logging level set to {log_level.name}")

    original_waypoints = [
        [ 246.0, 225.0, 194.0],
        [ 250.0, 175.0, 197.0], 
        [ 220.0, 165.0, 196.0],
        [ 200.0, 175.0, 196.0],
        [ 190.0, 165.0, 196.0],
        [ 170.0, 200.0, 196.0],
        [ 190.0, 220.0, 196.0],
        [ 150.0, 220.0, 196.0],
        [ 140.0, 225.0, 196.0],
    ]

    node.get_logger().info("Assuming Odometry provides LOCAL coordinates relative to start.")
    node.get_logger().info("Converting Absolute Waypoints to Relative Targets for VFHNavTo.")

    origin_pos_abs = np.array(original_waypoints[0])
    relative_step_targets = []
    for i in range(1, len(original_waypoints)):
        wp_abs = np.array(original_waypoints[i])
        
        relative_target = wp_abs
        relative_step_targets.append(relative_target)
        node.get_logger().info(f"Calculated Relative Target {i}: {vector_to_str(relative_target)}")

    nav_tolerance_xy = 1.0
    nav_tolerance_z = 0.5
    takeoff_tolerance = 0.5
    climb_rate = 1.0

    steps: list[Step] = []

    relative_takeoff_alt = relative_step_targets[0][2]
    node.get_logger().info(f"Desired relative takeoff altitude: {relative_takeoff_alt:.2f}m")

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
        simple_controller = SimpleController(node, steps, control_frequency=20.0, origin_pos_abs=[0,0,0])
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
                for _ in range(3): simple_controller.velocity_publisher.publish(ts)
                time.sleep(0.05)
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
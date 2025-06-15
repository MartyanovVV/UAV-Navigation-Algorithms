#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO

# --- DwaConfigStandalone ---
class DwaConfigStandalone:
    def __init__(self):
        self.robot_radius: float = 0.5
        self.max_speed: float = 15.0
        self.min_speed: float = 0.0
        self.max_yaw_rate: float = 4.5
        self.max_accel: float = 5.0
        self.max_dyaw_rate: float = 2.5
        self.v_resolution: float = 0.05
        self.yaw_rate_resolution: float = 0.01
        self.dt: float = 0.05
        self.predict_time: float = 2.5
        self.weight_heading: float = 0.2
        self.weight_clearance: float = 0.4
        self.weight_velocity: float = 0.4
        self.obstacle_max_distance: float = 5.0
        self.kp_z: float = 0.9
        self.max_vz: float = 5.0
    
        self.drone_vertical_clearance_half_height: float = self.robot_radius * 1.0
        
        self.deceleration_distance_xy: float = 3.0
        self.min_approach_speed_xy: float = 0.5

# --- Controller ABC ---
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
    def node(self) -> Node: pass
    @property
    @abstractmethod
    def logger(self): pass
    @property
    @abstractmethod
    def dwa_config(self) -> DwaConfigStandalone: pass
    @property
    @abstractmethod
    def current_state(self) -> State: pass
    @property
    @abstractmethod
    def arming_client(self) -> rclpy.client.Client: pass
    @property
    @abstractmethod
    def initial_pos_enu_abs(self) -> np.ndarray | None: pass

# --- Step ABC ---
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

def _normalize_angle_util(angle: float) -> float:
    while angle > math.pi: angle -= 2.0*math.pi
    while angle < -math.pi: angle += 2.0*math.pi
    return angle

# --- SimpleController (Merged DWA and YOLO/Camera Features) ---
class SimpleController(Controller):
    def __init__(self, node: Node, steps: list[Step], initial_global_pos_enu_abs: np.ndarray | None = None, control_frequency: float = 20.0):
        self._node_obj = node
        self.mission_steps_list = steps
        self.current_step: Step | None = None
        self._current_pose_local: Pose | None = None
        self._current_velocity_body: Twist | None = None
        
        self._current_goal_pose_for_pos_ctrl: Pose = Pose() 
        self._current_goal_pose_for_pos_ctrl.orientation.w = 1.0
        
        self._detected_obstacles_global_list: list = []
        self._internal_current_state = State()
        self._internal_dwa_config = DwaConfigStandalone()
        self.is_arming_req_sent = False
        self.is_offboard_req_sent = False
        self.last_state_check_time = self._node_obj.get_clock().now()
        self._state_access_lock = threading.Lock()
        
        self._initial_pos_enu_abs_internal: np.ndarray | None = None
        if initial_global_pos_enu_abs is not None:
            self._initial_pos_enu_abs_internal = np.array(initial_global_pos_enu_abs, dtype=float)
        
        self.model = YOLO('./tree_only_1-20.pt')
        if not self.model:
            self.logger.error("YOLO model not loaded! Check path: './classes_freezed_1-10.pt'")
            raise FileNotFoundError("YOLO model file not found.")

        self._image: np.ndarray | None = None
        self._display_image: np.ndarray | None = None
        self._display_lock = threading.Lock()
        self.bridge = CvBridge()
        
        self.camera_matrix_pencil: np.ndarray | None = None
        self.optical_to_camera_pencil: np.ndarray | None = None
        self.camera_to_base_pencil: np.ndarray | None = None
        self.init_camera_params_for_yolo()

        self.state_cb_group = ReentrantCallbackGroup()
        self._node_obj.get_logger().info("Starting SimpleController with DWA (TwistStamped output) and YOLO detection...")
        mavros_ns = '/mavros'
        qos_cmd_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.pose_cmd_publisher = self._node_obj.create_publisher(PoseStamped, f'{mavros_ns}/setpoint_position/local', qos_cmd_reliable)
        self._velocity_cmd_publisher_stamped = self._node_obj.create_publisher(TwistStamped, f'{mavros_ns}/setpoint_velocity/cmd_vel', qos_cmd_reliable)
        self._node_obj.get_logger().info(f"Using TwistStamped publisher: {self._velocity_cmd_publisher_stamped.topic_name}")

        qos_sensor=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=1)
        qos_state_sub=QoSProfile(reliability=ReliabilityPolicy.RELIABLE,history=HistoryPolicy.KEEP_LAST,depth=1)
        self.odom_sub=self._node_obj.create_subscription(Odometry,f'{mavros_ns}/local_position/odom',self.odom_callback,qos_sensor)
        
        self.obstacle_sub=self._node_obj.create_subscription(String,'/obstacles',self.obstacles_callback,10)
        
        self.state_sub=self._node_obj.create_subscription(State,f'{mavros_ns}/state',self.state_callback,qos_state_sub,callback_group=self.state_cb_group)
        self.set_mode_srv_client=self.create_client_and_wait(SetMode,f'{mavros_ns}/set_mode')
        self._arming_srv_client=self.create_client_and_wait(CommandBool,f'{mavros_ns}/cmd/arming')
        
        qos_viz = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._viz_publisher = self._node_obj.create_publisher(MarkerArray, '~/dwa_viz', qos_viz)

        self.camera_subscription = node.create_subscription(Image, '/camera', self.camera_subscription_callback, qos_sensor)

        timer_period=1.0/control_frequency
        self.control_timer_cb_group=ReentrantCallbackGroup()
        self.control_loop_timer=self._node_obj.create_timer(timer_period,self.timer_callback,callback_group=self.control_timer_cb_group)
        self._node_obj.create_timer(1.0/10.0, self.timer_callback_img)

        self._gui_thread = threading.Thread(target=self._gui_loop, daemon=True)
        self._gui_thread.start()

        self._node_obj.get_logger().info("SimpleController Initialized.")

    @property
    def detected_obstacles_global(self) -> list: return self._detected_obstacles_global_list
    @property
    def node(self) -> Node: return self._node_obj
    @property
    def logger(self): return self._node_obj.get_logger()
    @property
    def dwa_config(self) -> DwaConfigStandalone: return self._internal_dwa_config
    @property
    def current_state(self) -> State: return self._internal_current_state
    @property
    def arming_client(self) -> rclpy.client.Client: return self._arming_srv_client
    @property
    def initial_pos_enu_abs(self) -> np.ndarray | None: return self._initial_pos_enu_abs_internal
    @property
    def velocity_publisher(self) -> rclpy.publisher.Publisher: return self._velocity_cmd_publisher_stamped
    @property
    def viz_publisher(self) -> rclpy.publisher.Publisher: return self._viz_publisher


    def create_client_and_wait(self, srv_type, srv_name):
        client = self._node_obj.create_client(srv_type, srv_name)
        while not client.wait_for_service(timeout_sec=1.0) and rclpy.ok():
            self._node_obj.get_logger().info(f'{srv_name} service not available, waiting...')
        if not rclpy.ok(): raise SystemExit(f"RCLPY shutdown during service wait for {srv_name}")
        self._node_obj.get_logger().info(f'{srv_name} service available.')
        return client

    def odom_callback(self, msg: Odometry):
        self._current_pose_local = msg.pose.pose
        self._current_velocity_body = msg.twist.twist
        current_goal = self._current_goal_pose_for_pos_ctrl
        if current_goal.orientation.w == 0.0 and current_goal.orientation.x == 0.0 and \
           current_goal.orientation.y == 0.0 and current_goal.orientation.z == 0.0:
            if self._current_pose_local:
                self._current_goal_pose_for_pos_ctrl = copy(self._current_pose_local)
                cg_orient = self._current_goal_pose_for_pos_ctrl.orientation
                if cg_orient.w==0.0 and cg_orient.x==0.0 and cg_orient.y==0.0 and cg_orient.z==0.0: cg_orient.w=1.0
                self._node_obj.get_logger().debug(f"Initial goal_pose_for_pos_ctrl set from local odom.")

    def obstacles_callback(self, msg: String):
        """
        Parses obstacle data from a String message and adds it to the global obstacles list.
        Expected format: "name x y z sx sy sz"
        """
        new_obstacles_from_string = []
        parts = msg.data.strip().split()
        
        if not parts:
            if any(obs.get('source') == 'string' for obs in self._detected_obstacles_global_list):
                self.logger.info("External obstacles cleared (received empty message).")
            self._detected_obstacles_global_list = [obs for obs in self._detected_obstacles_global_list if obs.get('source') != 'string']
            return
            
        if len(parts) == 7:
            try:
                obs_name = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                sx, sy, sz = float(parts[4]), float(parts[5]), float(parts[6])
                
                obs_radius = max(sx, sy) / 2.0 
                
                new_obstacles_from_string.append({
                    'name': obs_name,
                    'pos': np.array([x, y, z]),
                    'radius': obs_radius,
                    'sz': sz, 
                    'source': 'string' 
                })
                self.logger.debug(f"Parsed external obstacle: Name='{obs_name}', Pos={vector_to_str(np.array([x,y,z]))}, Scale=[{sx:.2f},{sy:.2f},{sz:.2f}], Radius={obs_radius:.2f}")

            except ValueError as e:
                self.logger.error(f"Error parsing external obstacle data '{msg.data}': {e}")
                self._detected_obstacles_global_list = [obs for obs in self._detected_obstacles_global_list if obs.get('source') != 'string']
                return
        else:
            self.logger.warn(f"Invalid external obstacle message format. Expected 7 parts, got {len(parts)}. Data: '{msg.data}'")
            self._detected_obstacles_global_list = [obs for obs in self._detected_obstacles_global_list if obs.get('source') != 'string']
            return

        self._detected_obstacles_global_list = [obs for obs in self._detected_obstacles_global_list if obs.get('source') != 'string']
        self._detected_obstacles_global_list.extend(new_obstacles_from_string)

        if new_obstacles_from_string:
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            num_obs = len([o for o in self._detected_obstacles_global_list if o.get('source') == 'string'])
            obs_details = [f"'{o.get('name', 'N/A')}' at {vector_to_str(o['pos'])} R={o['radius']:.2f} SZ={o['sz']:.2f}" 
                           for o in new_obstacles_from_string[:2]] 
            if num_obs > 2: obs_details.append("...")
            self.logger.warn(f"{YELLOW}EXTERNAL OBSTACLE(S) DETECTED! Count: {num_obs}. Details: {'; '.join(obs_details)}{RESET}")
        elif not new_obstacles_from_string and any(obs.get('source') == 'string' for obs in self._detected_obstacles_global_list):
            GREEN = '\033[92m'
            RESET = '\033[0m'
            self.logger.info(f"{GREEN}EXTERNAL OBSTACLES CLEARED (valid empty or no obstacle message received).{RESET}")

    def _publish_zero_twist_stamped(self):
        ts_zero = TwistStamped()
        ts_zero.header.stamp = self.node.get_clock().now().to_msg()
        ts_zero.header.frame_id = 'odom' 
        self.velocity_publisher.publish(ts_zero)

    def state_callback(self, msg: State):
        with self._state_access_lock: self._internal_current_state = msg
        now = self._node_obj.get_clock().now()
        if not self.set_mode_srv_client.service_is_ready() or not self._arming_srv_client.service_is_ready(): return
        if (now - self.last_state_check_time) < Duration(seconds=1.5): return
        self.last_state_check_time = now
        if msg.mode != "OFFBOARD":
            if not self.is_offboard_req_sent:
                self._node_obj.get_logger().warn(f"Mode is {msg.mode}. Requesting OFFBOARD...")
                req = SetMode.Request(custom_mode="OFFBOARD")
                future = self.set_mode_srv_client.call_async(req)
                future.add_done_callback(self._set_mode_response_callback)
                self.is_offboard_req_sent = True
                self.is_arming_req_sent = False
        elif not msg.armed:
            if not self.is_arming_req_sent:
                self._node_obj.get_logger().warn("Mode OFFBOARD, not armed. Sending setpoints & ARM...")
                for _ in range(10): self._publish_zero_twist_stamped()
                time.sleep(0.02)
                req = CommandBool.Request(value=True)
                future = self._arming_srv_client.call_async(req)
                future.add_done_callback(self._arming_response_callback)
                self.is_arming_req_sent = True
        else: 
            if self.is_offboard_req_sent or self.is_arming_req_sent: self._node_obj.get_logger().info("State: ARMED and OFFBOARD. Ready.")
            self.is_offboard_req_sent=False
            self.is_arming_req_sent=False

    def _set_mode_response_callback(self, future):
        try:
            response = future.result()
            self._node_obj.get_logger().info(f"SetMode request result: mode_sent={response.mode_sent}")
            if not response.mode_sent: self.is_offboard_req_sent = False
        except Exception as e: self._node_obj.get_logger().error(f'SetMode service call failed: {e}')
        self.is_offboard_req_sent = False

    def _arming_response_callback(self, future):
        try:
            response = future.result()
            self._node_obj.get_logger().info(f"Arming request result: success={response.success}, result={response.result}")
            if not response.success: self.is_arming_req_sent = False
        except Exception as e: self._node_obj.get_logger().error(f'Arming service call failed: {e}')
        self.is_arming_req_sent = False
    
    def timer_callback(self):
        if not self._internal_current_state.armed or self._internal_current_state.mode != "OFFBOARD":
            if self._current_pose_local is not None and self._current_goal_pose_for_pos_ctrl.orientation.w != 0.0 :
                ps=PoseStamped()
                ps.header.stamp=self._node_obj.get_clock().now().to_msg()
                ps.header.frame_id="odom"
                ps.pose=self._current_goal_pose_for_pos_ctrl
                self.pose_cmd_publisher.publish(ps)
            self._publish_zero_twist_stamped()
            return

        if self._current_pose_local is None:
            self._publish_zero_twist_stamped()
            self._node_obj.get_logger().warn("Timer: Odom (pose) not ready, zero TwistStamped.", throttle_duration_sec=2.0)
            return

        if self.current_step is None:
            if not self.mission_steps_list:
                self._node_obj.get_logger().info("Mission complete. Sending zero TwistStamped.")
                self._publish_zero_twist_stamped()
                return
            self.current_step = self.mission_steps_list.pop(0)
            self._node_obj.get_logger().info(f"--- Starting step: {self.current_step} ---")
            try: 
                self.current_step.init(self)
            except Exception as e: 
                self._node_obj.get_logger().error(f"Step '{self.current_step}' init error: {e}\n{traceback.format_exc()}")
                self.current_step=None
                return

        step_completed = False
        if self.current_step:
            try: 
                step_completed = self.current_step.update(self)
            except Exception as e: 
                self._node_obj.get_logger().error(f"Step '{self.current_step}' update error: {e}\n{traceback.format_exc()}")
                self.current_step=None
                self._publish_zero_twist_stamped()
                return
        
        if self.current_step and not self.current_step.uses_velocity_control:
             if self._current_goal_pose_for_pos_ctrl.orientation.w != 0.0:
                pose_msg=PoseStamped()
                pose_msg.header.stamp=self._node_obj.get_clock().now().to_msg()
                pose_msg.header.frame_id="odom"
                pose_msg.pose=self._current_goal_pose_for_pos_ctrl
                self.pose_cmd_publisher.publish(pose_msg)
                self._publish_zero_twist_stamped()
             else:
                self._publish_zero_twist_stamped()

        if step_completed:
            self._node_obj.get_logger().info(f"--- Completed step: {self.current_step} ---")
            self.current_step = None
            self._publish_zero_twist_stamped()

    @property
    def position(self) -> np.ndarray | None:
        if self._current_pose_local: return np.array([self._current_pose_local.position.x, self._current_pose_local.position.y, self._current_pose_local.position.z])
        return None
    @property
    def orientation(self) -> np.ndarray | None:
        if self._current_pose_local: return np.array([self._current_pose_local.orientation.x, self._current_pose_local.orientation.y, self._current_pose_local.orientation.z, self._current_pose_local.orientation.w])
        return None
    @property
    def velocity(self) -> np.ndarray | None:
        if self._current_velocity_body: return np.array([self._current_velocity_body.linear.x,self._current_velocity_body.linear.y,self._current_velocity_body.linear.z,self._current_velocity_body.angular.x,self._current_velocity_body.angular.y,self._current_velocity_body.angular.z])
        return None

    @property
    def goal_position(self) -> np.ndarray | None:
        if self._current_goal_pose_for_pos_ctrl.orientation.w != 0.0: return np.array([self._current_goal_pose_for_pos_ctrl.position.x,self._current_goal_pose_for_pos_ctrl.position.y,self._current_goal_pose_for_pos_ctrl.position.z])
        return None
    @goal_position.setter
    def goal_position(self, value: np.ndarray) -> None:
        if len(value)==3: self._current_goal_pose_for_pos_ctrl.position = Point(x=float(value[0]),y=float(value[1]),z=float(value[2]))
        else: self._node_obj.get_logger().error("goal_position needs 3-elem array.")
    @property
    def goal_yaw(self) -> float | None:
        current_goal_orient = self._current_goal_pose_for_pos_ctrl.orientation
        if current_goal_orient.w==0.0 and current_goal_orient.x==0.0 and current_goal_orient.y==0.0 and current_goal_orient.z==0.0:
            if self.orientation is not None:
                try: return Rotation.from_quat(self.orientation).as_euler('xyz',degrees=False)[2]
                except ValueError: return 0.0
            return None
        quat = np.array([current_goal_orient.x,current_goal_orient.y,current_goal_orient.z,current_goal_orient.w])
        if abs(np.linalg.norm(quat)-1.0)>1e-3:
             if self.orientation is not None:
                try: return Rotation.from_quat(self.orientation).as_euler('xyz',degrees=False)[2]
                except ValueError: self._node_obj.get_logger().warn("Invalid current orientation for goal_yaw fallback.")
                return 0.0
             return 0.0
        try: return Rotation.from_quat(quat).as_euler('xyz',degrees=False)[2]
        except ValueError: self._node_obj.get_logger().warn(f"Invalid goal quat for goal_yaw: {quat}")
        return None
    @goal_yaw.setter
    def goal_yaw(self, value: float) -> None:
        current_goal_pos = self._current_goal_pose_for_pos_ctrl.position
        current_goal_orient = self._current_goal_pose_for_pos_ctrl.orientation
        if self.position is not None and \
           current_goal_pos.x==0.0 and current_goal_pos.y==0.0 and current_goal_pos.z==0.0 and \
           (abs(current_goal_orient.w-1.0)<1e-6 and abs(current_goal_orient.x)<1e-6 and abs(current_goal_orient.y)<1e-6 and abs(current_goal_orient.z)<1e-6 ):
            current_p=self.position
            self._current_goal_pose_for_pos_ctrl.position.x=current_p[0]
            self._current_goal_pose_for_pos_ctrl.position.y=current_p[1]
            self._current_goal_pose_for_pos_ctrl.position.z=current_p[2]
        quat_xyzw = Rotation.from_euler('xyz',[0,0,value]).as_quat()
        self._current_goal_pose_for_pos_ctrl.orientation = Quaternion(x=quat_xyzw[0],y=quat_xyzw[1],z=quat_xyzw[2],w=quat_xyzw[3])
    
    # --- Camera and YOLO Integration ---
    def camera_subscription_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._display_lock:
                self._image = cv_image.copy()
                self._display_image = cv_image.copy()
        except CvBridgeError as e:
            self.logger.error(f"CvBridge Error: {e}")

    def _gui_loop(self):
        """Loop for displaying images in a separate thread."""
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

    def timer_callback_img(self):
        """Processes images with YOLO and updates detected obstacles."""
        img_to_process = None
        with self._display_lock:
            if self._image is not None:
                img_to_process = cv2.resize(self._image, (640, 480)) 
                self._image = None

        if img_to_process is not None:
            annotated_image, bboxes = self.get_bbox_metrics(img_to_process, self.model, show=True)
            if bboxes:
                for box_data in bboxes:
                    center_x_640 = int(box_data['center_x_px'])
                    center_y_640 = int(box_data['center_y_px'])
                    
                    cv2.circle(annotated_image, (center_x_640, center_y_640), 10, (0,0,255), -1)

            display_annotated_image = cv2.resize(annotated_image, (1280, 720)) 
            
            with self._display_lock:
                self._display_image = display_annotated_image

            self.process_yolo_detections(bboxes)

    def get_bbox_metrics(self, image_np: np.ndarray, model: YOLO, show: bool = False):
        """
        Processes an image with YOLO and calculates bbox metrics.
        Returns:
            - annotated_image (np.ndarray): Image with YOLO detections drawn (if show=True).
            - bbox_metrics (list[dict]): List of dictionaries with bbox data.
        """
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
        
        results = model.predict(image_np, conf=0.55, verbose=False)
        
        annotated_image = results[0].plot() if show else image_np.copy()

        image_center_x = image_np.shape[1] / 2 # 640 / 2 = 320
        image_center_y = image_np.shape[0] / 2 # 480 / 2 = 240
        focal_length_px = 380
        
        image_height = 720
        camera_vertical_fov = math.radians(86.8)
        camera_f = image_height / (2 * math.tan(camera_vertical_fov / 2))
        w_real = 2.0
        
        bbox_metrics = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                center_x = (x1 + x2) / 2
                center_y = (y1 + y1) / 2 
                width = x2 - x1
                height = y2 - y1
                
                distance = 585.2 * height / width / width
                distance = (w_real * camera_f) / (width)

                bbox_metrics.append({
                    'center_x_px': center_x,
                    'center_y_px': center_y,
                    'width_px': width,
                    'height_px': height,
                    'distance_m': distance, 
                    'class_name': class_name
                })
        
        return annotated_image, bbox_metrics

    def init_camera_params_for_yolo(self):
        """Initializes camera intrinsic and extrinsic parameters for YOLO processing."""
        optical_to_camera_rot = np.array([
             [ 0,  0,  1],
             [-1,  0,  0],
             [ 0, -1,  0]
        ])
        
        camera_width = 640
        camera_height = 480
        
        camera_vertical_fov_rad = math.radians(86.8)
        camera_f = camera_height / (2.0 * math.tan(camera_vertical_fov_rad / 2.0))
        camera_c_x = camera_width / 2.0
        camera_c_y = camera_height / 2.0
        
        camera_matrix = np.array([
            [camera_f, 0, camera_c_x],
            [0, camera_f, camera_c_y],
            [0, 0, 1]
        ], dtype=np.float64)
        
        camera_to_base_tf = np.eye(4)
        pitch_rad = math.radians(-20.0) 
        camera_to_base_tf[:3, :3] = Rotation.from_euler('y', pitch_rad).as_matrix()
        camera_to_base_tf[:3, 3] = np.array([0.107, 0.0, 0.0])
        
        optical_to_camera_tf = np.eye(4)
        optical_to_camera_tf[:3, :3] = optical_to_camera_rot
        
        self.camera_matrix_pencil = camera_matrix
        self.optical_to_camera_pencil = optical_to_camera_tf
        self.camera_to_base_pencil = camera_to_base_tf
        
        self.logger.info("Camera parameters for YOLO processing initialized.")
        self.logger.info(f"Camera Matrix:\n{self.camera_matrix_pencil}")
        self.logger.info(f"Optical to Camera TF:\n{self.optical_to_camera_pencil}")
        self.logger.info(f"Camera to Base TF:\n{self.camera_to_base_pencil}")


    def process_yolo_detections(self, yolo_bboxes: list):
        """
        Transforms YOLO detections into global coordinates and updates self._detected_obstacles_global_list.
        Each bbox_data in yolo_bboxes is expected to be a dict:
            {'center_x_px', 'center_y_px', 'width_px', 'height_px', 'distance_m', 'class_name'}
        """
        if not yolo_bboxes:
            if any(obs.get('source') == 'yolo' for obs in self._detected_obstacles_global_list):
                self.logger.info("YOLO: No new detections, clearing previously YOLO-detected obstacles.")
            self._detected_obstacles_global_list = [obs for obs in self._detected_obstacles_global_list if obs.get('source') != 'yolo']
            return

        if self.camera_matrix_pencil is None or self.optical_to_camera_pencil is None or \
           self.camera_to_base_pencil is None or self.initial_pos_enu_abs is None:
            self.logger.warn("YOLO Processor: Camera parameters or initial_pos_enu_abs not initialized. Cannot process detections.")
            return

        new_yolo_obstacles_global = []
        current_pos_local = self.position
        current_orient_quat = self.orientation

        if current_pos_local is None or current_orient_quat is None:
            self.logger.warn("YOLO Processor: Missing drone odometry for transformation.")
            return

        try:
            R_local_world_to_base = Rotation.from_quat(current_orient_quat)
            T_local_world_base = np.eye(4)
            T_local_world_base[:3, :3] = R_local_world_to_base.as_matrix()
            T_local_world_base[:3, 3] = current_pos_local
        except ValueError:
            self.logger.warn("YOLO Processor: Invalid drone orientation quaternion.")
            return

        full_transform_optical_to_local_world = T_local_world_base @ self.camera_to_base_pencil @ self.optical_to_camera_pencil

        fx = self.camera_matrix_pencil[0, 0]
        fy = self.camera_matrix_pencil[1, 1]
        cx = self.camera_matrix_pencil[0, 2]
        cy = self.camera_matrix_pencil[1, 2]

        for bbox_data in yolo_bboxes:
            try:
                center_x_px = bbox_data['center_x_px']
                center_y_px = bbox_data['center_y_px']
                width_px = bbox_data['width_px']
                height_px = bbox_data['height_px']
                depth_Z_optical = bbox_data['distance_m'] 
                class_name = bbox_data['class_name']

                if depth_Z_optical <= 0.1 or depth_Z_optical > self.dwa_config.obstacle_max_distance:
                    self.logger.debug(f"YOLO: Obstacle '{class_name}' ignored, depth {depth_Z_optical:.2f}m out of range.")
                    continue

                obs_X_optical = (center_x_px - cx) * depth_Z_optical / fx
                obs_Y_optical = (center_y_px - cy) * depth_Z_optical / fy
                obs_pos_optical = np.array([obs_X_optical, obs_Y_optical, depth_Z_optical])
                self.logger.debug(f"  YOLO Obstacle '{class_name}': PxCoords=[{center_x_px:.1f},{center_y_px:.1f}], Depth={depth_Z_optical:.2f}m -> OpticalFrame={vector_to_str(obs_pos_optical)}")

                obs_pos_local_world_h = full_transform_optical_to_local_world @ np.append(obs_pos_optical, 1)
                obs_pos_local_world = obs_pos_local_world_h[:3]
                self.logger.debug(f"  -> LocalWorld (Odom) Frame: {vector_to_str(obs_pos_local_world)}")

                obs_radius_for_dwa = self.dwa_config.robot_radius * 2.0 
                obs_sz_for_dwa = self.dwa_config.robot_radius * 5.0

                if class_name == "gate":
                    gate_width_real = 2.0 
                    gate_height_real = 3.0 
                    if width_px > 0 and fx > 0:
                        re_estimated_depth = (gate_width_real * fx) / width_px
                        if re_estimated_depth > 0.1 and re_estimated_depth < self.dwa_config.obstacle_max_distance:
                            depth_Z_optical = re_estimated_depth
                            obs_X_optical = (center_x_px - cx) * depth_Z_optical / fx
                            obs_Y_optical = (center_y_px - cy) * depth_Z_optical / fy
                            obs_pos_optical = np.array([obs_X_optical, obs_Y_optical, depth_Z_optical])
                            obs_pos_local_world_h = full_transform_optical_to_local_world @ np.append(obs_pos_optical, 1)
                            obs_pos_local_world = obs_pos_local_world_h[:3]
                            self.logger.debug(f"  YOLO '{class_name}': Re-estimated Depth={depth_Z_optical:.2f}. New LocalPos={vector_to_str(obs_pos_local_world)}")

                    obs_radius_for_dwa = gate_width_real / 2.0
                    obs_sz_for_dwa = gate_height_real
                elif class_name == "tree":
                    obs_radius_for_dwa = 1.0 
                    obs_sz_for_dwa = 5.0 

                obs_pos_global_calculated = obs_pos_local_world + self.initial_pos_enu_abs

                new_yolo_obstacles_global.append({
                    'name': class_name,
                    'pos': obs_pos_global_calculated,
                    'radius': obs_radius_for_dwa,    
                    'sz': obs_sz_for_dwa,            
                    'source': 'yolo'                 
                })
                
                GREEN = '\033[92m'
                self.logger.info(f"{GREEN}YOLO Obstacle Processed: '{class_name}' GlobalPos={vector_to_str(obs_pos_global_calculated)} RadiusXY={obs_radius_for_dwa:.2f} SZ={obs_sz_for_dwa:.2f}m")
                RESET = '\033[0m'
                self.logger.info(f"{GREEN}Current Drone Local Pos={vector_to_str(current_pos_local)}{RESET}")

            except Exception as e:
                self.logger.error(f"Error processing YOLO bbox data {bbox_data}: {e}", exc_info=True)
                continue

        self._detected_obstacles_global_list = [obs for obs in self._detected_obstacles_global_list if obs.get('source') != 'yolo']
        self._detected_obstacles_global_list.extend(new_yolo_obstacles_global)

        if new_yolo_obstacles_global:
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            self.logger.warn(f"{YELLOW}YOLO ADDED/UPDATED {len(new_yolo_obstacles_global)} OBSTACLES! Total unique obstacles: {len(self._detected_obstacles_global_list)}{RESET}")

    def calculate_dwa_velocity_to_twist_stamped(self, target_pos_local_enu: np.ndarray) -> TwistStamped:
        ts_cmd = TwistStamped()
        ts_cmd.header.stamp = self.node.get_clock().now().to_msg()
        ts_cmd.header.frame_id = 'odom' 
        cfg = self.dwa_config
        current_vel_body_frame = self.velocity
        current_pos_local_enu = self.position
        current_orient_quat_local = self.orientation

        if current_vel_body_frame is None or current_pos_local_enu is None or current_orient_quat_local is None:
            self.node.get_logger().warn("DWA_TS: Missing odom. Returning zero TwistStamped.", throttle_duration_sec=2.0)
            return ts_cmd

        current_lin_vel_x_body = current_vel_body_frame[0]
        current_ang_vel_z_body = current_vel_body_frame[5]

        try:
            current_rotation_obj_local = Rotation.from_quat(current_orient_quat_local)
            current_local_yaw_rad = current_rotation_obj_local.as_euler('xyz', degrees=False)[2]
            rot_local_world_to_base = current_rotation_obj_local.inv()
        except ValueError:
            self.node.get_logger().warn("DWA_TS: Invalid local orientation quaternion.")
            return ts_cmd

        goal_delta_local_enu = target_pos_local_enu - current_pos_local_enu
        goal_pos_in_body_frame_flu_3d = rot_local_world_to_base.apply(goal_delta_local_enu)
        goal_pos_in_body_frame_flu_xy = goal_pos_in_body_frame_flu_3d[:2]

        self.node.get_logger().debug(
            f"DWA_TS CALC: TargetLocal={vector_to_str(target_pos_local_enu)}, CurrPosLocal={vector_to_str(current_pos_local_enu)}, "
            f"CurrLocYaw={math.degrees(current_local_yaw_rad):.1f}, GoalInBodyFLU_XY={vector_to_str(goal_pos_in_body_frame_flu_xy)}",
            throttle_duration_sec=0.5)

        dynamic_window = self._calculate_dynamic_window(
            current_lin_vel_x_body, 
            current_ang_vel_z_body, 
            current_pos_local_enu, 
            target_pos_local_enu   
        )
        v_min, v_max, w_min, w_max = dynamic_window
        
        obstacle_points_body_flu_with_radius = self._get_obstacle_points_for_dwa_local(rot_local_world_to_base, current_pos_local_enu)
        
        if obstacle_points_body_flu_with_radius.shape[0] > 0:
             self.node.get_logger().info(
                f"DWA_TS: Obstacles ACTIVE (count={obstacle_points_body_flu_with_radius.shape[0]}) for DWA calculation.", 
                throttle_duration_sec=1.0)

        best_score = -float('inf')
        best_v_body_raw, best_w_body_raw = 0.0, 0.0
        num_trajectories_evaluated = 0
        num_trajectories_rejected_clearance = 0
        
        v_range = np.arange(v_min, v_max + cfg.v_resolution/2, cfg.v_resolution)
        if len(v_range) == 0 and v_min == v_max : v_range = np.array([v_min])
        
        w_range = np.arange(w_min, w_max + cfg.yaw_rate_resolution/2, cfg.yaw_rate_resolution)
        if len(w_range) == 0 and w_min == w_max : w_range = np.array([w_min])

        angle_to_goal_body_flu_debug = math.atan2(goal_pos_in_body_frame_flu_xy[1], goal_pos_in_body_frame_flu_xy[0])
        self.node.get_logger().debug(
            f"DWA_TS Pre-Loop: GoalInBodyFLU_XY={vector_to_str(goal_pos_in_body_frame_flu_xy)}, "
            f"AngleToGoalBodyFLU_Deg={math.degrees(angle_to_goal_body_flu_debug):.1f}, "
            f"V_range samples: {len(v_range)}, W_range samples: {len(w_range)}", throttle_duration_sec=0.5)

        for v_sample in v_range:
            for w_sample in w_range:
                num_trajectories_evaluated += 1
                trajectory_flu = self._simulate_trajectory(v_sample, w_sample)
                if trajectory_flu is None or len(trajectory_flu)==0: 
                    continue
                h_score,c_score,vel_score = self._evaluate_trajectory_dwa(trajectory_flu,v_sample,goal_pos_in_body_frame_flu_xy,obstacle_points_body_flu_with_radius,w_sample)
                if c_score < 0: 
                    num_trajectories_rejected_clearance += 1
                    continue
                current_score = (cfg.weight_heading*h_score + cfg.weight_clearance*c_score + cfg.weight_velocity*vel_score)
                if current_score > best_score: 
                    best_score=current_score
                    best_v_body_raw=v_sample
                    best_w_body_raw=w_sample
        
        self.node.get_logger().debug(
            f"DWA_TS Trajs Summary: Eval={num_trajectories_evaluated}, RejClear={num_trajectories_rejected_clearance}. "
            f"BestRaw: v={best_v_body_raw:.2f}, w={best_w_body_raw:.2f}, score={best_score:.3f}", throttle_duration_sec=0.5)

        vx_body_final, vy_body_final, wz_body_final = 0.0, 0.0, 0.0

        if best_score > -float('inf'):
            wz_body_final = np.clip(best_w_body_raw, -cfg.max_yaw_rate, cfg.max_yaw_rate)
            vx_body_from_dwa = np.clip(best_v_body_raw, cfg.min_speed if best_v_body_raw >= cfg.min_speed else 0.0, cfg.max_speed)
            if abs(best_v_body_raw) < cfg.min_speed and best_v_body_raw != 0:
                 vx_body_from_dwa = np.sign(best_v_body_raw) * cfg.min_speed if abs(best_v_body_raw) > 1e-2 else 0.0
            vx_body_final = vx_body_from_dwa 
        else:
            self.node.get_logger().warn("DWA_TS: No valid DWA trajectory! Zero XY motion in body frame.", throttle_duration_sec=0.5)

        vx_odom = vx_body_final * math.cos(current_local_yaw_rad) - vy_body_final * math.sin(current_local_yaw_rad)
        vy_odom = vx_body_final * math.sin(current_local_yaw_rad) + vy_body_final * math.cos(current_local_yaw_rad)
        
        error_z = target_pos_local_enu[2] - current_pos_local_enu[2]
        vz_odom_cmd = np.clip(cfg.kp_z * error_z, -cfg.max_vz, cfg.max_vz)

        ts_cmd.twist.linear.x = vx_odom
        ts_cmd.twist.linear.y = vy_odom
        ts_cmd.twist.linear.z = vz_odom_cmd
        ts_cmd.twist.angular.x = 0.0
        ts_cmd.twist.angular.y = 0.0
        ts_cmd.twist.angular.z = wz_body_final
        
        self.node.get_logger().info(
            f"DWA_TS Out: Lin(odom) Vx={vx_odom:.2f} Vy={vy_odom:.2f} Vz={vz_odom_cmd:.2f} | AngZ(body)={wz_body_final:.2f}",
            throttle_duration_sec=0.2)
        return ts_cmd
    
    def _calculate_dynamic_window(self, v_curr_body: float, w_curr_body: float, 
                               current_pos_local_enu: np.ndarray, 
                               target_pos_local_enu: np.ndarray) -> tuple:
        cfg = self.dwa_config
        
        Vs = [0.0, cfg.max_speed, -cfg.max_yaw_rate, cfg.max_yaw_rate] 

        Vd=[v_curr_body-cfg.max_accel*cfg.dt, v_curr_body+cfg.max_accel*cfg.dt, 
            w_curr_body-cfg.max_dyaw_rate*cfg.dt, w_curr_body+cfg.max_dyaw_rate*cfg.dt]
        
        v_min_dw_accel = max(Vs[0],Vd[0])
        v_max_dw_accel = min(Vs[1],Vd[1]) 
        w_min_dw=max(Vs[2],Vd[2])
        w_max_dw=min(Vs[3],Vd[3])

        v_max_dw_accel = max(v_min_dw_accel, v_max_dw_accel) 
        w_max_dw = max(w_min_dw, w_max_dw)

        dist_xy_to_target = np.linalg.norm(current_pos_local_enu[:2] - target_pos_local_enu[:2])
        desired_v_xy = cfg.max_speed 

        if cfg.deceleration_distance_xy > 1e-3: 
            if dist_xy_to_target < cfg.deceleration_distance_xy:
                start_speed_for_interp = min(cfg.max_speed, v_max_dw_accel) 
                speed_range = start_speed_for_interp - cfg.min_approach_speed_xy
                
                distance_ratio = dist_xy_to_target / cfg.deceleration_distance_xy 
                distance_ratio = np.clip(distance_ratio, 0.0, 1.0) 
                
                desired_v_xy = cfg.min_approach_speed_xy + distance_ratio * speed_range
                
                critical_approach_dist = max(cfg.deceleration_distance_xy * 0.15, 2.5) 

                if dist_xy_to_target < critical_approach_dist:
                    desired_v_xy = min(desired_v_xy, cfg.min_approach_speed_xy * 1.2) 
                    self.node.get_logger().debug(
                        f"DWA_DW: CRITICAL APPROACH (dist={dist_xy_to_target:.2f} < {critical_approach_dist:.2f}). "
                        f"Forcing desired_v_xy to near min_approach ({desired_v_xy:.2f})",
                        throttle_duration_sec=1.0
                    )

        final_v_max_dw = min(v_max_dw_accel, desired_v_xy)
        final_v_min_dw = max(v_min_dw_accel, 0.0) 
                                               
        final_v_min_dw = min(final_v_min_dw, final_v_max_dw)
        
        self.node.get_logger().debug(
            f"DWA_DW: v_curr={v_curr_body:.2f}, w_curr={w_curr_body:.2f}, dist_target={dist_xy_to_target:.2f}\n"
            f"    Vd=[{Vd[0]:.2f}, {Vd[1]:.2f}, {Vd[2]:.2f}, {Vd[3]:.2f}]\n"
            f"    v_accel=[{v_min_dw_accel:.2f}, {v_max_dw_accel:.2f}], w_accel=[{w_min_dw:.2f}, {w_max_dw:.2f}]\n"
            f"    desired_v_xy_by_dist={desired_v_xy:.2f}\n"
            f"    FINAL_DW: v=[{final_v_min_dw:.2f}, {final_v_max_dw:.2f}], w=[{w_min_dw:.2f}, {w_max_dw:.2f}]",
            throttle_duration_sec=0.5 
        )

        return final_v_min_dw, final_v_max_dw, w_min_dw, w_max_dw

    def _simulate_trajectory(self, v: float, w: float) -> np.ndarray | None:
        cfg=self.dwa_config
        x,y,theta=0.0,0.0,0.0
        traj=[[x,y,theta]]
        curr_t=0.0
        if cfg.predict_time<=0 or cfg.dt<=0: 
            self.node.get_logger().warn("DWA Sim: predict_time or dt zero/neg.")
            return np.array(traj)
        n_steps=0
        max_s=int(cfg.predict_time/cfg.dt)+2
        while curr_t < cfg.predict_time and n_steps < max_s:
            dt_s=cfg.dt
            if abs(w)<1e-5: 
                x_n=x+v*dt_s*math.cos(theta)
                y_n=y+v*dt_s*math.sin(theta)
                th_n=theta
            else: 
                d_th=w*dt_s
                x_n=x+v*dt_s*math.cos(theta+0.5*d_th)
                y_n=y+v*dt_s*math.sin(theta+0.5*d_th)
                th_n=_normalize_angle_util(theta+d_th)
            x,y,theta=x_n,y_n,th_n
            traj.append([x,y,theta])
            curr_t+=dt_s
            n_steps+=1
        if n_steps>=max_s and curr_t<cfg.predict_time: self.node.get_logger().warn("DWA Sim: Max steps reached.")
        return np.array(traj)

    def _normalize_angle(self, angle: float) -> float: return _normalize_angle_util(angle)

    def _get_obstacle_points_for_dwa_local(self, rot_local_world_to_base: Rotation, current_pos_local_enu: np.ndarray) -> np.ndarray:
        """
        Processes global obstacles (from /obstacles topic or YOLO) and converts them
        to the drone's body frame (FLU) for DWA, applying distance and Z-filters.
        Returns a Nx3 numpy array: [[x_body, y_body, effective_radius], ...]
        """
        obstacles_for_dwa_body = []
        cfg = self.dwa_config 

        self.logger.debug(
            f"DWA_OBST_TRACE: Enter _get_obstacle_points_for_dwa_local. "
            f"Num global_obs: {len(self.detected_obstacles_global)}. "
            f"CurrPosLocal: {vector_to_str(current_pos_local_enu)}."
        )

        if current_pos_local_enu is None:
            self.logger.warn("DWA_OBST_TRACE: current_pos_local_enu is None! Cannot process.", throttle_duration_sec=5)
            return np.array([])
        
        initial_pos_abs = self.initial_pos_enu_abs 
        if initial_pos_abs is None: 
            self.logger.warn("DWA_OBST_TRACE: initial_pos_enu_abs is None! Cannot transform.", throttle_duration_sec=5)
            return np.array([])
        
        self.logger.debug(f"DWA_OBST_TRACE: initial_pos_abs: {vector_to_str(initial_pos_abs)}")

        for i_obs, obs_data_global in enumerate(self.detected_obstacles_global): 
            obs_pos_global_enu = obs_data_global['pos']
            
            obs_base_radius = 0.5
            obs_height_sz = 1.0

            if 'scale' in obs_data_global and len(obs_data_global['scale']) >= 3:
                obs_base_radius = max(obs_data_global['scale'][0], obs_data_global['scale'][1]) / 2.0
                obs_height_sz = obs_data_global['scale'][2]
                self.logger.debug(f"  Obs '{obs_data_global.get('name','N/A')}' (source: {obs_data_global.get('source', 'N/A')}) using 'scale': R={obs_base_radius:.2f}, SZ={obs_height_sz:.2f}")
            elif 'radius' in obs_data_global and 'sz' in obs_data_global:
                obs_base_radius = obs_data_global['radius']
                obs_height_sz = obs_data_global['sz']
                self.logger.debug(f"  Obs '{obs_data_global.get('name','N/A')}' (source: {obs_data_global.get('source', 'N/A')}) using 'radius'/'sz': R={obs_base_radius:.2f}, SZ={obs_height_sz:.2f}")
            else:
                self.logger.warn(f"  Obs '{obs_data_global.get('name','N/A')}' has no 'scale' or 'radius'/'sz'. Using default R={obs_base_radius:.2f}, SZ={obs_height_sz:.2f}.")

            self.logger.debug(
                f"DWA_OBST_TRACE: Proc global_obs[{i_obs}]: Name='{obs_data_global.get('name', 'N/A')}', "
                f"PosGlobENU={vector_to_str(obs_pos_global_enu)}, Extracted R={obs_base_radius:.2f}, SZ={obs_height_sz:.2f}"
            )
            
            obs_pos_local_enu = obs_pos_global_enu - initial_pos_abs
            self.logger.debug(f"DWA_OBST_TRACE:  -> ObsPosLocENU: {vector_to_str(obs_pos_local_enu)}")

            delta_local_world_to_obs_enu = obs_pos_local_enu - current_pos_local_enu
            self.logger.debug(f"DWA_OBST_TRACE:  -> DeltaDroneToObsLocENU: {vector_to_str(delta_local_world_to_obs_enu)}")
            
            dist_sq_to_obs_center_local_xy = np.sum(delta_local_world_to_obs_enu[:2]**2)
            if dist_sq_to_obs_center_local_xy > (cfg.obstacle_max_distance + obs_base_radius)**2:
                self.logger.debug(
                    f"DWA_OBST_TRACE:  -> FILTERED (MAX_DIST_XY): DistXY_loc={math.sqrt(dist_sq_to_obs_center_local_xy):.2f}m "
                    f"> CfgMaxDist+R={cfg.obstacle_max_distance + obs_base_radius:.2f}m"
                )
                continue

            pos_in_body_frame_flu_3d = rot_local_world_to_base.apply(delta_local_world_to_obs_enu)
            self.logger.debug(f"DWA_OBST_TRACE:  -> PosInBodyFLU3d (center of obs): {vector_to_str(pos_in_body_frame_flu_3d)}")
            
            obs_center_z_body = pos_in_body_frame_flu_3d[2]
            obs_bottom_z_body = obs_center_z_body - (obs_height_sz / 2.0)
            obs_top_z_body    = obs_center_z_body + (obs_height_sz / 2.0)

            drone_safe_layer_bottom_z_body = -cfg.drone_vertical_clearance_half_height
            drone_safe_layer_top_z_body    =  cfg.drone_vertical_clearance_half_height

            self.logger.debug(
                f"DWA_OBST_TRACE: Z-Filter Details: ObsZ_body(bot,top)=({obs_bottom_z_body:.2f}, {obs_top_z_body:.2f}), "
                f"DroneSafeZ_body(bot,top)=({drone_safe_layer_bottom_z_body:.2f}, {drone_safe_layer_top_z_body:.2f})"
            )

            is_obstacle_vertically_clear = (obs_bottom_z_body > drone_safe_layer_top_z_body) or \
                                           (obs_top_z_body < drone_safe_layer_bottom_z_body)

            if is_obstacle_vertically_clear:
                self.logger.debug(
                    f"DWA_OBST_TRACE:  -> FILTERED (Z_BODY_WITH_SCALE): Obstacle is vertically clear of drone's safe layer."
                )
                continue

            pos_in_body_frame_flu_xy = pos_in_body_frame_flu_3d[:2]
            
            effective_radius_for_dwa = obs_base_radius * 1.1 
            
            obstacles_for_dwa_body.append([pos_in_body_frame_flu_xy[0], pos_in_body_frame_flu_xy[1], effective_radius_for_dwa])
            self.logger.debug(
                f"DWA_OBST_TRACE:  -> ADDED to DWA: PosBodyXY={vector_to_str(pos_in_body_frame_flu_xy)}, EffRadius={effective_radius_for_dwa:.2f}"
            )
            
        if not obstacles_for_dwa_body and self.detected_obstacles_global: 
            self.logger.info("DWA_OBST_TRACE: Global obstacles detected, but NONE relevant/close for DWA after filtering.", throttle_duration_sec=2.0)
        
        self.logger.debug(f"DWA_OBST_TRACE: Exit. Returning {len(obstacles_for_dwa_body)} obstacles for DWA.")
        return np.array(obstacles_for_dwa_body)

    def _evaluate_trajectory_dwa(self, trajectory_flu: np.ndarray, v_sample: float,
                                 goal_pos_in_body_flu_xy: np.ndarray, 
                                 obstacle_points_body_flu_with_radius: np.ndarray,
                                 w_sample: float) -> tuple[float, float, float]:
        cfg = self.dwa_config
        angle_to_goal_body_flu = math.atan2(goal_pos_in_body_flu_xy[1], goal_pos_in_body_flu_xy[0])
        final_heading_trajectory_flu = trajectory_flu[-1,2] 
        heading_error = abs(_normalize_angle_util(angle_to_goal_body_flu - final_heading_trajectory_flu))
        heading_score = (math.cos(heading_error) + 1.0) / 2.0
        min_dist_to_obstacle_edge = float('inf')
        collision_detected = False
        traj_points_xy = trajectory_flu[:,:2] 
        if obstacle_points_body_flu_with_radius.shape[0] > 0:
            for obs_x_body, obs_y_body, obs_actual_radius in obstacle_points_body_flu_with_radius:
                obs_center_body_flu = np.array([obs_x_body, obs_y_body])
                min_dist_sq_traj_to_obs_center = np.min(np.sum((traj_points_xy - obs_center_body_flu)**2, axis=1))
                collision_threshold_sq = (cfg.robot_radius + obs_actual_radius)**2
                if min_dist_sq_traj_to_obs_center <= collision_threshold_sq: 
                    collision_detected=True
                    break 
                margin_this_obs = math.sqrt(min_dist_sq_traj_to_obs_center) - obs_actual_radius - cfg.robot_radius
                min_dist_to_obstacle_edge = min(min_dist_to_obstacle_edge, margin_this_obs)
        if collision_detected: 
            clearance_score = -1.0
        elif obstacle_points_body_flu_with_radius.shape[0]==0 : 
            clearance_score = 1.0
        elif min_dist_to_obstacle_edge == float('inf'): 
            clearance_score = 1.0
        else:
            desired_clearance = cfg.robot_radius * 2.0 
            clearance_score = np.clip(min_dist_to_obstacle_edge / desired_clearance, 0.0, 1.0)
            if min_dist_to_obstacle_edge < 0: 
                clearance_score = 0.0 
        
        velocity_score = np.clip(v_sample/cfg.max_speed if cfg.max_speed>0 else 0.0, 0.0, 1.0)
        return heading_score, clearance_score, velocity_score

# --- Takeoff, Land, Disarm Steps ---
class Takeoff(Step):
    def __init__(self, relative_altitude: float, tolerance: float = 0.3):
        self.relative_altitude=abs(relative_altitude)
        self.tolerance=tolerance
        self._initial_pos_z_local:float|None=None
        self._target_pos_z_local:float|None=None
    @property
    def uses_velocity_control(self) -> bool: return True
    def init(self, controller: Controller) -> None:
        current_pos_local = controller.position
        if current_pos_local is None: raise RuntimeError("Takeoff init: no_pos")
        self._initial_pos_z_local=current_pos_local[2]
        self._target_pos_z_local=self._initial_pos_z_local+self.relative_altitude
        controller.node.get_logger().info(f"Takeoff: StartLocZ={self._initial_pos_z_local:.2f},RelClimb={self.relative_altitude:.2f},TargetLocZ={self._target_pos_z_local:.2f}")
    def update(self, controller: Controller) -> bool:
        current_pos_local = controller.position
        if current_pos_local is None or self._target_pos_z_local is None:
            ts_zero = TwistStamped()
            ts_zero.header.stamp=controller.node.get_clock().now().to_msg()
            ts_zero.header.frame_id='odom'
            controller.velocity_publisher.publish(ts_zero)
            return False
        height_reached = abs(current_pos_local[2]-self._target_pos_z_local)<self.tolerance
        ts_cmd = TwistStamped()
        ts_cmd.header.stamp=controller.node.get_clock().now().to_msg()
        ts_cmd.header.frame_id='odom'
        if height_reached:
            controller.node.get_logger().info(f"Takeoff done: CurZ={current_pos_local[2]:.2f}")
            controller.velocity_publisher.publish(ts_cmd)
            return True
        else:
            kp_z=controller.dwa_config.kp_z
            max_vz=controller.dwa_config.max_vz
            vz_cmd_odom=np.clip(kp_z*(self._target_pos_z_local-current_pos_local[2]),-max_vz,max_vz)
            ts_cmd.twist.linear.z = vz_cmd_odom
            controller.velocity_publisher.publish(ts_cmd)
            return False
    def __str__(self) -> str:return f"Takeoff to rel_alt +{self.relative_altitude:.2f}m (Local)"

class Land(Step):
    def __init__(self, local_target_z: float, tolerance: float = 0.15):
        self.local_target_z=local_target_z
        self.tolerance=tolerance
    @property
    def uses_velocity_control(self) -> bool: return True
    def init(self, controller: Controller) -> None:
        current_pos_local=controller.position
        if current_pos_local is None: raise RuntimeError("Land init: no_pos")
        controller.node.get_logger().info(f"Land: TargetLocZ={self.local_target_z:.2f}. CurLocZ={current_pos_local[2]:.2f}")
    def update(self, controller: Controller) -> bool:
        current_pos_local=controller.position
        if current_pos_local is None: 
            ts_zero=TwistStamped()
            ts_zero.header.stamp=controller.node.get_clock().now().to_msg()
            ts_zero.header.frame_id='odom'
            controller.velocity_publisher.publish(ts_zero)
            return False
        height_reached = abs(current_pos_local[2]-self.local_target_z)<self.tolerance
        ts_cmd = TwistStamped()
        ts_cmd.header.stamp=controller.node.get_clock().now().to_msg()
        ts_cmd.header.frame_id='odom'
        if height_reached: 
            controller.node.get_logger().info(f"Land done: CurLocZ={current_pos_local[2]:.2f}")
            controller.velocity_publisher.publish(ts_cmd)
            return True
        else:
            kp_z=controller.dwa_config.kp_z
            max_vz_land=controller.dwa_config.max_vz*0.5 
            vz_cmd_odom=np.clip(kp_z*(self.local_target_z-current_pos_local[2]),-max_vz_land,max_vz_land)
            ts_cmd.twist.linear.z = vz_cmd_odom
            controller.velocity_publisher.publish(ts_cmd)
            return False
    def __str__(self)->str: return f"Land to LocZ={self.local_target_z:.2f}m"

class Disarm(Step):
    def __init__(self, attempts=5, delay_between_attempts=1.0):
        self.attempts_left=attempts
        self.delay_between_attempts=Duration(seconds=delay_between_attempts)
        self.last_attempt_time:rclpy.time.Time|None=None
        self.disarm_future:rclpy.task.Future|None=None
        self.disarm_req_sent_cycle=False
    @property
    def uses_velocity_control(self) -> bool: return False
    def init(self, controller:Controller) -> None:
        controller.node.get_logger().info(f"Disarm: ({self.attempts_left} attempts). Stop motion.") 
        ts_zero=TwistStamped()
        ts_zero.header.stamp=controller.node.get_clock().now().to_msg()
        ts_zero.header.frame_id='odom'
        controller.velocity_publisher.publish(ts_zero)
        self.last_attempt_time=None
        self.disarm_future=None
        self.disarm_req_sent_cycle=False
    def update(self, controller:Controller) -> bool:
        now=controller.node.get_clock().now()
        if not controller.current_state.armed:
            if self.disarm_req_sent_cycle or self.last_attempt_time is not None: 
                controller.node.get_logger().info("Disarm: Confirmed disarmed.")
                return True
        if self.attempts_left<=0:
            log_fn=controller.node.get_logger().error if controller.current_state.armed else controller.node.get_logger().info
            log_fn(f"Disarm: Attempts out. ARMED: {controller.current_state.armed}.")
            return True
        if self.disarm_future is None or self.disarm_future.done():
            if self.disarm_future and self.disarm_future.done():
                try:
                    resp=self.disarm_future.result()
                    log_fn=controller.node.get_logger().info if resp.success else controller.node.get_logger().warn
                    log_fn(f"Disarm ACK: succ={resp.success},res={resp.result}.")
                except Exception as e: controller.node.get_logger().error(f"Disarm srv ex: {e}")
                self.disarm_future=None
            if self.last_attempt_time is None or (now-self.last_attempt_time)>=self.delay_between_attempts:
                if self.attempts_left>0:
                    controller.node.get_logger().info(f"Disarm: Send cmd (left: {self.attempts_left})...")
                    if controller.arming_client.service_is_ready():
                        self.disarm_future=controller.arming_client.call_async(CommandBool.Request(value=False))
                        self.disarm_req_sent_cycle=True
                        self.attempts_left-=1
                        self.last_attempt_time=now
                    else: controller.node.get_logger().warn("Disarm: Arming srv not ready.")
        return False
    def __str__(self)->str: return "Disarm Motors"

# --- DwaNavTo Step ---
class DwaNavTo(Step):
    def __init__(self, x_local_enu: float, y_local_enu: float, z_local_enu: float, tolerance_xy: float = 1.0, tolerance_z: float = 0.5):
        self.target_pos_local_enu = np.array([x_local_enu, y_local_enu, z_local_enu])
        self.tolerance_xy = tolerance_xy
        self.tolerance_z = tolerance_z
    @property
    def uses_velocity_control(self) -> bool: return True
    def init(self, controller: Controller) -> None:
        controller.node.get_logger().info(f"DwaNavTo (TwistStamped): Init. Target Local ENU: {vector_to_str(self.target_pos_local_enu)}")
    def update(self, controller: Controller) -> bool:
        current_pos_local_enu = controller.position
        if current_pos_local_enu is None:
            controller.node.get_logger().warn("DwaNavTo (TwistStamped): Waiting for pos.", throttle_duration_sec=2.0)
            ts_zero=TwistStamped()
            ts_zero.header.stamp=controller.node.get_clock().now().to_msg()
            ts_zero.header.frame_id='odom'
            controller.velocity_publisher.publish(ts_zero)
            return False
        
        dist_xy = np.linalg.norm(current_pos_local_enu[:2] - self.target_pos_local_enu[:2])
        dist_z = abs(current_pos_local_enu[2] - self.target_pos_local_enu[2])

        if dist_xy < self.tolerance_xy and dist_z < self.tolerance_z:
            controller.node.get_logger().info(f"DwaNavTo (TwistStamped): Target Local {vector_to_str(self.target_pos_local_enu)} reached!")
            ts_zero=TwistStamped()
            ts_zero.header.stamp=controller.node.get_clock().now().to_msg()
            ts_zero.header.frame_id='odom'
            controller.velocity_publisher.publish(ts_zero)
            return True
        
        ts_cmd = controller.calculate_dwa_velocity_to_twist_stamped(self.target_pos_local_enu)
        controller.velocity_publisher.publish(ts_cmd)
        return False
    def __str__(self) -> str: return f"DwaNavTo (TwistStamped) -> Local {vector_to_str(self.target_pos_local_enu)}"

# --- Main Function ---
def main(args=None):
    rclpy.init(args=args)
    node = Node('dwa_controller_yolo_node') 
    
    node.get_logger().set_level(LoggingSeverity.INFO)
    node.get_logger().info(f"Logging level set to {node.get_logger().get_effective_level().name}")

    initial_pos_enu_abs = np.array([549.75, 207.8, 98.515]) 
    node.get_logger().info(f"Initial drone absolute ENU pos: {vector_to_str(initial_pos_enu_abs)}")
    
    raw_waypoints_enu_input = [ 
        [549.75,     207.8,      98.56658], # 0:
        [549.75,     207.8,     105.0],    # 1:
        [560.0,      257.0,     103.0],
        [560.0,      257.0,     100.0],    # 2:
        [570.0,      307.0,     85.0],
        [580.0,      357.0,     80.0],
        [580.0,      366.0,     105.0],    # 2:
        [624.55501,  598.40931,  50.0],    # 3:
        [928.84280,  947.73771,  95.0],    # 4:
        [965.0,     1238.0,     103.0],    # 5:
        [1108.28750, 1443.92409, 105.0],   # 6:
        [933.8,     1701.4,     140.0],    # 7:
        [933.8,     1701.4,     108.5]     # 8:
    ]

    takeoff_target_global_z = raw_waypoints_enu_input[1][2]
    
    nav_tolerance_xy = 1.0
    nav_tolerance_z = 2.5

    steps: list[Step] = []
    
    takeoff_relative_alt = takeoff_target_global_z - initial_pos_enu_abs[2] 
    if takeoff_relative_alt < 0.5: 
        takeoff_relative_alt = 0.5 
    node.get_logger().info(f"Adding Takeoff step to relative altitude: {takeoff_relative_alt:.2f}m")
    steps.append(Takeoff(relative_altitude=takeoff_relative_alt, tolerance=0.5))

    mission_waypoints_enu = raw_waypoints_enu_input[2:] 
    for i, wp_enu_abs_coords_list in enumerate(mission_waypoints_enu): 
        wp_enu_abs_coords_np = np.array(wp_enu_abs_coords_list, dtype=float) 

        local_target_for_nav = wp_enu_abs_coords_np - initial_pos_enu_abs
        
        node.get_logger().info(f"Adding DwaNavTo step {i+1}. AbsTargetENU: {vector_to_str(wp_enu_abs_coords_np)}, LocalTargetENU: {vector_to_str(local_target_for_nav)}")
        steps.append(DwaNavTo(local_target_for_nav[0], local_target_for_nav[1], local_target_for_nav[2], 
                              tolerance_xy=nav_tolerance_xy, tolerance_z=nav_tolerance_z))

    local_target_z_for_landing = 0.1 
    node.get_logger().info(f"Adding Land step to Local Z={local_target_z_for_landing:.2f}m")
    steps.append(Land(local_target_z=local_target_z_for_landing, tolerance=0.15))
    
    node.get_logger().info("Adding Disarm step.")
    steps.append(Disarm(attempts=5, delay_between_attempts=1.0))

    simple_controller = SimpleController(node, steps, initial_global_pos_enu_abs=initial_pos_enu_abs, control_frequency=20.0)
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        node.get_logger().info("Spinning DWA Controller node with YOLO...")
        executor.spin()
    except KeyboardInterrupt: 
        node.get_logger().info("Node stopped by KeyboardInterrupt.")
    except SystemExit as e: 
        node.get_logger().info(f"Node stopped by SystemExit: {e}")
    except Exception as e: 
        node.get_logger().fatal(f"Unhandled exception in main: {e}\n{traceback.format_exc()}")
    finally:
        node.get_logger().info("Shutting down. Sending zero velocity.")
        if 'simple_controller' in locals() and hasattr(simple_controller, 'velocity_publisher') and \
           simple_controller.velocity_publisher is not None and rclpy.ok():
            try: 
                ts_zero = TwistStamped()
                ts_zero.header.stamp = node.get_clock().now().to_msg()
                ts_zero.header.frame_id = 'odom'
                simple_controller.velocity_publisher.publish(ts_zero)
                time.sleep(0.1) 
            except Exception as e_pub: node.get_logger().error(f"Error sending zero velocity: {e_pub}")
        
        if rclpy.ok():
            if 'executor' in locals() and executor: executor.shutdown() 
            if 'node' in locals() and node.context.ok(): node.destroy_node() 
            rclpy.shutdown() 
    node.get_logger().info("Shutdown complete.")

if __name__ == '__main__':
    main()
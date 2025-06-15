#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.logging import LoggingSeverity

from geometry_msgs.msg import Twist, TwistStamped, Point, Pose, Quaternion, Vector3
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

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO
import random

def get_bbox_metrics(image_path, model, show=False):

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
    w_real = 2.0
    
    bbox_metrics = []

    for result in results:
        for box in result.boxes:
            class_index = int(box.cls.item())

            class_name = "unknown"
            if isinstance(model.names, dict):
                class_name = model.names.get(class_index, "unknown")
            elif isinstance(model.names, list) and 0 <= class_index < len(model.names):
                class_name = model.names[class_index]
                
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
                'class': class_name
            })
    if show:
        return annotated_image, bbox_metrics
    return bbox_metrics

model = YOLO('./tree_only_1-20.pt')

# --- Конфигурация движения ---
class MotionConfig:
    def __init__(self):
        self.robot_radius: float = 0.5
        self.max_xy_speed: float = 15.0
        self.max_z_speed: float = 5.0
        self.max_yaw_rate: float = 1.8
        self.xy_control_gain: float = 0.9
        self.z_control_gain: float = 0.9
        self.yaw_control_gain: float = 1.5
        self.target_reach_tolerance_xy: float = 0.7
        self.target_reach_tolerance_z: float = 0.5
        self.dt: float = 0.05

# --- Конфигурация планировщика JPS ---
class JPSPlannerConfig:
    def __init__(self):
        self.grid_resolution: float = 3.0
        self.robot_radius_jps: float = 0.5 
        self.grid_padding_cells: int = 5 
        self.allow_diagonal_movement: bool = True 
        self.heuristic_weight: float = 1.01
        
        self.max_obstacle_detection_distance_for_yolo: float = 20.0
        self.vertical_evasion_z_filter_threshold: float = 30.0

# --- Узел для JPS ---
class NodeJPS:
    """Представляет узел в сетке JPS."""
    def __init__(self, position: tuple[int, int, int], parent=None):
        self.position = position  # Координаты ячейки в сетке (gx, gy, gz)
        self.parent = parent      # Ссылка на родительский узел
        self.g = 0                # Стоимость от старта до текущего узла
        self.h = 0                # Эвристическая стоимость от текущего узла до цели
        self.f = 0                # Общая стоимость (g + h)

    def __eq__(self, other):
        """Сравнение узлов по их позиции."""
        return self.position == other.position

    def __lt__(self, other):
        """Сравнение узлов для использования в heapq (приоритетная очередь)."""
        return self.f < other.f

    def __hash__(self):
        """Хэш узла для использования в set/dict."""
        return hash(self.position)

# --- Планировщик Jump Point Search (3D) ---
class JumpPointSearchPlanner:
    """Реализация алгоритма JPS для 3D сетки."""
    def __init__(self, config: JPSPlannerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.grid: np.ndarray | None = None          # Трехмерная сетка препятствий 0 свободно, 1 препятствие
        self.grid_shape: tuple[int, int, int] | None = None
        self.grid_origin_world: np.ndarray | None = None 
        self.resolution: float = 0.0                
        self.goal_pos_grid: tuple[int,int,int] | None = None

    def _is_valid(self, x: int, y: int, z: int) -> bool:
        """Проверяет, находятся ли координаты ячейки в пределах сетки."""
        if self.grid_shape is None: return False
        return 0 <= x < self.grid_shape[0] and \
               0 <= y < self.grid_shape[1] and \
               0 <= z < self.grid_shape[2]

    def _is_obstacle(self, x: int, y: int, z: int) -> bool:
        """Проверяет, является ли ячейка препятствием (или выходит за границы)."""
        if not self._is_valid(x,y,z) or self.grid is None: return True
        return self.grid[x,y,z] == 1

    def _is_free(self, x: int, y: int, z: int) -> bool:
        """Проверяет, является ли ячейка свободной и находится ли в пределах сетки."""
        return self._is_valid(x,y,z) and not self._is_obstacle(x,y,z)

    def _heuristic(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        """
        Вычисляет эвристическое расстояние (Евклидово) между двумя ячейками сетки,
        переведенное в мировые единицы.
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dz = abs(a[2] - b[2])
        return math.sqrt(dx**2 + dy**2 + dz**2) * self.resolution

    def _get_natural_and_forced_neighbors(self, current_pos: tuple[int,int,int], parent_pos: tuple[int,int,int] | None) -> list[tuple[int,int,int]]:
        """
        Определяет "естественные" и "вынужденные" направления для поиска точек прыжка
        из текущего узла, используя правила прореживания JPS.
        Возвращает список кортежей (dx, dy, dz), представляющих направления.
        """
        px, py, pz = current_pos
        neighbors_directions = []

        # Если это стартовый узел, исследуем всех 26 соседей
        if parent_pos is None:
            for dx_n in [-1, 0, 1]:
                for dy_n in [-1, 0, 1]:
                    for dz_n in [-1, 0, 1]:
                        if dx_n == 0 and dy_n == 0 and dz_n == 0: continue
                        # Если диагональное движение запрещено, пропускаем диагонали
                        if not self.config.allow_diagonal_movement and (abs(dx_n) + abs(dy_n) + abs(dz_n) > 1): continue
                        # Проверяем, что соседняя ячейка свободна
                        if self._is_free(px + dx_n, py + dy_n, pz + dz_n):
                            neighbors_directions.append((dx_n, dy_n, dz_n))
            return list(set(neighbors_directions))

        dir_x = np.sign(px - parent_pos[0]) if px != parent_pos[0] else 0
        dir_y = np.sign(py - parent_pos[1]) if py != parent_pos[1] else 0
        dir_z = np.sign(pz - parent_pos[2]) if pz != parent_pos[2] else 0

        # --- ОСЕВОЕ ДВИЖЕНИЕ ---
        if abs(dir_x) + abs(dir_y) + abs(dir_z) == 1:
            # Естественный сосед: продолжение по оси
            if self._is_free(px + dir_x, py + dir_y, pz + dir_z):
                neighbors_directions.append((dir_x, dir_y, dir_z))

            # Вынужденные соседи для осевого движения (проверяем 'боковые' блокировки)
            if dir_x != 0:
                for dy_f in [-1, 1]: # Проверяем Forced Neighbors в Y-направлении
                    if self._is_free(px, py + dy_f, pz) and self._is_obstacle(px - dir_x, py + dy_f, pz):
                        neighbors_directions.append((0, dy_f, 0)) # Вынужденный осевой сосед
                        if self._is_free(px + dir_x, py + dy_f, pz): neighbors_directions.append((dir_x, dy_f, 0)) # Вынужденный диагональный
                for dz_f in [-1, 1]: # Проверяем Forced Neighbors в Z-направлении
                    if self._is_free(px, py, pz + dz_f) and self._is_obstacle(px - dir_x, py, pz + dz_f):
                        neighbors_directions.append((0, 0, dz_f)) # Вынужденный осевой сосед
                        if self._is_free(px + dir_x, py, pz + dz_f): neighbors_directions.append((dir_x, 0, dz_f)) # Вынужденный диагональный
            # Аналогично для движения по Y
            elif dir_y != 0:
                for dx_f in [-1, 1]:
                    if self._is_free(px + dx_f, py, pz) and self._is_obstacle(px + dx_f, py - dir_y, pz):
                        neighbors_directions.append((dx_f, 0, 0))
                        if self._is_free(px + dx_f, py + dir_y, pz): neighbors_directions.append((dx_f, dir_y, 0))
                for dz_f in [-1, 1]:
                    if self._is_free(px, py, pz + dz_f) and self._is_obstacle(px, py - dir_y, pz + dz_f):
                        neighbors_directions.append((0, 0, dz_f))
                        if self._is_free(px, py + dir_y, pz + dz_f): neighbors_directions.append((0, dir_y, dz_f))
            # Аналогично для движения по Z
            elif dir_z != 0:
                for dx_f in [-1, 1]:
                    if self._is_free(px + dx_f, py, pz) and self._is_obstacle(px + dx_f, py, pz - dir_z):
                        neighbors_directions.append((dx_f, 0, 0))
                        if self._is_free(px + dx_f, py, pz + dir_z): neighbors_directions.append((dx_f, 0, dir_z))
                for dy_f in [-1, 1]:
                    if self._is_free(px, py + dy_f, pz) and self._is_obstacle(px, py + dy_f, pz - dir_z):
                        neighbors_directions.append((0, dy_f, 0))
                        if self._is_free(px, py + dy_f, pz + dir_z): neighbors_directions.append((0, dy_f, dir_z))
        
        # --- ДИАГОНАЛЬНОЕ ДВИЖЕНИЕ ПО ГРАНИ ---
        elif abs(dir_x) + abs(dir_y) + abs(dir_z) == 2:
            # Естественные соседи: компоненты по осям и продолжение по диагонали
            if self._is_free(px + dir_x, py, pz): neighbors_directions.append((dir_x, 0, 0))
            if self._is_free(px, py + dir_y, pz): neighbors_directions.append((0, dir_y, 0))
            if self._is_free(px, py, pz + dir_z): neighbors_directions.append((0, 0, dir_z))

            if self._is_free(px + dir_x, py + dir_y, pz + dir_z):
                neighbors_directions.append((dir_x, dir_y, dir_z)) # Естественное продолжение диагонали
            
            # Вынужденные соседи для диагонального движения по грани
            if dir_x != 0 and dir_y != 0 and dir_z == 0:
                if self._is_free(px + dir_x, py, pz) and self._is_obstacle(px + dir_x, py - dir_y, pz):
                    neighbors_directions.append((dir_x, 0, 0))
                if self._is_free(px, py + dir_y, pz) and self._is_obstacle(px - dir_x, py + dir_y, pz):
                    neighbors_directions.append((0, dir_y, 0))
                
                # Добавлены Forced Neighbors для 2D-диагонального движения (в Z-направлении)
                for dz_f in [-1, 1]:
                    if self._is_free(px, py, pz + dz_f) and \
                       (self._is_obstacle(px - dir_x, py, pz + dz_f) or \
                        self._is_obstacle(px, py - dir_y, pz + dz_f) or \
                        self._is_obstacle(px - dir_x, py - dir_y, pz + dz_f)):
                        neighbors_directions.append((0, 0, dz_f))
                        if self._is_free(px + dir_x, py, pz + dz_f): neighbors_directions.append((dir_x, 0, dz_f))
                        if self._is_free(px, py + dir_y, pz + dz_f): neighbors_directions.append((0, dir_y, dz_f))
                        if self._is_free(px + dir_x, py + dir_y, pz + dz_f): neighbors_directions.append((dir_x, dir_y, dz_f))


            # Аналогично для XZ грани (dir_y = 0)
            elif dir_x != 0 and dir_z != 0 and dir_y == 0:
                if self._is_free(px + dir_x, py, pz) and self._is_obstacle(px + dir_x, py, pz - dir_z):
                    neighbors_directions.append((dir_x, 0, 0))
                if self._is_free(px, py, pz + dir_z) and self._is_obstacle(px - dir_x, py, pz + dir_z):
                    neighbors_directions.append((0, 0, dir_z))
                
                for dy_f in [-1, 1]:
                    if self._is_free(px, py + dy_f, pz) and \
                       (self._is_obstacle(px - dir_x, py + dy_f, pz) or \
                        self._is_obstacle(px, py + dy_f, pz - dir_z) or \
                        self._is_obstacle(px - dir_x, py + dy_f, pz - dir_z)):
                        neighbors_directions.append((0, dy_f, 0))
                        if self._is_free(px + dir_x, py + dy_f, pz): neighbors_directions.append((dir_x, dy_f, 0))
                        if self._is_free(px, py + dy_f, pz + dir_z): neighbors_directions.append((0, dy_f, dir_z))
                        if self._is_free(px + dir_x, py + dy_f, pz + dir_z): neighbors_directions.append((dir_x, dy_f, dir_z))
            
            # Аналогично для YZ грани (dir_x = 0)
            elif dir_y != 0 and dir_z != 0 and dir_x == 0:
                if self._is_free(px, py + dir_y, pz) and self._is_obstacle(px, py + dir_y, pz - dir_z):
                    neighbors_directions.append((0, dir_y, 0))
                if self._is_free(px, py, pz + dir_z) and self._is_obstacle(px, py - dir_y, pz + dir_z):
                    neighbors_directions.append((0, 0, dir_z))
                
                for dx_f in [-1, 1]:
                    if self._is_free(px + dx_f, py, pz) and \
                       (self._is_obstacle(px + dx_f, py - dir_y, pz) or \
                        self._is_obstacle(px + dx_f, py, pz - dir_z) or \
                        self._is_obstacle(px + dx_f, py - dir_y, pz - dir_z)):
                        neighbors_directions.append((dx_f, 0, 0))
                        if self._is_free(px + dx_f, py + dir_y, pz): neighbors_directions.append((dx_f, dir_y, 0))
                        if self._is_free(px + dx_f, py, pz + dir_z): neighbors_directions.append((dx_f, 0, dir_z))
                        if self._is_free(px + dx_f, py + dir_y, pz + dir_z): neighbors_directions.append((dx_f, dir_y, dir_z))

        # --- ГЛАВНАЯ ДИАГОНАЛЬНОЕ ДВИЖЕНИЕ ---
        elif abs(dir_x) + abs(dir_y) + abs(dir_z) == 3:
            # Естественные соседи: все компоненты и продолжение диагонали
            if self._is_free(px + dir_x, py, pz): neighbors_directions.append((dir_x, 0, 0))
            if self._is_free(px, py + dir_y, pz): neighbors_directions.append((0, dir_y, 0))
            if self._is_free(px, py, pz + dir_z): neighbors_directions.append((0, 0, dir_z))
            if self._is_free(px + dir_x, py + dir_y, pz): neighbors_directions.append((dir_x, dir_y, 0))
            if self._is_free(px + dir_x, py, pz + dir_z): neighbors_directions.append((dir_x, 0, dir_z))
            if self._is_free(px, py + dir_y, pz + dir_z): neighbors_directions.append((0, dir_y, dir_z))
            if self._is_free(px + dir_x, py + dir_y, pz + dir_z): neighbors_directions.append((dir_x, dir_y, dir_z))

            # Вынужденные соседи для главной диагонали
            if self._is_free(px, py + dir_y, pz + dir_z) and self._is_obstacle(px - dir_x, py + dir_y, pz + dir_z):
                neighbors_directions.append((0, dir_y, dir_z))
                if self._is_free(px + dir_x, py + dir_y, pz + dir_z): neighbors_directions.append((dir_x, dir_y, dir_z))
            
            # Проверка, является ли компонентная грань XZ вынужденной (из-за блокировки позади Y)
            if self._is_free(px + dir_x, py, pz + dir_z) and self._is_obstacle(px + dir_x, py - dir_y, pz + dir_z):
                neighbors_directions.append((dir_x, 0, dir_z))
                if self._is_free(px + dir_x, py + dir_y, pz + dir_z): neighbors_directions.append((dir_x, dir_y, dir_z))

            # Проверка, является ли компонентная грань XY вынужденной (из-за блокировки позади Z)
            if self._is_free(px + dir_x, py + dir_y, pz) and self._is_obstacle(px + dir_x, py + dir_y, pz - dir_z):
                neighbors_directions.append((dir_x, dir_y, 0))
                if self._is_free(px + dir_x, py + dir_y, pz + dir_z): neighbors_directions.append((dir_x, dir_y, dir_z))

            # Вынужденные осевые соседи (из-за блокировки двух компонент)
            if self._is_free(px + dir_x, py, pz) and \
               self._is_obstacle(px + dir_x, py - dir_y, pz) and \
               self._is_obstacle(px + dir_x, py, pz - dir_z):
                neighbors_directions.append((dir_x, 0, 0))
                if self._is_free(px+dir_x, py+dir_y, pz): neighbors_directions.append((dir_x,dir_y,0))
                if self._is_free(px+dir_x, py, pz+dir_z): neighbors_directions.append((dir_x,0,dir_z))
                if self._is_free(px+dir_x, py+dir_y, pz+dir_z): neighbors_directions.append((dir_x,dir_y,dir_z))

            if self._is_free(px, py + dir_y, pz) and \
               self._is_obstacle(px - dir_x, py + dir_y, pz) and \
               self._is_obstacle(px, py + dir_y, pz - dir_z):
                neighbors_directions.append((0, dir_y, 0))
                if self._is_free(px+dir_x, py+dir_y, pz): neighbors_directions.append((dir_x,dir_y,0))
                if self._is_free(px, py+dir_y, pz+dir_z): neighbors_directions.append((0,dir_y,dir_z))
                if self._is_free(px+dir_x, py+dir_y, pz+dir_z): neighbors_directions.append((dir_x,dir_y,dir_z))

            if self._is_free(px, py, pz + dir_z) and \
               self._is_obstacle(px - dir_x, py, pz + dir_z) and \
               self._is_obstacle(px, py - dir_y, pz + dir_z):
                neighbors_directions.append((0, 0, dir_z))
                if self._is_free(px+dir_x, py, pz+dir_z): neighbors_directions.append((dir_x,0,dir_z))
                if self._is_free(px, py+dir_y, pz+dir_z): neighbors_directions.append((0,dir_y,dir_z))
                if self._is_free(px+dir_x, py+dir_y, pz+dir_z): neighbors_directions.append((dir_x,dir_y,dir_z))

        unique_dirs = []
        for d_item in neighbors_directions:
            if d_item != (0,0,0) and d_item not in unique_dirs:
                unique_dirs.append(d_item)
        
        if not self.config.allow_diagonal_movement:
            unique_dirs = [d for d in unique_dirs if abs(d[0]) + abs(d[1]) + abs(d[2]) == 1]

        return unique_dirs
    
    def _jump(self, current_x: int, current_y: int, current_z: int,
              dx: int, dy: int, dz: int) -> tuple[int, int, int] | None:
        """
        Рекурсивно "прыгает" из (current_x,y,z) в направлении (dx,dy,dz)
        и возвращает координаты jump point или None, если прыжок невозможен.
        """
        next_x, next_y, next_z = current_x + dx, current_y + dy, current_z + dz

        if not self._is_free(next_x, next_y, next_z):
            return None

        if (next_x, next_y, next_z) == self.goal_pos_grid:
            return self.goal_pos_grid
        
        # --- Проверка для ОСЕВОГО движения ---
        if dx != 0 and dy == 0 and dz == 0:
            if (self._is_free(next_x, next_y + 1, next_z) and self._is_obstacle(next_x - dx, next_y + 1, next_z)) or \
               (self._is_free(next_x, next_y - 1, next_z) and self._is_obstacle(next_x - dx, next_y - 1, next_z)):
                return (next_x, next_y, next_z)
            if (self._is_free(next_x, next_y, next_z + 1) and self._is_obstacle(next_x - dx, next_y, next_z + 1)) or \
               (self._is_free(next_x, next_y, next_z - 1) and self._is_obstacle(next_x - dx, next_y, next_z - 1)):
                return (next_x, next_y, next_z)
        # Аналогично для Y
        elif dy != 0 and dx == 0 and dz == 0:
            if (self._is_free(next_x + 1, next_y, next_z) and self._is_obstacle(next_x + 1, next_y - dy, next_z)) or \
               (self._is_free(next_x - 1, next_y, next_z) and self._is_obstacle(next_x - 1, next_y - dy, next_z)):
                return (next_x, next_y, next_z)
            if (self._is_free(next_x, next_y, next_z + 1) and self._is_obstacle(next_x, next_y - dy, next_z + 1)) or \
               (self._is_free(next_x, next_y, next_z - 1) and self._is_obstacle(next_x, next_y - dy, next_z - 1)):
                return (next_x, next_y, next_z)
        # Аналогично для Z
        elif dz != 0 and dx == 0 and dy == 0:
             if (self._is_free(next_x + 1, next_y, next_z) and self._is_obstacle(next_x + 1, next_y, next_z - dz)) or \
                (self._is_free(next_x - 1, next_y, next_z) and self._is_obstacle(next_x - 1, next_y, next_z - dz)):
                 return (next_x, next_y, next_z)
             if (self._is_free(next_x, next_y + 1, next_z) and self._is_obstacle(next_x, next_y + 1, next_z - dz)) or \
                (self._is_free(next_x, next_y - 1, next_z) and self._is_obstacle(next_x, next_y - 1, next_z - dz)):
                 return (next_x, next_y, next_z)

        # --- Проверка для ДИАГОНАЛЬНОГО движения ---
        if dx != 0 and dy != 0 and dz == 0:
            # Рекурсивные вызовы для компонентных прыжков
            if self._jump(next_x, next_y, next_z, dx, 0, 0) is not None or \
               self._jump(next_x, next_y, next_z, 0, dy, 0) is not None:
                return (next_x, next_y, next_z)
            
            # Проверка на вынужденных соседей перпендикулярно плоскости движения (в Z-направлении)
            for dz_f in [-1, 1]:
                if self._is_free(next_x, next_y, next_z + dz_f) and ( \
                   self._is_obstacle(next_x - dx, next_y, next_z + dz_f) or \
                   self._is_obstacle(next_x, next_y - dy, next_z + dz_f) or \
                   self._is_obstacle(next_x - dx, next_y - dy, next_z + dz_f) \
                ): 
                    return (next_x, next_y, next_z)

        elif dx != 0 and dz != 0 and dy == 0:
            if self._jump(next_x, next_y, next_z, dx, 0, 0) is not None or \
               self._jump(next_x, next_y, next_z, 0, 0, dz) is not None:
                return (next_x, next_y, next_z)
            
            # Проверка на вынужденных соседей в Y-направлении
            for dy_f in [-1, 1]:
                if self._is_free(next_x, next_y + dy_f, next_z) and ( \
                   self._is_obstacle(next_x - dx, next_y + dy_f, next_z) or \
                   self._is_obstacle(next_x, next_y + dy_f, next_z - dz) or \
                   self._is_obstacle(next_x - dx, next_y + dy_f, next_z - dz) \
                ): 
                    return (next_x, next_y, next_z)

        elif dy != 0 and dz != 0 and dx == 0:
            if self._jump(next_x, next_y, next_z, 0, dy, 0) is not None or \
               self._jump(next_x, next_y, next_z, 0, 0, dz) is not None:
                return (next_x, next_y, next_z)
            
            # Проверка на вынужденных соседей в X-направлении
            for dx_f in [-1, 1]:
                if self._is_free(next_x + dx_f, next_y, next_z) and ( \
                   self._is_obstacle(next_x + dx_f, next_y - dy, next_z) or \
                   self._is_obstacle(next_x + dx_f, next_y, next_z - dz) or \
                   self._is_obstacle(next_x + dx_f, next_y - dy, next_z - dz) \
                ): 
                    return (next_x, next_y, next_z)

        # Движение по главной диагонали
        elif dx != 0 and dy != 0 and dz != 0:
            if self._jump(next_x, next_y, next_z, dx, 0, 0) is not None or \
               self._jump(next_x, next_y, next_z, 0, dy, 0) is not None or \
               self._jump(next_x, next_y, next_z, 0, 0, dz) is not None or \
               self._jump(next_x, next_y, next_z, dx, dy, 0) is not None or \
               self._jump(next_x, next_y, next_z, dx, 0, dz) is not None or \
               self._jump(next_x, next_y, next_z, 0, dy, dz) is not None:
                return (next_x, next_y, next_z)
            
        return self._jump(next_x, next_y, next_z, dx, dy, dz)
    

    def _identify_successors(self, current_node: NodeJPS) -> list[NodeJPS]:
        """
        Идентифицирует преемников (точки прыжка) для текущего узла.
        """
        successors = []
        parent_pos = current_node.parent.position if current_node.parent else None
        
        pruned_directions = self._get_natural_and_forced_neighbors(current_node.position, parent_pos)

        for dx, dy, dz in pruned_directions:
            jump_point_pos = self._jump(current_node.position[0], current_node.position[1], current_node.position[2], dx, dy, dz)
            
            if jump_point_pos:
                if not self._is_valid(*jump_point_pos):
                    continue

                jp_node = NodeJPS(jump_point_pos, current_node)
                # g-стоимость - это фактическое расстояние от current_node до jump_point_pos
                jp_node.g = current_node.g + self._heuristic(current_node.position, jump_point_pos)
                # h-стоимость - эвристическое расстояние от jump_point_pos до цели
                jp_node.h = self._heuristic(jump_point_pos, self.goal_pos_grid)
                # f-стоимость с учетом веса эвристики
                jp_node.f = jp_node.g + jp_node.h * self.config.heuristic_weight
                successors.append(jp_node)
        return successors
    
    def _reconstruct_path(self, current_node: NodeJPS) -> list[np.ndarray]:
        """
        Восстанавливает путь, состоящий только из Jump Points,
        и переводит их в мировые координаты.
        """
        jump_point_sequence_grid = []
        curr = current_node
        while curr is not None:
            jump_point_sequence_grid.append(curr.position)
            curr = curr.parent
        jump_point_sequence_grid.reverse()

        path_world = []
        for gx, gy, gz in jump_point_sequence_grid:
            wx = self.grid_origin_world[0] + (gx + 0.5) * self.resolution
            wy = self.grid_origin_world[1] + (gy + 0.5) * self.resolution
            wz = self.grid_origin_world[2] + (gz + 0.5) * self.resolution
            path_world.append(np.array([wx, wy, wz]))
        
        return path_world

    def _bresenham3d(self, p1_grid: tuple[int, int, int], p2_grid: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        """
        Реализация алгоритма Брезенхема для 3D, возвращает список ячеек,
        составляющих прямую линию между двумя точками в сетке.
        """
        path = []
        x1, y1, z1 = p1_grid
        x2, y2, z2 = p2_grid
        
        path.append((x1,y1,z1))

        dx_abs, dy_abs, dz_abs = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        sz = 1 if z1 < z2 else -1
        
        if (dx_abs >= dy_abs and dx_abs >= dz_abs):
            err_1, err_2 = 2 * dy_abs - dx_abs, 2 * dz_abs - dx_abs
            for _ in range(dx_abs):
                if err_1 > 0: y1 += sy
                err_1 -= 2 * dx_abs
                if err_2 > 0: z1 += sz
                err_2 -= 2 * dx_abs
                err_1 += 2 * dy_abs
                err_2 += 2 * dz_abs
                x1 += sx
                path.append((x1,y1,z1))
        elif (dy_abs >= dx_abs and dy_abs >= dz_abs):
            err_1, err_2 = 2 * dx_abs - dy_abs, 2 * dz_abs - dy_abs
            for _ in range(dy_abs):
                if err_1 > 0: x1 += sx
                err_1 -= 2 * dy_abs
                if err_2 > 0: z1 += sz
                err_2 -= 2 * dy_abs
                err_1 += 2 * dx_abs
                err_2 += 2 * dz_abs
                y1 += sy
                path.append((x1,y1,z1))
        else:
            err_1, err_2 = 2 * dy_abs - dz_abs, 2 * dx_abs - dz_abs
            for _ in range(dz_abs):
                if err_1 > 0: y1 += sy
                err_1 -= 2 * dz_abs
                if err_2 > 0: x1 += sx
                err_2 -= 2 * dz_abs
                err_1 += 2 * dy_abs
                err_2 += 2 * dx_abs
                z1 += sz
                path.append((x1,y1,z1))
        return path

    def plan(self, start_xyz_world: np.ndarray, goal_xyz_world: np.ndarray,
             obstacles_world_3d: list[dict]) -> list[np.ndarray] | None:
        """
        Планирует путь от стартовой до целевой точки в 3D с учетом препятствий.
        """
        self.resolution = self.config.grid_resolution
        robot_r_jps = self.config.robot_radius_jps 
        padding_dist = self.config.grid_padding_cells * self.resolution

        all_points_x = [start_xyz_world[0], goal_xyz_world[0]]
        all_points_y = [start_xyz_world[1], goal_xyz_world[1]]
        all_points_z = [start_xyz_world[2], goal_xyz_world[2]]

        for obs in obstacles_world_3d:
            inflated_r_xy = obs['radius_xy'] + robot_r_jps
            inflated_h_half_z = obs['half_height_z'] / 2.0 + robot_r_jps
            all_points_x.extend([obs['pos_center'][0] - inflated_r_xy, obs['pos_center'][0] + inflated_r_xy])
            all_points_y.extend([obs['pos_center'][1] - inflated_r_xy, obs['pos_center'][1] + inflated_r_xy])
            all_points_z.extend([obs['pos_center'][2] - inflated_h_half_z, obs['pos_center'][2] + inflated_h_half_z])

        min_x_w, max_x_w = min(all_points_x) - padding_dist, max(all_points_x) + padding_dist
        min_y_w, max_y_w = min(all_points_y) - padding_dist, max(all_points_y) + padding_dist
        min_z_w, max_z_w = min(all_points_z) - padding_dist, max(all_points_z) + padding_dist
        
        self.grid_origin_world = np.array([min_x_w, min_y_w, min_z_w])
        self.grid_shape = (int(math.ceil((max_x_w - min_x_w) / self.resolution)),
                           int(math.ceil((max_y_w - min_y_w) / self.resolution)),
                           int(math.ceil((max_z_w - min_z_w) / self.resolution)))

        if self.grid_shape[0] <= 0 or self.grid_shape[1] <= 0 or self.grid_shape[2] <= 0:
            self.logger.error(f"JPS Planner: Invalid grid dimensions calculated: {self.grid_shape}. Cannot plan.")
            return None
        
        self.logger.info(f"JPS Planner: Grid Size (cells): {self.grid_shape}. Grid Origin (world): {vector_to_str(self.grid_origin_world)}")
        self.grid = np.zeros(self.grid_shape, dtype=np.uint8)

        def world_to_grid_xyz(wx: float, wy: float, wz: float) -> tuple[int, int, int]:
            gx = int(math.floor((wx - self.grid_origin_world[0]) / self.resolution))
            gy = int(math.floor((wy - self.grid_origin_world[1]) / self.resolution))
            gz = int(math.floor((wz - self.grid_origin_world[2]) / self.resolution))
            return max(0, min(gx, self.grid_shape[0] - 1)), \
                   max(0, min(gy, self.grid_shape[1] - 1)), \
                   max(0, min(gz, self.grid_shape[2] - 1))

        for obs in obstacles_world_3d:
            obs_center_w, obs_r_xy, obs_sz_full = obs['pos_center'], obs['radius_xy'], obs['half_height_z']
            inflated_r_xy = obs_r_xy + robot_r_jps
            inflated_h_half_z = obs_sz_full / 2.0 + robot_r_jps
            inflated_obs_z_min_w = obs_center_w[2] - inflated_h_half_z
            inflated_obs_z_max_w = obs_center_w[2] + inflated_h_half_z
            
            min_gx_obs, min_gy_obs, min_gz_obs_approx = world_to_grid_xyz(
                obs_center_w[0] - inflated_r_xy, obs_center_w[1] - inflated_r_xy, inflated_obs_z_min_w)
            max_gx_obs, max_gy_obs, max_gz_obs_approx = world_to_grid_xyz(
                obs_center_w[0] + inflated_r_xy, obs_center_w[1] + inflated_r_xy, inflated_obs_z_max_w)
            
            for igx in range(min_gx_obs, max_gx_obs + 1):
                for igy in range(min_gy_obs, max_gy_obs + 1):
                    cell_center_xy_w_x = self.grid_origin_world[0] + (igx + 0.5) * self.resolution
                    cell_center_xy_w_y = self.grid_origin_world[1] + (igy + 0.5) * self.resolution
                    
                    if (cell_center_xy_w_x - obs_center_w[0])**2 + (cell_center_xy_w_y - obs_center_w[1])**2 <= inflated_r_xy**2:
                        for igz in range(min_gz_obs_approx, max_gz_obs_approx + 1):
                            cell_center_z_w = self.grid_origin_world[2] + (igz + 0.5) * self.resolution
                            if inflated_obs_z_min_w <= cell_center_z_w <= inflated_obs_z_max_w:
                                if self._is_valid(igx,igy,igz):
                                    self.grid[igx, igy, igz] = 1

        start_gx, start_gy, start_gz = world_to_grid_xyz(*start_xyz_world)
        self.goal_pos_grid = world_to_grid_xyz(*goal_xyz_world)

        start_node_pos_grid = (start_gx, start_gy, start_gz)
        start_node = NodeJPS(start_node_pos_grid)
        start_node.g = 0
        start_node.h = self._heuristic(start_node_pos_grid, self.goal_pos_grid)
        start_node.f = start_node.g + start_node.h * self.config.heuristic_weight

        if self._is_obstacle(*start_node.position):
            self.logger.warn(f"JPS Planner: Start position {start_node.position} (grid) is inside an obstacle. Trying to find a nearby free cell...")
            found_new_start = False
            search_radius_cells = 3
            for r_c in range(1, search_radius_cells + 1):
                for dx in range(-r_c, r_c + 1):
                    for dy in range(-r_c, r_c + 1):
                        for dz in range(-r_c, r_c + 1):
                            if abs(dx) + abs(dy) + abs(dz) == 0: continue
                            new_x, new_y, new_z = start_gx + dx, start_gy + dy, start_gz + dz
                            if self._is_free(new_x, new_y, new_z):
                                self.logger.info(f"JPS Planner: Found new free start at {new_x, new_y, new_z} (grid).")
                                start_node.position = (new_x, new_y, new_z)
                                start_node.g = 0
                                start_node.h = self._heuristic(start_node.position, self.goal_pos_grid)
                                start_node.f = start_node.g + start_node.h * self.config.heuristic_weight
                                found_new_start = True
                                break
                        if found_new_start: break
                    if found_new_start: break
            if not found_new_start:
                self.logger.error("JPS Planner: Start in obstacle, no free nearby cell found. Aborting plan.")
                return None

        if self._is_obstacle(*self.goal_pos_grid):
            self.logger.warn(f"JPS Planner: Goal position {self.goal_pos_grid} (grid) is inside an inflated obstacle. Path may lead to edge of obstacle.")

        open_list = [] 
        heapq.heappush(open_list, start_node)
        
        closed_set = {start_node.position: start_node.g} 

        while open_list:
            current_node = heapq.heappop(open_list)

            if current_node.position in closed_set and current_node.g > closed_set[current_node.position]:
                continue

            if current_node.position == self.goal_pos_grid:
                self.logger.info(f"JPS Planner: Path found to goal. Nodes evaluated: {len(closed_set)}")
                return self._reconstruct_path(current_node)

            successors = self._identify_successors(current_node)
            for successor_node in successors:
                if successor_node.position not in closed_set or \
                   successor_node.g < closed_set[successor_node.position]:
                    
                    closed_set[successor_node.position] = successor_node.g
                    heapq.heappush(open_list, successor_node)

        self.logger.warn("JPS Planner: Path not found. Open list exhausted.")
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
    def jps_planner(self) -> JumpPointSearchPlanner: pass
    @property
    @abstractmethod
    def new_obstacles_for_replan(self) -> bool: pass
    @new_obstacles_for_replan.setter
    @abstractmethod
    def new_obstacles_for_replan(self, value: bool) -> None: pass
    @property
    @abstractmethod
    def detected_obstacles_global(self) -> list: pass


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

        self.jps_planner_config = JPSPlannerConfig()
        self.jps_planner_config.robot_radius_jps = self._motion_config.robot_radius + 0.3
        self._jps_planner = JumpPointSearchPlanner(self.jps_planner_config, self.node.get_logger())
        
        self._path_marker_id_counter = 0
        self._new_obstacles_for_replan: bool = False

        self.node.get_logger().info("SimpleController (JPS + YOLO) инициализирован.")
        
        mavros_ns = '/mavros'
        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self._velocity_publisher = node.create_publisher(TwistStamped, f'{mavros_ns}/setpoint_velocity/cmd_vel', qos_cmd)
        
        qos_viz = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._viz_publisher = node.create_publisher(MarkerArray, '~/jps_path_viz', qos_viz)

        qos_odom = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.odom_subscription = node.create_subscription(Odometry, f'{mavros_ns}/local_position/odom', self.odom_callback, qos_odom)
        qos_state_sub = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_subscription = node.create_subscription(State, f'{mavros_ns}/state', self.state_callback, qos_state_sub, callback_group=self.state_callback_group)
        
        qos_obstacles = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.obstacle_subscription = node.create_subscription(String, '/obstacles', self.obstacles_callback, qos_obstacles)
        
        self.bridge = CvBridge()
        self.camera_matrix_pencil: np.ndarray | None = None
        self.optical_to_camera_pencil: np.ndarray | None = None
        self.camera_to_base_pencil: np.ndarray | None = None
        
        self.camera_subscription = node.create_subscription(Image, '/camera', self.camera_subscription_callback, 10)
        
        self._display_image = None
        self._display_lock = threading.Lock()
        self._gui_thread = threading.Thread(target=self._gui_loop, daemon=True)
        self._gui_thread.start()
        
        self._image_processing_active = True
        self._image_processing_thread = threading.Thread(target=self._image_processing_loop, daemon=True)
        self._image_processing_thread.start()
        
        self.init_camera_params_for_yolo()

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
        """
        Обрабатывает сообщения о препятствиях, приходящие в строковом формате (например, от симулятора).
        Обновляет _detected_obstacles_global, но только для препятствий 'sim' источника.
        """
        with self._state_lock:
            current_non_yolo_obstacles = [obs for obs in self._detected_obstacles_global if obs.get('source') != 'yolo']
            new_sim_obstacles_parsed = []
            parts = msg.data.split()
            
            if msg.data.strip() == "": 
                if any(obs.get('source') == 'sim' for obs in self._detected_obstacles_global):
                    self.node.get_logger().info("Simulator obstacles cleared by empty message.")
                    self._detected_obstacles_global = current_non_yolo_obstacles
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
                        new_sim_obstacles_parsed.append({
                            'name': obs_name,
                            'pos': np.array([x_abs, y_abs, z_abs]),
                            'radius': obs_radius_xy,
                            'sz': sz_full,
                            'source': 'sim'
                        })
                    except ValueError as e:
                        self.node.get_logger().error(f"Error parsing simulator obstacle data chunk '{' '.join(obs_chunk)}': {e}")
                        if any(obs.get('source') == 'sim' for obs in self._detected_obstacles_global):
                            self.node.get_logger().warn("Clearing existing simulator obstacles due to parsing error.")
                            self._detected_obstacles_global = current_non_yolo_obstacles
                            self.new_obstacles_for_replan = True
                        return 
                
                old_sim_obstacles = [obs for obs in self._detected_obstacles_global if obs.get('source') == 'sim']
                if str(new_sim_obstacles_parsed) != str(old_sim_obstacles):
                    self._detected_obstacles_global = current_non_yolo_obstacles + new_sim_obstacles_parsed
                    self.new_obstacles_for_replan = True 
                    YELLOW = '\033[93m'
                    RESET = '\033[0m'
                    num_obs = len([o for o in self._detected_obstacles_global if o.get('source') == 'sim'])
                    obs_summary_list = [f"'{o.get('name', 'N/A')}' @ GLOB [{vector_to_str(o['pos'])}]" for o in new_sim_obstacles_parsed[:3]]
                    if num_obs > 3: obs_summary_list.append("...")
                    self.node.get_logger().warn(f"{YELLOW}SIMULATOR OBSTACLES UPDATED! Count: {num_obs}. Details: {'; '.join(obs_summary_list)}{RESET}. Replanning will be triggered if in JPS Nav.")
            elif msg.data.strip():
                self.node.get_logger().warn(f"Invalid simulator obstacle message format: '{msg.data}'. Clearing simulator obstacles.")
                if any(obs.get('source') == 'sim' for obs in self._detected_obstacles_global):
                    self._detected_obstacles_global = current_non_yolo_obstacles
                    self.new_obstacles_for_replan = True

    def init_camera_params_for_yolo(self):
        """Метод для инициализации параметров камеры."""
        
        camera_width = 640
        camera_height = 480
        
        focal_length = 380.0
        cx = camera_width / 2.0
        cy = camera_height / 2.0
        
        self.camera_matrix_pencil = np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        optical_to_camera_rot = np.array([
             [ 0.,  0.,  1.],
             [-1.,  0.,  0.],
             [ 0., -1.,  0.]
        ])
        self.optical_to_camera_pencil = np.eye(4)
        self.optical_to_camera_pencil[:3, :3] = optical_to_camera_rot

        self.camera_to_base_pencil = np.eye(4)
        pitch_rad = math.radians(-20.0)
        self.camera_to_base_pencil[:3, :3] = Rotation.from_euler('y', pitch_rad).as_matrix()
        self.camera_to_base_pencil[:3, 3] = np.array([0.107, 0.0, 0.0])
        
        self.node.get_logger().info("Camera parameters for YOLO processing initialized.")

    def camera_subscription_callback(self, msg: Image):
        """Callback для получения изображений с камеры."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._display_lock:
                self._display_image = cv_image.copy()
        except CvBridgeError as e:
            self.node.get_logger().error(f"CvBridgeError: {e}")

    def _gui_loop(self):
        """Цикл для отображения изображений в отдельном потоке (GUI)."""
        while rclpy.ok():
            img_to_show = None
            with self._display_lock:
                if self._display_image is not None:
                    img_to_show = self._display_image.copy()

            if img_to_show is not None:
                img_to_show_resized = cv2.resize(img_to_show, (1280, 720)) 
                cv2.imshow("Drone Camera Feed (YOLO Detections)", img_to_show_resized)
            
            key = cv2.waitKey(20)
            if key == 27:
                 break
        cv2.destroyAllWindows()
        self.node.get_logger().info("GUI thread finished.")
                        
    def _image_processing_loop(self):
        """
        Цикл для обработки изображений YOLO в отдельном потоке.
        Это предотвращает блокировку основного потока управления.
        """
        if model is None:
            self.node.get_logger().error("YOLO model not loaded. Image processing thread will not run.")
            self._image_processing_active = False

        processing_rate = 5.0
        sleep_duration = 1.0 / processing_rate

        while rclpy.ok() and self._image_processing_active:
            img_to_process = None
            with self._display_lock:
                if self._display_image is not None:
                    img_to_process = self._display_image.copy()

            if img_to_process is not None:
                try:
                    annotated_img, bboxes_metrics = get_bbox_metrics(img_to_process, model, show=True)
                    
                    self.process_yolo_detections(bboxes_metrics) 
                    
                    if annotated_img is not None:
                        with self._display_lock:
                            self._display_image = annotated_img.copy()
                            self._display_image = cv2.resize(annotated_img, (1280, 720)) 
                    
                except Exception as e:
                    self.node.get_logger().error(f"Error in YOLO image processing thread: {e}\n{traceback.format_exc()}")
            
            time.sleep(sleep_duration)
        self.node.get_logger().info("YOLO image processing thread finished.")


    def process_yolo_detections(self, yolo_bboxes_metrics: list[dict]):
        """
        Преобразует обнаружения YOLO (bboxes_metrics) в ГЛОБАЛЬНЫЕ координаты и обновляет
        self._detected_obstacles_global.
        yolo_bboxes_metrics: список словарей, каждый элемент:
            {'center_x_px': float, 'center_y_px': float, 'width_px': float, 'height_px': float,
             'distance': float, 'class': str} # Обновлено: 'distance_m' на 'distance'
        """
        if not yolo_bboxes_metrics:
            with self._state_lock:
                old_yolo_obstacles = [obs for obs in self._detected_obstacles_global if obs.get('source') == 'yolo']
                if old_yolo_obstacles:
                    self.node.get_logger().info("YOLO: No new detections, clearing previously YOLO-detected obstacles.")
                    self._detected_obstacles_global = [obs for obs in self._detected_obstacles_global if obs.get('source') != 'yolo']
                    self.new_obstacles_for_replan = True
            return

        if self.camera_matrix_pencil is None or self.optical_to_camera_pencil is None or self.camera_to_base_pencil is None or self.origin_pos_abs is None:
            self.node.get_logger().warn("YOLO Processor: Camera parameters or origin_pos_abs not initialized. Cannot process detections.")
            return

        new_yolo_obstacles_global = []
        current_pos_local = self.position
        current_orient_quat = self.orientation

        if current_pos_local is None or current_orient_quat is None:
            self.node.get_logger().warn("YOLO Processor: Missing drone odometry. Cannot process detections.")
            return

        try:
            current_rot_from_quat = Rotation.from_quat(current_orient_quat)
            T_local_world_base = np.eye(4)
            T_local_world_base[:3, :3] = current_rot_from_quat.as_matrix()
            T_local_world_base[:3, 3] = current_pos_local
        except ValueError:
            self.node.get_logger().warn("YOLO Processor: Invalid drone orientation for transformations.")
            return

        full_transform_optical_to_local_world = T_local_world_base @ self.camera_to_base_pencil @ self.optical_to_camera_pencil

        fx = self.camera_matrix_pencil[0, 0]
        fy = self.camera_matrix_pencil[1, 1]
        cx = self.camera_matrix_pencil[0, 2]
        cy = self.camera_matrix_pencil[1, 2]

        for bbox_idx, bbox_data in enumerate(yolo_bboxes_metrics):
            try:
                center_x_px = float(bbox_data['center_x'])
                center_y_px = float(bbox_data['center_y'])
                width_px = float(bbox_data['width'])
                height_px = float(bbox_data['height'])
                
                depth_Z_optical = bbox_data['distance'] if bbox_data['distance'] > 0 else 5.0

                if depth_Z_optical <= 0.1 or depth_Z_optical > self.jps_planner_config.max_obstacle_detection_distance_for_yolo:
                    self.node.get_logger().debug(f"YOLO: Obstacle '{bbox_data['class']}' ignored, depth {depth_Z_optical:.2f}m out of range.")
                    continue

                gate_X_optical = (center_x_px - cx) * depth_Z_optical / fx
                gate_Y_optical = (center_y_px - cy) * depth_Z_optical / fy
                gate_Z_optical = depth_Z_optical

                obstacle_pos_optical = np.array([gate_X_optical, gate_Y_optical, gate_Z_optical])
                self.node.get_logger().debug(f"  YOLO Obstacle '{bbox_data['class']}': Depth={depth_Z_optical:.2f}, PxCoords=[{center_x_px:.1f},{center_y_px:.1f}] -> OpticalFrame={vector_to_str(obstacle_pos_optical)}")

                obstacle_pos_local_world_h = full_transform_optical_to_local_world @ np.append(obstacle_pos_optical, 1)
                obstacle_pos_local_world = obstacle_pos_local_world_h[:3]
                self.node.get_logger().debug(f"  -> LocalWorld (Odom) Frame: {vector_to_str(obstacle_pos_local_world)}")

                obs_radius_xy = 0.5
                obs_full_z = 5.0 
                
                if bbox_data['class'] == 'tree':
                    obs_radius_xy = 5.5
                    obs_full_z = 5.0
                elif bbox_data['class'] == 'leaf':
                    obs_radius_xy = 1.0
                    obs_full_z = 4.0
                else: 
                    continue

                self.node.get_logger().debug(f"  Estimated size for '{bbox_data['class']}': RadiusXY={obs_radius_xy:.2f}, FullZ={obs_full_z:.2f}")

                obs_pos_global_calculated = obstacle_pos_local_world + self.origin_pos_abs

                new_yolo_obstacles_global.append({
                    'name': bbox_data['class'],
                    'pos': obs_pos_global_calculated,
                    'radius': obs_radius_xy, 
                    'sz': obs_full_z,
                    'source': 'yolo'
                })
                
                self.node.get_logger().info(f"YOLO Obstacle Processed: '{bbox_data['class']}' GlobalPos={vector_to_str(obs_pos_global_calculated)} RadiusXY={obs_radius_xy:.2f} SZ={obs_full_z:.2f}")

            except Exception as e:
                self.node.get_logger().error(f"Error processing YOLO bbox {bbox_idx} data: {e}\n{traceback.format_exc()}")
                continue 

        with self._state_lock:
            current_non_yolo_obstacles = [obs for obs in self._detected_obstacles_global if obs.get('source') != 'yolo']
            old_yolo_obstacles_before_update = [obs for obs in self._detected_obstacles_global if obs.get('source') == 'yolo']

            self._detected_obstacles_global = current_non_yolo_obstacles + new_yolo_obstacles_global

            if str(new_yolo_obstacles_global) != str(old_yolo_obstacles_before_update):
                self.new_obstacles_for_replan = True
                YELLOW = '\033[93m'
                RESET = '\033[0m'
                self.node.get_logger().warn(f"{YELLOW}YOLO UPDATED {len(new_yolo_obstacles_global)} OBSTACLES! Total: {len(self._detected_obstacles_global)}{RESET}. Replanning triggered.")
            else:
                self.node.get_logger().debug("YOLO detections are identical to previous cycle. No replan triggered.")


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
            else: 
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
            self.node.get_logger().warn(f"Not ready to fly for current step ({self.step}). Holding...", throttle_duration_sec=5.0)
            if hasattr(self, '_velocity_publisher') and self._velocity_publisher:
                 ts = TwistStamped()
                 ts.header.stamp = self.node.get_clock().now().to_msg()
                 ts.header.frame_id = 'odom'
                 self._velocity_publisher.publish(ts)
            return
        elif not self.step and not is_ready_to_fly and self.steps: 
             self.node.get_logger().warn(f"Waiting to start next step, FCU not ready. Holding...", throttle_duration_sec=5.0)
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
                 self.logger.error(f"Error initializing step '{self.step}': {e}\n{traceback.format_exc()}")
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
                self.logger.error(f"Error executing step '{self.step}': {e}\n{traceback.format_exc()}")
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
    def detected_obstacles_global(self) -> list: 
        with self._state_lock:
            return self._detected_obstacles_global
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
    def jps_planner(self) -> JumpPointSearchPlanner: return self._jps_planner

    @property
    def new_obstacles_for_replan(self) -> bool: return self._new_obstacles_for_replan
    @new_obstacles_for_replan.setter
    def new_obstacles_for_replan(self, value: bool) -> None: self._new_obstacles_for_replan = value

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

    def publish_jps_path_markers(self, path_local_odom: list[np.ndarray] | None):
        if self.viz_publisher is None or path_local_odom is None or not path_local_odom:
            return
        marker_array = MarkerArray()
        
        line_marker = Marker()
        line_marker.header.frame_id = "odom"
        line_marker.header.stamp = self.node.get_clock().now().to_msg()
        line_marker.ns = "jps_path_line"
        line_marker.id = self._path_marker_id_counter
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.15
        line_marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8)
        
        for wp_local_odom in path_local_odom:
            p = Point(x=wp_local_odom[0], y=wp_local_odom[1], z=wp_local_odom[2])
            line_marker.points.append(p)
        marker_array.markers.append(line_marker)

        sphere_id_offset = self._path_marker_id_counter + 1
        for i, wp_local_odom in enumerate(path_local_odom):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "odom"
            sphere_marker.header.stamp = self.node.get_clock().now().to_msg()
            sphere_marker.ns = "jps_waypoints_spheres"
            sphere_marker.id = sphere_id_offset + i
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position = Point(x=wp_local_odom[0], y=wp_local_odom[1], z=wp_local_odom[2])
            sphere_marker.pose.orientation.w = 1.0 
            sphere_marker.scale = Vector3(x=0.35, y=0.35, z=0.35) 
            sphere_marker.color = ColorRGBA(r=0.8, g=0.0, b=0.8, a=0.7)
            marker_array.markers.append(sphere_marker)

        if marker_array.markers:
            self.viz_publisher.publish(marker_array)
            self._path_marker_id_counter = sphere_id_offset + len(path_local_odom)

    def clear_jps_path_markers(self):
        """Удаляет все маркеры пути JPS из RViz."""
        if self.viz_publisher is None: return
        marker_array = MarkerArray()
        
        delete_line_marker = Marker(action=Marker.DELETEALL, ns="jps_path_line", id=0)
        delete_line_marker.header.frame_id = "odom"
        delete_line_marker.header.stamp = self.node.get_clock().now().to_msg()
        marker_array.markers.append(delete_line_marker)

        delete_spheres_marker = Marker(action=Marker.DELETEALL, ns="jps_waypoints_spheres", id=0)
        delete_spheres_marker.header.frame_id = "odom"
        delete_spheres_marker.header.stamp = self.node.get_clock().now().to_msg()
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
        controller.clear_jps_path_markers()

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




class JPSNavToGlobalGoal(Step):
    def __init__(self, target_x_rel_mission: float, target_y_rel_mission: float, target_z_rel_mission: float,
                 final_tolerance_xy: float = 0.5, final_tolerance_z: float = 0.3,
                 lookahead_distance: float = 5.0):
        self.final_target_pos_local_odom = np.array([target_x_rel_mission, target_y_rel_mission, target_z_rel_mission])
        self.final_tolerance_xy = final_tolerance_xy
        self.final_tolerance_z = final_tolerance_z
        self.lookahead_distance = lookahead_distance

        self._jps_path_local_odom: list[np.ndarray] | None = None
        self._path_segment_lengths: list[float] | None = None 
        self._current_path_segment_idx: int = 0
        self._current_lookahead_target: np.ndarray | None = None
        self._last_replan_time: rclpy.time.Time | None = None
        self._replan_interval_seconds: float = 1.0

    @property
    def uses_velocity_control(self) -> bool: return True

    def _plan_path_jps(self, controller: Controller, start_pos_local_odom: np.ndarray) -> bool:
        """Выполняет планирование пути с использованием JPS."""
        jps_planner_instance = controller.jps_planner
        
        obstacles_for_jps_planning = controller.detected_obstacles_global
        
        adapted_obstacles_world_3d = []
        for obs_global in obstacles_for_jps_planning:
            obs_pos_center_local_odom = obs_global['pos'] - controller.origin_pos_abs
            adapted_obstacles_world_3d.append({
                'pos_center': obs_pos_center_local_odom,
                'radius_xy': obs_global['radius'], 
                'half_height_z': obs_global['sz'] / 2.0
            })
        
        is_goal_obstructed = False
        for obs_local in adapted_obstacles_world_3d:
            dist_xy_sq = (self.final_target_pos_local_odom[0] - obs_local['pos_center'][0])**2 + \
                         (self.final_target_pos_local_odom[1] - obs_local['pos_center'][1])**2
            effective_obs_radius_xy = obs_local['radius_xy'] + controller.motion_config.robot_radius
            
            if dist_xy_sq < effective_obs_radius_xy**2:
                obs_z_min = obs_local['pos_center'][2] - obs_local['half_height_z']
                obs_z_max = obs_local['pos_center'][2] + obs_local['half_height_z']
                
                if obs_z_min <= self.final_target_pos_local_odom[2] <= obs_z_max:
                    is_goal_obstructed = True
                    break
        
        if is_goal_obstructed:
            controller.logger.error(f"JPSNav: Critical - Final target {vector_to_str(self.final_target_pos_local_odom)} is INSIDE inflated obstacle. Aborting plan for this target.")
            return False

        controller.logger.info(f"JPSNav: Planning from LocalOdom {vector_to_str(start_pos_local_odom)} to FinalLocalOdom {vector_to_str(self.final_target_pos_local_odom)}")
        new_path = jps_planner_instance.plan(
            start_pos_local_odom,
            self.final_target_pos_local_odom,
            adapted_obstacles_world_3d
        )

        if not new_path:
            controller.logger.error("JPSNav: JPS planning FAILED!")
            self._jps_path_local_odom = None
            return False
        else:
            controller.logger.info(f"JPSNav: JPS plan SUCCESSFUL, {len(new_path)} 3D waypoints (Jump Points).")
            self._jps_path_local_odom = new_path
            
            self._path_segment_lengths = [0.0] * len(new_path)
            for i in range(1, len(new_path)):
                self._path_segment_lengths[i] = self._path_segment_lengths[i-1] + np.linalg.norm(new_path[i] - new_path[i-1])

            controller.clear_jps_path_markers()
            if controller.viz_publisher:
                 controller.publish_jps_path_markers(self._jps_path_local_odom)
            self._current_path_segment_idx = 0
            self._update_lookahead_target(controller, controller.position)
            self._last_replan_time = controller.node.get_clock().now()
            return True

    def init(self, controller: Controller):
        controller.clear_jps_path_markers()

        current_pos_local_odom = controller.position
        if current_pos_local_odom is None:
            raise RuntimeError("JPSNavToGlobalGoal init: current position unknown")

        if not self._plan_path_jps(controller, current_pos_local_odom):
            controller.logger.error("JPSNav: Initial planning failed. Step might not execute correctly. Attempting to navigate directly to final target.")
            self._current_lookahead_target = self.final_target_pos_local_odom
        controller.new_obstacles_for_replan = False

    def _update_lookahead_target(self, controller: Controller, current_pos_local_odom: np.ndarray):
        if not self._jps_path_local_odom or len(self._jps_path_local_odom) < 2:
            self._current_lookahead_target = self.final_target_pos_local_odom
            controller.logger.debug(f"JPSNav: Path too short or empty. Setting lookahead to final target: {vector_to_str(self._current_lookahead_target)}")
            return

        min_dist_sq_to_path = float('inf')
        closest_jp_idx_on_path = 0 
        for i in range(len(self._jps_path_local_odom)):
            dist_sq = np.sum((current_pos_local_odom - self._jps_path_local_odom[i])**2)
            if dist_sq < min_dist_sq_to_path:
                min_dist_sq_to_path = dist_sq
                closest_jp_idx_on_path = i
        
        self._current_path_segment_idx = max(self._current_path_segment_idx, closest_jp_idx_on_path)
        
        if self._current_path_segment_idx >= len(self._jps_path_local_odom) - 1:
            self._current_lookahead_target = self.final_target_pos_local_odom
            controller.logger.debug(f"JPSNav: At or past end of JPS path. Setting lookahead to final target: {vector_to_str(self._current_lookahead_target)}")
            return
        p_seg_start = self._jps_path_local_odom[self._current_path_segment_idx]
        p_seg_end = self._jps_path_local_odom[self._current_path_segment_idx + 1]
        
        segment_vector = p_seg_end - p_seg_start
        segment_length = np.linalg.norm(segment_vector)
        
        projection_t = 0.0
        if segment_length > 1e-3:
            vec_drone_from_p_seg_start = current_pos_local_odom - p_seg_start
            projection_t = np.dot(vec_drone_from_p_seg_start, segment_vector) / (segment_length**2)
            projection_t = np.clip(projection_t, 0.0, 1.0)
        dist_along_current_segment_to_projection = projection_t * segment_length
        
        cumulative_dist_to_drone_projection = self._path_segment_lengths[self._current_path_segment_idx] + dist_along_current_segment_to_projection

        target_cumulative_distance = cumulative_dist_to_drone_projection + self.lookahead_distance
        
        found_lookahead_point = False
        lookahead_search_idx = self._current_path_segment_idx
        
        while lookahead_search_idx < len(self._jps_path_local_odom) - 1:
            current_seg_start_jp = self._jps_path_local_odom[lookahead_search_idx]
            current_seg_end_jp = self._jps_path_local_odom[lookahead_search_idx + 1]
            current_seg_cum_dist_start = self._path_segment_lengths[lookahead_search_idx]
            current_seg_cum_dist_end = self._path_segment_lengths[lookahead_search_idx + 1]
            
            if target_cumulative_distance <= current_seg_cum_dist_end:
                local_t = (target_cumulative_distance - current_seg_cum_dist_start) / (current_seg_cum_dist_end - current_seg_cum_dist_start + 1e-6) 
                local_t = np.clip(local_t, 0.0, 1.0)
                self._current_lookahead_target = current_seg_start_jp + (current_seg_end_jp - current_seg_start_jp) * local_t
                found_lookahead_point = True
                break
            
            lookahead_search_idx += 1

        if not found_lookahead_point:
            self._current_lookahead_target = self.final_target_pos_local_odom
            controller.logger.debug(f"JPSNav: Lookahead distance exceeds path length. Snapping lookahead to final mission goal: {vector_to_str(self.final_target_pos_local_odom)}")
        
        if np.linalg.norm(self._current_lookahead_target - self.final_target_pos_local_odom) < controller.motion_config.target_reach_tolerance_xy * 1.5:
            self._current_lookahead_target = self.final_target_pos_local_odom
            controller.logger.debug(f"JPSNav: Lookahead target within snap range of final mission goal. Snapping to: {vector_to_str(self._current_lookahead_target)}")
            
        controller.logger.info(f"JPSNav: Lookahead update. Drone pos: {vector_to_str(current_pos_local_odom)}. Current path segment: {self._current_path_segment_idx}. Lookahead target: {vector_to_str(self._current_lookahead_target)}")

    def update(self, controller: Controller) -> bool:
        current_pos_local_odom = controller.position
        if current_pos_local_odom is None:
            controller.logger.warn("JPSNav: Waiting for position data.", throttle_duration_sec=2.0)
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return False

        if controller.new_obstacles_for_replan:
            now = controller.node.get_clock().now()
            if self._last_replan_time is None or (now - self._last_replan_time).nanoseconds / 1e9 > self._replan_interval_seconds:
                controller.logger.warn("JPSNav: New obstacles detected! Attempting to REPLAN...")
                if self._plan_path_jps(controller, current_pos_local_odom):
                    controller.logger.info("JPSNav: REPLAN successful. Following new path.")
                else:
                    controller.logger.error("JPSNav: REPLAN FAILED. Continuing on old path or moving to final target.")
                controller.new_obstacles_for_replan = False
            else:
                controller.logger.debug(f"JPSNav: New obstacles detected, but replanning is throttled.")


        self._update_lookahead_target(controller, current_pos_local_odom)
        target_to_pursue = self._current_lookahead_target

        if target_to_pursue is None:
            controller.logger.error("JPSNav: Lookahead target is None. Holding position.")
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            return False

        delta_final = self.final_target_pos_local_odom - current_pos_local_odom
        dist_xy_final = np.linalg.norm(delta_final[:2])
        dist_z_final = abs(delta_final[2])

        if dist_xy_final < self.final_tolerance_xy and dist_z_final < self.final_tolerance_z:
            controller.logger.info(f"JPSNav: Final target {vector_to_str(self.final_target_pos_local_odom)} REACHED!")
            ts = TwistStamped()
            ts.header.stamp = controller.node.get_clock().now().to_msg()
            ts.header.frame_id = 'odom'
            controller.velocity_publisher.publish(ts)
            if controller.viz_publisher: controller.clear_jps_path_markers()
            return True
        else:
            twist_cmd = controller.calculate_direct_velocity_to_target(target_to_pursue)
            controller.velocity_publisher.publish(twist_cmd)
            return False

    def __str__(self) -> str:
        return f"JPSNavToGlobalGoal -> FinalLocalOdom {vector_to_str(self.final_target_pos_local_odom)}"


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
        if controller.viz_publisher: controller.clear_jps_path_markers()

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
        if controller.viz_publisher: controller.clear_jps_path_markers()

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


def main(args=None):
    rclpy.init(args=args)
    log_level = LoggingSeverity.INFO
    node = Node('jps_3d_controller_node')
    node.get_logger().set_level(log_level)
    node.get_logger().info(f"Logging level set to {log_level.name}")

    original_waypoints_abs = [
        
        [ 246.0, 225.0, 194.0],
        [ 260.0, 205.0, 197.0],
        [ 250.0, 175.0, 197.0],
        [ 220.0, 165.0, 196.0],
        [ 200.0, 175.0, 196.0],
        [ 190.0, 165.0, 196.0],
        [ 170.0, 200.0, 196.0],
        [ 190.0, 220.0, 196.0],
        [ 150.0, 220.0, 196.0],
        [ 140.0, 225.0, 196.0],
    ]

    mission_origin_abs = np.array(original_waypoints_abs[0])
    node.get_logger().info(f"Absolute Mission Origin (Global Coords): {vector_to_str(mission_origin_abs)}")

    target_waypoints_local_odom = []
    for i, wp_abs_list in enumerate(original_waypoints_abs):
        wp_abs = np.array(wp_abs_list)
        target_local_odom = wp_abs
        target_waypoints_local_odom.append(target_local_odom)
        node.get_logger().info(f"Target {i} (LocalOdom, Rel.ToMissionOrigin): {vector_to_str(target_local_odom)}")

    jps_nav_final_tolerance_xy = 0.5 
    jps_nav_final_tolerance_z = 1.0 
    takeoff_tolerance = 0.3
    landing_tolerance = 0.15

    steps: list[Step] = []

    num_nav_goals = len(target_waypoints_local_odom)
    for i in range(2, num_nav_goals):
        target_nav_local_odom = target_waypoints_local_odom[i]
        node.get_logger().info(f"Adding Step {len(steps)+1}: JPSNavTo Target LocalOdom {vector_to_str(target_nav_local_odom)}")
        steps.append(JPSNavToGlobalGoal(
            target_x_rel_mission=target_nav_local_odom[0],
            target_y_rel_mission=target_nav_local_odom[1],
            target_z_rel_mission=target_nav_local_odom[2],
            final_tolerance_xy=jps_nav_final_tolerance_xy,
            final_tolerance_z=jps_nav_final_tolerance_z,
            lookahead_distance = 10
        ))

    landing_target_z_local_odom = target_waypoints_local_odom[-1][2] 
    steps.append(Land(target_z_local_odom_for_landing=landing_target_z_local_odom, tolerance=landing_tolerance))

    steps.append(Disarm(attempts=5, delay_between_attempts=1.0))

    try:
        simple_controller = SimpleController(node, steps,
                                             origin_pos_abs=mission_origin_abs,
                                             control_frequency=20.0)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        node.get_logger().info("Starting JPS(3D) Controller Node...")
        executor.spin()
    except KeyboardInterrupt: node.get_logger().info("Node stopped via KeyboardInterrupt.")
    except SystemExit as e: node.get_logger().info(f"Node stopped via SystemExit: {e}")
    except Exception as e:
        node.get_logger().fatal(f"Unhandled exception in main: {e}")
        node.get_logger().error(traceback.format_exc())
    finally:
        node.get_logger().info("Shutting down JPS(3D) Controller Node...")
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
        
        if 'simple_controller' in locals() and hasattr(simple_controller, 'clear_jps_path_markers') and simple_controller.viz_publisher:
             simple_controller.clear_jps_path_markers()

        if 'simple_controller' in locals() and hasattr(simple_controller, '_image_processing_active'):
            simple_controller._image_processing_active = False
            if simple_controller._image_processing_thread.is_alive():
                simple_controller._image_processing_thread.join(timeout=1.0)
                node.get_logger().info("Image processing thread joined.")
            if simple_controller._gui_thread.is_alive():
                simple_controller._gui_thread.join(timeout=1.0) 
                node.get_logger().info("GUI thread joined.")

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
#!/usr/bin/env python3
"""ML-based Navigation Node.

Uses trained model to predict target_angle from LiDAR data,
then applies PD control to generate velocity commands.

Compatible with minicar_navigation's input/output configuration.
"""

import math
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class AngleClassifierMLP(nn.Module):
    """MLP for angle classification from LiDAR."""

    def __init__(self, input_dim=360, hidden_dims=[256, 128, 64], num_classes=9):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AngleClassifierCNN(nn.Module):
    """1D CNN for angle classification from LiDAR with circular padding."""

    def __init__(self, input_dim=360, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 45, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def _circular_pad(self, x, pad):
        return torch.cat([x[:, :, -pad:], x, x[:, :, :pad]], dim=2)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self._circular_pad(x, 3)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        x = self._circular_pad(x, 2)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        x = self._circular_pad(x, 1)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def class_to_angle(class_idx, num_classes=9, angle_range=math.pi):
    """Convert class index back to angle (radians)."""
    class_width = angle_range / num_classes
    angle = -angle_range/2 + (class_idx + 0.5) * class_width
    return angle


def lidar_to_bev(lidar_ranges, image_size=64, max_range=12.0, view_range=3.0, robot_radius=2):
    """Convert LiDAR ranges to Bird's Eye View image."""
    img = np.zeros((image_size, image_size), dtype=np.float32)
    center = image_size // 2
    scale = center / view_range

    angles = np.linspace(0, 2*np.pi, len(lidar_ranges), endpoint=False)
    ranges_m = lidar_ranges * max_range
    valid_mask = ranges_m < view_range

    if valid_mask.sum() > 0:
        valid_ranges = ranges_m[valid_mask]
        valid_angles = angles[valid_mask]

        x_robot = valid_ranges * np.cos(valid_angles)
        y_robot = valid_ranges * np.sin(valid_angles)

        px = (center - y_robot * scale).astype(np.int32)
        py = (center - x_robot * scale).astype(np.int32)

        valid_pixels = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
        px = px[valid_pixels]
        py = py[valid_pixels]

        img[py, px] = 1.0

    cv2.circle(img, (center, center), robot_radius, 0.5, -1)

    return img


class MultiTaskBEVCNN(nn.Module):
    """2D CNN for multi-task BEV learning: path validity + angle classification (5 layers)."""

    def __init__(self, num_classes=9, image_size=64):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # Shared CNN backbone (5 conv layers)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # 64 -> 32 -> 16 -> 8 -> 4 -> 2
        feat_size = image_size // 32
        self.shared_fc = nn.Linear(256 * feat_size * feat_size, 512)
        self.bn_fc = nn.BatchNorm1d(512)

        # Path validity head (binary classification)
        self.path_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Angle classification head
        self.angle_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, 1, H, W)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 64->32
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 32->16
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # 16->8
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))  # 8->4
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))  # 4->2

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        shared = torch.relu(self.bn_fc(self.shared_fc(x)))
        shared = self.dropout(shared)

        path_logit = self.path_head(shared).squeeze(-1)
        angle_logits = self.angle_head(shared)

        return path_logit, angle_logits


class MLNavNode(Node):
    """ML-based navigation node."""

    # 緊急停止パラメータ (クラス定数 - local_nav_node と同一)
    EMERGENCY_STOP_DIST = 0.20  # 20cm以内に障害物があれば停止
    EMERGENCY_STOP_CONE_DEG = 45.0  # 前方±45°のコーンを検査
    EMERGENCY_STOP_RATIO = 0.3  # コーン内の30%が閾値以下で停止（ノイズ除去）

    def __init__(self):
        super().__init__("ml_nav_node", automatically_declare_parameters_from_overrides=True)

        # Input/Output namespace parameters (same as minicar_navigation)
        if not self.has_parameter("input_sim"):
            self.declare_parameter("input_sim", True)
        if not self.has_parameter("input_real"):
            self.declare_parameter("input_real", False)
        if not self.has_parameter("output_sim"):
            self.declare_parameter("output_sim", True)
        if not self.has_parameter("output_real"):
            self.declare_parameter("output_real", False)
        if not self.has_parameter("sim_ns"):
            self.declare_parameter("sim_ns", "sim_robot")
        if not self.has_parameter("real_ns"):
            self.declare_parameter("real_ns", "real_robot")
        if not self.has_parameter("robot_type"):
            self.declare_parameter("robot_type", "diff")

        # Model path (top-level param)
        if not self.has_parameter("model_path"):
            self.declare_parameter("model_path", "data/models/angle_predictor.pt")

        # Load model
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.max_range = 12.0  # LiDAR max range for normalization

        # Load control parameters from hierarchical structure
        self._load_control_params()

        # PD control state
        self.prev_angle_error = 0.0
        self.prev_time = None

        # LiDAR data
        self.latest_lidar_data = None
        self.received_scan = False

        # Setup input subscribers (same as minicar_navigation)
        self.scan_sub = self._setup_input_subscribers()

        # Setup output publishers (same as minicar_navigation)
        self.cmd_publishers = self._setup_output_publishers()

        # Control loop timer - 10 Hz (same as local_nav_node)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Dynamic parameter update callback
        self._ctrl_lock = threading.Lock()
        self.add_on_set_parameters_callback(self._on_parameters_updated)

        self.get_logger().info(f"MLNavNode started")
        self.get_logger().info(f"  Model: {model_path}")
        self.get_logger().info(f"  Device: {self.device}")
        self.get_logger().info(f"  Lookahead: {self.lookahead_distance}m (model fixed)")
        self.get_logger().info(f"  Velocity: target={self.target_velocity}, min={self.min_velocity}")
        self.get_logger().info(f"  PD control: kp={self.kp_angular}, kd={self.kd_angular}, k_curv={self.k_curvature}")
        self.get_logger().info(f"  Max angular vel: {self.max_angular_velocity} rad/s")

    def _get_parameters_as_dict(self, prefix: str) -> dict:
        """指定されたプレフィックスのパラメータを辞書として取得"""
        param_dict = {}
        try:
            params = self.get_parameters_by_prefix(prefix)
            for key, param in params.items():
                param_dict[key] = param.value
        except Exception as e:
            self.get_logger().warn(f"Failed to get parameters with prefix '{prefix}': {e}")
        return param_dict

    def _load_control_params(self):
        """Load control parameters from hierarchical structure (common.* and controllers.pd.*)."""
        common_params = self._get_parameters_as_dict('common')
        pd_params = self._get_parameters_as_dict('controllers.pd')

        # Common parameters
        self.lookahead_distance = common_params.get('lookahead_distance', 0.5)
        self.target_velocity = common_params.get('target_velocity', 0.3)
        self.max_angular_velocity = common_params.get('max_angular_velocity', 1.0)
        self.goal_tolerance = common_params.get('goal_tolerance', 0.1)

        # PD controller parameters
        self.kp_angular = pd_params.get('kp_angular', 2.0)
        self.kd_angular = pd_params.get('kd_angular', 0.5)
        self.k_curvature = pd_params.get('k_curvature', 1.0)
        self.min_velocity = pd_params.get('min_velocity', 0.1)

    def _on_parameters_updated(self, params: list) -> SetParametersResult:
        """パラメータ更新を検知して制御パラメータを再読み込み"""
        relevant = any(
            p.name.startswith("common.") or p.name.startswith("controllers.")
            for p in params
        )

        if not relevant:
            return SetParametersResult(successful=True)

        # パラメータが適用された「後」に再初期化したいので遅延実行
        def apply():
            with self._ctrl_lock:
                self._load_control_params()
                self.get_logger().info("Control params re-loaded due to parameter update.")
                self.get_logger().info(
                    f"  Velocity: target={self.target_velocity}, min={self.min_velocity}")
                self.get_logger().info(
                    f"  PD: kp={self.kp_angular}, kd={self.kd_angular}, k_curv={self.k_curvature}")
                self.get_logger().info(
                    f"  Max angular vel: {self.max_angular_velocity} rad/s")
            # タイマーを破棄して一度だけ実行
            if hasattr(self, '_param_update_timer') and self._param_update_timer:
                self._param_update_timer.cancel()
                self._param_update_timer = None

        # 既存のタイマーがあればキャンセル
        if hasattr(self, '_param_update_timer') and self._param_update_timer:
            self._param_update_timer.cancel()
        self._param_update_timer = self.create_timer(0.01, apply)
        return SetParametersResult(successful=True)

    def _setup_input_subscribers(self):
        """Setup input subscribers based on namespace parameters (same as minicar_navigation)."""
        input_sim = self.get_parameter('input_sim').get_parameter_value().bool_value
        input_real = self.get_parameter('input_real').get_parameter_value().bool_value
        sim_ns = self.get_parameter('sim_ns').get_parameter_value().string_value
        real_ns = self.get_parameter('real_ns').get_parameter_value().string_value

        if not (input_sim or input_real):
            raise ValueError("At least one of input_sim or input_real must be true")

        if input_sim:
            scan_topic = f"/{sim_ns}/scan" if sim_ns else "/scan"
            self.get_logger().info(f"Subscribing to sim LiDAR: {scan_topic}")
        else:
            scan_topic = f"/{real_ns}/scan" if real_ns else "/scan"
            self.get_logger().info(f"Subscribing to real LiDAR: {scan_topic}")

        return self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10,
        )

    def _setup_output_publishers(self):
        """Setup output publishers based on namespace parameters (same as minicar_navigation)."""
        output_sim = self.get_parameter('output_sim').get_parameter_value().bool_value
        output_real = self.get_parameter('output_real').get_parameter_value().bool_value
        sim_ns = self.get_parameter('sim_ns').get_parameter_value().string_value
        real_ns = self.get_parameter('real_ns').get_parameter_value().string_value
        robot_type = self.get_parameter('robot_type').get_parameter_value().string_value

        # Determine topic suffix based on robot type
        if robot_type == "ackermann":
            topic_suffix = "ackermann_steering_controller/reference_unstamped"
        else:
            topic_suffix = "diff_drive_controller/cmd_vel_unstamped"

        publishers = {}

        if output_sim:
            sim_topic = f"/{sim_ns}/{topic_suffix}"
            publishers['sim'] = self.create_publisher(Twist, sim_topic, 10)
            self.get_logger().info(f"Publishing to sim robot ({robot_type}): {sim_topic}")

        if output_real:
            real_topic = f"/{real_ns}/{topic_suffix}"
            publishers['real'] = self.create_publisher(Twist, real_topic, 10)
            self.get_logger().info(f"Publishing to real robot ({robot_type}): {real_topic}")

        if not publishers:
            self.get_logger().warn("No output publishers configured!")

        return publishers

    def _publish_cmd_vel(self, cmd_msg):
        """Publish command to all configured output publishers."""
        for name, publisher in self.cmd_publishers.items():
            publisher.publish(cmd_msg)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Check model type
            self.model_type = checkpoint.get('model_type', 'classifier')
            self.model_arch = checkpoint.get('model_arch', 'cnn')
            self.num_classes = checkpoint.get('num_classes', 9)
            self.angle_range = checkpoint.get('angle_range', math.pi)

            # Detect multitask and BEV from model_type or explicit flags
            self.is_multitask = checkpoint.get('is_multitask', 'multitask' in self.model_type)
            self.is_bev = checkpoint.get('is_bev', 'bev' in self.model_type)

            # BEV parameters (check both old and new keys for compatibility)
            self.bev_image_size = checkpoint.get('bev_image_size', checkpoint.get('image_size', 64))
            self.bev_view_range = checkpoint.get('bev_view_range', checkpoint.get('view_range', 3.0))

            # Create model based on architecture
            if self.is_bev and self.is_multitask:
                model = MultiTaskBEVCNN(
                    num_classes=self.num_classes,
                    image_size=self.bev_image_size
                )
            elif self.model_arch == 'cnn':
                model = AngleClassifierCNN(
                    input_dim=checkpoint.get('input_dim', 360),
                    num_classes=self.num_classes
                )
            else:
                model = AngleClassifierMLP(
                    input_dim=checkpoint.get('input_dim', 360),
                    hidden_dims=checkpoint.get('hidden_dims', [256, 128, 64]),
                    num_classes=self.num_classes
                )

            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            self.max_range = checkpoint.get('max_range', 12.0)
            self.get_logger().info(
                f"Model loaded: arch={self.model_arch}, classes={self.num_classes}, "
                f"multitask={self.is_multitask}, bev={self.is_bev}"
            )
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

    def scan_callback(self, msg: LaserScan):
        """Process LaserScan message."""
        n = len(msg.ranges)
        ranges = np.asarray(msg.ranges, dtype=np.float32)

        # Handle invalid values
        max_r = float(msg.range_max) if msg.range_max > 0.0 else self.max_range
        ranges = np.where(np.isfinite(ranges), ranges, max_r)

        rmin = float(msg.range_min) if msg.range_min > 0.0 else 0.0
        ranges = np.where((ranges >= rmin) & (ranges <= max_r), ranges, max_r)

        # Store angles for emergency stop calculation
        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment

        self.latest_lidar_data = {
            "ranges": ranges,
            "angles": angles,
        }

        if not self.received_scan:
            self.get_logger().info(f"Received LaserScan: beams={n}, max_range={max_r}")
            self.received_scan = True

    def control_loop(self):
        """Main control loop."""
        if self.latest_lidar_data is None:
            return

        lidar_data = self.latest_lidar_data

        # Emergency stop check
        if self._check_emergency_stop(lidar_data):
            self._publish_stop()
            return

        try:
            # Predict target angle and path validity
            target_angle, has_path = self._predict_angle(lidar_data["ranges"])

            # Compute control using PD control (lock for thread safety)
            with self._ctrl_lock:
                linear_vel, angular_vel = self._compute_pd_control(target_angle)

            # Publish command
            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            self._publish_cmd_vel(cmd)

            self.get_logger().info(
                f"Control: angle={math.degrees(target_angle):.1f}°, "
                f"linear={linear_vel:.3f}, angular={angular_vel:.3f}"
            )

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            self._publish_stop()

    def _predict_angle(self, ranges: np.ndarray) -> tuple:
        """Predict target angle from LiDAR ranges.

        Returns:
            (angle, has_path): angle in radians, has_path boolean
        """
        # Normalize
        normalized = np.clip(ranges, 0, self.max_range) / self.max_range

        with torch.no_grad():
            if self.is_bev:
                # Convert to BEV image
                bev_img = lidar_to_bev(
                    normalized,
                    image_size=self.bev_image_size,
                    max_range=self.max_range,
                    view_range=self.bev_view_range
                )
                x = torch.tensor(bev_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

            if self.is_multitask:
                path_logit, angle_logits = self.model(x)
                has_path = torch.sigmoid(path_logit).item() > 0.5
                pred_class = torch.argmax(angle_logits, dim=1).item()
            else:
                logits = self.model(x)
                pred_class = torch.argmax(logits, dim=1).item()
                has_path = True  # Single-task models always assume path exists

        # Convert class to angle
        angle = class_to_angle(pred_class, self.num_classes, self.angle_range)

        return angle, has_path

    def _compute_pd_control(self, target_angle: float) -> tuple:
        """Compute PD control for velocity commands.

        Same formula as minicar_navigation's pd_pursuit_controller.

        Args:
            target_angle: Target angle in radians (robot frame)

        Returns:
            (linear_velocity, angular_velocity)
        """
        current_time = time.time()

        # Angle error (in robot frame, current_yaw = 0)
        angle_error = target_angle

        # PD control for angular velocity
        if self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                angle_error_dot = (angle_error - self.prev_angle_error) / dt
            else:
                angle_error_dot = 0.0
        else:
            angle_error_dot = 0.0

        angular_vel = self.kp_angular * angle_error + self.kd_angular * angle_error_dot

        # Clamp angular velocity
        angular_vel = max(-self.max_angular_velocity,
                          min(self.max_angular_velocity, angular_vel))

        # Linear velocity (reduce when turning)
        # Same formula as minicar_navigation's pd_pursuit_controller
        curvature = 0.0  # MLモデルでは曲率計算なし
        curvature_factor = 1.0 / (1.0 + self.k_curvature * abs(curvature))
        heading_factor = max(0.5, 1.0 - abs(angle_error) / math.pi)
        linear_vel = self.target_velocity * curvature_factor * heading_factor
        linear_vel = max(self.min_velocity, linear_vel)

        # Update state
        self.prev_angle_error = angle_error
        self.prev_time = current_time

        return linear_vel, angular_vel

    def _check_emergency_stop(self, lidar_data: dict) -> bool:
        """Check if emergency stop is needed (same logic as minicar_navigation)."""
        ranges = lidar_data["ranges"]
        angles = lidar_data["angles"]

        # Front cone mask (handles 0-2π range like minicar_navigation)
        cone_rad = math.radians(self.EMERGENCY_STOP_CONE_DEG)
        front_mask = (angles <= cone_rad) | (angles >= 2 * math.pi - cone_rad)

        if not np.any(front_mask):
            return False

        front_ranges = ranges[front_mask]
        total_points = len(front_ranges)

        # Check if enough points are within emergency distance
        close_points = np.sum(front_ranges < self.EMERGENCY_STOP_DIST)
        close_ratio = close_points / total_points

        if close_ratio >= self.EMERGENCY_STOP_RATIO:
            min_front_dist = np.min(front_ranges)
            self.get_logger().warn(
                f"EMERGENCY STOP: {close_ratio:.0%} points at <{self.EMERGENCY_STOP_DIST}m "
                f"(min: {min_front_dist:.3f}m)"
            )
            return True

        return False

    def _publish_stop(self):
        """Publish stop command."""
        cmd = Twist()
        self._publish_cmd_vel(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MLNavNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node._publish_cmd_vel(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

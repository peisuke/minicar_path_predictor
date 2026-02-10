#!/usr/bin/env python3
"""ML-based Navigation Node.

Uses trained model to predict target_angle from LiDAR data,
then applies PD control to generate velocity commands.

Compatible with minicar_navigation's input/output configuration.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
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


class MLNavNode(Node):
    """ML-based navigation node."""

    # Emergency stop parameters
    EMERGENCY_STOP_DIST = 0.20  # Stop if obstacle within 20cm
    EMERGENCY_STOP_CONE_DEG = 45.0  # Check front ±45° cone
    EMERGENCY_STOP_RATIO = 0.3  # Stop if 30% of cone is below threshold

    def _declare_param_if_needed(self, name: str, default_value):
        """Declare parameter only if not already declared."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default_value)

    def __init__(self):
        super().__init__("ml_nav_node", automatically_declare_parameters_from_overrides=True)

        # Declare parameters (only if not already declared from config file)
        self._declare_param_if_needed("model_path", "data/models/angle_predictor.pt")
        self._declare_param_if_needed("lookahead_distance", 0.5)
        self._declare_param_if_needed("target_velocity", 0.3)
        self._declare_param_if_needed("max_angular_velocity", 1.0)
        self._declare_param_if_needed("kp_angular", 2.0)
        self._declare_param_if_needed("kd_angular", 0.5)
        self._declare_param_if_needed("control_rate", 10.0)

        # Input/Output namespace parameters (same as minicar_navigation)
        self._declare_param_if_needed("input_sim", True)
        self._declare_param_if_needed("input_real", False)
        self._declare_param_if_needed("output_sim", True)
        self._declare_param_if_needed("output_real", False)
        self._declare_param_if_needed("sim_ns", "sim_robot")
        self._declare_param_if_needed("real_ns", "real_robot")
        self._declare_param_if_needed("robot_type", "diff")

        # Get parameters
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.lookahead_distance = self.get_parameter("lookahead_distance").get_parameter_value().double_value
        self.target_velocity = self.get_parameter("target_velocity").get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter("max_angular_velocity").get_parameter_value().double_value
        self.kp_angular = self.get_parameter("kp_angular").get_parameter_value().double_value
        self.kd_angular = self.get_parameter("kd_angular").get_parameter_value().double_value
        control_rate = self.get_parameter("control_rate").get_parameter_value().double_value

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.max_range = 12.0  # LiDAR max range for normalization

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

        # Control loop timer
        self.timer = self.create_timer(1.0 / control_rate, self.control_loop)

        self.get_logger().info(f"MLNavNode started")
        self.get_logger().info(f"  Model: {model_path}")
        self.get_logger().info(f"  Device: {self.device}")
        self.get_logger().info(f"  Lookahead: {self.lookahead_distance}m")
        self.get_logger().info(f"  Target velocity: {self.target_velocity}m/s")

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

            # Create model based on architecture
            if self.model_arch == 'cnn':
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
            self.get_logger().info(f"Model loaded: arch={self.model_arch}, classes={self.num_classes}")
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
            # Predict target angle
            target_angle = self._predict_angle(lidar_data["ranges"])

            # Compute control using PD control
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

    def _predict_angle(self, ranges: np.ndarray) -> float:
        """Predict target angle from LiDAR ranges."""
        # Normalize
        normalized = np.clip(ranges, 0, self.max_range) / self.max_range

        # Convert to tensor
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(x)
            pred_class = torch.argmax(logits, dim=1).item()

        # Convert class to angle
        angle = class_to_angle(pred_class, self.num_classes, self.angle_range)

        return angle

    def _compute_pd_control(self, target_angle: float) -> tuple:
        """Compute PD control for velocity commands.

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
        heading_factor = max(0.5, 1.0 - abs(angle_error) / math.pi)
        linear_vel = self.target_velocity * heading_factor

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

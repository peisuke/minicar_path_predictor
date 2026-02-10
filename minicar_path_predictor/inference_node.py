#!/usr/bin/env python3
"""ROS2 node for ML-based path prediction.

This node replaces the traditional local_nav by using a trained neural network
to predict local waypoints from LiDAR data.

Usage:
    ros2 run minicar_path_predictor inference_node --ros-args -p model_path:=models/best_model.pth
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Header

from .model import create_model


class PathPredictorNode(Node):
    """ROS2 node for ML-based local path prediction."""

    def __init__(self):
        super().__init__('path_predictor_node')

        # Declare parameters
        self.declare_parameter('model_path', 'models/best_model.pth')
        self.declare_parameter('model_info_path', '')
        self.declare_parameter('device', 'auto')

        # Robot parameters
        self.declare_parameter('sim_ns', 'sim_robot')
        self.declare_parameter('real_ns', 'robot')
        self.declare_parameter('is_simulation', True)
        self.declare_parameter('robot_type', 'ackermann')  # 'ackermann' or 'diff'

        # Control parameters
        self.declare_parameter('max_linear_vel', 1.0)
        self.declare_parameter('max_angular_vel', 1.5)
        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('control_rate', 10.0)

        # Safety parameters
        self.declare_parameter('emergency_stop_distance', 0.20)
        self.declare_parameter('emergency_stop_angle', 45.0)  # degrees
        self.declare_parameter('emergency_stop_ratio', 0.3)

        # Get parameters
        model_path = Path(self.get_parameter('model_path').value)
        model_info_path = self.get_parameter('model_info_path').value
        device = self.get_parameter('device').value

        self.is_simulation = self.get_parameter('is_simulation').value
        self.robot_type = self.get_parameter('robot_type').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.control_rate = self.get_parameter('control_rate').value

        self.emergency_stop_distance = self.get_parameter('emergency_stop_distance').value
        self.emergency_stop_angle = math.radians(self.get_parameter('emergency_stop_angle').value)
        self.emergency_stop_ratio = self.get_parameter('emergency_stop_ratio').value

        # Setup namespace
        if self.is_simulation:
            self.ns = self.get_parameter('sim_ns').value
        else:
            self.ns = self.get_parameter('real_ns').value

        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.get_logger().info(f"Using device: {self.device}")

        # Load model
        self._load_model(model_path, model_info_path)

        # State
        self.latest_scan: Optional[LaserScan] = None
        self.lidar_angles: Optional[np.ndarray] = None

        # Setup QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        scan_topic = f"/{self.ns}/scan"
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            sensor_qos
        )
        self.get_logger().info(f"Subscribing to: {scan_topic}")

        # Publishers
        if self.robot_type == 'ackermann':
            cmd_topic = f"/{self.ns}/ackermann_steering_controller/reference_unstamped"
        else:
            cmd_topic = f"/{self.ns}/diff_drive_controller/cmd_vel_unstamped"

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.get_logger().info(f"Publishing to: {cmd_topic}")

        # Path visualization
        self.path_pub = self.create_publisher(NavPath, '/predicted_path', 10)

        # Control timer
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_callback)

        self.get_logger().info("Path predictor node initialized")

    def _load_model(self, model_path: Path, model_info_path: str):
        """Load trained model."""
        # Find model info
        if model_info_path:
            info_path = Path(model_info_path)
        else:
            info_path = model_path.parent / "model_info.json"

        if not info_path.exists():
            self.get_logger().error(f"Model info not found: {info_path}")
            raise FileNotFoundError(f"Model info not found: {info_path}")

        with open(info_path, 'r') as f:
            self.model_info = json.load(f)

        # Create model
        self.model = create_model(
            model_type=self.model_info['model_type'],
            input_dim=self.model_info['input_dim'],
            output_dim=self.model_info['output_dim'],
            hidden_dims=self.model_info.get('hidden_dims', [256, 128, 64])
        )

        # Load weights
        if not model_path.exists():
            self.get_logger().error(f"Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.num_waypoints = self.model_info['num_waypoints']
        self.max_range = 8.0  # LiDAR max range

        self.get_logger().info(
            f"Loaded model: {self.model_info['model_type']} "
            f"(waypoints: {self.num_waypoints})"
        )

    def scan_callback(self, msg: LaserScan):
        """Handle incoming LiDAR scan."""
        self.latest_scan = msg

        # Compute angles once
        if self.lidar_angles is None:
            num_rays = len(msg.ranges)
            self.lidar_angles = np.linspace(
                msg.angle_min,
                msg.angle_max,
                num_rays,
                endpoint=False
            )

    def check_emergency_stop(self, ranges: np.ndarray, angles: np.ndarray) -> bool:
        """Check if emergency stop is needed."""
        # Front cone check
        front_mask = np.abs(angles) < self.emergency_stop_angle

        if np.sum(front_mask) == 0:
            return False

        front_ranges = ranges[front_mask]
        valid_ranges = front_ranges[np.isfinite(front_ranges)]

        if len(valid_ranges) == 0:
            return False

        close_ratio = np.sum(valid_ranges < self.emergency_stop_distance) / len(valid_ranges)
        return close_ratio >= self.emergency_stop_ratio

    @torch.no_grad()
    def predict_waypoints(self, ranges: np.ndarray) -> np.ndarray:
        """Predict waypoints from LiDAR ranges."""
        # Normalize
        ranges_normalized = ranges / self.max_range
        ranges_tensor = torch.from_numpy(
            ranges_normalized.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        # Predict
        output = self.model(ranges_tensor)
        waypoints = output.cpu().numpy().reshape(self.num_waypoints, 2)

        return waypoints

    def compute_control(self, waypoints: np.ndarray) -> Tuple[float, float]:
        """Compute control command using pure pursuit.

        Args:
            waypoints: Predicted waypoints in robot frame (x forward, y left)

        Returns:
            Tuple of (linear_vel, angular_vel)
        """
        if len(waypoints) == 0:
            return 0.0, 0.0

        # Find lookahead point
        lookahead_point = None

        for wp in waypoints:
            dist = np.sqrt(wp[0]**2 + wp[1]**2)
            if dist >= self.lookahead_distance:
                lookahead_point = wp
                break

        if lookahead_point is None:
            lookahead_point = waypoints[-1]

        # Pure pursuit
        x, y = lookahead_point
        L = np.sqrt(x**2 + y**2)

        if L < 0.01:
            return 0.0, 0.0

        # Curvature
        curvature = 2.0 * y / (L * L)

        # Linear velocity (reduce near obstacles or high curvature)
        linear_vel = self.max_linear_vel
        linear_vel *= min(1.0, 1.0 / (1.0 + abs(curvature) * 2.0))

        # Angular velocity
        angular_vel = linear_vel * curvature
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)

        return float(linear_vel), float(angular_vel)

    def publish_path(self, waypoints: np.ndarray):
        """Publish predicted path for visualization."""
        path_msg = NavPath()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = f"{self.ns}/base_link"

        for wp in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def control_callback(self):
        """Main control loop."""
        if self.latest_scan is None:
            return

        # Get ranges
        ranges = np.array(self.latest_scan.ranges, dtype=np.float32)

        # Replace inf/nan with max range
        ranges = np.where(np.isfinite(ranges), ranges, self.max_range)
        ranges = np.clip(ranges, 0.0, self.max_range)

        # Emergency stop check
        if self.check_emergency_stop(ranges, self.lidar_angles):
            self.get_logger().warn("Emergency stop triggered!")
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # Predict waypoints
        try:
            waypoints = self.predict_waypoints(ranges)
        except Exception as e:
            self.get_logger().error(f"Prediction failed: {e}")
            return

        # Publish path visualization
        self.publish_path(waypoints)

        # Compute and publish control
        linear_vel, angular_vel = self.compute_control(waypoints)

        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = PathPredictorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

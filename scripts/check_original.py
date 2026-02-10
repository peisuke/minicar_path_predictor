#!/usr/bin/env python3
"""Check original minicar_navigation candidate count."""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    DistanceFieldGenerator, DistanceFieldParams
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig
from minicar_path_predictor.utils import world_to_pixel


def main():
    course_path = Path("data/images/course.png")

    lidar_config = LiDARConfig(
        num_rays=360, angle_min=0.0, angle_max=2 * np.pi,
        range_min=0.15, range_max=12.0, noise_std=0.005,
        blind_spot_start=115.0, blind_spot_end=246.0
    )

    print("Loading course...")
    generator = CourseDataGenerator(course_path, resolution=0.02, lidar_config=lidar_config)

    np.random.seed(42)
    candidates = generator.generate_pose_candidates_grid(grid_spacing=0.1, track_only=True)
    indices = np.random.choice(len(candidates), 3, replace=False)
    pose = candidates[indices[1]]  # Pose 2

    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    print(f"Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°")

    # Simulate LiDAR to get view_points
    lidar_ranges = generator.simulate_lidar(robot_x, robot_y, robot_yaw, add_noise=False)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    # Convert to view_points (in robot frame, then to image coordinates)
    local_size = 200
    scale = 1.0 / generator.resolution  # pixels per meter
    center_x, center_y = local_size // 2, local_size // 2

    view_points = []
    for i, (r, a) in enumerate(zip(lidar_ranges, angles)):
        if r < lidar_config.range_max - 0.1:
            # Robot frame: x forward, y left
            x_robot = r * np.cos(a)
            y_robot = r * np.sin(a)
            # Image frame: x right, y down
            px = center_x + x_robot * scale
            py = center_y - y_robot * scale
            view_points.append([px, py])

    view_points = np.array(view_points, dtype=np.int32)
    print(f"View points: {len(view_points)}")

    # Use original DistanceFieldGenerator
    params = DistanceFieldParams(
        H=local_size,
        W=local_size,
        R_in=np.array([25, 50, 75, 100, 125, 150], dtype=np.int32),
        ring_thickness=1,
        front_deg=90.0,
        dist_thresh=15.0,
        border_thickness=1
    )

    dist_field_gen = DistanceFieldGenerator()
    mask, dist_inside, dist_inside_norm, cx, cy, border = dist_field_gen.generate_distance_field(
        view_points, params
    )

    print(f"\nOriginal DistanceFieldGenerator results:")
    print(f"  mask sum: {np.sum(mask > 0)}")
    print(f"  dist_inside max: {dist_inside.max():.2f}")

    # Check candidates per ring using original method
    H, W = local_size, local_size
    yy, xx = np.mgrid[:H, :W]
    delta_x = xx - cx
    delta_y = yy - cy
    radius_squared = delta_x ** 2 + delta_y ** 2

    # Front mask (original uses delta_x, delta_y directly)
    front_deg = params.front_deg
    front_rad = np.radians(front_deg / 2)
    angles_grid = np.arctan2(-delta_y, delta_x)  # Note: -delta_y for image coords
    front_mask = np.abs(angles_grid) <= front_rad

    print(f"\nCandidate counts per ring (original method):")
    for i, R in enumerate(params.R_in):
        ring_mask = (radius_squared >= (R - params.ring_thickness) ** 2) & \
                    (radius_squared <= (R + params.ring_thickness) ** 2)
        candidates = ring_mask & front_mask & (mask > 0)
        n = np.sum(candidates)
        print(f"  Ring {i+1} (R={R}px): {n} candidates")


if __name__ == "__main__":
    main()

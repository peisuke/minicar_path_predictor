#!/usr/bin/env python3
"""Check original minicar_navigation peak count."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PeakDetector, PeakDetectorParams, PeakParams, PathPlannerConfig
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

    # Simulate LiDAR
    lidar_ranges = generator.simulate_lidar(robot_x, robot_y, robot_yaw, add_noise=False)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    local_size = 200
    scale = 1.0 / generator.resolution
    center_x, center_y = local_size // 2, local_size // 2

    view_points = []
    for r, a in zip(lidar_ranges, angles):
        if r < lidar_config.range_max - 0.1:
            x_robot = r * np.cos(a)
            y_robot = r * np.sin(a)
            px = center_x + x_robot * scale
            py = center_y - y_robot * scale
            view_points.append([px, py])

    view_points = np.array(view_points, dtype=np.int32)
    print(f"View points: {len(view_points)}")

    # Use PeakDetector
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    peak_params = PeakParams()
    peak_detector = PeakDetector(params=peak_params, config=config)

    detector_params = PeakDetectorParams(
        H=local_size,
        W=local_size,
        R_in=np.array([25, 50, 75, 100, 125, 150], dtype=np.int32),
        ring_thickness=1,
        front_deg=90.0,
        dist_thresh=15.0,
        border_thickness=1,
        min_area=10,
        kernel_sz=3,
        smooth=True,
        peak_offs=(1, 3, 5)
    )

    peaks, centered_points, sorted_labels, dist_inside, dist_norm, cx, cy = \
        peak_detector.detect_from_view_points(view_points, detector_params)

    print(f"\nOriginal PeakDetector results:")
    print(f"  Number of peaks (candidate points): {len(peaks)}")
    print(f"  Peaks: {peaks}")
    print(f"  Centered points shape: {centered_points.shape if len(centered_points) > 0 else 'empty'}")
    if len(centered_points) > 0:
        print(f"  Centered points:\n{centered_points}")


if __name__ == "__main__":
    main()

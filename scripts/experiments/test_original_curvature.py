#!/usr/bin/env python3
"""Test curvature calculation with original pipeline."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PathPipeline, PeakParams, PathPlannerConfig, PeakDetectorParams
)
from minicar_navigation.controller.pd_pursuit_controller import (
    PDPursuitController, PDPursuitControllerConfig
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig


def main():
    course_path = Path("data/images/course.png")
    lookahead_distance = 0.5

    lidar_config = LiDARConfig(
        num_rays=360, angle_min=0.0, angle_max=2 * np.pi,
        range_min=0.15, range_max=12.0, noise_std=0.005,
        blind_spot_start=115.0, blind_spot_end=246.0
    )

    print("Loading course...")
    generator = CourseDataGenerator(course_path, resolution=0.02, lidar_config=lidar_config)

    local_size = 200
    resolution = generator.resolution
    scale = 1.0 / resolution
    center_x, center_y = local_size // 2, local_size // 2

    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    pipeline = PathPipeline(params=PeakParams(), config=config)

    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    dist_thresh = 15.0

    np.random.seed(42)
    candidates = generator.generate_pose_candidates_grid(grid_spacing=0.1, track_only=True)
    indices = np.random.choice(len(candidates), 5, replace=False)

    controller_config = PDPursuitControllerConfig(LOOKAHEAD_DISTANCE=lookahead_distance)
    controller = PDPursuitController(controller_config)

    print(f"\nTesting ORIGINAL pipeline with lookahead={lookahead_distance}m...\n")

    for i, idx in enumerate(indices):
        pose = candidates[idx]
        robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw

        # Simulate LiDAR
        lidar_ranges = generator.simulate_lidar(robot_x, robot_y, robot_yaw, add_noise=False)
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

        view_points = []
        for r, a in zip(lidar_ranges, angles):
            if r < generator.lidar_config.range_max - 0.1:
                x_robot = r * np.cos(a)
                y_robot = r * np.sin(a)
                px = center_x + x_robot * scale
                py = center_y - y_robot * scale
                view_points.append([px, py])
        view_points = np.array(view_points, dtype=np.int32)

        # Run ORIGINAL pipeline
        result = pipeline.run_pipeline(
            view_points=view_points, H=local_size, W=local_size,
            R_in=R_in, ring_thickness=1, front_deg=90.0,
            path_front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
        )

        smoothed_paths = result['paths']  # centered coordinates

        print(f"Pose {i+1}: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°")

        if len(smoothed_paths) > 0:
            # Convert to robot frame (meters)
            path_centered = smoothed_paths[0]  # centered pixel coords
            path_robot = np.zeros_like(path_centered, dtype=np.float32)
            path_robot[:, 0] = path_centered[:, 0] * resolution
            path_robot[:, 1] = -path_centered[:, 1] * resolution

            print(f"  path_len={len(path_robot)}")
            print(f"  path[0]={path_robot[0]}, path[-1]={path_robot[-1]}")

            # Find lookahead point
            target_point, nearest_idx = controller._find_lookahead_point(path_robot, 0.0, 0.0)

            if target_point is not None:
                target_angle = np.arctan2(target_point[1], target_point[0])
                curvature = controller._compute_curvature(path_robot, nearest_idx)

                print(f"  nearest_idx={nearest_idx}")
                print(f"  target_angle={np.degrees(target_angle):.2f}°")
                print(f"  curvature={curvature:.4f} [1/m]")
            else:
                print("  target_point=None")
        else:
            print("  No paths generated")
        print()


if __name__ == "__main__":
    main()

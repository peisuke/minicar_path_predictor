#!/usr/bin/env python3
"""Debug ring selection with original parameters."""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig
from minicar_path_predictor.utils import world_to_pixel


def debug_pose(generator: CourseDataGenerator, pose):
    """Debug ring selection for a pose."""
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200

    robot_px, robot_py = world_to_pixel(robot_x, robot_y, generator.img_shape, generator.resolution)

    H_local, W_local = local_size, local_size
    center_x, center_y = W_local // 2, H_local // 2

    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)

    yy_local, xx_local = np.mgrid[:H_local, :W_local]
    dx_local = xx_local - center_x
    dy_local = yy_local - center_y

    dx_world = cos_yaw * dx_local - sin_yaw * (-dy_local)
    dy_world = sin_yaw * dx_local + cos_yaw * (-dy_local)

    src_x = (robot_px + dx_world).astype(np.float32)
    src_y = (robot_py - dy_world).astype(np.float32)

    local_walls = cv2.remap(
        generator.walls_mask.astype(np.float32),
        src_x, src_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    visibility = generator._create_visibility_mask(local_walls, center_x, center_y, local_size)
    wall_dist = distance_transform_edt(visibility > 0)

    radius_squared = dx_local ** 2 + dy_local ** 2
    angles = np.arctan2(-dy_local, dx_local)

    # Original parameters
    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    ring_thickness = 1  # Original uses 1
    front_deg = 90.0
    front_rad = math.radians(front_deg / 2)
    front_mask = np.abs(angles) <= front_rad

    print(f"Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°")
    print(f"Original params: R_in={R_in}, ring_thickness={ring_thickness}, front_deg={front_deg}")
    print()

    for i, R in enumerate(R_in):
        ring_mask = (radius_squared >= (R - ring_thickness) ** 2) & \
                    (radius_squared <= (R + ring_thickness) ** 2)

        candidates = ring_mask & front_mask & (visibility > 0)
        n_candidates = np.sum(candidates)

        print(f"Ring {i+1} (R={R}px = {R * generator.resolution:.2f}m): {n_candidates} candidates")


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
    debug_pose(generator, pose)

    return 0


if __name__ == "__main__":
    exit(main())

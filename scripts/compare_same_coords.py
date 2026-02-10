#!/usr/bin/env python3
"""Compare original vs new with same coordinate system."""

import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PeakDetector, PeakDetectorParams, PeakParams, PathPlannerConfig,
    DistanceFieldGenerator, DistanceFieldParams
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig
from minicar_path_predictor.utils import world_to_pixel


def compare_single_pose(generator, pose, output_path):
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    scale = 1.0 / generator.resolution
    center_x, center_y = local_size // 2, local_size // 2

    robot_px, robot_py = world_to_pixel(robot_x, robot_y, generator.img_shape, generator.resolution)

    # Simulate LiDAR
    lidar_ranges = generator.simulate_lidar(robot_x, robot_y, robot_yaw, add_noise=False)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    # Create view_points in robot frame (same as original)
    view_points = []
    for r, a in zip(lidar_ranges, angles):
        if r < generator.lidar_config.range_max - 0.1:
            x_robot = r * np.cos(a)
            y_robot = r * np.sin(a)
            # Robot frame to image: x->right, y->up becomes y->down
            px = center_x + x_robot * scale
            py = center_y - y_robot * scale
            view_points.append([px, py])
    view_points = np.array(view_points, dtype=np.int32)

    # ========== ORIGINAL: Get distance field ==========
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    dist_field_gen = DistanceFieldGenerator(config)
    df_params = DistanceFieldParams(
        H=local_size, W=local_size,
        R_in=np.array([25, 50, 75, 100, 125, 150], dtype=np.int32),
        ring_thickness=1, front_deg=90.0, dist_thresh=15.0, border_thickness=1
    )
    mask_orig, dist_orig, dist_norm_orig, cx, cy, candidates_orig = \
        dist_field_gen.generate_distance_field(view_points, df_params)

    # ========== NEW: Sample path distance in SAME coordinate system ==========
    # view_points are in robot-local image coords
    # We need to map each pixel in local image to world coords, then sample path_distance_field

    yy, xx = np.mgrid[:local_size, :local_size]
    # Local image coords: center is robot, +x is right (robot forward), +y is down (robot right)
    dx_img = xx - center_x  # pixels from center
    dy_img = yy - center_y

    # Convert to robot frame meters: x_robot = dx_img * res, y_robot = -dy_img * res
    x_robot_m = dx_img * generator.resolution
    y_robot_m = -dy_img * generator.resolution

    # Convert robot frame to world frame
    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)
    x_world = robot_x + x_robot_m * cos_yaw - y_robot_m * sin_yaw
    y_world = robot_y + x_robot_m * sin_yaw + y_robot_m * cos_yaw

    # Convert world to global image pixel coords
    src_x = (x_world / generator.resolution + generator.W / 2).astype(np.float32)
    src_y = (generator.H / 2 - y_world / generator.resolution).astype(np.float32)

    # Sample path distance field
    local_dist_to_path = cv2.remap(
        generator.path_distance_field,
        src_x, src_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1000
    )

    # Create path attraction: max(wall_dist_max - dist_to_path, 0)
    wall_dist_max = dist_orig.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    path_attraction_masked = path_attraction * (mask_orig > 0).astype(np.float32)

    # ========== Select waypoints using rings ==========
    radius_squared = dx_img ** 2 + dy_img ** 2
    # Front mask: +x direction in image (robot forward)
    angles_grid = np.arctan2(-dy_img, dx_img)  # -dy because y increases downward
    front_rad = math.radians(45)
    front_mask = np.abs(angles_grid) <= front_rad

    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    ring_thickness = 1

    # Original waypoints (wall distance based)
    wp_orig = []
    for R in R_in:
        ring_mask = (radius_squared >= (R - ring_thickness) ** 2) & \
                    (radius_squared <= (R + ring_thickness) ** 2)
        candidates = ring_mask & front_mask & (mask_orig > 0) & (dist_orig >= 15.0)
        if np.any(candidates):
            masked = np.where(candidates, dist_orig, -np.inf)
            best_y, best_x = np.unravel_index(np.argmax(masked), masked.shape)
            wp_orig.append((best_x, best_y))

    # New waypoints (path attraction based)
    wp_new = []
    for R in R_in:
        ring_mask = (radius_squared >= (R - ring_thickness) ** 2) & \
                    (radius_squared <= (R + ring_thickness) ** 2)
        candidates = ring_mask & front_mask & (mask_orig > 0)
        if np.any(candidates):
            masked = np.where(candidates, path_attraction_masked, -np.inf)
            best_y, best_x = np.unravel_index(np.argmax(masked), masked.shape)
            wp_new.append((best_x, best_y))

    # ========== Visualization ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original (wall distance)
    ax = axes[0, 0]
    ax.imshow(dist_orig, cmap='hot')
    ax.plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIGINAL: Wall Distance\nmax={dist_orig.max():.1f}')

    ax = axes[0, 1]
    ax.imshow(dist_orig, cmap='hot', alpha=0.7)
    for R in R_in:
        theta = np.linspace(-front_rad, front_rad, 50)
        ax.plot(center_x + R * np.cos(theta), center_y - R * np.sin(theta), 'c-', linewidth=1, alpha=0.5)
    for i, (wx, wy) in enumerate(wp_orig):
        ax.plot(wx, wy, 'go', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (wx, wy), textcoords="offset points", xytext=(5, 5),
                    color='white', fontweight='bold')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIGINAL: {len(wp_orig)} waypoints\n(green, max wall dist)')

    ax = axes[0, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    path_px = generator.ideal_path_pixels
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    arrow_len = 25
    adx = arrow_len * np.cos(robot_yaw)
    ady = -arrow_len * np.sin(robot_yaw)
    ax.arrow(robot_px, robot_py, adx, ady, head_width=8, head_length=5, fc='black', ec='black')
    cos_yaw_fwd = np.cos(robot_yaw)
    sin_yaw_fwd = np.sin(robot_yaw)
    for wx, wy in wp_orig:
        lx = (wx - center_x) * generator.resolution
        ly = -(wy - center_y) * generator.resolution
        world_x = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
        world_y = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
        wpx, wpy = world_to_pixel(world_x, world_y, generator.img_shape, generator.resolution)
        ax.plot(wpx, wpy, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.set_title('ORIGINAL on Course\n(green = corridor center)')
    ax.axis('off')

    # Row 2: New (path attraction)
    ax = axes[1, 0]
    ax.imshow(path_attraction_masked, cmap='hot')
    ax.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: Path Attraction\nmax={path_attraction_masked.max():.1f}')

    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)
    for R in R_in:
        theta = np.linspace(-front_rad, front_rad, 50)
        ax.plot(center_x + R * np.cos(theta), center_y - R * np.sin(theta), 'c-', linewidth=1, alpha=0.5)
    for i, (wx, wy) in enumerate(wp_new):
        ax.plot(wx, wy, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (wx, wy), textcoords="offset points", xytext=(5, 5),
                    color='white', fontweight='bold')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: {len(wp_new)} waypoints\n(red, toward ideal path)')

    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    ax.arrow(robot_px, robot_py, adx, ady, head_width=8, head_length=5, fc='black', ec='black')
    for wx, wy in wp_new:
        lx = (wx - center_x) * generator.resolution
        ly = -(wy - center_y) * generator.resolution
        world_x = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
        world_y = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
        wpx, wpy = world_to_pixel(world_x, world_y, generator.img_shape, generator.resolution)
        ax.plot(wpx, wpy, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.set_title('NEW on Course\n(red = toward ideal path)')
    ax.axis('off')

    plt.suptitle(f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    course_path = Path("data/images/course.png")
    output_dir = Path("data/sim_robot")

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

    for i, idx in enumerate(indices):
        pose = candidates[idx]
        compare_single_pose(generator, pose, output_dir / f"same_coords_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

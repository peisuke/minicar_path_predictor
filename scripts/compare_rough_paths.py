#!/usr/bin/env python3
"""Compare rough_paths (before smoothing) only."""

import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PathPipeline, PeakParams, PathPlannerConfig
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig
from minicar_path_predictor.utils import world_to_pixel


def compare_rough(generator, pose, output_path):
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    scale = 1.0 / generator.resolution
    center_x, center_y = local_size // 2, local_size // 2

    robot_px, robot_py = world_to_pixel(robot_x, robot_y, generator.img_shape, generator.resolution)

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

    # ORIGINAL pipeline
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    pipeline = PathPipeline(params=PeakParams(), config=config)
    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)

    result_orig = pipeline.run_pipeline(
        view_points=view_points, H=local_size, W=local_size,
        R_in=R_in, ring_thickness=1, front_deg=90.0,
        path_front_deg=90.0, dist_thresh=15.0, border_thickness=1
    )

    peaks_orig = result_orig['peaks']
    rough_orig = result_orig['rough_paths']  # coordinate_paths before smoothing
    dist_inside = result_orig['dist_inside']

    # NEW: path attraction + ring selection
    yy, xx = np.mgrid[:local_size, :local_size]
    dx_img = xx - center_x
    dy_img = yy - center_y

    x_robot_m = dx_img * generator.resolution
    y_robot_m = -dy_img * generator.resolution
    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)
    x_world = robot_x + x_robot_m * cos_yaw - y_robot_m * sin_yaw
    y_world = robot_y + x_robot_m * sin_yaw + y_robot_m * cos_yaw

    src_x = (x_world / generator.resolution + generator.W / 2).astype(np.float32)
    src_y = (generator.H / 2 - y_world / generator.resolution).astype(np.float32)

    local_dist_to_path = cv2.remap(
        generator.path_distance_field, src_x, src_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1000
    )

    wall_dist_max = dist_inside.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    mask = dist_inside > 0
    path_attraction_masked = path_attraction * mask.astype(np.float32)

    radius_squared = dx_img ** 2 + dy_img ** 2
    angles_grid = np.arctan2(-dy_img, dx_img)
    front_rad = np.radians(45)
    front_mask = np.abs(angles_grid) <= front_rad

    # Ring-based waypoint selection (same as NEW approach)
    wp_new = []
    for R in R_in:
        ring_mask = (radius_squared >= (R - 1) ** 2) & (radius_squared <= (R + 1) ** 2)
        candidates = ring_mask & front_mask & mask
        if np.any(candidates):
            masked_field = np.where(candidates, path_attraction_masked, -np.inf)
            best_y, best_x = np.unravel_index(np.argmax(masked_field), masked_field.shape)
            wp_new.append([best_x - center_x, best_y - center_y])  # centered coords

    wp_new = np.array(wp_new) if wp_new else np.empty((0, 2))

    # Visualization: 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: ORIGINAL
    ax = axes[0, 0]
    ax.imshow(dist_inside, cmap='hot', alpha=0.7)
    for i, (px, py, val, lab) in enumerate(peaks_orig):
        ax.plot(px, py, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIG: {len(peaks_orig)} peaks')

    ax = axes[0, 1]
    ax.imshow(dist_inside, cmap='hot', alpha=0.5)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rough_orig), 1)))
    for i, path in enumerate(rough_orig):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'o-', color=colors[i],
                    markersize=10, linewidth=2, label=f'Path {i+1}: {len(path)} pts')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'ORIG: {len(rough_orig)} rough_paths')

    ax = axes[0, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    path_px = generator.ideal_path_pixels
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    adx = 25 * np.cos(robot_yaw)
    ady = -25 * np.sin(robot_yaw)
    ax.arrow(robot_px, robot_py, adx, ady, head_width=8, head_length=5, fc='black', ec='black')
    cos_yaw_fwd = np.cos(robot_yaw)
    sin_yaw_fwd = np.sin(robot_yaw)
    for i, path in enumerate(rough_orig):
        if len(path) > 0:
            world_path = []
            for pt in path:
                lx = pt[0] * generator.resolution
                ly = -pt[1] * generator.resolution
                wx = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
                wy = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
                wpx, wpy = world_to_pixel(wx, wy, generator.img_shape, generator.resolution)
                world_path.append([wpx, wpy])
            world_path = np.array(world_path)
            ax.plot(world_path[:, 0], world_path[:, 1], 'o-', color=colors[i], markersize=8, linewidth=2)
    ax.set_title('ORIG on Course')
    ax.axis('off')

    # Row 2: NEW
    ax = axes[1, 0]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)
    if len(wp_new) > 0:
        for i, wp in enumerate(wp_new):
            ax.plot(wp[0] + center_x, wp[1] + center_y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: {len(wp_new)} waypoints')

    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.5)
    if len(wp_new) > 0:
        wp_img = wp_new + np.array([center_x, center_y])
        ax.plot(wp_img[:, 0], wp_img[:, 1], 'ro-', markersize=10, linewidth=2, label=f'Path: {len(wp_new)} pts')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'NEW: 1 path (ring-based)')

    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    ax.arrow(robot_px, robot_py, adx, ady, head_width=8, head_length=5, fc='black', ec='black')
    if len(wp_new) > 0:
        world_path = []
        for pt in wp_new:
            lx = pt[0] * generator.resolution
            ly = -pt[1] * generator.resolution
            wx = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
            wy = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
            wpx, wpy = world_to_pixel(wx, wy, generator.img_shape, generator.resolution)
            world_path.append([wpx, wpy])
        world_path = np.array(world_path)
        ax.plot(world_path[:, 0], world_path[:, 1], 'ro-', markersize=8, linewidth=2)
    ax.set_title('NEW on Course')
    ax.axis('off')

    plt.suptitle(f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°\n'
                 f'rough_paths (before smoothing)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    print(f"  ORIG rough_paths: {[len(p) for p in rough_orig]}")
    print(f"  NEW waypoints: {len(wp_new)}")


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
        compare_rough(generator, pose, output_dir / f"rough_paths_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

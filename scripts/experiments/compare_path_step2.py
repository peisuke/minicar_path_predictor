#!/usr/bin/env python3
"""Compare path building using original PathPipeline."""

import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PathPipeline, PeakParams, PathPlannerConfig, PotentialPathSmoother
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig
from minicar_path_predictor.utils import world_to_pixel


def compare_paths(generator, pose, output_path):
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

    # ========== ORIGINAL: Use PathPipeline directly ==========
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    pipeline = PathPipeline(params=PeakParams(), config=config)

    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    dist_thresh = 15.0

    result_orig = pipeline.run_pipeline(
        view_points=view_points,
        H=local_size, W=local_size,
        R_in=R_in,
        ring_thickness=1,
        front_deg=90.0,
        path_front_deg=90.0,
        dist_thresh=dist_thresh,
        border_thickness=1
    )

    peaks_orig = result_orig['peaks']
    centered_orig = result_orig['peaks_centered']
    paths_orig_smooth = result_orig['paths']  # smoothed
    paths_orig = result_orig['rough_paths']   # before smoothing
    dist_inside_orig = result_orig['dist_inside']
    # center is at local_size // 2
    cx_orig = center_x
    cy_orig = center_y

    print(f"  rough_paths: {[len(p) for p in paths_orig]}")
    print(f"  smoothed_paths: {[len(p) for p in paths_orig_smooth]}")

    # ========== NEW: Sample path distance field ==========
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

    # Create path attraction field
    wall_dist_max = dist_inside_orig.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)

    # Use same mask as original
    mask = dist_inside_orig > 0
    path_attraction_masked = path_attraction * mask.astype(np.float32)

    # For NEW, select waypoints per ring (simpler approach matching what we validated)
    radius_squared = dx_img ** 2 + dy_img ** 2
    angles_grid = np.arctan2(-dy_img, dx_img)
    front_rad = np.radians(45)
    front_mask = np.abs(angles_grid) <= front_rad

    wp_new = []
    for R in R_in:
        ring_mask = (radius_squared >= (R - 1) ** 2) & (radius_squared <= (R + 1) ** 2)
        candidates = ring_mask & front_mask & mask
        if np.any(candidates):
            masked = np.where(candidates, path_attraction_masked, -np.inf)
            best_y, best_x = np.unravel_index(np.argmax(masked), masked.shape)
            wp_new.append([best_x - center_x, best_y - center_y])  # centered coords

    wp_new = np.array(wp_new) if wp_new else np.empty((0, 2))

    # Apply same smoothing as original
    smoother = PotentialPathSmoother(config)
    if len(wp_new) > 0:
        wp_new_smooth = smoother.smooth_paths([wp_new], path_attraction_masked, local_size, local_size)
    else:
        wp_new_smooth = []

    # ========== Visualization ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original (use smoothed paths)
    ax = axes[0, 0]
    ax.imshow(dist_inside_orig, cmap='hot', alpha=0.7)
    for i, (px, py, val, lab) in enumerate(peaks_orig):
        ax.plot(px, py, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.plot(cx_orig, cy_orig, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIG: Wall Distance\n{len(peaks_orig)} peaks')

    ax = axes[0, 1]
    ax.imshow(dist_inside_orig, cmap='hot', alpha=0.5)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(paths_orig_smooth), 1)))
    for i, path in enumerate(paths_orig_smooth):
        if len(path) > 0:
            path_img = path + np.array([cx_orig, cy_orig])
            ax.plot(path_img[:, 0], path_img[:, 1], 'o-', color=colors[i],
                    markersize=6, linewidth=2)
    ax.plot(cx_orig, cy_orig, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIG: {len(paths_orig_smooth)} paths (smoothed)')

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
    for i, path in enumerate(paths_orig_smooth):
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
            ax.plot(world_path[:, 0], world_path[:, 1], 'o-', color=colors[i], markersize=4, linewidth=2)
    ax.set_title('ORIG on Course (green)')
    ax.axis('off')

    # Row 2: New (use smoothed paths)
    ax = axes[1, 0]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)
    for i, wp in enumerate(wp_new):
        ax.plot(wp[0] + center_x, wp[1] + center_y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: Path Attraction\n{len(wp_new)} waypoints')

    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.5)
    for i, path in enumerate(wp_new_smooth):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'ro-', markersize=6, linewidth=2)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: {len(wp_new_smooth)} paths (smoothed)')

    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    ax.arrow(robot_px, robot_py, adx, ady, head_width=8, head_length=5, fc='black', ec='black')
    for i, path in enumerate(wp_new_smooth):
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
            ax.plot(world_path[:, 0], world_path[:, 1], 'ro-', markersize=4, linewidth=2)
    ax.set_title('NEW on Course (red)')
    ax.axis('off')

    plt.suptitle(f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    print(f"  ORIG: {len(peaks_orig)} peaks -> {len(paths_orig_smooth)} paths (smoothed)")
    print(f"  NEW:  {len(wp_new)} waypoints -> {len(wp_new_smooth)} paths (smoothed)")


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
        compare_paths(generator, pose, output_dir / f"path_compare_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

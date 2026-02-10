#!/usr/bin/env python3
"""Overlay raw waypoints and smoothed paths."""

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


def compare_overlay(generator, pose, output_path):
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
    rough_orig = result_orig['rough_paths']
    smooth_orig = result_orig['paths']
    dist_inside = result_orig['dist_inside']

    # NEW: path attraction
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

    wp_new_raw = []
    for R in R_in:
        ring_mask = (radius_squared >= (R - 1) ** 2) & (radius_squared <= (R + 1) ** 2)
        candidates = ring_mask & front_mask & mask
        if np.any(candidates):
            masked = np.where(candidates, path_attraction_masked, -np.inf)
            best_y, best_x = np.unravel_index(np.argmax(masked), masked.shape)
            wp_new_raw.append([best_x - center_x, best_y - center_y])

    wp_new_raw = np.array(wp_new_raw) if wp_new_raw else np.empty((0, 2))

    smoother = PotentialPathSmoother(config)
    if len(wp_new_raw) > 0:
        wp_new_smooth = smoother.smooth_paths([wp_new_raw], path_attraction_masked, local_size, local_size)
    else:
        wp_new_smooth = []

    # Visualization: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Top left: ORIG raw + smooth overlay
    ax = axes[0, 0]
    ax.imshow(dist_inside, cmap='hot', alpha=0.5)
    for i, path in enumerate(rough_orig):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'go-', markersize=12, linewidth=3, label=f'Raw {i+1}' if i == 0 else None)
    for i, path in enumerate(smooth_orig):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'c.-', markersize=4, linewidth=1, label=f'Smooth {i+1}' if i == 0 else None)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.legend()
    ax.set_title(f'ORIG: Raw (green) vs Smooth (cyan)\nRaw: {[len(p) for p in rough_orig]}, Smooth: {[len(p) for p in smooth_orig]}')

    # Top right: NEW raw + smooth overlay
    ax = axes[0, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.5)
    if len(wp_new_raw) > 0:
        wp_img = wp_new_raw + np.array([center_x, center_y])
        ax.plot(wp_img[:, 0], wp_img[:, 1], 'ro-', markersize=12, linewidth=3, label='Raw')
    for i, path in enumerate(wp_new_smooth):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'm.-', markersize=4, linewidth=1, label=f'Smooth' if i == 0 else None)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.legend()
    ax.set_title(f'NEW: Raw (red) vs Smooth (magenta)\nRaw: {len(wp_new_raw)}, Smooth: {[len(p) for p in wp_new_smooth]}')

    # Bottom: on course
    ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    path_px = generator.ideal_path_pixels
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    adx = 25 * np.cos(robot_yaw)
    ady = -25 * np.sin(robot_yaw)
    ax.arrow(robot_px, robot_py, adx, ady, head_width=8, head_length=5, fc='black', ec='black')
    cos_yaw_fwd = np.cos(robot_yaw)
    sin_yaw_fwd = np.sin(robot_yaw)
    # ORIG raw
    for path in rough_orig:
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
            ax.plot(world_path[:, 0], world_path[:, 1], 'go-', markersize=10, linewidth=2)
    # ORIG smooth
    for path in smooth_orig:
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
            ax.plot(world_path[:, 0], world_path[:, 1], 'c.-', markersize=3, linewidth=1)
    ax.set_title('ORIG on Course')
    ax.axis('off')

    ax = axes[1, 1]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    ax.arrow(robot_px, robot_py, adx, ady, head_width=8, head_length=5, fc='black', ec='black')
    # NEW raw
    if len(wp_new_raw) > 0:
        world_path = []
        for pt in wp_new_raw:
            lx = pt[0] * generator.resolution
            ly = -pt[1] * generator.resolution
            wx = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
            wy = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
            wpx, wpy = world_to_pixel(wx, wy, generator.img_shape, generator.resolution)
            world_path.append([wpx, wpy])
        world_path = np.array(world_path)
        ax.plot(world_path[:, 0], world_path[:, 1], 'ro-', markersize=10, linewidth=2)
    # NEW smooth
    for path in wp_new_smooth:
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
            ax.plot(world_path[:, 0], world_path[:, 1], 'm.-', markersize=3, linewidth=1)
    ax.set_title('NEW on Course')
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
        compare_overlay(generator, pose, output_dir / f"smooth_overlay_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

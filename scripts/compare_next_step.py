#!/usr/bin/env python3
"""Compare next steps: threshold -> connected regions -> peaks."""

import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    DistanceFieldGenerator, DistanceFieldParams, PathPlannerConfig
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig
from minicar_path_predictor.utils import world_to_pixel


def compare_steps(generator, pose, output_path):
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

    # Get distance fields
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    dist_field_gen = DistanceFieldGenerator(config)

    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    dist_thresh = 15.0

    df_params = DistanceFieldParams(
        H=local_size, W=local_size, R_in=R_in,
        ring_thickness=1, front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
    )
    mask_orig, dist_orig, dist_norm_orig, cx, cy, candidates_orig = \
        dist_field_gen.generate_distance_field(view_points, df_params)

    # Sample path distance field in same coords
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

    wall_dist_max = dist_orig.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    path_attraction_masked = path_attraction * (mask_orig > 0).astype(np.float32)

    # Compute ring and front masks
    radius_squared = dx_img ** 2 + dy_img ** 2
    angles_grid = np.arctan2(-dy_img, dx_img)
    front_rad = np.radians(45)
    front_mask = np.abs(angles_grid) <= front_rad

    ring_mask = np.zeros((local_size, local_size), dtype=bool)
    for R in R_in:
        ring_mask |= (radius_squared >= (R - 1) ** 2) & (radius_squared <= (R + 1) ** 2)

    # ========== Step 1: Ring + Front mask ==========
    ring_front_orig = ring_mask & front_mask & (mask_orig > 0)
    ring_front_new = ring_mask & front_mask & (mask_orig > 0)

    # ========== Step 2: Threshold filter ==========
    thresh_orig = ring_front_orig & (dist_orig >= dist_thresh)
    # For NEW, use same threshold concept but on path_attraction
    path_thresh = path_attraction_masked.max() * 0.3  # relative threshold
    thresh_new = ring_front_new & (path_attraction_masked >= path_thresh)

    # ========== Step 3: Connected components ==========
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    thresh_orig_uint8 = (thresh_orig.astype(np.uint8) * 255)
    thresh_orig_closed = cv2.morphologyEx(thresh_orig_uint8, cv2.MORPH_CLOSE, kernel)
    num_orig, labels_orig = cv2.connectedComponents(thresh_orig_closed)

    thresh_new_uint8 = (thresh_new.astype(np.uint8) * 255)
    thresh_new_closed = cv2.morphologyEx(thresh_new_uint8, cv2.MORPH_CLOSE, kernel)
    num_new, labels_new = cv2.connectedComponents(thresh_new_closed)

    # ========== Step 4: Find peaks in each region ==========
    def find_peaks_per_region(labels, dist_field, num_labels):
        peaks = []
        for lab in range(1, num_labels):
            region = labels == lab
            if np.sum(region) < 10:  # min_area
                continue
            masked = np.where(region, dist_field, -np.inf)
            best_idx = np.unravel_index(np.argmax(masked), masked.shape)
            peaks.append((best_idx[1], best_idx[0], dist_field[best_idx]))  # x, y, value
        return peaks

    peaks_orig = find_peaks_per_region(labels_orig, dist_orig, num_orig)
    peaks_new = find_peaks_per_region(labels_new, path_attraction_masked, num_new)

    # ========== Visualization ==========
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Original
    ax = axes[0, 0]
    ax.imshow(dist_orig, cmap='hot')
    ax.contour(ring_front_orig, levels=[0.5], colors='cyan', linewidths=1)
    ax.set_title(f'ORIG Step1: Ring+Front\n{np.sum(ring_front_orig)} pixels')
    ax.plot(center_x, center_y, 'w+', markersize=10)

    ax = axes[0, 1]
    ax.imshow(dist_orig, cmap='hot')
    ax.contour(thresh_orig, levels=[0.5], colors='yellow', linewidths=1)
    ax.set_title(f'ORIG Step2: Threshold >= {dist_thresh}\n{np.sum(thresh_orig)} pixels')
    ax.plot(center_x, center_y, 'w+', markersize=10)

    ax = axes[0, 2]
    ax.imshow(labels_orig, cmap='tab20')
    ax.set_title(f'ORIG Step3: Connected\n{num_orig - 1} regions')
    ax.plot(center_x, center_y, 'w+', markersize=10)

    ax = axes[0, 3]
    ax.imshow(dist_orig, cmap='hot', alpha=0.7)
    for i, (px, py, val) in enumerate(peaks_orig):
        ax.plot(px, py, 'go', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (px, py), textcoords="offset points", xytext=(5, 5),
                    color='white', fontweight='bold', fontsize=12)
    ax.set_title(f'ORIG Step4: Peaks\n{len(peaks_orig)} peaks')
    ax.plot(center_x, center_y, 'w+', markersize=10)

    # Row 2: New
    ax = axes[1, 0]
    ax.imshow(path_attraction_masked, cmap='hot')
    ax.contour(ring_front_new, levels=[0.5], colors='cyan', linewidths=1)
    ax.set_title(f'NEW Step1: Ring+Front\n{np.sum(ring_front_new)} pixels')
    ax.plot(center_x, center_y, 'w+', markersize=10)

    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot')
    ax.contour(thresh_new, levels=[0.5], colors='yellow', linewidths=1)
    ax.set_title(f'NEW Step2: Threshold >= {path_thresh:.1f}\n{np.sum(thresh_new)} pixels')
    ax.plot(center_x, center_y, 'w+', markersize=10)

    ax = axes[1, 2]
    ax.imshow(labels_new, cmap='tab20')
    ax.set_title(f'NEW Step3: Connected\n{num_new - 1} regions')
    ax.plot(center_x, center_y, 'w+', markersize=10)

    ax = axes[1, 3]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)
    for i, (px, py, val) in enumerate(peaks_new):
        ax.plot(px, py, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (px, py), textcoords="offset points", xytext=(5, 5),
                    color='white', fontweight='bold', fontsize=12)
    ax.set_title(f'NEW Step4: Peaks\n{len(peaks_new)} peaks')
    ax.plot(center_x, center_y, 'w+', markersize=10)

    plt.suptitle(f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°\n'
                 f'ORIGINAL (top) vs NEW (bottom)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    print(f"  ORIG: {np.sum(ring_front_orig)} -> {np.sum(thresh_orig)} -> {num_orig-1} regions -> {len(peaks_orig)} peaks")
    print(f"  NEW:  {np.sum(ring_front_new)} -> {np.sum(thresh_new)} -> {num_new-1} regions -> {len(peaks_new)} peaks")


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
        compare_steps(generator, pose, output_dir / f"steps_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

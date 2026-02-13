#!/usr/bin/env python3
"""Compare original PeakDetector vs new path attraction method."""

import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

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

    view_points = []
    for r, a in zip(lidar_ranges, angles):
        if r < generator.lidar_config.range_max - 0.1:
            x_robot = r * np.cos(a)
            y_robot = r * np.sin(a)
            px = center_x + x_robot * scale
            py = center_y - y_robot * scale
            view_points.append([px, py])
    view_points = np.array(view_points, dtype=np.int32)

    # ========== ORIGINAL: PeakDetector ==========
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    peak_detector = PeakDetector(params=PeakParams(), config=config)

    detector_params = PeakDetectorParams(
        H=local_size, W=local_size,
        R_in=np.array([25, 50, 75, 100, 125, 150], dtype=np.int32),
        ring_thickness=1, front_deg=90.0, dist_thresh=15.0,
        border_thickness=1, min_area=10, kernel_sz=3,
        smooth=True, peak_offs=(1, 3, 5)
    )

    peaks, centered_points, sorted_labels, dist_inside, dist_norm, cx, cy = \
        peak_detector.detect_from_view_points(view_points, detector_params)

    # Also get the distance field for visualization
    dist_field_gen = DistanceFieldGenerator(config)
    df_params = DistanceFieldParams(
        H=local_size, W=local_size,
        R_in=np.array([25, 50, 75, 100, 125, 150], dtype=np.int32),
        ring_thickness=1, front_deg=90.0, dist_thresh=15.0, border_thickness=1
    )
    mask_orig, dist_orig, dist_norm_orig, _, _, candidates_orig = \
        dist_field_gen.generate_distance_field(view_points, df_params)

    # ========== NEW: Path Attraction ==========
    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)

    yy, xx = np.mgrid[:local_size, :local_size]
    dx_local = xx - center_x
    dy_local = yy - center_y

    dx_world = cos_yaw * dx_local - sin_yaw * (-dy_local)
    dy_world = sin_yaw * dx_local + cos_yaw * (-dy_local)

    src_x = (robot_px + dx_world).astype(np.float32)
    src_y = (robot_py - dy_world).astype(np.float32)

    local_dist_to_path = cv2.remap(
        generator.path_distance_field,
        src_x, src_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1000
    )

    # Use original's mask
    wall_dist_max = dist_orig.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    path_attraction_masked = path_attraction * (mask_orig > 0).astype(np.float32)

    # Select waypoints using rings (same as original)
    radius_squared = dx_local ** 2 + dy_local ** 2
    angles_grid = np.arctan2(-dy_local, dx_local)
    front_rad = math.radians(45)
    front_mask = np.abs(angles_grid) <= front_rad

    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    ring_thickness = 1

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

    # Row 1: Original
    ax = axes[0, 0]
    ax.imshow(dist_orig, cmap='hot')
    ax.plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIGINAL: Wall Distance\nmax={dist_orig.max():.1f}')

    ax = axes[0, 1]
    ax.imshow(dist_orig, cmap='hot', alpha=0.7)
    # Show candidate region
    ax.contour(candidates_orig, levels=[0.5], colors='cyan', linewidths=1)
    # Show peaks
    for i, (px, py, val, lab) in enumerate(peaks):
        ax.plot(px, py, 'go', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (px, py), textcoords="offset points", xytext=(5, 5),
                    color='white', fontweight='bold', fontsize=12)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIGINAL: {len(peaks)} peaks\n(green, cyan=candidates)')

    ax = axes[0, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    path_px = generator.ideal_path_pixels
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    arrow_len = 25
    dx = arrow_len * np.cos(robot_yaw)
    dy = -arrow_len * np.sin(robot_yaw)
    ax.arrow(robot_px, robot_py, dx, dy, head_width=8, head_length=5, fc='black', ec='black')
    # Convert peaks to world
    cos_yaw_fwd = np.cos(robot_yaw)
    sin_yaw_fwd = np.sin(robot_yaw)
    for px, py, val, lab in peaks:
        lx = (px - center_x) * generator.resolution
        ly = -(py - center_y) * generator.resolution
        wx = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
        wy = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
        wpx, wpy = world_to_pixel(wx, wy, generator.img_shape, generator.resolution)
        ax.plot(wpx, wpy, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.set_title(f'ORIGINAL on Course\n({len(peaks)} green points)')
    ax.axis('off')

    # Row 2: New
    ax = axes[1, 0]
    ax.imshow(path_attraction_masked, cmap='hot')
    ax.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: Path Attraction\nmax={path_attraction_masked.max():.1f}')

    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)
    # Show rings
    for R in R_in:
        theta = np.linspace(-front_rad, front_rad, 50)
        ax.plot(center_x + R * np.cos(theta), center_y - R * np.sin(theta), 'c-', linewidth=1, alpha=0.5)
    for i, (wx, wy) in enumerate(wp_new):
        ax.plot(wx, wy, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (wx, wy), textcoords="offset points", xytext=(5, 5),
                    color='white', fontweight='bold', fontsize=10)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: {len(wp_new)} waypoints\n(red, per ring)')

    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    ax.arrow(robot_px, robot_py, dx, dy, head_width=8, head_length=5, fc='black', ec='black')
    for wx, wy in wp_new:
        lx = (wx - center_x) * generator.resolution
        ly = -(wy - center_y) * generator.resolution
        world_x = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
        world_y = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
        wpx, wpy = world_to_pixel(world_x, world_y, generator.img_shape, generator.resolution)
        ax.plot(wpx, wpy, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
    ax.set_title(f'NEW on Course\n({len(wp_new)} red points)')
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
        compare_single_pose(generator, pose, output_dir / f"orig_vs_new_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

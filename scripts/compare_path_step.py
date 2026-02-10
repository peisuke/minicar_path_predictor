#!/usr/bin/env python3
"""Compare path building step: peaks -> graph -> paths."""

import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    DistanceFieldGenerator, DistanceFieldParams, PathPlannerConfig,
    PeakDetector, PeakDetectorParams, PeakParams,
    GraphPathSearcher
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

    # ========== ORIGINAL: Full pipeline ==========
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    peak_detector = PeakDetector(params=PeakParams(), config=config)
    graph_searcher = GraphPathSearcher(config)

    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    dist_thresh = 15.0

    detector_params = PeakDetectorParams(
        H=local_size, W=local_size, R_in=R_in,
        ring_thickness=1, front_deg=90.0, dist_thresh=dist_thresh,
        border_thickness=1, min_area=10, kernel_sz=3,
        smooth=True, peak_offs=(1, 3, 5)
    )

    peaks_orig, centered_points_orig, sorted_labels_orig, dist_inside_orig, dist_norm_orig, cx, cy = \
        peak_detector.detect_from_view_points(view_points, detector_params)

    # Build paths from peaks
    if len(centered_points_orig) > 0:
        # Get labels for path building
        df_params = detector_params.to_distance_field_params()
        dist_field_gen = DistanceFieldGenerator(config)
        mask_orig, _, _, _, _, candidates_orig = dist_field_gen.generate_distance_field(view_points, df_params)

        # Connected components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cand_uint8 = (candidates_orig.astype(np.uint8) * 255)
        cand_closed = cv2.morphologyEx(cand_uint8, cv2.MORPH_CLOSE, kernel)
        num_labels, labels_orig = cv2.connectedComponents(cand_closed)

        try:
            paths_orig = graph_searcher.build_and_deduplicate_paths(
                labels=labels_orig,
                centered_points=centered_points_orig,
                dist_inside=dist_inside_orig,
                center_x=cx, center_y=cy,
                dist_thresh=dist_thresh,
                path_front_deg=90.0,
                start_ids=(0, 1, 2)
            )
        except Exception as e:
            print(f"  ORIG path building failed: {e}")
            paths_orig = []
    else:
        paths_orig = []
        labels_orig = np.zeros((local_size, local_size), dtype=np.int32)

    # ========== NEW: Path attraction based ==========
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

    # Get mask from original
    df_params = DistanceFieldParams(
        H=local_size, W=local_size, R_in=R_in,
        ring_thickness=1, front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
    )
    dist_field_gen = DistanceFieldGenerator(config)
    mask_new, dist_wall, _, _, _, _ = dist_field_gen.generate_distance_field(view_points, df_params)

    wall_dist_max = dist_wall.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    path_attraction_masked = path_attraction * (mask_new > 0).astype(np.float32)

    # Ring and front masks
    radius_squared = dx_img ** 2 + dy_img ** 2
    angles_grid = np.arctan2(-dy_img, dx_img)
    front_rad = np.radians(45)
    front_mask = np.abs(angles_grid) <= front_rad

    ring_mask = np.zeros((local_size, local_size), dtype=bool)
    for R in R_in:
        ring_mask |= (radius_squared >= (R - 1) ** 2) & (radius_squared <= (R + 1) ** 2)

    # Threshold and connected components for NEW
    path_thresh = path_attraction_masked.max() * 0.3
    thresh_new = ring_mask & front_mask & (mask_new > 0) & (path_attraction_masked >= path_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_new_uint8 = (thresh_new.astype(np.uint8) * 255)
    thresh_new_closed = cv2.morphologyEx(thresh_new_uint8, cv2.MORPH_CLOSE, kernel)
    num_new, labels_new = cv2.connectedComponents(thresh_new_closed)

    # Find peaks for NEW
    peaks_new = []
    centered_new = []
    for lab in range(1, num_new):
        region = labels_new == lab
        if np.sum(region) < 10:
            continue
        masked = np.where(region, path_attraction_masked, -np.inf)
        best_idx = np.unravel_index(np.argmax(masked), masked.shape)
        peaks_new.append((best_idx[1], best_idx[0], path_attraction_masked[best_idx], lab))
        centered_new.append([best_idx[1] - center_x, best_idx[0] - center_y])

    centered_new = np.array(centered_new) if centered_new else np.empty((0, 2))

    # Build paths for NEW using same graph searcher
    if len(centered_new) > 0:
        try:
            paths_new = graph_searcher.build_and_deduplicate_paths(
                labels=labels_new,
                centered_points=centered_new,
                dist_inside=path_attraction_masked,
                center_x=center_x, center_y=center_y,
                dist_thresh=path_thresh,
                path_front_deg=90.0,
                start_ids=(0, 1, 2)
            )
        except Exception as e:
            print(f"  NEW path building failed: {e}")
            paths_new = []
    else:
        paths_new = []

    # ========== Visualization ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original
    ax = axes[0, 0]
    ax.imshow(dist_inside_orig, cmap='hot', alpha=0.7)
    for i, (px, py, val, lab) in enumerate(peaks_orig):
        ax.plot(px, py, 'go', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (px, py), textcoords="offset points", xytext=(5, 5),
                    color='white', fontweight='bold')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIG: {len(peaks_orig)} peaks')

    ax = axes[0, 1]
    ax.imshow(dist_inside_orig, cmap='hot', alpha=0.5)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(paths_orig), 1)))
    for i, path in enumerate(paths_orig):
        if len(path) > 0:
            # path is in centered coords, convert to image coords
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'o-', color=colors[i],
                    markersize=8, linewidth=2, label=f'Path {i+1}')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIG: {len(paths_orig)} paths')
    if paths_orig:
        ax.legend(loc='upper right', fontsize=8)

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
    for i, path in enumerate(paths_orig):
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
            ax.plot(world_path[:, 0], world_path[:, 1], 'o-', color=colors[i], markersize=6, linewidth=2)
    ax.set_title('ORIG on Course')
    ax.axis('off')

    # Row 2: New
    ax = axes[1, 0]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)
    for i, (px, py, val, lab) in enumerate(peaks_new):
        ax.plot(px, py, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (px, py), textcoords="offset points", xytext=(5, 5),
                    color='white', fontweight='bold')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: {len(peaks_new)} peaks')

    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.5)
    colors_new = plt.cm.tab10(np.linspace(0, 1, max(len(paths_new), 1)))
    for i, path in enumerate(paths_new):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'o-', color=colors_new[i],
                    markersize=8, linewidth=2, label=f'Path {i+1}')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: {len(paths_new)} paths')
    if paths_new:
        ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    ax.arrow(robot_px, robot_py, adx, ady, head_width=8, head_length=5, fc='black', ec='black')
    for i, path in enumerate(paths_new):
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
            ax.plot(world_path[:, 0], world_path[:, 1], 'o-', color=colors_new[i], markersize=6, linewidth=2)
    ax.set_title('NEW on Course')
    ax.axis('off')

    plt.suptitle(f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    print(f"  ORIG: {len(peaks_orig)} peaks -> {len(paths_orig)} paths")
    print(f"  NEW:  {len(peaks_new)} peaks -> {len(paths_new)} paths")


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
        compare_paths(generator, pose, output_dir / f"paths_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

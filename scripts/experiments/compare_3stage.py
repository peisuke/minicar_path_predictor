#!/usr/bin/env python3
"""Compare 3 stages: peaks -> rough_paths -> smoothed_paths."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PathPipeline, PeakParams, PathPlannerConfig,
    DistanceFieldGenerator, DistanceFieldParams,
    PeakDetector, GraphPathSearcher, PotentialPathSmoother
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig


def compare_3stage(generator, pose, output_path):
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    scale = 1.0 / generator.resolution
    center_x, center_y = local_size // 2, local_size // 2

    # Simulate LiDAR -> view_points
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

    # Common config
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    dist_thresh = 15.0

    # ========== ORIGINAL: wall distance ==========
    pipeline_orig = PathPipeline(params=PeakParams(), config=config)
    result_orig = pipeline_orig.run_pipeline(
        view_points=view_points, H=local_size, W=local_size,
        R_in=R_in, ring_thickness=1, front_deg=90.0,
        path_front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
    )

    peaks_orig = result_orig['peaks']
    centered_orig = result_orig['peaks_centered']
    rough_orig = result_orig['rough_paths']
    smooth_orig = result_orig['paths']
    dist_inside_orig = result_orig['dist_inside']

    # ========== NEW: path attraction ==========
    # 1. Create path_attraction field
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

    # Use same mask as original
    mask = dist_inside_orig > 0
    wall_dist_max = dist_inside_orig.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    path_attraction_masked = path_attraction * mask.astype(np.float32)

    # Normalize for peak detection
    if path_attraction_masked.max() > 0:
        path_attraction_norm = (path_attraction_masked / path_attraction_masked.max() * 255).astype(np.uint8)
    else:
        path_attraction_norm = np.zeros_like(path_attraction_masked, dtype=np.uint8)

    # 2. Peak detection on path_attraction
    dist_field_gen = DistanceFieldGenerator(config)
    df_params = DistanceFieldParams(
        H=local_size, W=local_size, R_in=R_in,
        ring_thickness=1, front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
    )
    _, _, _, _, _, path_candidate_pixels = dist_field_gen.generate_distance_field(view_points, df_params)

    # Use path_attraction threshold instead
    _, _, delta_x, delta_y, radius_squared = dist_field_gen._compute_center_grid(local_size, local_size)
    ring_mask = dist_field_gen._create_ring_mask(radius_squared, R_in, thickness=1)
    front_mask = dist_field_gen._create_front_mask(delta_x, delta_y, 90.0)
    path_candidate_new = ring_mask & front_mask & (path_attraction_masked >= dist_thresh)

    num_labels, labels_new, _ = dist_field_gen.connect_regions_and_label(path_candidate_new, kernel_sz=3)

    peak_detector = PeakDetector(config=config)
    peaks_new, centered_new, sorted_labels_new = peak_detector._detect_path_candidates(
        labels_new, path_attraction_norm, center_x, center_y
    )

    # 3. Path building
    path_searcher = GraphPathSearcher(config)
    if len(centered_new) > 0:
        rough_new = path_searcher.build_and_deduplicate_paths(
            labels=sorted_labels_new,
            centered_points=centered_new,
            dist_inside=path_attraction_masked,
            center_x=center_x, center_y=center_y,
            dist_thresh=dist_thresh,
            path_front_deg=90.0,
            start_ids=(0, 1, 2)
        )
    else:
        rough_new = []

    # 4. Path smoothing
    smoother = PotentialPathSmoother(config)
    if len(rough_new) > 0:
        smooth_new = smoother.smooth_paths(rough_new, path_attraction_masked, local_size, local_size)
    else:
        smooth_new = []

    # ========== Visualization: 2 rows x 3 cols ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: ORIGINAL
    ax = axes[0, 0]
    ax.imshow(dist_inside_orig, cmap='hot', alpha=0.7)
    for i, (px, py, val, lab) in enumerate(peaks_orig):
        ax.plot(px, py, 'go', markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIG Stage1: {len(peaks_orig)} peaks')

    ax = axes[0, 1]
    ax.imshow(dist_inside_orig, cmap='hot', alpha=0.5)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rough_orig), 1)))
    for i, path in enumerate(rough_orig):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'o-', color=colors[i],
                    markersize=10, linewidth=2, label=f'Path {i+1}: {len(path)} pts')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'ORIG Stage2: {len(rough_orig)} rough_paths')

    ax = axes[0, 2]
    ax.imshow(dist_inside_orig, cmap='hot', alpha=0.5)
    for i, path in enumerate(smooth_orig):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], '.-', color=colors[i],
                    markersize=4, linewidth=1)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIG Stage3: {len(smooth_orig)} smoothed_paths')

    # Row 2: NEW
    ax = axes[1, 0]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)
    for i, (px, py, val, lab) in enumerate(peaks_new):
        ax.plot(px, py, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW Stage1: {len(peaks_new)} peaks')

    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.5)
    colors_new = plt.cm.tab10(np.linspace(0, 1, max(len(rough_new), 1)))
    for i, path in enumerate(rough_new):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'o-', color=colors_new[i],
                    markersize=10, linewidth=2, label=f'Path {i+1}: {len(path)} pts')
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'NEW Stage2: {len(rough_new)} rough_paths')

    ax = axes[1, 2]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.5)
    for i, path in enumerate(smooth_new):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], '.-', color=colors_new[i],
                    markersize=4, linewidth=1)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW Stage3: {len(smooth_new)} smoothed_paths')

    plt.suptitle(f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°\n'
                 f'3 Stages: peaks -> rough_paths -> smoothed_paths', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    print(f"  ORIG: {len(peaks_orig)} peaks -> {len(rough_orig)} rough -> {len(smooth_orig)} smooth")
    print(f"  NEW:  {len(peaks_new)} peaks -> {len(rough_new)} rough -> {len(smooth_new)} smooth")


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
        compare_3stage(generator, pose, output_dir / f"3stage_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

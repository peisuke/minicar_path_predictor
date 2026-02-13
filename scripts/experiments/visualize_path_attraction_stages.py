#!/usr/bin/env python3
"""Visualize PathPipeline 3 stages with path_attraction field."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PathPipeline, PeakParams, PathPlannerConfig, PeakDetectorParams,
    DistanceFieldGenerator, DistanceFieldParams,
    PeakDetector, GraphPathSearcher, PotentialPathSmoother
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig


def visualize_stages(generator, pose, output_path):
    """Visualize 3 stages with path_attraction field in robot frame."""
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    resolution = generator.resolution  # 0.02 m/pixel
    scale = 1.0 / resolution  # pixel/m
    center_x, center_y = local_size // 2, local_size // 2
    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    dist_thresh = 15.0

    # 1. Simulate LiDAR -> view_points
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

    # 2. Get mask from original distance field (for visibility)
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    dist_field_gen = DistanceFieldGenerator(config)
    df_params = DistanceFieldParams(
        H=local_size, W=local_size, R_in=R_in,
        ring_thickness=1, front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
    )
    mask, dist_inside_orig, dist_norm_orig, cx, cy, _ = \
        dist_field_gen.generate_distance_field(view_points, df_params)

    # 3. Create path_attraction field (THIS IS THE ONLY CHANGE)
    yy, xx = np.mgrid[:local_size, :local_size]
    dx_img = xx - center_x
    dy_img = yy - center_y

    x_robot_m = dx_img * resolution
    y_robot_m = -dy_img * resolution
    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)
    x_world = robot_x + x_robot_m * cos_yaw - y_robot_m * sin_yaw
    y_world = robot_y + x_robot_m * sin_yaw + y_robot_m * cos_yaw

    src_x = (x_world / resolution + generator.W / 2).astype(np.float32)
    src_y = (generator.H / 2 - y_world / resolution).astype(np.float32)

    local_dist_to_path = cv2.remap(
        generator.path_distance_field, src_x, src_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1000
    )

    # path_attraction = max(wall_dist_max - dist_to_path, 0)
    wall_dist_max = dist_inside_orig.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    
    # Apply mask (same visibility as original)
    dist_inside = path_attraction * (mask > 0).astype(np.float32)

    # Normalize for peak detection (same as original dist_norm)
    if dist_inside.max() > 0:
        dist_norm = (dist_inside / dist_inside.max() * 255).astype(np.uint8)
    else:
        dist_norm = np.zeros_like(dist_inside, dtype=np.uint8)

    # ========== Stage 1: Peak detection (same logic, different field) ==========
    _, _, delta_x, delta_y, radius_squared = dist_field_gen._compute_center_grid(local_size, local_size)
    ring_mask = dist_field_gen._create_ring_mask(radius_squared, R_in, thickness=1)
    front_mask = dist_field_gen._create_front_mask(delta_x, delta_y, 90.0)
    path_candidate_pixels = ring_mask & front_mask & (dist_inside >= dist_thresh)

    num_labels, labels, _ = dist_field_gen.connect_regions_and_label(path_candidate_pixels, kernel_sz=3)

    peak_detector = PeakDetector(config=config)
    peaks, centered_points, sorted_labels = peak_detector._detect_path_candidates(
        labels, dist_norm, center_x, center_y
    )

    # ========== Stage 2: Path building (SKIP lookahead trimming) ==========
    path_searcher = GraphPathSearcher(config)
    if len(centered_points) > 0:
        # 1. Build paths
        result_paths = path_searcher.build_valid_paths_coords(
            labels=sorted_labels,
            centered_pts=centered_points,
            dist_map=dist_inside,
            cx=center_x,
            cy=center_y,
            thr=dist_thresh,
            front_deg=90.0,
            start_ids=list(config.START_IDS),
        )

        # 2. Deduplicate paths
        paths_ids = result_paths["paths_ids"]
        deduplicated_paths_ids = path_searcher.deduplicator.deduplicate_paths(
            paths_ids, centered_points, result_paths["flat_to_pts"]
        )

        # 3. SKIP: _trim_paths_by_lookahead (commented out)
        # trimmed_paths_ids = path_searcher._trim_paths_by_lookahead(...)

        # 4. Convert IDs to coordinates (use deduplicated, not trimmed)
        coordinate_paths = path_searcher._paths_to_coords(
            deduplicated_paths_ids, centered_points, result_paths["flat_to_pts"]
        )
    else:
        coordinate_paths = []

    # ========== Stage 3: Path smoothing (same logic) ==========
    smoother = PotentialPathSmoother(config)
    if len(coordinate_paths) > 0:
        smoothed_paths = smoother.smooth_paths(coordinate_paths, dist_inside, local_size, local_size)
    else:
        smoothed_paths = []

    # ========== Convert to Robot Frame (meters) ==========
    def img_to_robot(pts_img):
        pts = np.array(pts_img, dtype=np.float32)
        robot_pts = np.zeros_like(pts)
        robot_pts[:, 0] = pts[:, 0] * resolution
        robot_pts[:, 1] = -pts[:, 1] * resolution
        return robot_pts

    if len(centered_points) > 0:
        peaks_robot = img_to_robot(centered_points)
    else:
        peaks_robot = np.empty((0, 2))

    rough_paths_robot = [img_to_robot(p) for p in coordinate_paths if len(p) > 0]
    smooth_paths_robot = [img_to_robot(p) for p in smoothed_paths if len(p) > 0]

    # ========== Visualization ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    extent = [-local_size/2 * resolution, local_size/2 * resolution,
              -local_size/2 * resolution, local_size/2 * resolution]

    dist_robot = np.flipud(dist_inside)

    # Stage 1
    ax = axes[0]
    ax.imshow(dist_robot, cmap='hot', alpha=0.7, extent=extent, origin='lower')
    if len(peaks_robot) > 0:
        ax.scatter(peaks_robot[:, 0], peaks_robot[:, 1], c='lime', s=100,
                   edgecolors='white', linewidths=2, zorder=5)
    ax.plot(0, 0, 'w+', markersize=20, markeredgewidth=3)
    ax.arrow(0, 0, 0.3, 0, head_width=0.1, head_length=0.05, fc='white', ec='white')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.set_title(f'Stage 1: path_attraction + peaks\n{len(peaks_robot)} peaks')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Stage 2
    ax = axes[1]
    ax.imshow(dist_robot, cmap='hot', alpha=0.5, extent=extent, origin='lower')
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rough_paths_robot), 1)))
    for i, path in enumerate(rough_paths_robot):
        ax.plot(path[:, 0], path[:, 1], 'o-', color=colors[i], markersize=10, linewidth=2,
                label=f'Path {i+1}: {len(path)} pts')
    ax.plot(0, 0, 'w+', markersize=20, markeredgewidth=3)
    ax.arrow(0, 0, 0.3, 0, head_width=0.1, head_length=0.05, fc='white', ec='white')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.set_title(f'Stage 2: coordinate_paths\n{len(rough_paths_robot)} paths')
    if rough_paths_robot:
        ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Stage 3
    ax = axes[2]
    ax.imshow(dist_robot, cmap='hot', alpha=0.5, extent=extent, origin='lower')
    for i, path in enumerate(smooth_paths_robot):
        ax.plot(path[:, 0], path[:, 1], '.-', color=colors[i], markersize=4, linewidth=1,
                label=f'Path {i+1}: {len(path)} pts')
    ax.plot(0, 0, 'w+', markersize=20, markeredgewidth=3)
    ax.arrow(0, 0, 0.3, 0, head_width=0.1, head_length=0.05, fc='white', ec='white')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.set_title(f'Stage 3: smoothed_paths\n{len(smooth_paths_robot)} paths')
    if smooth_paths_robot:
        ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'PathPipeline with path_attraction Field\n'
                 f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  Stage 1: {len(peaks_robot)} peaks")
    print(f"  Stage 2: {len(rough_paths_robot)} coordinate_paths, points: {[len(p) for p in rough_paths_robot]}")
    print(f"  Stage 3: {len(smooth_paths_robot)} smoothed_paths, points: {[len(p) for p in smooth_paths_robot]}")


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
        visualize_stages(generator, pose, output_dir / f"path_attraction_stages_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

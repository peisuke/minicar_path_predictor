#!/usr/bin/env python3
"""Visualize PathPipeline 3 stages in robot frame."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PathPipeline, PeakParams, PathPlannerConfig, PeakDetectorParams
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig


def visualize_stages(generator, pose, output_path):
    """Visualize 3 stages in robot coordinate frame."""
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    resolution = generator.resolution  # 0.02 m/pixel
    scale = 1.0 / resolution  # pixel/m
    center_x, center_y = local_size // 2, local_size // 2

    # 1. Simulate LiDAR -> view_points (in local image coords)
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

    # 2. Run PathPipeline._execute_core_pipeline equivalent
    config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
    pipeline = PathPipeline(params=PeakParams(), config=config)
    
    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    dist_thresh = 15.0
    
    # Stage 1: peaks detection
    peak_detector_params = PeakDetectorParams(
        H=local_size, W=local_size, R_in=R_in,
        ring_thickness=1, front_deg=90.0,
        dist_thresh=dist_thresh, border_thickness=1
    )
    peaks, centered_points, sorted_labels, dist_inside, dist_norm, cx, cy = \
        pipeline.peak_detector.detect_from_view_points(view_points, peak_detector_params)
    
    # Stage 2: path building (coordinate_paths = rough_paths)
    coordinate_paths = pipeline.path_searcher.build_and_deduplicate_paths(
        sorted_labels, centered_points, dist_inside, cx, cy,
        dist_thresh, 90.0, config.START_IDS
    )
    
    # Stage 3: path smoothing
    smoothed_paths = pipeline.smoother.smooth_paths(coordinate_paths, dist_inside, local_size, local_size)

    # ========== Convert to Robot Frame (meters) ==========
    # Robot frame: X=forward, Y=left, origin at robot
    # Image frame: center at (center_x, center_y), X=right, Y=down
    # Conversion: x_robot = (px - center_x) * resolution
    #             y_robot = -(py - center_y) * resolution
    
    def img_to_robot(pts_img):
        """Convert image coords (pixels, centered) to robot frame (meters)."""
        pts = np.array(pts_img, dtype=np.float32)
        robot_pts = np.zeros_like(pts)
        robot_pts[:, 0] = pts[:, 0] * resolution   # x
        robot_pts[:, 1] = -pts[:, 1] * resolution  # y (flip)
        return robot_pts
    
    # Convert centered_points to robot frame
    if len(centered_points) > 0:
        peaks_robot = img_to_robot(centered_points)
    else:
        peaks_robot = np.empty((0, 2))
    
    # Convert coordinate_paths to robot frame
    rough_paths_robot = []
    for path in coordinate_paths:
        if len(path) > 0:
            rough_paths_robot.append(img_to_robot(path))
    
    # Convert smoothed_paths to robot frame
    smooth_paths_robot = []
    for path in smoothed_paths:
        if len(path) > 0:
            smooth_paths_robot.append(img_to_robot(path))

    # ========== Create Robot Frame Distance Field ==========
    # Create coordinate grid in robot frame
    x_range = np.linspace(-local_size/2 * resolution, local_size/2 * resolution, local_size)
    y_range = np.linspace(local_size/2 * resolution, -local_size/2 * resolution, local_size)  # flip Y
    
    # ========== Visualization ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    extent = [-local_size/2 * resolution, local_size/2 * resolution,
              -local_size/2 * resolution, local_size/2 * resolution]
    
    # Stage 1: dist_inside + centered_points (peaks)
    ax = axes[0]
    # Flip dist_inside vertically for robot frame display
    dist_robot = np.flipud(dist_inside)
    ax.imshow(dist_robot, cmap='hot', alpha=0.7, extent=extent, origin='lower')
    if len(peaks_robot) > 0:
        ax.scatter(peaks_robot[:, 0], peaks_robot[:, 1], c='lime', s=100, 
                   edgecolors='white', linewidths=2, zorder=5, label=f'{len(peaks_robot)} peaks')
    ax.plot(0, 0, 'w+', markersize=20, markeredgewidth=3)
    ax.arrow(0, 0, 0.3, 0, head_width=0.1, head_length=0.05, fc='white', ec='white')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.set_title(f'Stage 1: dist_inside + centered_points\n{len(peaks_robot)} peaks')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Stage 2: dist_inside + coordinate_paths (rough_paths)
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
    ax.set_title(f'Stage 2: coordinate_paths (rough)\n{len(rough_paths_robot)} paths')
    if rough_paths_robot:
        ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Stage 3: dist_inside + smoothed_paths
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

    plt.suptitle(f'PathPipeline Stages in Robot Frame\n'
                 f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"  Stage 1: {len(peaks_robot)} peaks (centered_points)")
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
        visualize_stages(generator, pose, output_dir / f"pipeline_stages_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

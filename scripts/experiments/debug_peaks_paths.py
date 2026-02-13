#!/usr/bin/env python3
"""Debug: overlay peaks and rough_paths points."""

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


def debug_peaks_paths(generator, pose, output_path):
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    scale = 1.0 / generator.resolution
    center_x, center_y = local_size // 2, local_size // 2

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

    result = pipeline.run_pipeline(
        view_points=view_points, H=local_size, W=local_size,
        R_in=R_in, ring_thickness=1, front_deg=90.0,
        path_front_deg=90.0, dist_thresh=15.0, border_thickness=1
    )

    peaks = result['peaks']
    peaks_centered = result['peaks_centered']
    rough_paths = result['rough_paths']
    dist_inside = result['dist_inside']

    print(f"\n=== Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}° ===")
    print(f"peaks (image coords): {[(p[0], p[1]) for p in peaks]}")
    print(f"peaks_centered: {peaks_centered.tolist() if len(peaks_centered) > 0 else []}")
    print(f"rough_paths points:")
    for i, path in enumerate(rough_paths):
        print(f"  Path {i}: {path.tolist()}")

    # Check if peaks_centered matches rough_paths points
    print(f"\npeaks_centered to image coords (+ center):")
    if len(peaks_centered) > 0:
        for i, pc in enumerate(peaks_centered):
            img_x = pc[0] + center_x
            img_y = pc[1] + center_y
            print(f"  Peak {i}: centered=({pc[0]}, {pc[1]}) -> img=({img_x}, {img_y})")

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(dist_inside, cmap='hot', alpha=0.7)

    # Draw peaks (image coords) - large green circles
    for i, (px, py, val, lab) in enumerate(peaks):
        ax.plot(px, py, 'go', markersize=20, markeredgecolor='white', markeredgewidth=3,
                label='peaks' if i == 0 else None)
        ax.annotate(f'P{i}', (px, py), textcoords="offset points", xytext=(10, 10),
                    color='green', fontsize=12, fontweight='bold')

    # Draw peaks_centered (converted to image coords) - cyan squares
    if len(peaks_centered) > 0:
        for i, pc in enumerate(peaks_centered):
            img_x = pc[0] + center_x
            img_y = pc[1] + center_y
            ax.plot(img_x, img_y, 'cs', markersize=15, markeredgecolor='white', markeredgewidth=2,
                    label='peaks_centered' if i == 0 else None)

    # Draw rough_paths - red/orange lines with markers
    colors = ['red', 'orange', 'yellow']
    for i, path in enumerate(rough_paths):
        if len(path) > 0:
            path_img = path + np.array([center_x, center_y])
            ax.plot(path_img[:, 0], path_img[:, 1], 'o-', color=colors[i % len(colors)],
                    markersize=12, linewidth=3, label=f'rough_path {i}')
            for j, pt in enumerate(path_img):
                ax.annotate(f'{i}.{j}', (pt[0], pt[1]), textcoords="offset points", xytext=(-15, -15),
                            color=colors[i % len(colors)], fontsize=10)

    ax.plot(center_x, center_y, 'w+', markersize=20, markeredgewidth=3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'Peaks vs rough_paths\nGreen=peaks, Cyan=peaks_centered, Red/Orange=rough_paths')

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
        debug_peaks_paths(generator, pose, output_dir / f"debug_peaks_{i+1}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

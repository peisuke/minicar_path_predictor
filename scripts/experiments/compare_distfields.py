#!/usr/bin/env python3
"""Compare original (wall distance) vs new (path attraction) distance fields."""

import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig
from minicar_path_predictor.utils import world_to_pixel


def compare_distfields_single_pose(generator: CourseDataGenerator, pose, output_path: Path):
    """Compare original and new distance fields for a single pose."""
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    waypoint_spacing = 0.3
    num_points = 5

    # Convert robot position to pixel coordinates
    robot_px, robot_py = world_to_pixel(robot_x, robot_y, generator.img_shape, generator.resolution)

    # Create local view
    H_local, W_local = local_size, local_size
    center_x, center_y = W_local // 2, H_local // 2

    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)

    yy_local, xx_local = np.mgrid[:H_local, :W_local]
    dx_local = xx_local - center_x
    dy_local = yy_local - center_y

    dx_world = cos_yaw * dx_local - sin_yaw * (-dy_local)
    dy_world = sin_yaw * dx_local + cos_yaw * (-dy_local)

    src_x = (robot_px + dx_world).astype(np.float32)
    src_y = (robot_py - dy_world).astype(np.float32)

    # Sample walls mask
    local_walls = cv2.remap(
        generator.walls_mask.astype(np.float32),
        src_x, src_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    # Create visibility mask
    visibility = generator._create_visibility_mask(local_walls, center_x, center_y, local_size)

    # ========== ORIGINAL: Wall distance field ==========
    # Distance from walls (high in corridor center)
    wall_dist_original = distance_transform_edt(visibility > 0)

    # ========== NEW: Path attraction field ==========
    # Sample distance-to-path
    local_dist_to_path = cv2.remap(
        generator.path_distance_field,
        src_x, src_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1000
    )
    # Invert: max(wall_max - dist_to_path, 0)
    wall_dist_max = wall_dist_original.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    path_attraction_masked = path_attraction * (visibility > 0).astype(np.float32)

    # Apply visibility mask to original too
    wall_dist_masked = wall_dist_original * (visibility > 0).astype(np.float32)

    # Waypoint selection setup
    radius_squared = dx_local ** 2 + dy_local ** 2
    angles = np.arctan2(-dy_local, dx_local)
    front_rad = math.radians(45)
    front_mask = np.abs(angles) <= front_rad
    ring_thickness = 3
    ring_distances = [int((i + 1) * waypoint_spacing / generator.resolution) for i in range(num_points)]

    # Select waypoints with ORIGINAL field
    wp_original = []
    for R in ring_distances:
        ring_mask = (radius_squared >= (R - ring_thickness) ** 2) & \
                    (radius_squared <= (R + ring_thickness) ** 2)
        candidates = ring_mask & front_mask & (visibility > 0)
        if np.any(candidates):
            masked = np.where(candidates, wall_dist_masked, -np.inf)
            best_y, best_x = np.unravel_index(np.argmax(masked), masked.shape)
            wp_original.append((best_x, best_y))
        else:
            wp_original.append((center_x + R, center_y))

    # Select waypoints with NEW field
    wp_new = []
    for R in ring_distances:
        ring_mask = (radius_squared >= (R - ring_thickness) ** 2) & \
                    (radius_squared <= (R + ring_thickness) ** 2)
        candidates = ring_mask & front_mask & (visibility > 0)
        if np.any(candidates):
            masked = np.where(candidates, path_attraction_masked, -np.inf)
            best_y, best_x = np.unravel_index(np.argmax(masked), masked.shape)
            wp_new.append((best_x, best_y))
        else:
            wp_new.append((center_x + R, center_y))

    # Create figure: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original (wall distance)
    ax = axes[0, 0]
    im = ax.imshow(wall_dist_masked, cmap='hot')
    ax.plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=2)
    ax.set_title(f'ORIGINAL: Wall Distance\n(high at corridor center)\nmax={wall_dist_masked.max():.1f}')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    ax.imshow(wall_dist_masked, cmap='hot', alpha=0.7)
    for i, (wx, wy) in enumerate(wp_original):
        ax.plot(wx, wy, 'go', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (wx, wy), textcoords="offset points", xytext=(5, 5), color='white', fontweight='bold')
    # Draw rings
    for R in ring_distances:
        theta = np.linspace(-front_rad, front_rad, 50)
        ring_x = center_x + R * np.cos(theta)
        ring_y = center_y - R * np.sin(theta)
        ax.plot(ring_x, ring_y, 'c-', linewidth=1, alpha=0.5)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title('ORIGINAL: Waypoints\n(green = corridor center)')
    ax.set_xlim(0, local_size)
    ax.set_ylim(local_size, 0)

    ax = axes[0, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    path_px = generator.ideal_path_pixels
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5, label='Ideal Path')
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    arrow_len = 25
    dx = arrow_len * np.cos(robot_yaw)
    dy = -arrow_len * np.sin(robot_yaw)
    ax.arrow(robot_px, robot_py, dx, dy, head_width=8, head_length=5, fc='black', ec='black')
    # Convert original waypoints to world
    cos_yaw_fwd = np.cos(robot_yaw)
    sin_yaw_fwd = np.sin(robot_yaw)
    for wx, wy in wp_original:
        lx = (wx - center_x) * generator.resolution
        ly = -(wy - center_y) * generator.resolution
        world_x = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
        world_y = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
        wpx, wpy = world_to_pixel(world_x, world_y, generator.img_shape, generator.resolution)
        ax.plot(wpx, wpy, 'go', markersize=8, markeredgecolor='white', markeredgewidth=1)
    ax.set_title('ORIGINAL on Course\n(green = corridor center)')
    ax.axis('off')

    # Row 2: New (path attraction)
    ax = axes[1, 0]
    im = ax.imshow(path_attraction_masked, cmap='hot')
    ax.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW: Path Attraction\n(high near ideal path)\nmax={path_attraction_masked.max():.1f}')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)
    for i, (wx, wy) in enumerate(wp_new):
        ax.plot(wx, wy, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'{i+1}', (wx, wy), textcoords="offset points", xytext=(5, 5), color='white', fontweight='bold')
    for R in ring_distances:
        theta = np.linspace(-front_rad, front_rad, 50)
        ring_x = center_x + R * np.cos(theta)
        ring_y = center_y - R * np.sin(theta)
        ax.plot(ring_x, ring_y, 'c-', linewidth=1, alpha=0.5)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title('NEW: Waypoints\n(red = toward ideal path)')
    ax.set_xlim(0, local_size)
    ax.set_ylim(local_size, 0)

    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    ax.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=1, alpha=0.5, label='Ideal Path')
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    ax.arrow(robot_px, robot_py, dx, dy, head_width=8, head_length=5, fc='black', ec='black')
    # Convert new waypoints to world
    for wx, wy in wp_new:
        lx = (wx - center_x) * generator.resolution
        ly = -(wy - center_y) * generator.resolution
        world_x = robot_x + lx * cos_yaw_fwd - ly * sin_yaw_fwd
        world_y = robot_y + lx * sin_yaw_fwd + ly * cos_yaw_fwd
        wpx, wpy = world_to_pixel(world_x, world_y, generator.img_shape, generator.resolution)
        ax.plot(wpx, wpy, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1)
    ax.set_title('NEW on Course\n(red = toward ideal path)')
    ax.axis('off')

    plt.suptitle(f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    course_path = Path("data/images/course.png")
    output_dir = Path("data/sim_robot")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not course_path.exists():
        print(f"Error: Course image not found at {course_path}")
        return 1

    lidar_config = LiDARConfig(
        num_rays=360, angle_min=0.0, angle_max=2 * np.pi,
        range_min=0.15, range_max=12.0, noise_std=0.005,
        blind_spot_start=115.0, blind_spot_end=246.0
    )

    print("Loading course...")
    generator = CourseDataGenerator(course_path, resolution=0.02, lidar_config=lidar_config)

    # Sample poses
    np.random.seed(42)
    candidates = generator.generate_pose_candidates_grid(grid_spacing=0.1, track_only=True)
    indices = np.random.choice(len(candidates), 3, replace=False)

    for i, idx in enumerate(indices):
        pose = candidates[idx]
        compare_distfields_single_pose(generator, pose, output_dir / f"compare_{i+1}.png")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())

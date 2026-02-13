#!/usr/bin/env python3
"""Visualize distance field and ring-based waypoint selection."""

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


def visualize_single_pose(generator: CourseDataGenerator, pose, output_path: Path):
    """Visualize distance field and ring selection for a single pose."""
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    waypoint_spacing = 0.3
    num_points = 5

    # Convert robot position to pixel coordinates
    robot_px, robot_py = world_to_pixel(robot_x, robot_y, generator.img_shape, generator.resolution)

    # Create local view rotated to robot frame
    H_local, W_local = local_size, local_size
    center_x, center_y = W_local // 2, H_local // 2

    # Rotation matrix
    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)

    # Create coordinate grids
    yy_local, xx_local = np.mgrid[:H_local, :W_local]
    dx_local = xx_local - center_x
    dy_local = yy_local - center_y

    # Transform to world pixel coordinates
    dx_world = cos_yaw * dx_local - sin_yaw * (-dy_local)
    dy_world = sin_yaw * dx_local + cos_yaw * (-dy_local)

    src_x = (robot_px + dx_world).astype(np.float32)
    src_y = (robot_py - dy_world).astype(np.float32)

    # Sample distance-to-path from global field
    local_dist_to_path = cv2.remap(
        generator.path_distance_field,
        src_x, src_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1000
    )

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

    # Wall distance from visibility boundary
    wall_dist_local = distance_transform_edt(visibility > 0)
    wall_dist_max = wall_dist_local.max()

    # Path attraction field: max(wall_dist_max - dist_to_path, 0)
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    path_attraction_masked = path_attraction * (visibility > 0).astype(np.float32)

    # Radius and angle grids
    radius_squared = dx_local ** 2 + dy_local ** 2
    angles = np.arctan2(-dy_local, dx_local)
    front_rad = math.radians(45)  # 90 degree total cone
    front_mask = np.abs(angles) <= front_rad

    # Ring distances
    ring_distances = [int((i + 1) * waypoint_spacing / generator.resolution) for i in range(num_points)]
    ring_thickness = 3

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Distance to path (raw)
    ax = axes[0, 0]
    im = ax.imshow(local_dist_to_path, cmap='viridis')
    ax.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
    ax.set_title(f'Distance to Path\n(min={local_dist_to_path.min():.1f}, max={local_dist_to_path.max():.1f})')
    plt.colorbar(im, ax=ax)

    # 2. Visibility mask with wall distance
    ax = axes[0, 1]
    im = ax.imshow(wall_dist_local, cmap='hot')
    ax.plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=2)
    ax.set_title(f'Wall Distance (from LiDAR)\n(max={wall_dist_max:.1f})')
    plt.colorbar(im, ax=ax)

    # 3. Path attraction field (inverted)
    ax = axes[0, 2]
    im = ax.imshow(path_attraction_masked, cmap='hot')
    ax.plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=2)
    ax.set_title(f'Path Attraction\n(wall_max - dist_to_path)\nmax={path_attraction_masked.max():.1f}')
    plt.colorbar(im, ax=ax)

    # 4. Rings + front mask overlay
    ax = axes[1, 0]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)

    # Draw rings
    for i, R in enumerate(ring_distances):
        theta = np.linspace(0, 2 * np.pi, 100)
        ring_x = center_x + R * np.cos(theta)
        ring_y = center_y - R * np.sin(theta)
        ax.plot(ring_x, ring_y, 'c-', linewidth=1, alpha=0.7)

    # Draw front cone
    cone_len = max(ring_distances) + 10
    ax.plot([center_x, center_x + cone_len * np.cos(front_rad)],
            [center_y, center_y - cone_len * np.sin(front_rad)], 'g--', linewidth=2)
    ax.plot([center_x, center_x + cone_len * np.cos(-front_rad)],
            [center_y, center_y - cone_len * np.sin(-front_rad)], 'g--', linewidth=2)
    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title('Rings + Front Mask')
    ax.set_xlim(0, local_size)
    ax.set_ylim(local_size, 0)

    # 5. Waypoint selection per ring
    ax = axes[1, 1]
    ax.imshow(path_attraction_masked, cmap='hot', alpha=0.5)

    waypoints = []
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))

    for i, R in enumerate(ring_distances):
        ring_mask = (radius_squared >= (R - ring_thickness) ** 2) & \
                    (radius_squared <= (R + ring_thickness) ** 2)
        candidates = ring_mask & front_mask & (visibility > 0)

        # Highlight candidate region
        candidate_overlay = np.zeros((local_size, local_size, 4))
        candidate_overlay[candidates, :3] = colors[i][:3]
        candidate_overlay[candidates, 3] = 0.3
        ax.imshow(candidate_overlay)

        if np.any(candidates):
            masked_field = np.where(candidates, path_attraction_masked, -np.inf)
            best_idx = np.unravel_index(np.argmax(masked_field), masked_field.shape)
            best_y, best_x = best_idx
            waypoints.append((best_x, best_y))
            ax.plot(best_x, best_y, 'o', color=colors[i], markersize=12, markeredgecolor='white', markeredgewidth=2)
            ax.annotate(f'{i+1}', (best_x, best_y), textcoords="offset points",
                        xytext=(8, 8), fontsize=10, color='white', fontweight='bold')

    ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title('Waypoint Selection\n(colored by ring)')
    ax.set_xlim(0, local_size)
    ax.set_ylim(local_size, 0)

    # 6. Final result on course image
    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.7)
    path_px = generator.ideal_path_pixels
    ax.plot(path_px[:, 0], path_px[:, 1], 'g-', linewidth=1, alpha=0.5)

    # Robot position and direction
    ax.plot(robot_px, robot_py, 'ko', markersize=10)
    arrow_len = 25
    dx = arrow_len * np.cos(robot_yaw)
    dy = -arrow_len * np.sin(robot_yaw)
    ax.arrow(robot_px, robot_py, dx, dy, head_width=8, head_length=5,
             fc='black', ec='black', linewidth=2)

    # Convert waypoints to world frame
    cos_yaw_fwd = np.cos(robot_yaw)
    sin_yaw_fwd = np.sin(robot_yaw)
    for i, (wx, wy) in enumerate(waypoints):
        local_x = (wx - center_x) * generator.resolution
        local_y = -(wy - center_y) * generator.resolution
        world_x = robot_x + local_x * cos_yaw_fwd - local_y * sin_yaw_fwd
        world_y = robot_y + local_x * sin_yaw_fwd + local_y * cos_yaw_fwd
        wpx, wpy = world_to_pixel(world_x, world_y, generator.img_shape, generator.resolution)
        ax.plot(wpx, wpy, 'o', color=colors[i], markersize=10, markeredgecolor='white', markeredgewidth=2)

    ax.set_title(f'Result on Course\n({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°')
    ax.axis('off')

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

    # Sample a few poses
    np.random.seed(42)
    candidates = generator.generate_pose_candidates_grid(grid_spacing=0.1, track_only=True)
    indices = np.random.choice(len(candidates), 3, replace=False)

    for i, idx in enumerate(indices):
        pose = candidates[idx]
        visualize_single_pose(generator, pose, output_dir / f"distfield_ring_{i+1}.png")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""Debug ring selection for a specific pose."""

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


def debug_pose(generator: CourseDataGenerator, pose, output_path: Path):
    """Debug ring selection for a pose."""
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    waypoint_spacing = 0.3
    num_points = 5

    robot_px, robot_py = world_to_pixel(robot_x, robot_y, generator.img_shape, generator.resolution)

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

    local_walls = cv2.remap(
        generator.walls_mask.astype(np.float32),
        src_x, src_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    visibility = generator._create_visibility_mask(local_walls, center_x, center_y, local_size)

    # Distance fields
    wall_dist_original = distance_transform_edt(visibility > 0)

    local_dist_to_path = cv2.remap(
        generator.path_distance_field,
        src_x, src_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1000
    )
    wall_dist_max = wall_dist_original.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    path_attraction_masked = path_attraction * (visibility > 0).astype(np.float32)

    # Setup
    radius_squared = dx_local ** 2 + dy_local ** 2
    angles = np.arctan2(-dy_local, dx_local)
    front_rad = math.radians(45)
    front_mask = np.abs(angles) <= front_rad
    ring_thickness = 3
    ring_distances = [int((i + 1) * waypoint_spacing / generator.resolution) for i in range(num_points)]

    print(f"Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°")
    print(f"Ring distances (pixels): {ring_distances}")
    print(f"Front angle: ±{math.degrees(front_rad):.0f}°")
    print()

    # Create figure
    fig, axes = plt.subplots(2, num_points, figsize=(4 * num_points, 8))

    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))

    for i, R in enumerate(ring_distances):
        ring_mask = (radius_squared >= (R - ring_thickness) ** 2) & \
                    (radius_squared <= (R + ring_thickness) ** 2)

        ring_and_front = ring_mask & front_mask
        candidates = ring_mask & front_mask & (visibility > 0)

        n_ring = np.sum(ring_mask)
        n_ring_front = np.sum(ring_and_front)
        n_candidates = np.sum(candidates)

        print(f"Ring {i+1} (R={R}px = {R * generator.resolution:.2f}m):")
        print(f"  ring_mask pixels: {n_ring}")
        print(f"  ring & front_mask: {n_ring_front}")
        print(f"  ring & front & visible: {n_candidates}")

        # Top row: Show ring mask
        ax = axes[0, i]
        display = np.zeros((local_size, local_size, 3))
        display[ring_mask, 0] = 0.3  # Ring in dark red
        display[ring_and_front, 1] = 0.5  # Ring+front in green
        display[candidates, 2] = 1.0  # Candidates in blue
        ax.imshow(display)
        ax.plot(center_x, center_y, 'w+', markersize=10)
        # Draw ring
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(center_x + R * np.cos(theta), center_y - R * np.sin(theta), 'y-', linewidth=0.5, alpha=0.5)
        ax.set_title(f'Ring {i+1}: R={R}px\nring={n_ring}, front={n_ring_front}\ncandidates={n_candidates}')
        ax.set_xlim(0, local_size)
        ax.set_ylim(local_size, 0)

        # Bottom row: Show field values on candidates
        ax = axes[1, i]
        ax.imshow(path_attraction_masked, cmap='hot', alpha=0.7)

        if n_candidates > 0:
            masked_field = np.where(candidates, path_attraction_masked, -np.inf)
            best_idx = np.unravel_index(np.argmax(masked_field), masked_field.shape)
            best_y, best_x = best_idx
            best_val = path_attraction_masked[best_y, best_x]

            # Show all candidate values
            cand_vals = path_attraction_masked[candidates]
            print(f"  Field values: min={cand_vals.min():.2f}, max={cand_vals.max():.2f}, mean={cand_vals.mean():.2f}")
            print(f"  Best point: ({best_x}, {best_y}), value={best_val:.2f}")

            ax.plot(best_x, best_y, 'go', markersize=15, markeredgecolor='white', markeredgewidth=2)
            ax.set_title(f'Best: ({best_x}, {best_y})\nvalue={best_val:.2f}')
        else:
            print(f"  NO CANDIDATES!")
            ax.set_title('NO CANDIDATES')

        ax.plot(center_x, center_y, 'w+', markersize=10)
        ax.set_xlim(0, local_size)
        ax.set_ylim(local_size, 0)
        print()

    plt.suptitle(f'Pose: ({robot_x:.2f}, {robot_y:.2f}), yaw={np.degrees(robot_yaw):.0f}°\n'
                 f'Red=ring, Green=ring+front, Blue=candidates', fontsize=12)
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

    # Use same seed to get Pose 2
    np.random.seed(42)
    candidates = generator.generate_pose_candidates_grid(grid_spacing=0.1, track_only=True)
    indices = np.random.choice(len(candidates), 3, replace=False)

    # Debug Pose 2 (index 1)
    pose = candidates[indices[1]]
    debug_pose(generator, pose, output_dir / "debug_rings_pose2.png")

    return 0


if __name__ == "__main__":
    exit(main())

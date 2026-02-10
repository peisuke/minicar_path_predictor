#!/usr/bin/env python3
"""Test waypoint generation with updated distance field approach."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig
from minicar_path_predictor.utils import world_to_pixel


def test_multiple_poses(generator: CourseDataGenerator, output_path: Path, num_poses: int = 9):
    """Test waypoint generation with multiple poses."""
    np.random.seed(42)

    # Sample random poses from track
    candidates = generator.generate_pose_candidates_grid(
        grid_spacing=0.1, track_only=True
    )

    if len(candidates) > num_poses:
        indices = np.random.choice(len(candidates), num_poses, replace=False)
        test_poses = [candidates[i] for i in indices]
    else:
        test_poses = candidates[:num_poses]

    # Create figure with subplots
    rows = int(np.ceil(np.sqrt(num_poses)))
    cols = int(np.ceil(num_poses / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i, pose in enumerate(test_poses):
        ax = axes[i]

        # Show course background
        ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.6)

        # Show ideal path
        path_px = generator.ideal_path_pixels
        ax.plot(path_px[:, 0], path_px[:, 1], 'g-', linewidth=1, alpha=0.5, label='Ideal Path')

        # Robot position
        robot_px, robot_py = world_to_pixel(
            pose.x, pose.y, generator.img_shape, generator.resolution
        )
        ax.plot(robot_px, robot_py, 'ko', markersize=10)

        # Robot direction arrow
        arrow_len = 25
        dx = arrow_len * np.cos(pose.yaw)
        dy = -arrow_len * np.sin(pose.yaw)
        ax.arrow(robot_px, robot_py, dx, dy, head_width=8, head_length=5,
                 fc='black', ec='black', linewidth=2)

        # Generate waypoints using new method
        try:
            waypoints = generator.find_target_waypoints_distance_field(
                pose.x, pose.y, pose.yaw,
                num_points=5, waypoint_spacing=0.3
            )

            # Convert robot frame to world frame
            cos_yaw = np.cos(pose.yaw)
            sin_yaw = np.sin(pose.yaw)

            wp_world = []
            for wp in waypoints:
                wx = pose.x + wp[0] * cos_yaw - wp[1] * sin_yaw
                wy = pose.y + wp[0] * sin_yaw + wp[1] * cos_yaw
                wpx, wpy = world_to_pixel(wx, wy, generator.img_shape, generator.resolution)
                wp_world.append([wpx, wpy])
            wp_world = np.array(wp_world)

            ax.plot(wp_world[:, 0], wp_world[:, 1], 'r.-',
                    markersize=10, linewidth=2, label='Waypoints')

            # Number the waypoints
            for j, (wpx, wpy) in enumerate(wp_world):
                ax.annotate(f'{j+1}', (wpx, wpy), textcoords="offset points",
                            xytext=(5, 5), fontsize=8, color='red')
        except Exception as e:
            ax.set_title(f"Pose {i+1}: Error - {e}")
            continue

        ax.set_title(f"Pose {i+1}: ({pose.x:.2f}, {pose.y:.2f}), yaw={np.degrees(pose.yaw):.0f}°")
        ax.axis('off')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for i in range(len(test_poses), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    course_path = Path("data/images/course.png")
    output_dir = Path("data/sim_robot")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not course_path.exists():
        print(f"Error: Course image not found at {course_path}")
        return 1

    # Configure LiDAR
    lidar_config = LiDARConfig(
        num_rays=360,
        angle_min=0.0,
        angle_max=2 * np.pi,
        range_min=0.15,
        range_max=12.0,
        noise_std=0.005,
        blind_spot_start=115.0,
        blind_spot_end=246.0
    )

    print("Loading course image...")
    generator = CourseDataGenerator(
        course_path,
        resolution=0.02,
        lidar_config=lidar_config
    )
    print(f"  Image size: {generator.W}x{generator.H} pixels")
    print(f"  Ideal path points: {len(generator.ideal_path_pixels)}")

    # Test with multiple poses
    print("\nTesting waypoint generation with multiple poses...")
    test_multiple_poses(generator, output_dir / "waypoint_test.png", num_poses=9)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())

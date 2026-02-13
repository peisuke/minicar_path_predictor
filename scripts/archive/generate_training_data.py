#!/usr/bin/env python3
"""Generate training data from course image.

Usage:
    python3 generate_training_data.py
    python3 generate_training_data.py --input data/images/course.png --output data/sim_robot/training_data.npz
    python3 generate_training_data.py --resolution 0.02 --grid-spacing 0.05
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training data from course image"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/images/course.png"),
        help="Input course image (default: data/images/course.png)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/sim_robot/training_data.npz"),
        help="Output NPZ file path (default: data/sim_robot/training_data.npz)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.02,
        help="Meters per pixel (default: 0.02)"
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.05,
        help="Spacing between sample positions in meters (default: 0.05)"
    )
    parser.add_argument(
        "--num-orientations",
        type=int,
        default=36,
        help="Number of orientations per position (default: 36 = every 10 degrees)"
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=5,
        help="Number of target waypoints (default: 5)"
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=0.3,
        help="Lookahead distance for first waypoint in meters (default: 0.3)"
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable sensor noise"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images"
    )
    return parser.parse_args()


def visualize_course(generator: CourseDataGenerator, output_dir: Path):
    """Save course visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Original image with masks overlay
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(generator.walls_mask, cmap='gray')
    axes[0, 1].set_title("Walls Mask")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(generator.path_mask, cmap='gray')
    axes[0, 2].set_title("Ideal Path Mask")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(generator.wall_distance, cmap='viridis')
    axes[1, 0].set_title("Distance from Walls")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(generator.track_mask, cmap='gray')
    axes[1, 1].set_title("Track Mask")
    axes[1, 1].axis('off')

    # Path distance field (combined field for gradient ascent)
    axes[1, 2].imshow(generator.path_distance_field, cmap='hot')
    axes[1, 2].set_title("Path Distance Field\n(high=close to path)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "course_analysis.png", dpi=150)
    plt.close()

    # Ideal path
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB))
    path_px = generator.ideal_path_pixels
    ax.plot(path_px[:, 0], path_px[:, 1], 'g-', linewidth=2, label='Ideal Path')
    ax.scatter(path_px[0, 0], path_px[0, 1], c='blue', s=100, marker='o', label='Start')
    ax.set_title("Extracted Ideal Path")
    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "ideal_path.png", dpi=150)
    plt.close()

    print(f"Visualizations saved to {output_dir}")


def visualize_waypoint_comparison(generator: CourseDataGenerator, output_dir: Path, num_samples: int = 6):
    """Compare gradient-based and direct waypoint generation."""
    from minicar_path_predictor.utils import world_to_pixel

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample some random poses from track
    np.random.seed(42)
    candidates = generator.generate_pose_candidates_grid(grid_spacing=0.1, track_only=True)
    if len(candidates) > num_samples:
        indices = np.random.choice(len(candidates), num_samples, replace=False)
        candidates = [candidates[i] for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, pose in enumerate(candidates[:num_samples]):
        ax = axes[i]

        # Show course background
        ax.imshow(cv2.cvtColor(generator.image, cv2.COLOR_BGR2RGB), alpha=0.5)

        # Show ideal path
        path_px = generator.ideal_path_pixels
        ax.plot(path_px[:, 0], path_px[:, 1], 'g-', linewidth=1, alpha=0.5, label='Ideal Path')

        # Robot position in pixels
        robot_px, robot_py = world_to_pixel(pose.x, pose.y, generator.img_shape, generator.resolution)
        ax.plot(robot_px, robot_py, 'ko', markersize=8)

        # Robot direction arrow
        arrow_len = 20  # pixels
        dx = arrow_len * np.cos(pose.yaw)
        dy = -arrow_len * np.sin(pose.yaw)  # Flip Y for image
        ax.arrow(robot_px, robot_py, dx, dy, head_width=5, head_length=3, fc='black', ec='black')

        # Direct waypoints (old method)
        try:
            wp_direct = generator.find_target_waypoints(pose.x, pose.y, pose.yaw, num_points=5, lookahead_dist=0.3)
            # Convert robot frame to world then to pixel
            cos_yaw = np.cos(pose.yaw)
            sin_yaw = np.sin(pose.yaw)
            wp_direct_world = []
            for wp in wp_direct:
                wx = pose.x + wp[0] * cos_yaw - wp[1] * sin_yaw
                wy = pose.y + wp[0] * sin_yaw + wp[1] * cos_yaw
                wpx, wpy = world_to_pixel(wx, wy, generator.img_shape, generator.resolution)
                wp_direct_world.append([wpx, wpy])
            wp_direct_world = np.array(wp_direct_world)
            ax.plot(wp_direct_world[:, 0], wp_direct_world[:, 1], 'r.-', markersize=6, linewidth=2, label='Direct')
        except Exception as e:
            pass

        # Gradient waypoints (new method)
        try:
            wp_gradient = generator.find_target_waypoints_gradient(pose.x, pose.y, pose.yaw, num_points=5, waypoint_spacing=0.3)
            # Convert robot frame to world then to pixel
            wp_grad_world = []
            for wp in wp_gradient:
                wx = pose.x + wp[0] * cos_yaw - wp[1] * sin_yaw
                wy = pose.y + wp[0] * sin_yaw + wp[1] * cos_yaw
                wpx, wpy = world_to_pixel(wx, wy, generator.img_shape, generator.resolution)
                wp_grad_world.append([wpx, wpy])
            wp_grad_world = np.array(wp_grad_world)
            ax.plot(wp_grad_world[:, 0], wp_grad_world[:, 1], 'b.-', markersize=6, linewidth=2, label='Gradient')
        except Exception as e:
            pass

        ax.set_title(f"Pose {i+1}: ({pose.x:.2f}, {pose.y:.2f}), {np.degrees(pose.yaw):.0f}°")
        ax.axis('off')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "waypoint_comparison.png", dpi=150)
    plt.close()

    print(f"Waypoint comparison saved to {output_dir / 'waypoint_comparison.png'}")


def visualize_samples(generator: CourseDataGenerator, samples, output_dir: Path, num_samples: int = 5):
    """Visualize sample LiDAR scans and waypoints."""
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)

    for i, idx in enumerate(indices):
        sample = samples[idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # LiDAR visualization (polar plot)
        ax1 = axes[0]
        angles = np.linspace(0, 2 * np.pi, len(sample.lidar_ranges), endpoint=False)
        # Convert to Cartesian for visualization
        x = sample.lidar_ranges * np.cos(angles)
        y = sample.lidar_ranges * np.sin(angles)

        ax1.scatter(x, y, c='blue', s=1, alpha=0.5)
        ax1.plot([0], [0], 'ro', markersize=10, label='Robot')
        ax1.arrow(0, 0, 0.5, 0, head_width=0.1, head_length=0.05, fc='red', ec='red')
        ax1.set_xlim(-8, 8)
        ax1.set_ylim(-8, 8)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"LiDAR Scan\n(x={sample.robot_x:.2f}, y={sample.robot_y:.2f}, yaw={np.degrees(sample.robot_yaw):.0f}°)")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")

        # Waypoints visualization
        ax2 = axes[1]
        wp = sample.target_waypoints
        ax2.plot([0], [0], 'ro', markersize=10, label='Robot')
        ax2.arrow(0, 0, 0.3, 0, head_width=0.05, head_length=0.03, fc='red', ec='red')
        ax2.plot(wp[:, 0], wp[:, 1], 'g.-', markersize=10, linewidth=2, label='Target Waypoints')
        for j, (wx, wy) in enumerate(wp):
            ax2.annotate(f'{j+1}', (wx, wy), textcoords="offset points", xytext=(5, 5))
        ax2.set_xlim(-2, 3)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Target Waypoints (Robot Frame)")
        ax2.set_xlabel("X (m) - Forward")
        ax2.set_ylabel("Y (m) - Left")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i+1}.png", dpi=150)
        plt.close()

    print(f"Sample visualizations saved to {output_dir}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Training Data Generator")
    print("=" * 60)

    # Check input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Configure LiDAR
    lidar_config = LiDARConfig(
        num_rays=360,
        angle_min=0.0,
        angle_max=2 * np.pi,
        range_min=0.15,
        range_max=12.0,  # 12m range
        noise_std=0.005,
        blind_spot_start=115.0,
        blind_spot_end=246.0
    )

    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.resolution} m/pixel")
    print(f"Grid spacing: {args.grid_spacing} m")
    print(f"Orientations: {args.num_orientations}")
    print(f"Waypoints: {args.num_waypoints}")
    print(f"Lookahead: {args.lookahead} m")
    print(f"Noise: {'disabled' if args.no_noise else 'enabled'}")

    # Create generator
    print("\nLoading course image...")
    generator = CourseDataGenerator(
        args.input,
        resolution=args.resolution,
        lidar_config=lidar_config
    )

    print(f"  Image size: {generator.W}x{generator.H} pixels")
    print(f"  Course size: {generator.W * args.resolution:.2f}m x {generator.H * args.resolution:.2f}m")
    print(f"  Ideal path points: {len(generator.ideal_path_pixels)}")

    # Visualize course
    if args.visualize:
        vis_dir = args.output.parent / "visualizations"
        visualize_course(generator, vis_dir)
        visualize_waypoint_comparison(generator, vis_dir)

    # Generate samples
    print("\nGenerating training samples...")

    def progress_callback(count, total):
        percent = count / total * 100
        print(f"  Progress: {count}/{total} ({percent:.1f}%)", end='\r')

    samples = generator.generate_samples(
        grid_spacing=args.grid_spacing,
        num_orientations=args.num_orientations,
        num_waypoints=args.num_waypoints,
        lookahead_dist=args.lookahead,
        min_wall_distance=0.1,
        add_noise=not args.no_noise,
        progress_callback=progress_callback
    )

    print(f"\n  Generated {len(samples)} samples")

    # Visualize samples
    if args.visualize and samples:
        vis_dir = args.output.parent / "visualizations"
        visualize_samples(generator, samples, vis_dir)

    # Save samples
    print(f"\nSaving to {args.output}...")
    generator.save_samples(samples, args.output)

    # Print statistics
    print("\nDataset Statistics:")
    lidar_data = np.array([s.lidar_ranges for s in samples])
    waypoints_data = np.array([s.target_waypoints for s in samples])

    print(f"  LiDAR shape: {lidar_data.shape}")
    print(f"  Waypoints shape: {waypoints_data.shape}")
    print(f"  LiDAR range: [{lidar_data.min():.3f}, {lidar_data.max():.3f}] m")
    print(f"  Waypoint X range: [{waypoints_data[:,:,0].min():.3f}, {waypoints_data[:,:,0].max():.3f}] m")
    print(f"  Waypoint Y range: [{waypoints_data[:,:,1].min():.3f}, {waypoints_data[:,:,1].max():.3f}] m")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

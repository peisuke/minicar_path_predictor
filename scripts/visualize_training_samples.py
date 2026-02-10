#!/usr/bin/env python3
"""Visualize training samples on course map.

Shows:
- Course map with robot position
- LiDAR scan overlay
- Smoothed path
- Target point and angle
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    data_dir = Path("data/training")
    course_path = Path("data/images/course.png")
    output_dir = Path("data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load course image
    course_img = cv2.imread(str(course_path))
    course_img = cv2.cvtColor(course_img, cv2.COLOR_BGR2RGB)
    H, W = course_img.shape[:2]
    resolution = 0.02  # meters per pixel

    # Load training data
    lidar_data = np.load(data_dir / "lidar_data.npy")
    target_angles = np.load(data_dir / "target_angles.npy")
    poses = np.load(data_dir / "poses.npy")
    smoothed_paths = np.load(data_dir / "smoothed_paths.npy")
    target_points = np.load(data_dir / "target_points.npy")

    num_samples = len(lidar_data)
    print(f"Loaded {num_samples} samples")

    # LiDAR config
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    max_range = 12.0

    # Create figure with subplots
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        ax = axes[i]

        # Robot pose (world coordinates)
        robot_x, robot_y, robot_yaw = poses[i]

        # Convert world to image coordinates
        robot_px = W / 2 + robot_x / resolution
        robot_py = H / 2 - robot_y / resolution

        # Show course image (zoomed to robot area)
        zoom_size = 150  # pixels
        x1 = int(max(0, robot_px - zoom_size))
        x2 = int(min(W, robot_px + zoom_size))
        y1 = int(max(0, robot_py - zoom_size))
        y2 = int(min(H, robot_py + zoom_size))

        ax.imshow(course_img[y1:y2, x1:x2], extent=[x1, x2, y2, y1])

        # Draw robot position
        ax.plot(robot_px, robot_py, 'go', markersize=10, label='Robot')

        # Draw robot heading
        heading_len = 20  # pixels
        hx = robot_px + heading_len * np.cos(robot_yaw)
        hy = robot_py - heading_len * np.sin(robot_yaw)
        ax.arrow(robot_px, robot_py, hx - robot_px, hy - robot_py,
                 head_width=5, head_length=3, fc='green', ec='green')

        # Draw LiDAR points
        lidar_ranges = lidar_data[i]
        valid_mask = lidar_ranges < max_range - 0.1
        for j, (r, a) in enumerate(zip(lidar_ranges, angles)):
            if valid_mask[j]:
                # Convert to robot frame
                lx_robot = r * np.cos(a)
                ly_robot = r * np.sin(a)
                # Convert to world frame
                lx_world = robot_x + lx_robot * np.cos(robot_yaw) - ly_robot * np.sin(robot_yaw)
                ly_world = robot_y + lx_robot * np.sin(robot_yaw) + ly_robot * np.cos(robot_yaw)
                # Convert to image coordinates
                lpx = W / 2 + lx_world / resolution
                lpy = H / 2 - ly_world / resolution
                ax.plot(lpx, lpy, 'r.', markersize=1, alpha=0.5)

        # Draw smoothed path (robot frame -> world frame -> image)
        path_robot = smoothed_paths[i]  # (15, 2) in robot frame
        for j, (px, py) in enumerate(path_robot):
            # Convert to world frame
            wx = robot_x + px * np.cos(robot_yaw) - py * np.sin(robot_yaw)
            wy = robot_y + px * np.sin(robot_yaw) + py * np.cos(robot_yaw)
            # Convert to image
            ipx = W / 2 + wx / resolution
            ipy = H / 2 - wy / resolution
            if j == 0:
                ax.plot(ipx, ipy, 'b.', markersize=4)
            else:
                ax.plot(ipx, ipy, 'b.', markersize=4)

        # Connect path points
        path_px = []
        path_py = []
        for px, py in path_robot:
            wx = robot_x + px * np.cos(robot_yaw) - py * np.sin(robot_yaw)
            wy = robot_y + px * np.sin(robot_yaw) + py * np.cos(robot_yaw)
            path_px.append(W / 2 + wx / resolution)
            path_py.append(H / 2 - wy / resolution)
        ax.plot(path_px, path_py, 'b-', linewidth=2, alpha=0.7, label='Path')

        # Draw target point
        tp = target_points[i]  # robot frame
        tp_wx = robot_x + tp[0] * np.cos(robot_yaw) - tp[1] * np.sin(robot_yaw)
        tp_wy = robot_y + tp[0] * np.sin(robot_yaw) + tp[1] * np.cos(robot_yaw)
        tp_px = W / 2 + tp_wx / resolution
        tp_py = H / 2 - tp_wy / resolution
        ax.plot(tp_px, tp_py, 'mo', markersize=8, label='Target')

        # Draw target angle direction
        target_angle = target_angles[i]
        angle_world = robot_yaw + target_angle
        ta_len = 30
        ta_x = robot_px + ta_len * np.cos(angle_world)
        ta_y = robot_py - ta_len * np.sin(angle_world)
        ax.arrow(robot_px, robot_py, ta_x - robot_px, ta_y - robot_py,
                 head_width=4, head_length=3, fc='magenta', ec='magenta', alpha=0.8)

        ax.set_title(f"Sample {i+1}: angle={np.degrees(target_angle):.1f}°")
        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
        ax.set_aspect('equal')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path = output_dir / "training_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize ML output (target_angle, curvature) from path_attraction pipeline."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PathPlannerConfig,
    DistanceFieldGenerator, DistanceFieldParams,
    PeakDetector, GraphPathSearcher, PotentialPathSmoother
)
from minicar_navigation.controller.pd_pursuit_controller import (
    PDPursuitController, PDPursuitControllerConfig
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig


def smooth_path_follow_waypoints(coordinate_path: np.ndarray, num_points: int = 15) -> np.ndarray:
    """Smooth path that follows intermediate waypoints using Catmull-Rom spline.

    Unlike the original Hermite/Bezier method that aims at the goal,
    this method passes through all waypoints smoothly.

    Args:
        coordinate_path: Waypoints in centered coordinates, shape (N, 2)
        num_points: Number of output points

    Returns:
        Smoothed path, shape (num_points, 2)
    """
    if len(coordinate_path) < 2:
        return coordinate_path

    # Add robot position (0, 0) at the start
    path_with_start = np.vstack([[0.0, 0.0], coordinate_path])

    if len(path_with_start) < 3:
        # Not enough points for spline, just interpolate linearly
        t = np.linspace(0, 1, num_points)
        return (1 - t)[:, np.newaxis] * path_with_start[0] + t[:, np.newaxis] * path_with_start[-1]

    # Catmull-Rom spline interpolation
    # Add phantom points at start and end for boundary conditions
    p0 = 2 * path_with_start[0] - path_with_start[1]  # Phantom before start
    p_end = 2 * path_with_start[-1] - path_with_start[-2]  # Phantom after end

    extended_path = np.vstack([p0, path_with_start, p_end])

    # Calculate total path length for parameterization
    segment_lengths = np.linalg.norm(np.diff(path_with_start, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_lengths[-1]

    if total_length < 1e-6:
        return np.tile(path_with_start[0], (num_points, 1))

    # Generate uniformly spaced points along the path
    target_lengths = np.linspace(0, total_length, num_points)

    result = []
    for target_len in target_lengths:
        # Find which segment this point falls in
        seg_idx = np.searchsorted(cumulative_lengths[1:], target_len)
        seg_idx = min(seg_idx, len(path_with_start) - 2)

        # Local parameter within segment [0, 1]
        seg_start_len = cumulative_lengths[seg_idx]
        seg_len = segment_lengths[seg_idx] if segment_lengths[seg_idx] > 1e-6 else 1e-6
        t = (target_len - seg_start_len) / seg_len
        t = np.clip(t, 0, 1)

        # Catmull-Rom spline formula
        # P(t) = 0.5 * [(2*P1) + (-P0 + P2)*t + (2*P0 - 5*P1 + 4*P2 - P3)*t^2 + (-P0 + 3*P1 - 3*P2 + P3)*t^3]
        i = seg_idx + 1  # Index in extended_path (offset by 1 due to phantom point)
        P0 = extended_path[i - 1]
        P1 = extended_path[i]
        P2 = extended_path[i + 1]
        P3 = extended_path[i + 2]

        t2 = t * t
        t3 = t2 * t

        point = 0.5 * (
            2 * P1 +
            (-P0 + P2) * t +
            (2 * P0 - 5 * P1 + 4 * P2 - P3) * t2 +
            (-P0 + 3 * P1 - 3 * P2 + P3) * t3
        )
        result.append(point)

    return np.array(result)


def compute_path_and_control(generator, pose, config, lookahead_distance=0.5):
    """Compute path and control outputs using existing logic."""
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
    dist_field_gen = DistanceFieldGenerator(config)
    df_params = DistanceFieldParams(
        H=local_size, W=local_size, R_in=R_in,
        ring_thickness=1, front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
    )
    mask, dist_inside_orig, dist_norm_orig, cx, cy, _ = \
        dist_field_gen.generate_distance_field(view_points, df_params)

    # 3. Create path_attraction field
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

    wall_dist_max = dist_inside_orig.max()
    path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
    dist_inside = path_attraction * (mask > 0).astype(np.float32)

    if dist_inside.max() > 0:
        dist_norm = (dist_inside / dist_inside.max() * 255).astype(np.uint8)
    else:
        dist_norm = np.zeros_like(dist_inside, dtype=np.uint8)

    # ========== Stage 1: Peak detection ==========
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

        paths_ids = result_paths["paths_ids"]
        deduplicated_paths_ids = path_searcher.deduplicator.deduplicate_paths(
            paths_ids, centered_points, result_paths["flat_to_pts"]
        )

        coordinate_paths = path_searcher._paths_to_coords(
            deduplicated_paths_ids, centered_points, result_paths["flat_to_pts"]
        )
    else:
        coordinate_paths = []

    # ========== Stage 3: Path smoothing (ORIGINAL) ==========
    smoother = PotentialPathSmoother(config)
    if len(coordinate_paths) > 0:
        smoothed_paths = smoother.smooth_paths(coordinate_paths, dist_inside, local_size, local_size)
    else:
        smoothed_paths = []

    # ========== Stage 3b: Path smoothing (NEW - follow waypoints) ==========
    smoothed_paths_new = []
    for coord_path in coordinate_paths:
        if len(coord_path) > 0:
            smoothed_paths_new.append(smooth_path_follow_waypoints(coord_path, num_points=15))

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
    smooth_paths_new_robot = [img_to_robot(p) for p in smoothed_paths_new if len(p) > 0]

    # ========== Compute ML output using existing controller logic ==========
    target_angle = None
    curvature = None
    target_point = None

    if len(smooth_paths_robot) > 0:
        # Use the first (best) path
        path = smooth_paths_robot[0]

        # Use PDPursuitController's logic directly
        controller_config = PDPursuitControllerConfig(
            LOOKAHEAD_DISTANCE=lookahead_distance
        )
        controller = PDPursuitController(controller_config)

        # Find lookahead point (robot at origin)
        target_point_tuple, nearest_idx = controller._find_lookahead_point(path, 0.0, 0.0)

        if target_point_tuple is not None:
            target_point = np.array(target_point_tuple)
            # Compute target_angle (robot frame, so current_yaw = 0)
            target_angle = np.arctan2(target_point[1], target_point[0])

            # Curvature: hardcode to 0 for compatibility with existing controller
            curvature = 0.0

    # ========== Compute ML output for NEW smoothing ==========
    target_angle_new = None
    target_point_new = None

    if len(smooth_paths_new_robot) > 0:
        path_new = smooth_paths_new_robot[0]
        controller_config = PDPursuitControllerConfig(LOOKAHEAD_DISTANCE=lookahead_distance)
        controller = PDPursuitController(controller_config)
        target_point_tuple_new, _ = controller._find_lookahead_point(path_new, 0.0, 0.0)

        if target_point_tuple_new is not None:
            target_point_new = np.array(target_point_tuple_new)
            target_angle_new = np.arctan2(target_point_new[1], target_point_new[0])

    return {
        'dist_inside': dist_inside,
        'peaks_robot': peaks_robot,
        'rough_paths_robot': rough_paths_robot,
        'smooth_paths_robot': smooth_paths_robot,
        'smooth_paths_new_robot': smooth_paths_new_robot,
        'target_point': target_point,
        'target_angle': target_angle,
        'target_point_new': target_point_new,
        'target_angle_new': target_angle_new,
        'curvature': curvature,
        'lidar_ranges': lidar_ranges,
        'resolution': resolution,
        'local_size': local_size,
        'pose': pose,
    }


def visualize_ml_output(result, output_path, lookahead_distance):
    """Visualize path stages and ML output comparing original vs new smoothing."""
    local_size = result['local_size']
    resolution = result['resolution']
    pose = result['pose']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    extent = [-local_size/2 * resolution, local_size/2 * resolution,
              -local_size/2 * resolution, local_size/2 * resolution]

    dist_robot = np.flipud(result['dist_inside'])
    theta_circle = np.linspace(0, 2*np.pi, 100)

    # ========== Left: Path stages (rough + both smoothing) ==========
    ax = axes[0]
    ax.imshow(dist_robot, cmap='hot', alpha=0.6, extent=extent, origin='lower')

    # Peaks
    peaks_robot = result['peaks_robot']
    if len(peaks_robot) > 0:
        ax.scatter(peaks_robot[:, 0], peaks_robot[:, 1], c='lime', s=80,
                   edgecolors='white', linewidths=1.5, zorder=5, label=f'peaks ({len(peaks_robot)})')

    # Rough paths
    for i, path in enumerate(result['rough_paths_robot']):
        ax.plot(path[:, 0], path[:, 1], 'o-', color='orange', markersize=8,
                linewidth=2, alpha=0.7, label=f'rough ({len(path)} pts)' if i == 0 else None)

    # Original smoothed paths
    for i, path in enumerate(result['smooth_paths_robot']):
        ax.plot(path[:, 0], path[:, 1], '.-', color='cyan', markersize=3,
                linewidth=2, label=f'smooth_orig ({len(path)} pts)' if i == 0 else None)

    # New smoothed paths
    for i, path in enumerate(result['smooth_paths_new_robot']):
        ax.plot(path[:, 0], path[:, 1], '.-', color='magenta', markersize=3,
                linewidth=2, label=f'smooth_new ({len(path)} pts)' if i == 0 else None)

    ax.plot(0, 0, 'w+', markersize=15, markeredgewidth=2)
    ax.arrow(0, 0, 0.2, 0, head_width=0.08, head_length=0.04, fc='white', ec='white')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.set_title('Path Comparison\ncyan=original, magenta=new')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ========== Middle: Original smoothing + ML output ==========
    ax = axes[1]
    ax.imshow(dist_robot, cmap='hot', alpha=0.4, extent=extent, origin='lower')

    if len(result['smooth_paths_robot']) > 0:
        path = result['smooth_paths_robot'][0]
        ax.plot(path[:, 0], path[:, 1], 'c.-', markersize=4, linewidth=2, label='smooth_orig')

    ax.plot(0, 0, 'wo', markersize=12, markeredgecolor='black', markeredgewidth=2)
    ax.arrow(0, 0, 0.3, 0, head_width=0.08, head_length=0.04, fc='white', ec='black', linewidth=2)
    ax.plot(lookahead_distance * np.cos(theta_circle),
            lookahead_distance * np.sin(theta_circle),
            'g--', linewidth=1.5, alpha=0.7)

    target_point = result['target_point']
    target_angle = result['target_angle']
    if target_point is not None:
        ax.plot(target_point[0], target_point[1], 'r*', markersize=20,
                markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        ax.annotate('', xy=(target_point[0], target_point[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    info_text = f"angle: {np.degrees(target_angle):.1f}°" if target_angle is not None else "angle: N/A"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.set_title(f'ORIGINAL Smoothing\nangle={np.degrees(target_angle):.1f}°' if target_angle else 'ORIGINAL Smoothing')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ========== Right: New smoothing + ML output ==========
    ax = axes[2]
    ax.imshow(dist_robot, cmap='hot', alpha=0.4, extent=extent, origin='lower')

    if len(result['smooth_paths_new_robot']) > 0:
        path = result['smooth_paths_new_robot'][0]
        ax.plot(path[:, 0], path[:, 1], 'm.-', markersize=4, linewidth=2, label='smooth_new')

    ax.plot(0, 0, 'wo', markersize=12, markeredgecolor='black', markeredgewidth=2)
    ax.arrow(0, 0, 0.3, 0, head_width=0.08, head_length=0.04, fc='white', ec='black', linewidth=2)
    ax.plot(lookahead_distance * np.cos(theta_circle),
            lookahead_distance * np.sin(theta_circle),
            'g--', linewidth=1.5, alpha=0.7)

    target_point_new = result['target_point_new']
    target_angle_new = result['target_angle_new']
    if target_point_new is not None:
        ax.plot(target_point_new[0], target_point_new[1], 'r*', markersize=20,
                markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        ax.annotate('', xy=(target_point_new[0], target_point_new[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    info_text = f"angle: {np.degrees(target_angle_new):.1f}°" if target_angle_new is not None else "angle: N/A"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.set_title(f'NEW Smoothing (Catmull-Rom)\nangle={np.degrees(target_angle_new):.1f}°' if target_angle_new else 'NEW Smoothing')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Pose: ({pose.x:.2f}, {pose.y:.2f}), yaw={np.degrees(pose.yaw):.0f}°\n'
                 f'lookahead={lookahead_distance}m', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    target_angle = result['target_angle']
    target_angle_new = result['target_angle_new']
    print(f"Saved: {output_path}")
    print(f"  ORIGINAL angle: {np.degrees(target_angle):.2f}°" if target_angle is not None else "  ORIGINAL angle: N/A")
    print(f"  NEW angle:      {np.degrees(target_angle_new):.2f}°" if target_angle_new is not None else "  NEW angle: N/A")


def main():
    course_path = Path("data/images/course.png")
    output_dir = Path("data/sim_robot")
    output_dir.mkdir(parents=True, exist_ok=True)

    lookahead_distance = 0.5  # Fixed lookahead distance

    lidar_config = LiDARConfig(
        num_rays=360, angle_min=0.0, angle_max=2 * np.pi,
        range_min=0.15, range_max=12.0, noise_std=0.005,
        blind_spot_start=115.0, blind_spot_end=246.0
    )

    print("Loading course...")
    generator = CourseDataGenerator(course_path, resolution=0.02, lidar_config=lidar_config)

    config = PathPlannerConfig(HEIGHT=200, WIDTH=200)

    np.random.seed(42)
    candidates = generator.generate_pose_candidates_grid(grid_spacing=0.1, track_only=True)
    indices = np.random.choice(len(candidates), 5, replace=False)

    print(f"\nProcessing {len(indices)} poses with lookahead={lookahead_distance}m...")

    for i, idx in enumerate(indices):
        pose = candidates[idx]
        result = compute_path_and_control(generator, pose, config, lookahead_distance)
        visualize_ml_output(result, output_dir / f"ml_output_{i+1}.png", lookahead_distance)

    print("\nDone!")


if __name__ == "__main__":
    main()

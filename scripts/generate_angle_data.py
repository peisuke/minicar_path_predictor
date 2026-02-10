#!/usr/bin/env python3
"""Generate training data for angle prediction ML model.

Uses path_attraction field + Catmull-Rom smoothing.
Output: LiDAR (360) -> target_angle (1)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")

from minicar_navigation.planner.local_path_planner import (
    PathPlannerConfig,
    DistanceFieldGenerator, DistanceFieldParams,
    PeakDetector, GraphPathSearcher
)
from minicar_navigation.controller.pd_pursuit_controller import (
    PDPursuitController, PDPursuitControllerConfig
)
from minicar_path_predictor.data_generator import CourseDataGenerator, LiDARConfig


def smooth_path_catmull_rom(coordinate_path: np.ndarray, num_points: int = 15) -> np.ndarray:
    """Smooth path using Catmull-Rom spline that follows intermediate waypoints."""
    if len(coordinate_path) < 2:
        return coordinate_path

    path_with_start = np.vstack([[0.0, 0.0], coordinate_path])

    if len(path_with_start) < 3:
        t = np.linspace(0, 1, num_points)
        return (1 - t)[:, np.newaxis] * path_with_start[0] + t[:, np.newaxis] * path_with_start[-1]

    p0 = 2 * path_with_start[0] - path_with_start[1]
    p_end = 2 * path_with_start[-1] - path_with_start[-2]
    extended_path = np.vstack([p0, path_with_start, p_end])

    segment_lengths = np.linalg.norm(np.diff(path_with_start, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_lengths[-1]

    if total_length < 1e-6:
        return np.tile(path_with_start[0], (num_points, 1))

    target_lengths = np.linspace(0, total_length, num_points)

    result = []
    for target_len in target_lengths:
        seg_idx = np.searchsorted(cumulative_lengths[1:], target_len)
        seg_idx = min(seg_idx, len(path_with_start) - 2)

        seg_start_len = cumulative_lengths[seg_idx]
        seg_len = segment_lengths[seg_idx] if segment_lengths[seg_idx] > 1e-6 else 1e-6
        t = (target_len - seg_start_len) / seg_len
        t = np.clip(t, 0, 1)

        i = seg_idx + 1
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


def generate_sample(generator, pose, config, lookahead_distance=0.5):
    """Generate a single training sample: (lidar_data, target_angle)."""
    robot_x, robot_y, robot_yaw = pose.x, pose.y, pose.yaw
    local_size = 200
    resolution = generator.resolution
    scale = 1.0 / resolution
    center_x, center_y = local_size // 2, local_size // 2
    R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
    dist_thresh = 15.0

    # 1. Simulate LiDAR
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

    if len(view_points) < 10:
        return None

    # 2. Distance field
    dist_field_gen = DistanceFieldGenerator(config)
    df_params = DistanceFieldParams(
        H=local_size, W=local_size, R_in=R_in,
        ring_thickness=1, front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
    )
    mask, dist_inside_orig, _, cx, cy, _ = \
        dist_field_gen.generate_distance_field(view_points, df_params)

    # 3. Path attraction field
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

    if dist_inside.max() <= 0:
        return None

    dist_norm = (dist_inside / dist_inside.max() * 255).astype(np.uint8)

    # 4. Peak detection
    _, _, delta_x, delta_y, radius_squared = dist_field_gen._compute_center_grid(local_size, local_size)
    ring_mask = dist_field_gen._create_ring_mask(radius_squared, R_in, thickness=1)
    front_mask = dist_field_gen._create_front_mask(delta_x, delta_y, 90.0)
    path_candidate_pixels = ring_mask & front_mask & (dist_inside >= dist_thresh)

    _, labels, _ = dist_field_gen.connect_regions_and_label(path_candidate_pixels, kernel_sz=3)

    peak_detector = PeakDetector(config=config)
    peaks, centered_points, sorted_labels = peak_detector._detect_path_candidates(
        labels, dist_norm, center_x, center_y
    )

    if len(centered_points) == 0:
        return None

    # 5. Path building (skip lookahead trimming)
    path_searcher = GraphPathSearcher(config)
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
    if len(paths_ids) == 0:
        return None

    deduplicated_paths_ids = path_searcher.deduplicator.deduplicate_paths(
        paths_ids, centered_points, result_paths["flat_to_pts"]
    )

    if len(deduplicated_paths_ids) == 0:
        return None

    coordinate_paths = path_searcher._paths_to_coords(
        deduplicated_paths_ids, centered_points, result_paths["flat_to_pts"]
    )

    if len(coordinate_paths) == 0:
        return None

    # 6. Catmull-Rom smoothing
    coord_path = coordinate_paths[0]
    smoothed_path = smooth_path_catmull_rom(coord_path, num_points=15)

    # Convert to robot frame (meters)
    path_robot = np.zeros_like(smoothed_path, dtype=np.float32)
    path_robot[:, 0] = smoothed_path[:, 0] * resolution
    path_robot[:, 1] = -smoothed_path[:, 1] * resolution

    # 7. Compute target_angle using controller logic
    controller_config = PDPursuitControllerConfig(LOOKAHEAD_DISTANCE=lookahead_distance)
    controller = PDPursuitController(controller_config)
    target_point_tuple, _ = controller._find_lookahead_point(path_robot, 0.0, 0.0)

    if target_point_tuple is None:
        return None

    target_point = np.array(target_point_tuple)
    target_angle = np.arctan2(target_point[1], target_point[0])

    return {
        'lidar_ranges': lidar_ranges.astype(np.float32),
        'target_angle': np.float32(target_angle),
        # Debug info (not used for training)
        'pose': np.array([robot_x, robot_y, robot_yaw], dtype=np.float32),
        'smoothed_path': path_robot.astype(np.float32),
        'target_point': target_point.astype(np.float32),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--grid-spacing", type=float, default=0.05, help="Grid spacing for pose candidates")
    args = parser.parse_args()

    course_path = Path("data/images/course.png")
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    lookahead_distance = 0.5
    num_samples = args.num_samples

    lidar_config = LiDARConfig(
        num_rays=360, angle_min=0.0, angle_max=2 * np.pi,
        range_min=0.15, range_max=12.0, noise_std=0.005,
        blind_spot_start=115.0, blind_spot_end=246.0
    )

    print("Loading course...")
    generator = CourseDataGenerator(course_path, resolution=0.02, lidar_config=lidar_config)
    config = PathPlannerConfig(HEIGHT=200, WIDTH=200)

    print(f"Generating {num_samples} training samples...")
    candidates = generator.generate_pose_candidates_grid(grid_spacing=args.grid_spacing, track_only=True)
    print(f"  Total pose candidates: {len(candidates)}")

    np.random.seed(42)
    if len(candidates) > num_samples:
        indices = np.random.choice(len(candidates), num_samples, replace=False)
    else:
        indices = np.arange(len(candidates))

    lidar_data_list = []
    target_angle_list = []
    # Debug info
    pose_list = []
    smoothed_path_list = []
    target_point_list = []

    valid_count = 0
    skip_count = 0

    for i, idx in enumerate(indices):
        pose = candidates[idx]
        sample = generate_sample(generator, pose, config, lookahead_distance)

        if sample is not None:
            lidar_data_list.append(sample['lidar_ranges'])
            target_angle_list.append(sample['target_angle'])
            # Debug info
            pose_list.append(sample['pose'])
            smoothed_path_list.append(sample['smoothed_path'])
            target_point_list.append(sample['target_point'])
            valid_count += 1
        else:
            skip_count += 1

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(indices)}, valid={valid_count}, skipped={skip_count}")

    # Save as numpy arrays
    lidar_data = np.array(lidar_data_list)
    target_angles = np.array(target_angle_list)
    poses = np.array(pose_list)
    smoothed_paths = np.array(smoothed_path_list)
    target_points = np.array(target_point_list)

    np.save(output_dir / "lidar_data.npy", lidar_data)
    np.save(output_dir / "target_angles.npy", target_angles)
    # Debug info
    np.save(output_dir / "poses.npy", poses)
    np.save(output_dir / "smoothed_paths.npy", smoothed_paths)
    np.save(output_dir / "target_points.npy", target_points)

    print(f"\nSaved {valid_count} samples to {output_dir}")
    print(f"  lidar_data.npy: shape={lidar_data.shape}")
    print(f"  target_angles.npy: shape={target_angles.shape}")
    print(f"  angle range: [{np.degrees(target_angles.min()):.1f}, {np.degrees(target_angles.max()):.1f}] degrees")
    print(f"  [debug] poses.npy: shape={poses.shape}")
    print(f"  [debug] smoothed_paths.npy: shape={smoothed_paths.shape}")
    print(f"  [debug] target_points.npy: shape={target_points.shape}")


if __name__ == "__main__":
    main()

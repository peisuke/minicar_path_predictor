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


def get_path_direction_at_robot(local_dist_to_path: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
    """Get the forward direction along the ideal path at robot position.

    Uses gradient of distance field to find path tangent direction.
    Disambiguates using robot forward direction (assumed to be roughly aligned with path).

    Returns:
        Unit vector in centered image coordinates (dx_img, dy_img)
    """
    # Compute gradient of distance to path (points perpendicular to path)
    grad_y, grad_x = np.gradient(local_dist_to_path)
    gx = grad_x[center_y, center_x]
    gy = grad_y[center_y, center_x]

    # Path tangent is perpendicular to gradient (two options)
    tangent1 = np.array([-gy, gx])   # rotate 90° CCW
    tangent2 = np.array([gy, -gx])   # rotate 90° CW

    # Robot forward in image coords: +x_img (since x_robot -> x_img in this code)
    robot_forward_img = np.array([1.0, 0.0])

    # Choose tangent aligned with robot forward
    if np.dot(tangent1, robot_forward_img) > np.dot(tangent2, robot_forward_img):
        path_direction = tangent1
    else:
        path_direction = tangent2

    # Normalize
    norm = np.linalg.norm(path_direction)
    if norm < 1e-6:
        return robot_forward_img  # fallback to robot forward
    return path_direction / norm


def filter_backwards_waypoints(coord_path: np.ndarray, path_direction: np.ndarray,
                               cos_threshold: float = -0.5) -> np.ndarray:
    """Filter waypoints that go backwards relative to path direction.

    Args:
        coord_path: Waypoints in centered image coordinates
        path_direction: Unit vector of path forward direction
        cos_threshold: Filter if dot product < threshold (default -0.5 = 120°)

    Returns:
        Filtered waypoints
    """
    if len(coord_path) == 0:
        return coord_path

    filtered = []
    for wp in coord_path:
        dist = np.linalg.norm(wp)
        if dist < 1e-6:
            continue
        wp_dir = wp / dist
        dot = np.dot(wp_dir, path_direction)
        if dot >= cos_threshold:  # Not going backwards
            filtered.append(wp)

    if len(filtered) == 0:
        return np.array([]).reshape(0, 2)
    return np.array(filtered)


def smooth_path_catmull_rom(coordinate_path: np.ndarray, num_points: int = 15,
                            min_distance: float = 0.0) -> np.ndarray:
    """Smooth path using Catmull-Rom spline that follows intermediate waypoints.

    Args:
        coordinate_path: Waypoints in centered coordinates (origin = robot)
        num_points: Number of output points
        min_distance: Minimum distance from robot to consider (filter closer points)
    """
    if len(coordinate_path) < 2:
        return coordinate_path

    # Filter out points closer than min_distance
    if min_distance > 0:
        distances = np.linalg.norm(coordinate_path, axis=1)
        far_mask = distances >= min_distance
        coordinate_path = coordinate_path[far_mask]

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


def generate_sample(generator, pose, config, lookahead_distances=[0.5], include_no_path=False):
    """Generate a single training sample: (lidar_data, target_angles, has_path).

    Args:
        lookahead_distances: List of lookahead distances to compute target angles for
        include_no_path: If True, return samples even when no valid path is found (has_path=False)
    """
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

    # Helper to return no-path sample
    def no_path_sample():
        if include_no_path:
            return {
                'lidar_ranges': lidar_ranges.astype(np.float32),
                'target_angles': np.zeros(len(lookahead_distances), dtype=np.float32),  # dummy
                'has_path': False,
                'pose': np.array([robot_x, robot_y, robot_yaw], dtype=np.float32),
            }
        return None

    if len(view_points) < 10:
        return no_path_sample()

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
        return no_path_sample()

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
        return no_path_sample()

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
        return no_path_sample()

    deduplicated_paths_ids = path_searcher.deduplicator.deduplicate_paths(
        paths_ids, centered_points, result_paths["flat_to_pts"]
    )

    if len(deduplicated_paths_ids) == 0:
        return no_path_sample()

    coordinate_paths = path_searcher._paths_to_coords(
        deduplicated_paths_ids, centered_points, result_paths["flat_to_pts"]
    )

    if len(coordinate_paths) == 0:
        return no_path_sample()

    # 6. Filter backwards waypoints and apply Catmull-Rom smoothing
    coord_path = coordinate_paths[0]
    min_dist_pixels = 0.5 / resolution  # 50cm in pixels (=25 pixels at 0.02m/px)

    # 6a. Get path direction at robot position and filter backwards waypoints
    path_direction = get_path_direction_at_robot(local_dist_to_path, center_x, center_y)
    coord_path = filter_backwards_waypoints(coord_path, path_direction, cos_threshold=-0.5)

    if len(coord_path) < 2:
        return no_path_sample()

    # 6b. Extract valid waypoints (>= 50cm) for debug
    distances_px = np.linalg.norm(coord_path, axis=1)
    valid_mask = distances_px >= min_dist_pixels
    valid_waypoints_px = coord_path[valid_mask]

    if len(valid_waypoints_px) < 1:
        return no_path_sample()

    # Convert valid waypoints to robot frame (meters)
    valid_waypoints_robot = np.zeros_like(valid_waypoints_px, dtype=np.float32)
    valid_waypoints_robot[:, 0] = valid_waypoints_px[:, 0] * resolution  # x (forward)
    valid_waypoints_robot[:, 1] = -valid_waypoints_px[:, 1] * resolution  # y (left)

    # 6c. Catmull-Rom smoothing (ignore points closer than 50cm)
    smoothed_path = smooth_path_catmull_rom(coord_path, num_points=15, min_distance=min_dist_pixels)

    # Convert to robot frame (meters)
    path_robot = np.zeros_like(smoothed_path, dtype=np.float32)
    path_robot[:, 0] = smoothed_path[:, 0] * resolution
    path_robot[:, 1] = -smoothed_path[:, 1] * resolution

    # 7. Compute target_angles for each lookahead distance using controller logic
    target_angles = []
    target_points = []
    for lookahead_dist in lookahead_distances:
        controller_config = PDPursuitControllerConfig(LOOKAHEAD_DISTANCE=lookahead_dist)
        controller = PDPursuitController(controller_config)
        target_point_tuple, _ = controller._find_lookahead_point(path_robot, 0.0, 0.0)

        if target_point_tuple is None:
            return no_path_sample()

        target_point = np.array(target_point_tuple)
        target_angle = np.arctan2(target_point[1], target_point[0])
        target_angles.append(target_angle)
        target_points.append(target_point)

    return {
        'lidar_ranges': lidar_ranges.astype(np.float32),
        'target_angles': np.array(target_angles, dtype=np.float32),
        'has_path': True,
        # Debug info (not used for training)
        'pose': np.array([robot_x, robot_y, robot_yaw], dtype=np.float32),
        'smoothed_path': path_robot.astype(np.float32),
        'target_points': np.array(target_points, dtype=np.float32),
        'valid_waypoints': valid_waypoints_robot.astype(np.float32),  # waypoints >= 50cm
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--grid-spacing", type=float, default=0.05, help="Grid spacing for pose candidates")
    parser.add_argument("--include-no-path", action="store_true", help="Include samples where no valid path was found")
    parser.add_argument("--lookahead-distances", type=str, default="0.5",
                        help="Comma-separated lookahead distances in meters (e.g., '0.25,0.5,0.75,1.0')")
    args = parser.parse_args()

    course_path = Path("data/images/course.png")
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    lookahead_distances = [float(x.strip()) for x in args.lookahead_distances.split(",")]
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
    print(f"  Lookahead distances: {lookahead_distances} m")
    print(f"  Include no-path samples: {args.include_no_path}")
    candidates = generator.generate_pose_candidates_grid(grid_spacing=args.grid_spacing, track_only=True)
    print(f"  Total pose candidates: {len(candidates)}")

    np.random.seed(42)
    if len(candidates) > num_samples:
        indices = np.random.choice(len(candidates), num_samples, replace=False)
    else:
        indices = np.arange(len(candidates))

    lidar_data_list = []
    target_angles_list = []  # Now a list of arrays, one per sample
    has_path_list = []
    # Debug info
    pose_list = []
    smoothed_path_list = []
    target_points_list = []  # Now arrays with shape (num_lookahead_distances, 2)
    valid_waypoints_list = []  # waypoints >= 50cm (variable length)

    path_count = 0
    no_path_count = 0
    skip_count = 0

    for i, idx in enumerate(indices):
        pose = candidates[idx]
        sample = generate_sample(generator, pose, config, lookahead_distances,
                                 include_no_path=args.include_no_path)

        if sample is not None:
            lidar_data_list.append(sample['lidar_ranges'])
            target_angles_list.append(sample['target_angles'])
            has_path_list.append(sample['has_path'])
            pose_list.append(sample['pose'])

            if sample['has_path']:
                smoothed_path_list.append(sample['smoothed_path'])
                target_points_list.append(sample['target_points'])
                valid_waypoints_list.append(sample['valid_waypoints'])
                path_count += 1
            else:
                no_path_count += 1
        else:
            skip_count += 1

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(indices)}, path={path_count}, no_path={no_path_count}, skipped={skip_count}")

    # Save as numpy arrays
    lidar_data = np.array(lidar_data_list)
    target_angles = np.array(target_angles_list)  # shape: (N, num_lookahead_distances)
    has_path = np.array(has_path_list, dtype=np.bool_)
    poses = np.array(pose_list)
    lookahead_dists_array = np.array(lookahead_distances, dtype=np.float32)

    np.save(output_dir / "lidar_data.npy", lidar_data)
    np.save(output_dir / "target_angles.npy", target_angles)
    np.save(output_dir / "has_path.npy", has_path)
    np.save(output_dir / "poses.npy", poses)
    np.save(output_dir / "lookahead_distances.npy", lookahead_dists_array)

    # Debug info (only for samples with valid path)
    if len(smoothed_path_list) > 0:
        smoothed_paths = np.array(smoothed_path_list)
        target_points = np.array(target_points_list)  # shape: (N, num_lookahead_distances, 2)
        np.save(output_dir / "smoothed_paths.npy", smoothed_paths)
        np.save(output_dir / "target_points.npy", target_points)
        np.save(output_dir / "valid_waypoints.npy", np.array(valid_waypoints_list, dtype=object))

    total_samples = path_count + no_path_count
    print(f"\nSaved {total_samples} samples to {output_dir}")
    print(f"  lidar_data.npy: shape={lidar_data.shape}")
    print(f"  target_angles.npy: shape={target_angles.shape}")
    print(f"  lookahead_distances.npy: {lookahead_distances} m")
    print(f"  has_path.npy: {path_count} with path, {no_path_count} without path")
    if path_count > 0:
        valid_angles = target_angles[has_path]
        for i, dist in enumerate(lookahead_distances):
            angles_i = valid_angles[:, i]
            print(f"  angle range (dist={dist}m): [{np.degrees(angles_i.min()):.1f}, {np.degrees(angles_i.max()):.1f}] degrees")
    print(f"  [debug] poses.npy: shape={poses.shape}")
    if len(smoothed_path_list) > 0:
        print(f"  [debug] smoothed_paths.npy: shape={smoothed_paths.shape}")
        print(f"  [debug] target_points.npy: shape={target_points.shape}")
        waypoint_counts = [len(wp) for wp in valid_waypoints_list]
        print(f"  [debug] valid_waypoints.npy: {len(valid_waypoints_list)} samples, {min(waypoint_counts)}-{max(waypoint_counts)} waypoints each")


if __name__ == "__main__":
    main()

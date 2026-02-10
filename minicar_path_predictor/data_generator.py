"""Generate training data from course image."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev

from .utils import (
    load_course_image,
    extract_walls_mask,
    extract_path_mask,
    get_drivable_area_mask,
    pixel_to_world,
    world_to_pixel,
    world_to_robot,
    normalize_angle,
    apply_blind_spot_mask,
)


@dataclass
class LiDARConfig:
    """LiDAR sensor configuration."""
    num_rays: int = 360
    angle_min: float = 0.0
    angle_max: float = 2 * math.pi
    range_min: float = 0.15
    range_max: float = 12.0  # 12m range
    noise_std: float = 0.005
    blind_spot_start: float = 115.0  # degrees
    blind_spot_end: float = 246.0  # degrees


@dataclass
class PoseCandidate:
    """A candidate robot pose for training data."""
    x: float  # World X position (meters)
    y: float  # World Y position (meters)
    yaw: float  # Orientation (radians)
    path_index: int  # Index on ideal path (for reference)
    forward_yaw: float  # Forward direction at this position (radians)


@dataclass
class TrainingSample:
    """A single training sample."""
    robot_x: float  # World position
    robot_y: float
    robot_yaw: float
    lidar_ranges: np.ndarray  # Shape: (num_rays,)
    target_waypoints: np.ndarray  # Shape: (num_points, 2) in robot frame


class CourseDataGenerator:
    """Generate training data from course image."""

    def __init__(
        self,
        course_image_path: Path,
        resolution: float = 0.02,
        lidar_config: Optional[LiDARConfig] = None,
    ):
        self.resolution = resolution
        self.lidar_config = lidar_config or LiDARConfig()

        # Load and process course image
        self.image = load_course_image(course_image_path)
        self.img_shape = self.image.shape[:2]
        self.H, self.W = self.img_shape

        # Extract masks
        self.walls_mask = extract_walls_mask(self.image)
        self.path_mask = extract_path_mask(self.image)
        self.drivable_mask = get_drivable_area_mask(self.walls_mask)

        # Extract and order ideal path points (ensure clockwise)
        self.ideal_path_pixels = self._extract_ordered_path()
        self.ideal_path_world = self._convert_path_to_world()
        self._ensure_clockwise()

        # Precompute forward directions at each path point
        self.path_forward_yaw = self._compute_path_directions()

        # Create distance transform for wall detection
        self.wall_distance = distance_transform_edt(self.drivable_mask)

        # Create track-only mask (flood fill from path to get actual course area)
        self.track_mask = self._compute_track_mask()

        # Create combined distance field for waypoint generation
        # High values = close to ideal path AND far from walls
        self.path_distance_field = self._create_path_distance_field()

        # Precompute LiDAR angles
        self.lidar_angles = np.linspace(
            self.lidar_config.angle_min,
            self.lidar_config.angle_max,
            self.lidar_config.num_rays,
            endpoint=False
        )

    def _extract_ordered_path(self) -> np.ndarray:
        """Extract and order ideal path points."""
        # Find contour of red path
        contours, _ = cv2.findContours(
            self.path_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            raise ValueError("No path found in image")

        # Get the longest contour
        longest = max(contours, key=len)
        points = longest.reshape(-1, 2).astype(np.float64)

        # Smooth and resample the path using spline
        try:
            # Fit spline to path
            tck, u = splprep([points[:, 0], points[:, 1]], s=100, per=True)

            # Resample at uniform intervals
            u_new = np.linspace(0, 1, 1000)
            x_new, y_new = splev(u_new, tck)
            points = np.column_stack([x_new, y_new])
        except Exception:
            # Fall back to original points if spline fails
            pass

        return points

    def _convert_path_to_world(self) -> np.ndarray:
        """Convert path pixels to world coordinates."""
        world_points = []
        for px, py in self.ideal_path_pixels:
            x, y = pixel_to_world(px, py, self.img_shape, self.resolution)
            world_points.append([x, y])
        return np.array(world_points)

    def _ensure_clockwise(self):
        """Ensure the path is ordered clockwise (reverse if needed)."""
        # Calculate signed area to determine winding direction
        # Positive = counter-clockwise, Negative = clockwise
        path = self.ideal_path_world
        n = len(path)
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += path[i, 0] * path[j, 1]
            signed_area -= path[j, 0] * path[i, 1]
        signed_area /= 2.0

        # If counter-clockwise (positive), reverse to make clockwise
        if signed_area > 0:
            self.ideal_path_pixels = self.ideal_path_pixels[::-1].copy()
            self.ideal_path_world = self.ideal_path_world[::-1].copy()

    def _compute_path_directions(self) -> np.ndarray:
        """Compute forward direction (yaw) at each path point."""
        path = self.ideal_path_world
        n = len(path)
        directions = np.zeros(n)

        for i in range(n):
            # Use next point to compute forward direction
            j = (i + 1) % n
            dx = path[j, 0] - path[i, 0]
            dy = path[j, 1] - path[i, 1]
            directions[i] = math.atan2(dy, dx)

        return directions

    def _compute_track_mask(self) -> np.ndarray:
        """Compute mask for track area only (flood fill from ideal path)."""
        # Use a point on the ideal path as seed
        seed_x, seed_y = self.ideal_path_world[0]
        seed_px, seed_py = world_to_pixel(seed_x, seed_y, self.img_shape, self.resolution)

        # Flood fill from seed
        flood_mask = np.zeros((self.H + 2, self.W + 2), dtype=np.uint8)
        drivable_copy = self.drivable_mask.copy()
        cv2.floodFill(drivable_copy, flood_mask, (seed_px, seed_py), 128)

        # Extract filled region as track mask
        track_mask = (drivable_copy == 128).astype(np.uint8) * 255

        return track_mask

    def _create_path_distance_field(self) -> np.ndarray:
        """Create distance-to-path field for waypoint generation.

        Creates a field where:
        - The ideal path has value 0
        - Values increase with distance from the ideal path

        This field is used by find_target_waypoints_distance_field() which
        inverts it to create an attraction field toward the ideal path.

        Returns:
            Distance field with shape (H, W) - distance in pixels from ideal path
        """
        # Create binary mask where ideal path = 0, everywhere else = 1
        path_binary = np.ones((self.H, self.W), dtype=np.uint8)
        for px, py in self.ideal_path_pixels:
            px_int, py_int = int(round(px)), int(round(py))
            if 0 <= px_int < self.W and 0 <= py_int < self.H:
                path_binary[py_int, px_int] = 0
        # Dilate the path slightly to make it easier to hit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        path_dilated = cv2.erode(path_binary, kernel)  # erode 1s = dilate 0s

        # Distance from ideal path (0 on path, increasing away from it)
        dist_to_path = distance_transform_edt(path_dilated)

        return dist_to_path.astype(np.float32)

    def generate_pose_candidates(
        self,
        path_spacing: float = 0.1,
        lateral_offsets: List[float] = None,
        yaw_offsets: List[float] = None,
    ) -> List[PoseCandidate]:
        """Generate robot pose candidates along the ideal path.

        Args:
            path_spacing: Distance between samples along path (meters)
            lateral_offsets: List of lateral offsets from path centerline (meters)
                           Positive = left, Negative = right
                           Default: [0.0] (centerline only)
            yaw_offsets: List of yaw offsets from forward direction (radians)
                        Range should be within [-pi/2, pi/2]
                        Default: [-pi/2, -pi/4, 0, pi/4, pi/2]

        Returns:
            List of PoseCandidate objects
        """
        if lateral_offsets is None:
            lateral_offsets = [0.0]

        if yaw_offsets is None:
            # Default: ±90° in 5 steps
            yaw_offsets = [
                -math.pi / 2,
                -math.pi / 4,
                0.0,
                math.pi / 4,
                math.pi / 2,
            ]

        candidates = []
        path = self.ideal_path_world
        n = len(path)

        # Calculate cumulative distance along path
        cum_dist = np.zeros(n)
        for i in range(1, n):
            cum_dist[i] = cum_dist[i - 1] + np.linalg.norm(path[i] - path[i - 1])
        total_length = cum_dist[-1] + np.linalg.norm(path[0] - path[-1])

        # Sample positions along path at regular intervals
        sample_distances = np.arange(0, total_length, path_spacing)

        for dist in sample_distances:
            # Find path index for this distance
            idx = np.searchsorted(cum_dist, dist % cum_dist[-1])
            if idx >= n:
                idx = n - 1

            # Get position and forward direction
            base_x, base_y = path[idx]
            forward_yaw = self.path_forward_yaw[idx]

            # Compute perpendicular direction (left is +90 degrees from forward)
            perp_yaw = forward_yaw + math.pi / 2

            for lat_offset in lateral_offsets:
                # Apply lateral offset
                x = base_x + lat_offset * math.cos(perp_yaw)
                y = base_y + lat_offset * math.sin(perp_yaw)

                # Check if position is valid (inside drivable area)
                px, py = world_to_pixel(x, y, self.img_shape, self.resolution)
                if not (0 <= px < self.W and 0 <= py < self.H):
                    continue
                if self.wall_distance[py, px] < 0.1 / self.resolution:
                    continue

                for yaw_offset in yaw_offsets:
                    yaw = normalize_angle(forward_yaw + yaw_offset)

                    candidates.append(PoseCandidate(
                        x=x,
                        y=y,
                        yaw=yaw,
                        path_index=idx,
                        forward_yaw=forward_yaw,
                    ))

        return candidates

    def find_closest_path_index(self, x: float, y: float) -> int:
        """Find the closest point index on the ideal path."""
        pos = np.array([x, y])
        distances = np.linalg.norm(self.ideal_path_world - pos, axis=1)
        return int(np.argmin(distances))

    def generate_pose_candidates_grid(
        self,
        grid_spacing: float = 0.05,
        min_wall_distance: float = 0.1,
        yaw_offsets: List[float] = None,
        track_only: bool = True,
    ) -> List[PoseCandidate]:
        """Generate robot pose candidates from drivable area.

        Samples positions on a grid within the drivable area, and for each
        position determines the forward direction from the closest point
        on the ideal path.

        Args:
            grid_spacing: Grid spacing for position sampling (meters)
            min_wall_distance: Minimum distance from walls (meters)
            yaw_offsets: List of yaw offsets from forward direction (radians)
                        Range should be within [-pi/2, pi/2]
                        Default: -90° to +90° in 10° steps (19 orientations)
            track_only: If True, only sample from track area (inside course).
                       If False, sample from all drivable area.

        Returns:
            List of PoseCandidate objects
        """
        if yaw_offsets is None:
            # Default: -90° to +90° in 10° steps (19 orientations)
            yaw_offsets = [math.radians(deg) for deg in range(-90, 91, 10)]

        min_dist_px = min_wall_distance / self.resolution
        candidates = []

        # Choose which mask to use
        area_mask = self.track_mask if track_only else self.drivable_mask

        # Grid sampling in world coordinates
        x_min = -self.W / 2 * self.resolution
        x_max = self.W / 2 * self.resolution
        y_min = -self.H / 2 * self.resolution
        y_max = self.H / 2 * self.resolution

        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                px, py = world_to_pixel(x, y, self.img_shape, self.resolution)

                # Check bounds and area mask
                if 0 <= px < self.W and 0 <= py < self.H:
                    if area_mask[py, px] > 0 and self.wall_distance[py, px] >= min_dist_px:
                        # Find closest path point for forward direction
                        path_idx = self.find_closest_path_index(x, y)
                        forward_yaw = self.path_forward_yaw[path_idx]

                        # Generate poses with yaw offsets
                        for yaw_offset in yaw_offsets:
                            yaw = normalize_angle(forward_yaw + yaw_offset)
                            candidates.append(PoseCandidate(
                                x=x,
                                y=y,
                                yaw=yaw,
                                path_index=path_idx,
                                forward_yaw=forward_yaw,
                            ))

                y += grid_spacing
            x += grid_spacing

        return candidates

    def simulate_lidar(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        add_noise: bool = True
    ) -> np.ndarray:
        """Simulate LiDAR scan from given pose.

        Args:
            robot_x: Robot X position in world frame (meters)
            robot_y: Robot Y position in world frame (meters)
            robot_yaw: Robot orientation (radians)
            add_noise: Whether to add Gaussian noise

        Returns:
            Array of range measurements (meters)
        """
        ranges = np.full(self.lidar_config.num_rays, self.lidar_config.range_max)

        # Convert robot position to pixel
        robot_px, robot_py = world_to_pixel(
            robot_x, robot_y, self.img_shape, self.resolution
        )

        for i, angle in enumerate(self.lidar_angles):
            # World angle = robot_yaw + local_angle
            world_angle = robot_yaw + angle

            # Ray march to find intersection
            range_val = self._ray_march(
                robot_px, robot_py, world_angle,
                self.lidar_config.range_max / self.resolution
            )

            # Convert to meters
            ranges[i] = range_val * self.resolution

        # Clamp to valid range
        ranges = np.clip(ranges, self.lidar_config.range_min, self.lidar_config.range_max)

        # Add noise
        if add_noise:
            noise = np.random.normal(0, self.lidar_config.noise_std, ranges.shape)
            ranges = ranges + noise
            ranges = np.clip(ranges, self.lidar_config.range_min, self.lidar_config.range_max)

        # Apply blind spot mask
        ranges = apply_blind_spot_mask(
            ranges,
            self.lidar_angles,
            self.lidar_config.blind_spot_start,
            self.lidar_config.blind_spot_end,
            self.lidar_config.range_max
        )

        return ranges.astype(np.float32)

    def _ray_march(
        self,
        start_px: int,
        start_py: int,
        angle: float,
        max_dist_px: float
    ) -> float:
        """Ray march to find wall intersection."""
        dx = math.cos(angle)
        dy = -math.sin(angle)  # Flip Y for image coordinates

        # Use Bresenham-like stepping
        step_size = 0.5  # pixels
        dist = 0.0

        while dist < max_dist_px:
            px = int(start_px + dx * dist)
            py = int(start_py + dy * dist)

            # Check bounds
            if px < 0 or px >= self.W or py < 0 or py >= self.H:
                return dist

            # Check wall collision
            if self.walls_mask[py, px] > 0:
                return dist

            dist += step_size

        return max_dist_px

    def find_target_waypoints(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        num_points: int = 5,
        lookahead_dist: float = 0.5
    ) -> np.ndarray:
        """Find target waypoints on ideal path from robot pose.

        Returns waypoints in robot frame.
        """
        # Find closest point on ideal path
        robot_pos = np.array([robot_x, robot_y])
        distances = np.linalg.norm(self.ideal_path_world - robot_pos, axis=1)
        closest_idx = np.argmin(distances)

        # Determine direction along path (forward vs backward)
        # Use robot orientation to pick the right direction
        next_idx = (closest_idx + 1) % len(self.ideal_path_world)
        prev_idx = (closest_idx - 1) % len(self.ideal_path_world)

        to_next = self.ideal_path_world[next_idx] - self.ideal_path_world[closest_idx]
        to_prev = self.ideal_path_world[prev_idx] - self.ideal_path_world[closest_idx]

        robot_forward = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])

        # Choose direction that aligns with robot orientation
        if np.dot(to_next, robot_forward) > np.dot(to_prev, robot_forward):
            direction = 1
        else:
            direction = -1

        # Collect waypoints at fixed distances ahead
        waypoints_world = []
        current_idx = closest_idx
        accumulated_dist = 0.0
        target_dists = np.linspace(lookahead_dist, lookahead_dist * num_points, num_points)
        target_idx = 0

        for _ in range(len(self.ideal_path_world)):
            next_idx = (current_idx + direction) % len(self.ideal_path_world)
            segment_dist = np.linalg.norm(
                self.ideal_path_world[next_idx] - self.ideal_path_world[current_idx]
            )
            accumulated_dist += segment_dist
            current_idx = next_idx

            while target_idx < len(target_dists) and accumulated_dist >= target_dists[target_idx]:
                waypoints_world.append(self.ideal_path_world[current_idx].copy())
                target_idx += 1

            if target_idx >= len(target_dists):
                break

        # Fill remaining if not enough points
        while len(waypoints_world) < num_points:
            waypoints_world.append(self.ideal_path_world[current_idx].copy())

        # Convert to robot frame
        waypoints_robot = []
        for wp in waypoints_world:
            local_x, local_y = world_to_robot(
                wp[0], wp[1], robot_x, robot_y, robot_yaw
            )
            waypoints_robot.append([local_x, local_y])

        return np.array(waypoints_robot, dtype=np.float32)

    def find_target_waypoints_distance_field(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        local_size: int = 200
    ) -> np.ndarray:
        """Find target waypoints using PathPipeline logic with path attraction field.

        Uses the exact same 3-stage logic as minicar_navigation's PathPipeline:
        Stage 1: peaks = peak_detector.detect_from_view_points()
        Stage 2: rough_paths = path_searcher.build_and_deduplicate_paths()
        Stage 3: smoothed_paths = smoother.smooth_paths()

        The only difference is the distance field:
        - Original: wall distance (distance from polygon boundary)
        - This: path_attraction = max(wall_dist_max - dist_to_path, 0)

        Args:
            robot_x: Robot X position in world frame (meters)
            robot_y: Robot Y position in world frame (meters)
            robot_yaw: Robot orientation (radians)
            local_size: Size of local view in pixels

        Returns:
            Waypoints in robot frame, shape (N, 2) in meters
        """
        # Import PathPipeline components
        import sys
        sys.path.insert(0, "/home/ubuntu/ros2_ws/src/minicar_navigation")
        from minicar_navigation.planner.local_path_planner import (
            PathPlannerConfig, DistanceFieldParams,
            DistanceFieldGenerator, PeakDetector, GraphPathSearcher, PotentialPathSmoother
        )

        center_x, center_y = local_size // 2, local_size // 2
        scale = 1.0 / self.resolution
        R_in = np.array([25, 50, 75, 100, 125, 150], dtype=np.int32)
        dist_thresh = 15.0

        # 1. Simulate LiDAR -> view_points (same as comparison scripts)
        lidar_ranges = self.simulate_lidar(robot_x, robot_y, robot_yaw, add_noise=False)
        angles = np.linspace(0, 2 * np.pi, self.lidar_config.num_rays, endpoint=False)

        view_points = []
        for r, a in zip(lidar_ranges, angles):
            if r < self.lidar_config.range_max - 0.1:
                x_robot = r * np.cos(a)
                y_robot = r * np.sin(a)
                px = center_x + x_robot * scale
                py = center_y - y_robot * scale
                view_points.append([px, py])
        view_points = np.array(view_points, dtype=np.int32)

        if len(view_points) < 3:
            return self._fallback_waypoints()

        # 2. Create path_attraction field (replacing dist_inside)
        yy, xx = np.mgrid[:local_size, :local_size]
        dx_img = xx - center_x
        dy_img = yy - center_y

        x_robot_m = dx_img * self.resolution
        y_robot_m = -dy_img * self.resolution
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        x_world = robot_x + x_robot_m * cos_yaw - y_robot_m * sin_yaw
        y_world = robot_y + x_robot_m * sin_yaw + y_robot_m * cos_yaw

        src_x = (x_world / self.resolution + self.W / 2).astype(np.float32)
        src_y = (self.H / 2 - y_world / self.resolution).astype(np.float32)

        local_dist_to_path = cv2.remap(
            self.path_distance_field, src_x, src_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1000
        )

        # Get mask from original distance field
        config = PathPlannerConfig(HEIGHT=local_size, WIDTH=local_size)
        dist_field_gen = DistanceFieldGenerator(config)
        df_params = DistanceFieldParams(
            H=local_size, W=local_size, R_in=R_in,
            ring_thickness=1, front_deg=90.0, dist_thresh=dist_thresh, border_thickness=1
        )
        mask, dist_inside, _, cx, cy, _ = dist_field_gen.generate_distance_field(view_points, df_params)

        # path_attraction = max(wall_dist_max - dist_to_path, 0)
        wall_dist_max = dist_inside.max()
        path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)
        path_attraction_masked = path_attraction * (mask > 0).astype(np.float32)

        # Normalize for peak detection
        if path_attraction_masked.max() > 0:
            path_attraction_norm = (path_attraction_masked / path_attraction_masked.max() * 255).astype(np.uint8)
        else:
            return self._fallback_waypoints()

        # Stage 1: Peak detection (same as PeakDetector._detect_path_candidates)
        _, _, delta_x, delta_y, radius_squared = dist_field_gen._compute_center_grid(local_size, local_size)
        ring_mask = dist_field_gen._create_ring_mask(radius_squared, R_in, thickness=1)
        front_mask = dist_field_gen._create_front_mask(delta_x, delta_y, 90.0)
        path_candidate_pixels = ring_mask & front_mask & (path_attraction_masked >= dist_thresh)

        num_labels, labels, _ = dist_field_gen.connect_regions_and_label(path_candidate_pixels, kernel_sz=3)

        peak_detector = PeakDetector(config=config)
        peaks, centered_points, sorted_labels = peak_detector._detect_path_candidates(
            labels, path_attraction_norm, center_x, center_y
        )

        if len(centered_points) == 0:
            return self._fallback_waypoints()

        # Stage 2: Path building (same as GraphPathSearcher.build_and_deduplicate_paths)
        path_searcher = GraphPathSearcher(config)
        rough_paths = path_searcher.build_and_deduplicate_paths(
            labels=sorted_labels,
            centered_points=centered_points,
            dist_inside=path_attraction_masked,
            center_x=center_x,
            center_y=center_y,
            dist_thresh=dist_thresh,
            path_front_deg=90.0,
            start_ids=(0, 1, 2)
        )

        if len(rough_paths) == 0:
            return self._fallback_waypoints()

        # Stage 3: Path smoothing (same as PotentialPathSmoother.smooth_paths)
        smoother = PotentialPathSmoother(config)
        smoothed_paths = smoother.smooth_paths(rough_paths, path_attraction_masked, local_size, local_size)

        if len(smoothed_paths) == 0:
            return self._fallback_waypoints()

        # Select best path and convert centered coords (pixels) to robot frame (meters)
        best_path = smoothed_paths[0]
        waypoints_robot = np.zeros_like(best_path, dtype=np.float32)
        waypoints_robot[:, 0] = best_path[:, 0] * self.resolution   # x: pixel -> meters
        waypoints_robot[:, 1] = -best_path[:, 1] * self.resolution  # y: pixel -> meters (flip Y)

        return waypoints_robot

    def _fallback_waypoints(self) -> np.ndarray:
        """Generate fallback waypoints when path planning fails."""
        spacing = 0.5  # meters
        waypoints = [[spacing * (i + 1), 0.0] for i in range(6)]
        return np.array(waypoints, dtype=np.float32)

    def get_valid_positions(
        self,
        grid_spacing: float = 0.05,
        min_wall_distance: float = 0.1
    ) -> List[Tuple[float, float]]:
        """Get valid robot positions inside drivable area."""
        positions = []
        min_dist_px = min_wall_distance / self.resolution

        # Grid sampling in world coordinates
        x_min = -self.W / 2 * self.resolution
        x_max = self.W / 2 * self.resolution
        y_min = -self.H / 2 * self.resolution
        y_max = self.H / 2 * self.resolution

        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                px, py = world_to_pixel(x, y, self.img_shape, self.resolution)

                # Check bounds
                if 0 <= px < self.W and 0 <= py < self.H:
                    # Check if in drivable area with margin from walls
                    if self.wall_distance[py, px] >= min_dist_px:
                        positions.append((x, y))

                y += grid_spacing
            x += grid_spacing

        return positions

    def generate_samples(
        self,
        grid_spacing: float = 0.05,
        num_orientations: int = 36,
        num_waypoints: int = 5,
        lookahead_dist: float = 0.5,
        min_wall_distance: float = 0.1,
        add_noise: bool = True,
        progress_callback=None,
        use_gradient_waypoints: bool = True
    ) -> List[TrainingSample]:
        """Generate training samples.

        Args:
            grid_spacing: Spacing between sample positions (meters)
            num_orientations: Number of orientations per position
            num_waypoints: Number of target waypoints
            lookahead_dist: Distance to first waypoint (meters)
            min_wall_distance: Minimum distance from walls (meters)
            add_noise: Add sensor noise
            progress_callback: Callback for progress updates
            use_gradient_waypoints: Use gradient-based collision-free waypoint generation

        Returns:
            List of TrainingSample objects
        """
        positions = self.get_valid_positions(grid_spacing, min_wall_distance)
        orientations = np.linspace(0, 2 * math.pi, num_orientations, endpoint=False)

        samples = []
        total = len(positions) * len(orientations)
        count = 0

        for x, y in positions:
            for yaw in orientations:
                # Simulate LiDAR
                lidar_ranges = self.simulate_lidar(x, y, yaw, add_noise)

                # Find target waypoints using distance field method
                try:
                    if use_gradient_waypoints:
                        waypoints = self.find_target_waypoints_distance_field(
                            x, y, yaw, num_waypoints,
                            waypoint_spacing=lookahead_dist
                        )
                    else:
                        waypoints = self.find_target_waypoints(
                            x, y, yaw, num_waypoints, lookahead_dist
                        )
                except Exception:
                    continue

                sample = TrainingSample(
                    robot_x=x,
                    robot_y=y,
                    robot_yaw=yaw,
                    lidar_ranges=lidar_ranges,
                    target_waypoints=waypoints
                )
                samples.append(sample)

                count += 1
                if progress_callback and count % 1000 == 0:
                    progress_callback(count, total)

        return samples

    def save_samples(self, samples: List[TrainingSample], output_path: Path):
        """Save samples to NPZ file."""
        lidar_data = np.array([s.lidar_ranges for s in samples])
        waypoints_data = np.array([s.target_waypoints for s in samples])
        poses_data = np.array([[s.robot_x, s.robot_y, s.robot_yaw] for s in samples])

        np.savez_compressed(
            output_path,
            lidar=lidar_data,
            waypoints=waypoints_data,
            poses=poses_data,
            resolution=self.resolution,
            lidar_config={
                'num_rays': self.lidar_config.num_rays,
                'range_min': self.lidar_config.range_min,
                'range_max': self.lidar_config.range_max,
            }
        )
        print(f"Saved {len(samples)} samples to {output_path}")

    @staticmethod
    def load_samples(input_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load samples from NPZ file."""
        data = np.load(input_path, allow_pickle=True)
        return data['lidar'], data['waypoints'], data['poses']

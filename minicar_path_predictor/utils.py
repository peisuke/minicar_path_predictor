"""Utility functions for path predictor."""

import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_course_image(image_path: Path) -> np.ndarray:
    """Load course image as BGR."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img


def extract_walls_mask(img: np.ndarray, threshold: int = 50) -> np.ndarray:
    """Extract black walls as binary mask (walls = 255)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def extract_path_mask(img: np.ndarray) -> np.ndarray:
    """Extract red ideal path as binary mask (path = 255)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color ranges in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    return mask1 | mask2


def extract_path_centerline(path_mask: np.ndarray) -> np.ndarray:
    """Extract ordered centerline points from path mask."""
    # Skeletonize to get centerline
    skeleton = cv2.ximgproc.thinning(path_mask) if hasattr(cv2, 'ximgproc') else path_mask

    # Find contours
    contours, _ = cv2.findContours(
        path_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return np.array([])

    # Get the longest contour
    longest = max(contours, key=len)
    points = longest.reshape(-1, 2)

    return points


def get_drivable_area_mask(walls_mask: np.ndarray) -> np.ndarray:
    """Get drivable area (inverse of walls)."""
    return cv2.bitwise_not(walls_mask)


def pixel_to_world(
    px: float, py: float,
    img_shape: Tuple[int, int],
    resolution: float
) -> Tuple[float, float]:
    """Convert pixel coordinates to world coordinates (meters).

    Origin is at image center, X right, Y up.
    """
    H, W = img_shape[:2]
    cx, cy = W / 2.0, H / 2.0
    x = (px - cx) * resolution
    y = -(py - cy) * resolution  # Flip Y axis
    return x, y


def world_to_pixel(
    x: float, y: float,
    img_shape: Tuple[int, int],
    resolution: float
) -> Tuple[int, int]:
    """Convert world coordinates (meters) to pixel coordinates."""
    H, W = img_shape[:2]
    cx, cy = W / 2.0, H / 2.0
    px = int(x / resolution + cx)
    py = int(-y / resolution + cy)
    return px, py


def robot_to_world(
    local_x: float, local_y: float,
    robot_x: float, robot_y: float,
    robot_yaw: float
) -> Tuple[float, float]:
    """Transform from robot frame to world frame."""
    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)

    world_x = robot_x + local_x * cos_yaw - local_y * sin_yaw
    world_y = robot_y + local_x * sin_yaw + local_y * cos_yaw

    return world_x, world_y


def world_to_robot(
    world_x: float, world_y: float,
    robot_x: float, robot_y: float,
    robot_yaw: float
) -> Tuple[float, float]:
    """Transform from world frame to robot frame."""
    dx = world_x - robot_x
    dy = world_y - robot_y

    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)

    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw

    return local_x, local_y


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def apply_blind_spot_mask(
    ranges: np.ndarray,
    angles: np.ndarray,
    blind_start_deg: float,
    blind_end_deg: float,
    max_range: float
) -> np.ndarray:
    """Apply blind spot mask to LiDAR ranges."""
    angles_deg = np.rad2deg(angles) % 360

    # Create mask for blind spot
    if blind_start_deg < blind_end_deg:
        blind_mask = (angles_deg >= blind_start_deg) & (angles_deg <= blind_end_deg)
    else:
        blind_mask = (angles_deg >= blind_start_deg) | (angles_deg <= blind_end_deg)

    # Set blind spot ranges to max_range (or inf)
    masked_ranges = ranges.copy()
    masked_ranges[blind_mask] = max_range

    return masked_ranges

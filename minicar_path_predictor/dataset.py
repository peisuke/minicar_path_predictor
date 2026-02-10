"""PyTorch dataset for path prediction."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class PathPredictionDataset(Dataset):
    """Dataset for LiDAR to path prediction."""

    def __init__(
        self,
        lidar_data: np.ndarray,
        waypoints_data: np.ndarray,
        normalize_lidar: bool = True,
        max_range: float = 8.0
    ):
        """Initialize dataset.

        Args:
            lidar_data: LiDAR ranges, shape (N, num_rays)
            waypoints_data: Target waypoints, shape (N, num_points, 2)
            normalize_lidar: Whether to normalize LiDAR data to [0, 1]
            max_range: Maximum LiDAR range for normalization
        """
        self.lidar_data = lidar_data.astype(np.float32)
        self.waypoints_data = waypoints_data.astype(np.float32)
        self.normalize_lidar = normalize_lidar
        self.max_range = max_range

        # Flatten waypoints for output
        self.waypoints_flat = self.waypoints_data.reshape(len(self.waypoints_data), -1)

        # Compute normalization statistics
        if normalize_lidar:
            self.lidar_data = self.lidar_data / max_range

        # Compute waypoint statistics for normalization
        self.waypoint_mean = np.mean(self.waypoints_flat, axis=0)
        self.waypoint_std = np.std(self.waypoints_flat, axis=0) + 1e-6

    def __len__(self) -> int:
        return len(self.lidar_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lidar = torch.from_numpy(self.lidar_data[idx])
        waypoints = torch.from_numpy(self.waypoints_flat[idx])
        return lidar, waypoints

    @classmethod
    def from_npz(
        cls,
        path: Path,
        normalize_lidar: bool = True
    ) -> "PathPredictionDataset":
        """Load dataset from NPZ file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            data['lidar'],
            data['waypoints'],
            normalize_lidar=normalize_lidar,
            max_range=data['lidar_config'].item()['range_max']
        )


def create_data_loaders(
    dataset: PathPredictionDataset,
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders.

    Args:
        dataset: Full dataset
        batch_size: Batch size
        train_split: Fraction for training
        val_split: Fraction for validation
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


class AugmentedPathDataset(Dataset):
    """Dataset with data augmentation."""

    def __init__(
        self,
        base_dataset: PathPredictionDataset,
        noise_std: float = 0.01,
        dropout_prob: float = 0.05
    ):
        """Initialize augmented dataset.

        Args:
            base_dataset: Base dataset
            noise_std: Standard deviation of Gaussian noise to add
            dropout_prob: Probability of dropping each LiDAR ray
        """
        self.base_dataset = base_dataset
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lidar, waypoints = self.base_dataset[idx]

        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(lidar) * self.noise_std
            lidar = lidar + noise
            lidar = torch.clamp(lidar, 0, 1)

        # Random dropout (simulate sensor failures)
        if self.dropout_prob > 0:
            dropout_mask = torch.rand_like(lidar) > self.dropout_prob
            lidar = lidar * dropout_mask + (1.0 * ~dropout_mask)  # Set dropped to max range

        return lidar, waypoints

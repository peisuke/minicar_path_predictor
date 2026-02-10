#!/usr/bin/env python3
"""Run inference with trained model.

Usage:
    python3 inference.py
    python3 inference.py --model weights/best_model.pth --data data/sim_robot/training_data.npz --visualize
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minicar_path_predictor.model import create_model
from minicar_path_predictor.dataset import PathPredictionDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with trained model"
    )
    parser.add_argument(
        "--model", "-m",
        type=Path,
        default=Path("weights/best_model.pth"),
        help="Path to trained model (default: weights/best_model.pth)"
    )
    parser.add_argument(
        "--model-info",
        type=Path,
        help="Path to model_info.json (default: weights/model_info.json)"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path("data/sim_robot/training_data.npz"),
        help="Test data NPZ file (default: data/sim_robot/training_data.npz)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda, cpu, or auto"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize predictions"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for visualizations"
    )
    return parser.parse_args()


class PathPredictor:
    """Inference wrapper for path prediction model."""

    def __init__(
        self,
        model_path: Path,
        model_info_path: Path = None,
        device: str = "auto"
    ):
        """Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            model_info_path: Path to model_info.json
            device: Device to use
        """
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model info
        if model_info_path is None:
            model_info_path = model_path.parent / "model_info.json"

        if not model_info_path.exists():
            raise FileNotFoundError(f"Model info not found: {model_info_path}")

        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)

        # Create model
        self.model = create_model(
            model_type=self.model_info['model_type'],
            input_dim=self.model_info['input_dim'],
            output_dim=self.model_info['output_dim'],
            hidden_dims=self.model_info.get('hidden_dims', [256, 128, 64])
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.num_waypoints = self.model_info['num_waypoints']
        self.max_range = 8.0  # LiDAR max range for normalization

    @torch.no_grad()
    def predict(self, lidar_ranges: np.ndarray) -> np.ndarray:
        """Predict waypoints from LiDAR scan.

        Args:
            lidar_ranges: LiDAR ranges, shape (num_rays,) or (batch, num_rays)

        Returns:
            Predicted waypoints in robot frame, shape (num_waypoints, 2) or (batch, num_waypoints, 2)
        """
        # Handle single sample
        single = lidar_ranges.ndim == 1
        if single:
            lidar_ranges = lidar_ranges[np.newaxis, :]

        # Normalize
        lidar_normalized = lidar_ranges / self.max_range
        lidar_tensor = torch.from_numpy(lidar_normalized.astype(np.float32)).to(self.device)

        # Predict
        output = self.model(lidar_tensor)
        waypoints = output.cpu().numpy()

        # Reshape
        waypoints = waypoints.reshape(-1, self.num_waypoints, 2)

        if single:
            return waypoints[0]
        return waypoints

    def predict_single(self, lidar_ranges: np.ndarray) -> np.ndarray:
        """Predict waypoints for a single LiDAR scan."""
        return self.predict(lidar_ranges)


def visualize_predictions(
    predictor: PathPredictor,
    lidar_data: np.ndarray,
    target_waypoints: np.ndarray,
    output_dir: Path,
    num_samples: int = 10
):
    """Visualize model predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, len(lidar_data))
    indices = np.random.choice(len(lidar_data), num_samples, replace=False)

    for i, idx in enumerate(indices):
        lidar = lidar_data[idx]
        target = target_waypoints[idx].reshape(-1, 2)
        pred = predictor.predict_single(lidar)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # LiDAR visualization
        ax1 = axes[0]
        angles = np.linspace(0, 2 * np.pi, len(lidar), endpoint=False)
        x = lidar * np.cos(angles)
        y = lidar * np.sin(angles)

        ax1.scatter(x, y, c='blue', s=1, alpha=0.5, label='LiDAR')
        ax1.plot([0], [0], 'ko', markersize=10)
        ax1.arrow(0, 0, 0.5, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
        ax1.set_xlim(-8, 8)
        ax1.set_ylim(-8, 8)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title("LiDAR Scan")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")

        # Waypoints comparison
        ax2 = axes[1]
        ax2.plot([0], [0], 'ko', markersize=10, label='Robot')
        ax2.arrow(0, 0, 0.2, 0, head_width=0.05, head_length=0.03, fc='black', ec='black')

        # Ground truth
        ax2.plot(target[:, 0], target[:, 1], 'g.-', markersize=12, linewidth=2, label='Ground Truth')

        # Prediction
        ax2.plot(pred[:, 0], pred[:, 1], 'r.-', markersize=12, linewidth=2, label='Prediction')

        # Error lines
        for t, p in zip(target, pred):
            ax2.plot([t[0], p[0]], [t[1], p[1]], 'k--', alpha=0.3)

        ax2.set_xlim(-2, 3)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Waypoint Prediction")
        ax2.set_xlabel("X (m) - Forward")
        ax2.set_ylabel("Y (m) - Left")
        ax2.legend()

        # Compute error
        error = np.sqrt(np.sum((pred - target) ** 2, axis=1))
        avg_error = error.mean()
        ax2.text(0.02, 0.98, f'Avg Error: {avg_error:.3f} m',
                 transform=ax2.transAxes, va='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        plt.savefig(output_dir / f"prediction_{i+1}.png", dpi=150)
        plt.close()

    print(f"Visualizations saved to {output_dir}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Path Prediction Inference")
    print("=" * 60)

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    model_info_path = args.model_info or (args.model.parent / "model_info.json")

    predictor = PathPredictor(
        model_path=args.model,
        model_info_path=model_info_path,
        device=device
    )

    print(f"  Model type: {predictor.model_info['model_type']}")
    print(f"  Input dim: {predictor.model_info['input_dim']}")
    print(f"  Output dim: {predictor.model_info['output_dim']}")
    print(f"  Num waypoints: {predictor.num_waypoints}")

    # Evaluate on data if provided
    if args.data:
        print(f"\nLoading test data from {args.data}...")
        dataset = PathPredictionDataset.from_npz(args.data, normalize_lidar=False)

        lidar_data = dataset.lidar_data
        waypoints_data = dataset.waypoints_data

        print(f"  Samples: {len(lidar_data)}")

        # Batch prediction
        print("\nRunning inference...")
        predictions = predictor.predict(lidar_data)

        # Compute metrics
        errors = np.sqrt(np.sum((predictions - waypoints_data) ** 2, axis=2))
        ade = errors.mean()
        fde = errors[:, -1].mean()

        print(f"\nResults:")
        print(f"  Average Displacement Error (ADE): {ade:.4f} m")
        print(f"  Final Displacement Error (FDE): {fde:.4f} m")
        print(f"  Per-waypoint errors: {errors.mean(axis=0)}")

        # Visualize if requested
        if args.visualize:
            output_dir = args.output or args.model.parent / "visualizations"
            visualize_predictions(
                predictor,
                lidar_data,
                waypoints_data,
                output_dir,
                args.num_samples
            )

    # Interactive mode demo
    print("\n" + "-" * 60)
    print("Model ready for inference!")
    print("Use PathPredictor.predict(lidar_ranges) for predictions.")
    print("-" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

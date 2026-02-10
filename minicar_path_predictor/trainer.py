"""Training utilities for path predictor."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    save_best: bool = True
    save_every: int = 10


class WaypointLoss(nn.Module):
    """Custom loss for waypoint prediction."""

    def __init__(
        self,
        num_waypoints: int = 5,
        distance_weight: float = 1.0,
        direction_weight: float = 0.5
    ):
        """Initialize loss.

        Args:
            num_waypoints: Number of waypoints
            distance_weight: Weight for distance MSE loss
            direction_weight: Weight for direction consistency loss
        """
        super().__init__()
        self.num_waypoints = num_waypoints
        self.distance_weight = distance_weight
        self.direction_weight = direction_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss.

        Args:
            pred: Predicted waypoints, shape (batch, num_waypoints * 2)
            target: Target waypoints, shape (batch, num_waypoints * 2)

        Returns:
            Total loss and dict of component losses
        """
        batch_size = pred.size(0)

        # Reshape to (batch, num_waypoints, 2)
        pred = pred.view(batch_size, self.num_waypoints, 2)
        target = target.view(batch_size, self.num_waypoints, 2)

        # Position MSE loss
        position_loss = self.mse(pred, target)

        # Direction consistency loss (encourage smooth path)
        if self.num_waypoints > 1:
            pred_dirs = pred[:, 1:] - pred[:, :-1]
            target_dirs = target[:, 1:] - target[:, :-1]
            direction_loss = self.mse(pred_dirs, target_dirs)
        else:
            direction_loss = torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.distance_weight * position_loss +
            self.direction_weight * direction_loss
        )

        losses = {
            'total': total_loss.item(),
            'position': position_loss.item(),
            'direction': direction_loss.item()
        }

        return total_loss, losses


class Trainer:
    """Model trainer."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
        output_dir: Optional[Path] = None
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            device: Device to use
            output_dir: Directory for saving models and logs
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir or Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            min_lr=config.min_lr
        )

        # Infer num_waypoints from model output
        num_waypoints = model.output_dim // 2
        self.criterion = WaypointLoss(num_waypoints=num_waypoints)

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_position_loss = 0.0
        total_direction_loss = 0.0
        num_batches = 0

        for lidar, waypoints in train_loader:
            lidar = lidar.to(self.device)
            waypoints = waypoints.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(lidar)
            loss, losses = self.criterion(pred, waypoints)

            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += losses['total']
            total_position_loss += losses['position']
            total_direction_loss += losses['direction']
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'position_loss': total_position_loss / num_batches,
            'direction_loss': total_direction_loss / num_batches
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_position_loss = 0.0
        total_direction_loss = 0.0
        num_batches = 0

        for lidar, waypoints in val_loader:
            lidar = lidar.to(self.device)
            waypoints = waypoints.to(self.device)

            pred = self.model(lidar)
            loss, losses = self.criterion(pred, waypoints)

            total_loss += losses['total']
            total_position_loss += losses['position']
            total_direction_loss += losses['direction']
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'position_loss': total_position_loss / num_batches,
            'direction_loss': total_direction_loss / num_batches
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Print progress

        Returns:
            Training history
        """
        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)

            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0

                if self.config.save_best:
                    self.save_model('best_model.pth')
            else:
                self.patience_counter += 1

            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} "
                    f"[{epoch_time:.1f}s] - "
                    f"Train: {train_metrics['loss']:.6f} "
                    f"(pos: {train_metrics['position_loss']:.6f}) - "
                    f"Val: {val_metrics['loss']:.6f} "
                    f"(pos: {val_metrics['position_loss']:.6f}) - "
                    f"LR: {current_lr:.2e}"
                )

            # Save periodic checkpoints
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Best validation loss: {self.best_val_loss:.6f}")

        return self.history

    def save_model(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, path)

    def load_model(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    num_waypoints: int = 5
) -> Dict[str, float]:
    """Evaluate model on test set.

    Returns various metrics including:
    - MSE, MAE for positions
    - Average endpoint error
    - Trajectory error
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []

    for lidar, waypoints in test_loader:
        lidar = lidar.to(device)
        pred = model(lidar)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(waypoints.numpy())

    import numpy as np
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Reshape
    all_preds = all_preds.reshape(-1, num_waypoints, 2)
    all_targets = all_targets.reshape(-1, num_waypoints, 2)

    # MSE
    mse = np.mean((all_preds - all_targets) ** 2)

    # MAE
    mae = np.mean(np.abs(all_preds - all_targets))

    # Endpoint error (distance to final waypoint)
    endpoint_error = np.sqrt(
        np.sum((all_preds[:, -1] - all_targets[:, -1]) ** 2, axis=1)
    ).mean()

    # Average displacement error (ADE)
    per_point_distances = np.sqrt(
        np.sum((all_preds - all_targets) ** 2, axis=2)
    )
    ade = per_point_distances.mean()

    # Final displacement error (FDE)
    fde = per_point_distances[:, -1].mean()

    return {
        'mse': float(mse),
        'mae': float(mae),
        'ade': float(ade),  # Average Displacement Error
        'fde': float(fde),  # Final Displacement Error
        'endpoint_error': float(endpoint_error)
    }

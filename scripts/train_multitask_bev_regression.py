#!/usr/bin/env python3
"""Train multi-task model using Bird's Eye View images with angle REGRESSION.

Input: LiDAR (360) -> BEV image
Output: has_path (binary), target_angle (continuous, radians)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import cv2

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_run_name_with_timestamp(base_name: str) -> str:
    """Generate run name with timestamp suffix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_name:
        return f"{base_name}_{timestamp}"
    return f"run_{timestamp}"


def lidar_to_bev(lidar_ranges, image_size=64, max_range=12.0, view_range=3.0, robot_radius=2):
    """Convert LiDAR ranges to Bird's Eye View image."""
    img = np.zeros((image_size, image_size), dtype=np.float32)
    center = image_size // 2
    scale = center / view_range

    angles = np.linspace(0, 2*np.pi, len(lidar_ranges), endpoint=False)
    ranges_m = lidar_ranges * max_range
    valid_mask = ranges_m < view_range

    if valid_mask.sum() > 0:
        valid_ranges = ranges_m[valid_mask]
        valid_angles = angles[valid_mask]

        x_robot = valid_ranges * np.cos(valid_angles)
        y_robot = valid_ranges * np.sin(valid_angles)

        px = (center - y_robot * scale).astype(np.int32)
        py = (center - x_robot * scale).astype(np.int32)

        valid_pixels = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
        px = px[valid_pixels]
        py = py[valid_pixels]

        img[py, px] = 1.0

    cv2.circle(img, (center, center), robot_radius, 0.5, -1)

    return img


class MultiTaskBEVRegressionDataset(Dataset):
    """Dataset for multi-task BEV learning: path validity + angle regression."""

    def __init__(self, lidar_data: np.ndarray, target_angles: np.ndarray, has_path: np.ndarray,
                 image_size=64, view_range=3.0, max_range=12.0, angle_range=np.pi,
                 augment=False):
        self.lidar_data = lidar_data
        self.has_path = torch.tensor(has_path, dtype=torch.float32)

        # Normalize angles to [-1, 1] for better training
        # angle_range is ±90° = π radians total
        normalized_angles = np.clip(target_angles, -angle_range/2, angle_range/2) / (angle_range/2)
        self.target_angles = torch.tensor(normalized_angles, dtype=torch.float32)

        self.image_size = image_size
        self.view_range = view_range
        self.max_range = max_range
        self.angle_range = angle_range
        self.augment = augment

    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, idx):
        lidar = self.lidar_data[idx]
        has_path = self.has_path[idx]
        target_angle = self.target_angles[idx]

        bev_img = lidar_to_bev(lidar, self.image_size, self.max_range, self.view_range)

        if self.augment:
            # Random shift (±2 pixels)
            shift_y = np.random.randint(-2, 3)
            shift_x = np.random.randint(-2, 3)
            if shift_y != 0 or shift_x != 0:
                bev_img = np.roll(bev_img, shift_y, axis=0)
                bev_img = np.roll(bev_img, shift_x, axis=1)

            # Random noise
            noise = np.random.randn(*bev_img.shape) * 0.02
            bev_img = np.clip(bev_img + noise, 0, 1)

        bev_tensor = torch.tensor(bev_img, dtype=torch.float32).unsqueeze(0)

        return bev_tensor, has_path, target_angle


class MultiTaskBEVRegressionCNN(nn.Module):
    """2D CNN for multi-task BEV learning: path validity + angle regression (5 layers)."""

    def __init__(self, image_size=64):
        super().__init__()
        self.image_size = image_size

        # Shared CNN backbone (5 conv layers)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # 64 -> 32 -> 16 -> 8 -> 4 -> 2
        feat_size = image_size // 32
        self.shared_fc = nn.Linear(256 * feat_size * feat_size, 512)
        self.bn_fc = nn.BatchNorm1d(512)

        # Path validity head (binary classification)
        self.path_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Angle regression head (outputs single value in [-1, 1])
        self.angle_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        # x shape: (batch, 1, H, W)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 64->32
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 32->16
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # 16->8
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))  # 8->4
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))  # 4->2

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        shared = torch.relu(self.bn_fc(self.shared_fc(x)))
        shared = self.dropout(shared)

        path_logit = self.path_head(shared).squeeze(-1)
        angle_pred = self.angle_head(shared).squeeze(-1)  # [-1, 1]

        return path_logit, angle_pred


def train_epoch(model, dataloader, criterion_path, criterion_angle, optimizer, device, angle_loss_weight=1.0):
    """Train for one epoch with multi-task loss."""
    model.train()
    total_loss = 0.0
    path_correct = 0
    angle_errors = []
    total = 0

    for bev_img, has_path, target_angle in dataloader:
        bev_img = bev_img.to(device)
        has_path = has_path.to(device)
        target_angle = target_angle.to(device)

        optimizer.zero_grad()

        path_logit, angle_pred = model(bev_img)

        # Path loss (always computed)
        path_loss = criterion_path(path_logit, has_path)

        # Angle loss (only for samples with valid path)
        path_mask = has_path > 0.5
        if path_mask.sum() > 0:
            angle_loss = criterion_angle(angle_pred[path_mask], target_angle[path_mask])
        else:
            angle_loss = torch.tensor(0.0, device=device)

        # Combined loss
        loss = path_loss + angle_loss_weight * angle_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bev_img.size(0)

        # Path accuracy
        path_pred = (torch.sigmoid(path_logit) > 0.5).float()
        path_correct += (path_pred == has_path).sum().item()

        # Angle error (MAE in normalized space, only for valid paths)
        if path_mask.sum() > 0:
            with torch.no_grad():
                ae = torch.abs(angle_pred[path_mask] - target_angle[path_mask])
                angle_errors.extend(ae.cpu().numpy())

        total += bev_img.size(0)

    avg_loss = total_loss / total
    path_acc = path_correct / total
    avg_mae = np.mean(angle_errors) if angle_errors else 0.0

    return avg_loss, path_acc, avg_mae


def validate(model, dataloader, criterion_path, criterion_angle, device, angle_range=np.pi):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    path_correct = 0
    angle_errors_deg = []
    total = 0

    with torch.no_grad():
        for bev_img, has_path, target_angle in dataloader:
            bev_img = bev_img.to(device)
            has_path = has_path.to(device)
            target_angle = target_angle.to(device)

            path_logit, angle_pred = model(bev_img)

            # Path loss
            path_loss = criterion_path(path_logit, has_path)

            # Angle loss
            path_mask = has_path > 0.5
            if path_mask.sum() > 0:
                angle_loss = criterion_angle(angle_pred[path_mask], target_angle[path_mask])
            else:
                angle_loss = torch.tensor(0.0, device=device)

            loss = path_loss + angle_loss
            total_loss += loss.item() * bev_img.size(0)

            # Path accuracy
            path_pred = (torch.sigmoid(path_logit) > 0.5).float()
            path_correct += (path_pred == has_path).sum().item()

            # Angle MAE in degrees (only for valid paths)
            if path_mask.sum() > 0:
                # Convert from normalized [-1, 1] to radians then degrees
                pred_rad = angle_pred[path_mask] * (angle_range / 2)
                target_rad = target_angle[path_mask] * (angle_range / 2)
                ae_deg = torch.abs(pred_rad - target_rad) * 180 / np.pi
                angle_errors_deg.extend(ae_deg.cpu().numpy())

            total += bev_img.size(0)

    avg_loss = total_loss / total
    path_acc = path_correct / total
    avg_mae_deg = np.mean(angle_errors_deg) if angle_errors_deg else 0.0

    return avg_loss, path_acc, avg_mae_deg


def main():
    parser = argparse.ArgumentParser(description="Train multi-task BEV model with angle regression")
    parser.add_argument("--data-dir", type=str, default="data/training", help="Training data directory")
    parser.add_argument("--output-dir", type=str, default="data/models", help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--image-size", type=int, default=64, help="BEV image size")
    parser.add_argument("--view-range", type=float, default=3.0, help="BEV view range in meters")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--angle-loss-weight", type=float, default=1.0, help="Weight for angle loss")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="minicar-angle-predictor", help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default="", help="wandb run name (timestamp added)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    # Generate run name with timestamp
    run_name = get_run_name_with_timestamp(args.wandb_run_name or "multitask-bev-regression")

    # Create model directory
    model_dir = Path(args.output_dir) / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    print(f"Run name: {run_name}")
    print(f"Model directory: {model_dir}")

    # Load data
    data_dir = Path(args.data_dir)
    print("Loading data...")

    lidar_data = np.load(data_dir / "lidar_data.npy")
    target_angles = np.load(data_dir / "target_angles.npy")
    has_path = np.load(data_dir / "has_path.npy")

    # target_angles is already in radians from generate_angle_data.py
    target_angles_rad = target_angles

    max_range = 12.0
    image_size = args.image_size
    view_range = args.view_range
    angle_range = np.pi  # ±90°

    n_samples = len(lidar_data)
    n_with_path = has_path.sum()
    n_no_path = n_samples - n_with_path

    print(f"  LiDAR shape: {lidar_data.shape}")
    print(f"  has_path: {n_with_path} with path, {n_no_path} without path")
    print(f"  Regression output: continuous angle in [-π/2, π/2]")
    print(f"  BEV image size: {image_size}x{image_size}, view_range: {view_range}m")
    print(f"  Augment: {args.augment}, Angle loss weight: {args.angle_loss_weight}")

    # Train/val split
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = MultiTaskBEVRegressionDataset(
        lidar_data[train_idx], target_angles_rad[train_idx], has_path[train_idx],
        image_size=image_size, view_range=view_range, max_range=max_range,
        angle_range=angle_range, augment=args.augment
    )
    val_dataset = MultiTaskBEVRegressionDataset(
        lidar_data[val_idx], target_angles_rad[val_idx], has_path[val_idx],
        image_size=image_size, view_range=view_range, max_range=max_range,
        angle_range=angle_range, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_with_path = has_path[train_idx].sum()
    val_with_path = has_path[val_idx].sum()
    print(f"  Train samples: {len(train_idx)} ({train_with_path} with path)")
    print(f"  Val samples: {len(val_idx)} ({val_with_path} with path)")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskBEVRegressionCNN(image_size=image_size).to(device)

    print(f"  Device: {device}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    criterion_path = nn.BCEWithLogitsLoss()
    criterion_angle = nn.MSELoss()  # MSE for regression

    # Optimizer with scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')
    best_val_path_acc = 0.0
    best_mae = float('inf')
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(args.epochs):
        train_loss, train_path_acc, train_mae = train_epoch(
            model, train_loader, criterion_path, criterion_angle,
            optimizer, device, args.angle_loss_weight
        )

        val_loss, val_path_acc, val_mae_deg = validate(
            model, val_loader, criterion_path, criterion_angle, device, angle_range
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"    Train Loss: {train_loss:.4f}, Path Acc: {train_path_acc*100:.1f}%")
        print(f"    Val Loss: {val_loss:.4f}, Path Acc: {val_path_acc*100:.1f}%, MAE: {val_mae_deg:.2f}°")

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/path_accuracy": train_path_acc,
                "val/loss": val_loss,
                "val/path_accuracy": val_path_acc,
                "val/mae_degrees": val_mae_deg,
                "learning_rate": current_lr,
            })

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_path_acc = val_path_acc
            best_mae = val_mae_deg
            patience_counter = 0
            print(f"    -> New best model!")

            # Save best checkpoint
            best_checkpoint_path = model_dir / "best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_path_acc': val_path_acc,
                'val_mae_deg': val_mae_deg,
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Save periodic checkpoint
        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = model_dir / f"checkpoint_epoch{epoch + 1:03d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_path_acc': val_path_acc,
                'val_mae_deg': val_mae_deg,
            }, checkpoint_path)
            print(f"    -> Saved checkpoint: {checkpoint_path.name}")

    # Load best model
    best_checkpoint = torch.load(model_dir / "best.pt", weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"\nRestored best model (val_loss={best_val_loss:.4f}, path_acc={best_val_path_acc*100:.1f}%, mae={best_mae:.2f}°)")

    # Save final model
    model_path = model_dir / "model_final.pt"
    model_metadata = {
        'model_state_dict': model.state_dict(),
        'model_type': 'multitask_bev_regression',
        'model_arch': 'bev_cnn_regression',
        'is_bev': True,
        'is_multitask': True,
        'is_regression': True,
        'input_dim': 360,
        'bev_image_size': image_size,
        'bev_view_range': view_range,
        'image_size': image_size,
        'view_range': view_range,
        'angle_range': float(angle_range),
        'max_range': float(max_range),
        'best_val_loss': float(best_val_loss),
        'best_val_path_acc': float(best_val_path_acc),
        'best_mae_deg': float(best_mae),
        'run_name': run_name,
    }
    torch.save(model_metadata, model_path)
    print(f"\nModel saved to {model_path}")

    # Upload to wandb Artifacts
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"model-{run_name}",
            type="model",
            metadata={
                "val_loss": best_val_loss,
                "path_accuracy": best_val_path_acc,
                "mae_degrees": best_mae,
            }
        )
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        print(f"Uploaded model to wandb Artifacts: model-{run_name}")

        wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train multi-task model for path validity + angle prediction.

Input: LiDAR (360)
Output: has_path (binary), target_angle (9 classes, only when has_path=True)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

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


def angle_to_class(angle_rad, num_classes=9, angle_range=np.pi):
    """Convert angle (radians) to class index."""
    angle_rad = np.clip(angle_rad, -angle_range/2, angle_range/2)
    normalized = (angle_rad + angle_range/2) / angle_range
    class_idx = int(normalized * num_classes)
    return min(class_idx, num_classes - 1)


def class_to_angle(class_idx, num_classes=9, angle_range=np.pi):
    """Convert class index back to angle (radians)."""
    class_width = angle_range / num_classes
    angle = -angle_range/2 + (class_idx + 0.5) * class_width
    return angle


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning: path validity + angle classification."""

    def __init__(self, lidar_data: np.ndarray, target_angles: np.ndarray, has_path: np.ndarray,
                 num_classes=9, augment=False, noise_std=0.005, shift_range=5, scale_range=(0.95, 1.05)):
        self.lidar_data = torch.tensor(lidar_data, dtype=torch.float32)
        self.has_path = torch.tensor(has_path, dtype=torch.float32)

        # Convert angles to class labels (only meaningful when has_path=True)
        class_labels = np.array([angle_to_class(a, num_classes) for a in target_angles])
        self.target_classes = torch.tensor(class_labels, dtype=torch.long)

        self.num_classes = num_classes
        self.augment = augment
        self.noise_std = noise_std
        self.shift_range = shift_range
        self.scale_range = scale_range

    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, idx):
        lidar = self.lidar_data[idx].clone()
        has_path = self.has_path[idx]
        angle_class = self.target_classes[idx]

        if self.augment:
            # Random global scale
            scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
            lidar = lidar * scale

            # Add Gaussian noise
            if self.noise_std > 0:
                noise = torch.randn_like(lidar) * self.noise_std
                lidar = lidar + noise

            # Clamp to valid range
            lidar = torch.clamp(lidar, 0, 1)

            # Random circular shift
            if self.shift_range > 0:
                shift = torch.randint(-self.shift_range, self.shift_range + 1, (1,)).item()
                lidar = torch.roll(lidar, shift, dims=0)

        return lidar, has_path, angle_class


class MultiTaskCNN(nn.Module):
    """1D CNN for multi-task learning: path validity + angle classification (5 layers)."""

    def __init__(self, input_dim=360, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        # Shared CNN backbone with circular padding (5 layers)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Feature size: 360 -> 180 -> 90 -> 45 -> 22 -> 11
        self.shared_fc = nn.Linear(256 * 11, 512)
        self.bn_fc = nn.BatchNorm1d(512)

        # Path validity head (binary classification)
        self.path_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Angle classification head
        self.angle_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def _circular_pad(self, x, pad):
        """Apply circular padding for 1D convolution."""
        return torch.cat([x[:, :, -pad:], x, x[:, :, :pad]], dim=2)

    def forward(self, x):
        # x shape: (batch, 360)
        x = x.unsqueeze(1)  # (batch, 1, 360)

        # Conv1-5 with circular padding
        x = self._circular_pad(x, 3)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        x = self._circular_pad(x, 2)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        x = self._circular_pad(x, 1)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = self._circular_pad(x, 1)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = self._circular_pad(x, 1)
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))

        # Shared features
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        shared = torch.relu(self.bn_fc(self.shared_fc(x)))
        shared = self.dropout(shared)

        # Two heads
        path_logit = self.path_head(shared).squeeze(-1)
        angle_logits = self.angle_head(shared)

        return path_logit, angle_logits


def train_epoch(model, dataloader, criterion_path, criterion_angle, optimizer, device, angle_loss_weight=1.0):
    """Train for one epoch with multi-task loss."""
    model.train()
    total_loss = 0.0
    path_correct = 0
    angle_correct = 0
    angle_total = 0
    total = 0

    for lidar, has_path, angle_class in dataloader:
        lidar = lidar.to(device)
        has_path = has_path.to(device)
        angle_class = angle_class.to(device)

        optimizer.zero_grad()
        path_logit, angle_logits = model(lidar)

        # Path validity loss (all samples)
        loss_path = criterion_path(path_logit, has_path)

        # Angle loss (only for samples with valid path)
        path_mask = has_path > 0.5
        if path_mask.sum() > 0:
            loss_angle = criterion_angle(angle_logits[path_mask], angle_class[path_mask])
        else:
            loss_angle = torch.tensor(0.0, device=device)

        # Combined loss
        loss = loss_path + angle_loss_weight * loss_angle
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Path accuracy
        path_pred = (torch.sigmoid(path_logit) > 0.5).float()
        path_correct += (path_pred == has_path).sum().item()
        total += has_path.size(0)

        # Angle accuracy (only for valid paths)
        if path_mask.sum() > 0:
            _, angle_pred = torch.max(angle_logits[path_mask], 1)
            angle_correct += (angle_pred == angle_class[path_mask]).sum().item()
            angle_total += path_mask.sum().item()

    path_acc = path_correct / total if total > 0 else 0
    angle_acc = angle_correct / angle_total if angle_total > 0 else 0
    return total_loss / len(dataloader), path_acc, angle_acc


def evaluate(model, dataloader, criterion_path, criterion_angle, device, num_classes=9):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    path_correct = 0
    angle_correct = 0
    angle_total = 0
    total = 0
    all_angle_preds = []
    all_angle_targets = []

    with torch.no_grad():
        for lidar, has_path, angle_class in dataloader:
            lidar = lidar.to(device)
            has_path = has_path.to(device)
            angle_class = angle_class.to(device)

            path_logit, angle_logits = model(lidar)

            # Path validity loss
            loss_path = criterion_path(path_logit, has_path)

            # Angle loss (only for samples with valid path)
            path_mask = has_path > 0.5
            if path_mask.sum() > 0:
                loss_angle = criterion_angle(angle_logits[path_mask], angle_class[path_mask])
            else:
                loss_angle = torch.tensor(0.0, device=device)

            total_loss += (loss_path + loss_angle).item()

            # Path accuracy
            path_pred = (torch.sigmoid(path_logit) > 0.5).float()
            path_correct += (path_pred == has_path).sum().item()
            total += has_path.size(0)

            # Angle accuracy
            if path_mask.sum() > 0:
                _, angle_pred = torch.max(angle_logits[path_mask], 1)
                angle_correct += (angle_pred == angle_class[path_mask]).sum().item()
                angle_total += path_mask.sum().item()
                all_angle_preds.extend(angle_pred.cpu().numpy())
                all_angle_targets.extend(angle_class[path_mask].cpu().numpy())

    path_acc = path_correct / total if total > 0 else 0
    angle_acc = angle_correct / angle_total if angle_total > 0 else 0

    # Compute MAE for angle
    if len(all_angle_preds) > 0:
        pred_angles = np.array([class_to_angle(p, num_classes) for p in all_angle_preds])
        target_angles = np.array([class_to_angle(t, num_classes) for t in all_angle_targets])
        mae_degrees = np.degrees(np.mean(np.abs(pred_angles - target_angles)))
    else:
        mae_degrees = 0.0

    return total_loss / len(dataloader), path_acc, angle_acc, mae_degrees


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--num-classes", type=int, default=9, help="Number of angle classes")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--angle-loss-weight", type=float, default=1.0, help="Weight for angle loss")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="minicar-angle-predictor", help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs (0 to disable)")
    args = parser.parse_args()

    # Generate run name with timestamp
    run_name = get_run_name_with_timestamp(args.wandb_run_name or "multitask")

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, logging disabled")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": "multitask_cnn",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "num_classes": args.num_classes,
                "augment": args.augment,
                "angle_loss_weight": args.angle_loss_weight,
            }
        )

    num_classes = args.num_classes
    data_dir = Path("data/training")
    model_dir = Path("data/models") / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name: {run_name}")
    print(f"Model directory: {model_dir}")

    # Load data
    print("Loading data...")
    lidar_data = np.load(data_dir / "lidar_data.npy")
    target_angles = np.load(data_dir / "target_angles.npy")
    has_path = np.load(data_dir / "has_path.npy")

    print(f"  LiDAR shape: {lidar_data.shape}")
    print(f"  has_path: {has_path.sum()} with path, {(~has_path).sum()} without path")
    print(f"  Classification: {num_classes} classes")
    print(f"  Augment: {args.augment}, Angle loss weight: {args.angle_loss_weight}")

    # Normalize LiDAR data
    max_range = 12.0
    lidar_data = np.clip(lidar_data, 0, max_range) / max_range

    # Split data (80% train, 20% val)
    n_samples = len(lidar_data)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = MultiTaskDataset(
        lidar_data[train_indices], target_angles[train_indices], has_path[train_indices],
        num_classes, augment=args.augment
    )
    val_dataset = MultiTaskDataset(
        lidar_data[val_indices], target_angles[val_indices], has_path[val_indices],
        num_classes, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"  Train samples: {len(train_dataset)} ({train_dataset.has_path.sum().item():.0f} with path)")
    print(f"  Val samples: {len(val_dataset)} ({val_dataset.has_path.sum().item():.0f} with path)")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = MultiTaskCNN(input_dim=360, num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Loss functions
    criterion_path = nn.BCEWithLogitsLoss()
    criterion_angle = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(args.epochs):
        train_loss, train_path_acc, train_angle_acc = train_epoch(
            model, train_loader, criterion_path, criterion_angle, optimizer, device, args.angle_loss_weight
        )
        val_loss, val_path_acc, val_angle_acc, mae_deg = evaluate(
            model, val_loader, criterion_path, criterion_angle, device, num_classes
        )

        scheduler.step(val_loss)

        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"    Train Loss: {train_loss:.4f}, Path Acc: {train_path_acc:.1%}, Angle Acc: {train_angle_acc:.1%}")
        print(f"    Val Loss: {val_loss:.4f}, Path Acc: {val_path_acc:.1%}, Angle Acc: {val_angle_acc:.1%}, MAE: {mae_deg:.1f}°")

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/path_accuracy": train_path_acc,
                "train/angle_accuracy": train_angle_acc,
                "val/loss": val_loss,
                "val/path_accuracy": val_path_acc,
                "val/angle_accuracy": val_angle_acc,
                "val/mae_degrees": mae_deg,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_path_acc = val_path_acc
            best_val_angle_acc = val_angle_acc
            best_mae = mae_deg
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"    -> New best model!")

            # Save best checkpoint
            best_checkpoint_path = model_dir / "best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_path_acc': val_path_acc,
                'val_angle_acc': val_angle_acc,
                'mae_deg': mae_deg,
            }, best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
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
                'val_angle_acc': val_angle_acc,
                'mae_deg': mae_deg,
            }, checkpoint_path)
            print(f"    -> Saved checkpoint: {checkpoint_path.name}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (val_loss={best_val_loss:.4f}, path_acc={best_val_path_acc:.1%}, angle_acc={best_val_angle_acc:.1%})")

    # Save final model
    model_path = model_dir / "model_final.pt"
    model_metadata = {
        'model_state_dict': model.state_dict(),
        'model_type': 'multitask',
        'input_dim': 360,
        'num_classes': num_classes,
        'max_range': max_range,
        'best_val_loss': best_val_loss,
        'best_val_path_acc': best_val_path_acc,
        'best_val_angle_acc': best_val_angle_acc,
        'best_mae': best_mae,
        'run_name': run_name,
    }
    torch.save(model_metadata, model_path)
    print(f"\nModel saved to {model_path}")

    # Upload to wandb Artifacts
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"model-{run_name}",
            type="model",
            description="Multi-task CNN (path validity + angle)",
            metadata={
                "val_path_accuracy": best_val_path_acc,
                "val_angle_accuracy": best_val_angle_acc,
                "val_loss": best_val_loss,
                "mae_degrees": best_mae,
                "num_classes": num_classes,
            }
        )
        artifact.add_file(str(model_path))
        artifact.add_file(str(model_dir / "best.pt"))
        wandb.log_artifact(artifact)
        print(f"Uploaded model to wandb Artifacts: model-{run_name}")

    # Log final metrics and finish wandb
    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_val_path_accuracy"] = best_val_path_acc
        wandb.summary["best_val_angle_accuracy"] = best_val_angle_acc
        wandb.summary["best_mae_degrees"] = best_mae
        wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()

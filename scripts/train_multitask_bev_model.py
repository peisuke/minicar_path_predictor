#!/usr/bin/env python3
"""Train multi-task model using Bird's Eye View images.

Input: LiDAR (360) -> BEV image
Output: has_path (binary), target_angle (9 classes, only when has_path=True)
"""

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


def get_nonuniform_bins(angle_range_deg=90, num_classes=13):
    """Get non-uniform bin boundaries with finer resolution near center.

    13-class (default): Center-concentrated with 2° resolution within ±5°.
        Boundaries: [-45, -30, -20, -10, -5, -3, -1, +1, +3, +5, +10, +20, +30, +45]
        Resolution: 2° (center) -> 5° -> 10° -> 15° (edge)

    9-class (legacy): Finer resolution near center with 5° minimum.
        Boundaries: [-45, -32.5, -17.5, -7.5, -2.5, +2.5, +7.5, +17.5, +32.5, +45]
        Resolution: 5° (center) -> 10° -> 15° -> edge

    Returns boundaries and centers in radians.
    """
    half = angle_range_deg / 2  # e.g., 45° for ±45° range

    if num_classes == 13:
        # 13-class: center-concentrated
        # Resolution from center: 2° -> 2° -> 2° -> 5° -> 10° -> 10° -> 15°
        boundaries_deg = np.array([-half, -30, -20, -10, -5, -3, -1, 1, 3, 5, 10, 20, 30, half])
    else:
        # 9-class: legacy
        boundaries_deg = np.array([-half, -32.5, -17.5, -7.5, -2.5, 2.5, 7.5, 17.5, 32.5, half])

    boundaries_deg = np.clip(boundaries_deg, -half, half)
    boundaries_deg = np.unique(boundaries_deg)

    boundaries_rad = np.radians(boundaries_deg)
    centers_rad = (boundaries_rad[:-1] + boundaries_rad[1:]) / 2
    return boundaries_rad, centers_rad


def angle_to_class(angle_rad, num_classes=9, angle_range=np.pi, bin_boundaries=None):
    """Convert angle (radians) to class index.

    Args:
        angle_rad: Angle in radians
        num_classes: Number of classes (used only for uniform bins)
        angle_range: Total angle range in radians (used only for uniform bins)
        bin_boundaries: Optional array of bin boundaries for non-uniform bins
    """
    if bin_boundaries is not None:
        # Non-uniform bins: use searchsorted
        angle_rad = float(np.clip(angle_rad, bin_boundaries[0], bin_boundaries[-1]))
        class_idx = int(np.searchsorted(bin_boundaries[1:], angle_rad, side='right'))
        max_idx = len(bin_boundaries) - 2
        return min(class_idx, max_idx)
    else:
        # Uniform bins
        angle_rad = np.clip(angle_rad, -angle_range/2, angle_range/2)
        normalized = (angle_rad + angle_range/2) / angle_range
        class_idx = int(normalized * num_classes)
        return min(class_idx, num_classes - 1)


def class_to_angle(class_idx, num_classes=9, angle_range=np.pi, bin_centers=None):
    """Convert class index back to angle (radians).

    Args:
        class_idx: Class index
        num_classes: Number of classes (used only for uniform bins)
        angle_range: Total angle range in radians (used only for uniform bins)
        bin_centers: Optional array of bin centers for non-uniform bins
    """
    if bin_centers is not None:
        # Non-uniform bins: use precomputed centers
        return bin_centers[class_idx]
    else:
        # Uniform bins
        class_width = angle_range / num_classes
        angle = -angle_range/2 + (class_idx + 0.5) * class_width
        return angle


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


class MultiTaskBEVDataset(Dataset):
    """Dataset for multi-task BEV learning: path validity + angle classification."""

    def __init__(self, lidar_data: np.ndarray, target_angles: np.ndarray, has_path: np.ndarray,
                 num_classes=9, image_size=64, view_range=3.0, max_range=12.0,
                 angle_range=np.pi, augment=False, bin_boundaries=None):
        self.lidar_data = lidar_data
        self.has_path = torch.tensor(has_path, dtype=torch.float32)
        self.angle_range = angle_range
        self.bin_boundaries = bin_boundaries

        class_labels = np.array([angle_to_class(a, num_classes, angle_range, bin_boundaries) for a in target_angles])
        self.target_classes = torch.tensor(class_labels, dtype=torch.long)

        self.num_classes = num_classes
        self.image_size = image_size
        self.view_range = view_range
        self.max_range = max_range
        self.augment = augment

    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, idx):
        lidar = self.lidar_data[idx]
        has_path = self.has_path[idx]
        angle_class = self.target_classes[idx]

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

        return bev_tensor, has_path, angle_class


class MultiTaskBEVCNN(nn.Module):
    """2D CNN for multi-task BEV learning: path validity + angle classification (5 layers)."""

    def __init__(self, num_classes=9, image_size=64):
        super().__init__()
        self.num_classes = num_classes
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

        # Angle classification head
        self.angle_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
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

    for bev_img, has_path, angle_class in dataloader:
        bev_img = bev_img.to(device)
        has_path = has_path.to(device)
        angle_class = angle_class.to(device)

        optimizer.zero_grad()
        path_logit, angle_logits = model(bev_img)

        loss_path = criterion_path(path_logit, has_path)

        path_mask = has_path > 0.5
        if path_mask.sum() > 0:
            loss_angle = criterion_angle(angle_logits[path_mask], angle_class[path_mask])
        else:
            loss_angle = torch.tensor(0.0, device=device)

        loss = loss_path + angle_loss_weight * loss_angle
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        path_pred = (torch.sigmoid(path_logit) > 0.5).float()
        path_correct += (path_pred == has_path).sum().item()
        total += has_path.size(0)

        if path_mask.sum() > 0:
            _, angle_pred = torch.max(angle_logits[path_mask], 1)
            angle_correct += (angle_pred == angle_class[path_mask]).sum().item()
            angle_total += path_mask.sum().item()

    path_acc = path_correct / total if total > 0 else 0
    angle_acc = angle_correct / angle_total if angle_total > 0 else 0
    return total_loss / len(dataloader), path_acc, angle_acc


def evaluate(model, dataloader, criterion_path, criterion_angle, device,
             num_classes=9, angle_range=np.pi, bin_centers=None):
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
        for bev_img, has_path, angle_class in dataloader:
            bev_img = bev_img.to(device)
            has_path = has_path.to(device)
            angle_class = angle_class.to(device)

            path_logit, angle_logits = model(bev_img)

            loss_path = criterion_path(path_logit, has_path)

            path_mask = has_path > 0.5
            if path_mask.sum() > 0:
                loss_angle = criterion_angle(angle_logits[path_mask], angle_class[path_mask])
            else:
                loss_angle = torch.tensor(0.0, device=device)

            total_loss += (loss_path + loss_angle).item()

            path_pred = (torch.sigmoid(path_logit) > 0.5).float()
            path_correct += (path_pred == has_path).sum().item()
            total += has_path.size(0)

            if path_mask.sum() > 0:
                _, angle_pred = torch.max(angle_logits[path_mask], 1)
                angle_correct += (angle_pred == angle_class[path_mask]).sum().item()
                angle_total += path_mask.sum().item()
                all_angle_preds.extend(angle_pred.cpu().numpy())
                all_angle_targets.extend(angle_class[path_mask].cpu().numpy())

    path_acc = path_correct / total if total > 0 else 0
    angle_acc = angle_correct / angle_total if angle_total > 0 else 0

    if len(all_angle_preds) > 0:
        pred_angles = np.array([class_to_angle(p, num_classes, angle_range, bin_centers) for p in all_angle_preds])
        target_angles = np.array([class_to_angle(t, num_classes, angle_range, bin_centers) for t in all_angle_targets])
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
    parser.add_argument("--angle-range", type=float, default=180.0, help="Angle range in degrees (e.g., 180 for ±90°, 90 for ±45°)")
    parser.add_argument("--image-size", type=int, default=64, help="BEV image size")
    parser.add_argument("--view-range", type=float, default=3.0, help="View range in meters")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--angle-loss-weight", type=float, default=1.0, help="Weight for angle loss")
    parser.add_argument("--nonuniform-bins", action="store_true",
                        help="Use non-uniform bins with finer resolution near center (0,±5,±10,±25,±40°)")
    parser.add_argument("--lookahead-idx", type=int, default=None,
                        help="Index of lookahead distance to use (if data has multiple). None=auto-detect or use column 0")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="minicar-angle-predictor", help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    # Generate run name with timestamp
    run_name = get_run_name_with_timestamp(args.wandb_run_name or "multitask-bev")

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, logging disabled")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": "multitask_bev_cnn",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "num_classes": args.num_classes,
                "angle_range_deg": args.angle_range,
                "image_size": args.image_size,
                "view_range": args.view_range,
                "augment": args.augment,
                "angle_loss_weight": args.angle_loss_weight,
                "nonuniform_bins": args.nonuniform_bins,
            }
        )

    num_classes = args.num_classes
    image_size = args.image_size
    view_range = args.view_range
    angle_range = np.radians(args.angle_range)  # Convert to radians

    # Setup bin boundaries for non-uniform bins
    bin_boundaries = None
    bin_centers = None
    if args.nonuniform_bins:
        bin_boundaries, bin_centers = get_nonuniform_bins(args.angle_range, args.num_classes)
        num_classes = len(bin_centers)  # Override num_classes based on boundaries
        print(f"  Using non-uniform bins: {len(bin_centers)} classes")
        print(f"  Boundaries (deg): {np.degrees(bin_boundaries).astype(int)}")
        print(f"  Centers (deg): {np.degrees(bin_centers).round(1)}")

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
    print(f"  target_angles shape: {target_angles.shape}")

    # Handle multi-column target_angles (multiple lookahead distances)
    lookahead_dist = None
    if target_angles.ndim == 2:
        # Load lookahead distances if available
        lookahead_dists_file = data_dir / "lookahead_distances.npy"
        if lookahead_dists_file.exists():
            lookahead_dists = np.load(lookahead_dists_file)
            print(f"  Available lookahead distances: {lookahead_dists} m")
        else:
            lookahead_dists = None

        # Select which column to use
        lookahead_idx = args.lookahead_idx if args.lookahead_idx is not None else 0
        if lookahead_idx >= target_angles.shape[1]:
            print(f"  Warning: --lookahead-idx {lookahead_idx} >= {target_angles.shape[1]}, using 0")
            lookahead_idx = 0

        if lookahead_dists is not None:
            lookahead_dist = lookahead_dists[lookahead_idx]
            print(f"  Using lookahead distance: {lookahead_dist}m (index {lookahead_idx})")
        else:
            print(f"  Using lookahead index: {lookahead_idx}")

        target_angles = target_angles[:, lookahead_idx]

    print(f"  has_path: {has_path.sum()} with path, {(~has_path).sum()} without path")
    print(f"  Classification: {num_classes} classes, angle_range: ±{args.angle_range/2:.0f}°")
    print(f"  BEV image size: {image_size}x{image_size}, view_range: {view_range}m")
    print(f"  Augment: {args.augment}, Angle loss weight: {args.angle_loss_weight}")

    # Normalize LiDAR data
    max_range = 12.0
    lidar_data = np.clip(lidar_data, 0, max_range) / max_range

    # Split data
    n_samples = len(lidar_data)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = MultiTaskBEVDataset(
        lidar_data[train_indices], target_angles[train_indices], has_path[train_indices],
        num_classes, image_size, view_range, max_range, angle_range, augment=args.augment,
        bin_boundaries=bin_boundaries
    )
    val_dataset = MultiTaskBEVDataset(
        lidar_data[val_indices], target_angles[val_indices], has_path[val_indices],
        num_classes, image_size, view_range, max_range, angle_range, augment=False,
        bin_boundaries=bin_boundaries
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"  Train samples: {len(train_dataset)} ({train_dataset.has_path.sum().item():.0f} with path)")
    print(f"  Val samples: {len(val_dataset)} ({val_dataset.has_path.sum().item():.0f} with path)")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = MultiTaskBEVCNN(num_classes=num_classes, image_size=image_size).to(device)
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
    best_mae = 0.0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(args.epochs):
        train_loss, train_path_acc, train_angle_acc = train_epoch(
            model, train_loader, criterion_path, criterion_angle, optimizer, device, args.angle_loss_weight
        )
        val_loss, val_path_acc, val_angle_acc, mae_deg = evaluate(
            model, val_loader, criterion_path, criterion_angle, device, num_classes, angle_range, bin_centers
        )

        scheduler.step(val_loss)

        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"    Train Loss: {train_loss:.4f}, Path Acc: {train_path_acc:.1%}, Angle Acc: {train_angle_acc:.1%}")
        print(f"    Val Loss: {val_loss:.4f}, Path Acc: {val_path_acc:.1%}, Angle Acc: {val_angle_acc:.1%}, MAE: {mae_deg:.1f}°")

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_path_acc = val_path_acc
            best_val_angle_acc = val_angle_acc
            best_mae = mae_deg
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"    -> New best model!")

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
        'model_type': 'multitask_bev',
        'model_arch': 'bev_cnn',
        'is_bev': True,
        'is_multitask': True,
        'input_dim': 360,
        'bev_image_size': image_size,
        'bev_view_range': view_range,
        'image_size': image_size,
        'view_range': view_range,
        'num_classes': num_classes,
        'angle_range': float(angle_range),
        'max_range': max_range,
        'best_val_loss': best_val_loss,
        'best_val_path_acc': best_val_path_acc,
        'best_val_angle_acc': best_val_angle_acc,
        'best_mae': best_mae,
        'run_name': run_name,
        # Non-uniform bins (None if uniform)
        'bin_boundaries': bin_boundaries.tolist() if bin_boundaries is not None else None,
        'bin_centers': bin_centers.tolist() if bin_centers is not None else None,
        'nonuniform_bins': args.nonuniform_bins,
        # Lookahead distance used for training
        'lookahead_distance': float(lookahead_dist) if lookahead_dist is not None else None,
    }
    torch.save(model_metadata, model_path)
    print(f"\nModel saved to {model_path}")

    # Upload to wandb Artifacts
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"model-{run_name}",
            type="model",
            description="Multi-task BEV CNN (path validity + angle)",
            metadata={
                "val_path_accuracy": best_val_path_acc,
                "val_angle_accuracy": best_val_angle_acc,
                "val_loss": best_val_loss,
                "mae_degrees": best_mae,
                "num_classes": num_classes,
                "image_size": image_size,
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

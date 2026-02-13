#!/usr/bin/env python3
"""Train angle prediction model using Bird's Eye View images.

Converts LiDAR data to 2D BEV images and uses 2D CNN for classification.
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


def lidar_to_bev(lidar_ranges, image_size=64, max_range=12.0, view_range=3.0, robot_radius=2):
    """Convert LiDAR ranges to Bird's Eye View image.

    Args:
        lidar_ranges: Array of 360 range values (normalized 0-1)
        image_size: Output image size (square)
        max_range: Maximum LiDAR range in meters (for denormalization)
        view_range: View range in meters (zoom level - smaller = more zoomed in)
        robot_radius: Robot marker radius in pixels

    Returns:
        BEV image (image_size x image_size), single channel, float32 [0, 1]
    """
    # Create empty image
    img = np.zeros((image_size, image_size), dtype=np.float32)

    # Robot is at center
    center = image_size // 2

    # Scale: pixels per meter (based on view_range, not max_range)
    scale = (image_size / 2 - 2) / view_range

    # LiDAR angles (0 to 2*pi)
    num_rays = len(lidar_ranges)
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    # Convert normalized ranges back to meters
    ranges_m = lidar_ranges * max_range

    # Convert polar to cartesian (robot frame)
    # LiDAR convention: angle=0 is forward (+x_robot), angle=π/2 is left (+y_robot)
    x_robot = ranges_m * np.cos(angles)  # forward direction
    y_robot = ranges_m * np.sin(angles)  # left direction

    # Convert to image coordinates (forward = up, left = left)
    px = (center - y_robot * scale).astype(np.int32)  # left (+y_robot) → image left (-x)
    py = (center - x_robot * scale).astype(np.int32)  # forward (+x_robot) → image up (-y)

    # Draw LiDAR points
    valid_mask = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
    valid_mask &= (ranges_m < max_range - 0.1)  # Only draw actual obstacles

    for i in range(num_rays):
        if valid_mask[i]:
            cv2.circle(img, (px[i], py[i]), 1, 1.0, -1)

    # Draw robot position at center (always fixed)
    cv2.circle(img, (center, center), robot_radius, 0.5, -1)

    return img


class BEVDataset(Dataset):
    """Dataset for BEV image -> angle classification."""

    def __init__(self, lidar_data: np.ndarray, target_angles: np.ndarray, num_classes=9,
                 image_size=64, view_range=3.0, augment=False):
        self.lidar_data = lidar_data  # Already normalized [0, 1]
        class_labels = np.array([angle_to_class(a, num_classes) for a in target_angles])
        self.target_classes = torch.tensor(class_labels, dtype=torch.long)
        self.num_classes = num_classes
        self.image_size = image_size
        self.view_range = view_range
        self.augment = augment

    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, idx):
        lidar = self.lidar_data[idx]
        target = self.target_classes[idx]

        # Data augmentation (before BEV conversion)
        if self.augment:
            # Random scale (0.95 ~ 1.05)
            scale = np.random.uniform(0.95, 1.05)
            lidar = lidar * scale

            # Random noise
            noise = np.random.randn(len(lidar)) * 0.005
            lidar = np.clip(lidar + noise, 0, 1)

        # Convert to BEV image
        bev_img = lidar_to_bev(lidar, self.image_size, view_range=self.view_range)

        # Data augmentation (after BEV conversion)
        if self.augment:
            # Random shift (1-2 pixels up/down/left/right)
            shift_y = np.random.randint(-2, 3)  # -2 to +2
            shift_x = np.random.randint(-2, 3)
            if shift_y != 0 or shift_x != 0:
                bev_img = np.roll(bev_img, shift_y, axis=0)
                bev_img = np.roll(bev_img, shift_x, axis=1)

        # Add channel dimension: (H, W) -> (1, H, W)
        bev_tensor = torch.tensor(bev_img, dtype=torch.float32).unsqueeze(0)

        return bev_tensor, target


class BEVClassifierCNN(nn.Module):
    """2D CNN for BEV image classification (5 layers)."""

    def __init__(self, num_classes=9, image_size=64):
        super().__init__()
        self.num_classes = num_classes

        # Convolutional layers (5 layers)
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

        # Calculate feature size after convolutions
        # 64 -> 32 -> 16 -> 8 -> 4 -> 2
        feat_size = image_size // 32
        self.fc1 = nn.Linear(256 * feat_size * feat_size, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv layers with pooling
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 64->32
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 32->16
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # 16->8
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))  # 8->4
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))  # 4->2

        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device, num_classes=9):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    # Compute angle error
    pred_angles = np.array([class_to_angle(p, num_classes) for p in all_predictions])
    target_angles = np.array([class_to_angle(t, num_classes) for t in all_targets])
    mae_degrees = np.degrees(np.mean(np.abs(pred_angles - target_angles)))

    return avg_loss, accuracy, mae_degrees


def visualize_samples(dataset, num_samples=9, output_path="data/visualizations/bev_samples.png"):
    """Visualize BEV samples."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        img, target = dataset[idx]
        img = img.squeeze().numpy()

        angle_deg = np.degrees(class_to_angle(target.item(), dataset.num_classes))

        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"Class {target.item()} ({angle_deg:+.0f}°)")
        axes[i].axis('off')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved BEV samples to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--num-classes", type=int, default=9, help="Number of angle classes")
    parser.add_argument("--image-size", type=int, default=64, help="BEV image size")
    parser.add_argument("--view-range", type=float, default=3.0, help="View range in meters (smaller = more zoomed in)")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--class-weights", action="store_true", help="Use class weights")
    parser.add_argument("--visualize", action="store_true", help="Visualize samples before training")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="minicar-angle-predictor", help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs (0 to disable)")
    args = parser.parse_args()

    # Generate run name with timestamp
    run_name = get_run_name_with_timestamp(args.wandb_run_name or "bev-cnn")

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, logging disabled")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": "bev_cnn",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "num_classes": args.num_classes,
                "image_size": args.image_size,
                "view_range": args.view_range,
                "augment": args.augment,
                "class_weights": args.class_weights,
            }
        )

    num_classes = args.num_classes
    image_size = args.image_size
    view_range = args.view_range

    data_dir = Path("data/training")
    model_dir = Path("data/models") / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name: {run_name}")
    print(f"Model directory: {model_dir}")

    # Load data
    print("Loading data...")
    lidar_data = np.load(data_dir / "lidar_data.npy")
    target_angles = np.load(data_dir / "target_angles.npy")

    print(f"  LiDAR shape: {lidar_data.shape}")
    print(f"  Angles shape: {target_angles.shape}")
    print(f"  Classification: {num_classes} classes")
    print(f"  BEV image size: {image_size}x{image_size}, view_range: {view_range}m")
    print(f"  Augment: {args.augment}, Class weights: {args.class_weights}")

    # Normalize LiDAR data
    max_range = 12.0
    lidar_data = np.clip(lidar_data, 0, max_range) / max_range

    # Split data
    n_samples = len(lidar_data)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = BEVDataset(
        lidar_data[train_indices], target_angles[train_indices], num_classes,
        image_size=image_size, view_range=view_range, augment=args.augment
    )
    val_dataset = BEVDataset(
        lidar_data[val_indices], target_angles[val_indices], num_classes,
        image_size=image_size, view_range=view_range, augment=False
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Visualize samples
    if args.visualize:
        visualize_samples(train_dataset)

    # Class distribution and weights
    train_classes = train_dataset.target_classes.numpy()
    class_counts = np.bincount(train_classes, minlength=num_classes)
    print(f"  Class distribution:")
    for c in range(num_classes):
        angle_center = np.degrees(class_to_angle(c, num_classes))
        print(f"    Class {c} ({angle_center:+6.1f}°): {class_counts[c]:5d} ({100*class_counts[c]/len(train_classes):.1f}%)")

    if args.class_weights:
        class_weights = len(train_classes) / (num_classes * class_counts + 1e-6)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = BEVClassifierCNN(num_classes=num_classes, image_size=image_size)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training
    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    epochs_without_improvement = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, mae_deg = evaluate(model, val_loader, criterion, device, num_classes)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"    Train Loss: {train_loss:.4f}, Acc: {train_acc:.1%}")
        print(f"    Val Loss: {val_loss:.4f}, Acc: {val_acc:.1%}, MAE: {mae_deg:.1f}°")

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/mae_degrees": mae_deg,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
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
                'val_acc': val_acc,
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
                'val_acc': val_acc,
                'mae_deg': mae_deg,
            }, checkpoint_path)
            print(f"    -> Saved checkpoint: {checkpoint_path.name}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.1%})")

    # Save final model
    model_path = model_dir / "model_final.pt"
    model_metadata = {
        'model_state_dict': model.state_dict(),
        'model_type': 'classifier',
        'model_arch': 'bev_cnn',
        'input_dim': 360,
        'image_size': image_size,
        'num_classes': num_classes,
        'angle_range': np.pi,
        'max_range': max_range,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
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
            description="2D CNN BEV angle predictor",
            metadata={
                "val_accuracy": best_val_acc,
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

    # Test predictions
    print("\n=== Test Predictions ===")
    model.eval()
    with torch.no_grad():
        for i in range(min(10, len(val_dataset))):
            img, target_class = val_dataset[i]
            img = img.unsqueeze(0).to(device)
            logits = model(img)
            pred_class = torch.argmax(logits, dim=1).item()

            pred_angle = class_to_angle(pred_class, num_classes)
            target_angle = class_to_angle(target_class.item(), num_classes)

            print(f"  Sample {i + 1}: pred={pred_class} ({np.degrees(pred_angle):+6.1f}°), "
                  f"target={target_class.item()} ({np.degrees(target_angle):+6.1f}°), "
                  f"{'OK' if pred_class == target_class.item() else 'MISS'}")

    # Log final metrics and finish wandb
    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.summary["best_mae_degrees"] = best_mae
        wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()

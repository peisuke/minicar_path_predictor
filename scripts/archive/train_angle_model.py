#!/usr/bin/env python3
"""Train angle prediction model.

Input: LiDAR (360) -> Output: target_angle (1)
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
    """Convert angle (radians) to class index.

    Args:
        angle_rad: Angle in radians
        num_classes: Number of classes (default 9)
        angle_range: Total range (default π for ±90°)

    Returns:
        Class index (0 to num_classes-1)
    """
    # Clip to range [-angle_range/2, angle_range/2]
    angle_rad = np.clip(angle_rad, -angle_range/2, angle_range/2)
    # Normalize to [0, 1]
    normalized = (angle_rad + angle_range/2) / angle_range
    # Convert to class index
    class_idx = int(normalized * num_classes)
    return min(class_idx, num_classes - 1)


def class_to_angle(class_idx, num_classes=9, angle_range=np.pi):
    """Convert class index back to angle (radians).

    Returns the center angle of the class.
    """
    class_width = angle_range / num_classes
    angle = -angle_range/2 + (class_idx + 0.5) * class_width
    return angle


class LiDARDataset(Dataset):
    """Dataset for LiDAR -> angle classification with data augmentation."""

    def __init__(self, lidar_data: np.ndarray, target_angles: np.ndarray, num_classes=9,
                 augment=False, noise_std=0.005, shift_range=5, scale_range=(0.95, 1.05)):
        self.lidar_data = torch.tensor(lidar_data, dtype=torch.float32)
        # Convert angles to class labels
        class_labels = np.array([angle_to_class(a, num_classes) for a in target_angles])
        self.target_classes = torch.tensor(class_labels, dtype=torch.long)
        self.target_angles = target_angles  # Keep original for reference
        self.num_classes = num_classes
        self.augment = augment
        self.noise_std = noise_std
        self.shift_range = shift_range
        self.scale_range = scale_range

    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, idx):
        lidar = self.lidar_data[idx].clone()
        target = self.target_classes[idx]

        if self.augment:
            # Random global scale (0.95 ~ 1.05)
            scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
            lidar = lidar * scale

            # Add Gaussian noise (std=0.005 -> ~6cm)
            if self.noise_std > 0:
                noise = torch.randn_like(lidar) * self.noise_std
                lidar = lidar + noise

            # Clamp to valid range
            lidar = torch.clamp(lidar, 0, 1)

            # Random circular shift (simulates slight rotation)
            if self.shift_range > 0:
                shift = torch.randint(-self.shift_range, self.shift_range + 1, (1,)).item()
                lidar = torch.roll(lidar, shift, dims=0)

        return lidar, target


class AngleClassifierMLP(nn.Module):
    """MLP for angle classification from LiDAR."""

    def __init__(self, input_dim=360, hidden_dims=[256, 128, 64], num_classes=9):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AngleClassifierCNN(nn.Module):
    """1D CNN for angle classification from LiDAR with circular padding (5 layers)."""

    def __init__(self, input_dim=360, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        # CNN layers with circular padding (5 layers)
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

        # Calculate feature size after convolutions
        # After conv1 (k=7): 360 -> pool -> 180
        # After conv2 (k=5): 180 -> pool -> 90
        # After conv3 (k=3): 90 -> pool -> 45
        # After conv4 (k=3): 45 -> pool -> 22
        # After conv5 (k=3): 22 -> pool -> 11
        self.fc1 = nn.Linear(256 * 11, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

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

        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Alias for backward compatibility
AngleClassifier = AngleClassifierCNN


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for lidar, labels in dataloader:
        lidar = lidar.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(lidar)
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
        for lidar, labels in dataloader:
            lidar = lidar.to(device)
            labels = labels.to(device)

            logits = model(lidar)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Compute angle error (convert class to angle)
    pred_angles = np.array([class_to_angle(p, num_classes) for p in predictions])
    target_angles = np.array([class_to_angle(t, num_classes) for t in targets])
    angle_errors = np.abs(pred_angles - target_angles)
    mae_degrees = np.degrees(np.mean(angle_errors))

    return avg_loss, accuracy, mae_degrees, predictions, targets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-classes", type=int, default=9, help="Number of angle classes")
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn"], help="Model type")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--class-weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="minicar-angle-predictor", help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs (0 to disable)")
    args = parser.parse_args()

    # Generate run name with timestamp
    run_name = get_run_name_with_timestamp(args.wandb_run_name or "1d-cnn")

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, logging disabled")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "num_classes": args.num_classes,
                "augment": args.augment,
                "class_weights": args.class_weights,
            }
        )

    num_classes = args.num_classes
    angle_range = np.pi  # ±90°

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
    print(f"  Angle range: [{np.degrees(target_angles.min()):.1f}°, {np.degrees(target_angles.max()):.1f}°]")
    print(f"  Classification: {num_classes} classes (each {180/num_classes:.1f}°)")
    print(f"  Model: {args.model.upper()}, Augment: {args.augment}, Class weights: {args.class_weights}")

    # Normalize LiDAR data (clip to max range and scale to [0, 1])
    max_range = 12.0
    lidar_data = np.clip(lidar_data, 0, max_range) / max_range

    # Split data (80% train, 20% val)
    n_samples = len(lidar_data)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = LiDARDataset(
        lidar_data[train_indices], target_angles[train_indices], num_classes,
        augment=args.augment, noise_std=0.005, shift_range=5, scale_range=(0.95, 1.05)
    )
    val_dataset = LiDARDataset(
        lidar_data[val_indices], target_angles[val_indices], num_classes,
        augment=False  # No augmentation for validation
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Show class distribution and compute class weights
    train_classes = train_dataset.target_classes.numpy()
    class_counts = np.bincount(train_classes, minlength=num_classes)
    print(f"  Class distribution (train):")
    for c in range(num_classes):
        angle_center = np.degrees(class_to_angle(c, num_classes))
        print(f"    Class {c} ({angle_center:+6.1f}°): {class_counts[c]:5d} ({100*class_counts[c]/len(train_classes):.1f}%)")

    # Compute class weights (inverse frequency)
    if args.class_weights:
        class_weights = len(train_classes) / (num_classes * class_counts + 1e-6)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"  Class weights: {class_weights.numpy().round(2)}")
    else:
        class_weights = None

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if args.model == "cnn":
        model = AngleClassifierCNN(input_dim=360, num_classes=num_classes)
    else:
        model = AngleClassifierMLP(input_dim=360, hidden_dims=[256, 128, 64], num_classes=num_classes)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training with early stopping
    num_epochs = args.epochs
    patience = args.patience
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    print(f"\nTraining for up to {num_epochs} epochs (early stopping patience={patience})...")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, mae_deg, predictions, targets = evaluate(model, val_loader, criterion, device, num_classes)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch + 1}/{num_epochs}")
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

        # Early stopping check
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
            if epochs_without_improvement >= patience:
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
        'model_arch': args.model,  # 'cnn' or 'mlp'
        'input_dim': 360,
        'hidden_dims': [256, 128, 64],
        'num_classes': num_classes,
        'angle_range': angle_range,
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
            description=f"1D CNN angle predictor ({args.model})",
            metadata={
                "val_accuracy": best_val_acc,
                "val_loss": best_val_loss,
                "mae_degrees": best_mae,
                "num_classes": num_classes,
            }
        )
        artifact.add_file(str(model_path))
        artifact.add_file(str(model_dir / "best.pt"))
        wandb.log_artifact(artifact)
        print(f"Uploaded model to wandb Artifacts: model-{run_name}")

    # Test prediction
    print("\n=== Test Predictions ===")
    model.eval()
    with torch.no_grad():
        # Take first 10 validation samples
        for i in range(min(10, len(val_dataset))):
            lidar, target_class = val_dataset[i]
            lidar = lidar.unsqueeze(0).to(device)
            logits = model(lidar)
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

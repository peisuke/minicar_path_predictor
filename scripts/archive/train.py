#!/usr/bin/env python3
"""Train path prediction model.

Usage:
    python3 train.py
    python3 train.py --data data/sim_robot/training_data.npz --output weights/
    python3 train.py --model cnn --epochs 200
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minicar_path_predictor.dataset import PathPredictionDataset, AugmentedPathDataset, create_data_loaders
from minicar_path_predictor.model import create_model, count_parameters
from minicar_path_predictor.trainer import Trainer, TrainingConfig, evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train path prediction model"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path("data/sim_robot/training_data.npz"),
        help="Training data NPZ file (default: data/sim_robot/training_data.npz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("weights"),
        help="Output directory for models (default: weights/)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn", "transformer"],
        default="mlp",
        help="Model architecture (default: mlp)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda, cpu, or auto (default: auto)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation"
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="256,128,64",
        help="Hidden layer dimensions for MLP (default: 256,128,64)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def plot_training_history(history, output_path: Path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate plot
    axes[1].plot(history['learning_rate'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    print("=" * 60)
    print("Path Prediction Model Training")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\nDevice: {device}")

    # Load data
    print(f"\nLoading data from {args.data}...")
    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        return 1

    dataset = PathPredictionDataset.from_npz(args.data)
    print(f"  Total samples: {len(dataset)}")
    print(f"  Input dim: {dataset.lidar_data.shape[1]}")
    print(f"  Output dim: {dataset.waypoints_flat.shape[1]}")

    # Determine dimensions
    input_dim = dataset.lidar_data.shape[1]
    output_dim = dataset.waypoints_flat.shape[1]
    num_waypoints = output_dim // 2

    # Apply augmentation if requested
    if args.augment:
        print("  Data augmentation: enabled")
        dataset = AugmentedPathDataset(dataset)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        num_workers=4,
        seed=args.seed
    )

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Create model
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    print(f"\nCreating {args.model.upper()} model...")
    model = create_model(
        model_type=args.model,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims
    )

    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Training configuration
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.0001,
        early_stopping_patience=15,
        scheduler_patience=5,
        scheduler_factor=0.5,
        gradient_clip=1.0,
        save_best=True,
        save_every=20
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=args.output
    )

    # Train
    print("\n" + "-" * 60)
    print("Training...")
    print("-" * 60 + "\n")

    history = trainer.train(train_loader, val_loader, verbose=True)

    # Plot training history
    plot_path = args.output / "training_history.png"
    plot_training_history(history, plot_path)
    print(f"\nTraining history saved to {plot_path}")

    # Evaluate on test set
    print("\n" + "-" * 60)
    print("Evaluating on test set...")
    print("-" * 60)

    # Load best model
    trainer.load_model('best_model.pth')

    test_metrics = evaluate_model(
        trainer.model,
        test_loader,
        device=device,
        num_waypoints=num_waypoints
    )

    print(f"\nTest Results:")
    print(f"  MSE: {test_metrics['mse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  Average Displacement Error: {test_metrics['ade']:.4f} m")
    print(f"  Final Displacement Error: {test_metrics['fde']:.4f} m")

    # Save model info
    model_info = {
        'model_type': args.model,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'num_waypoints': num_waypoints,
        'num_parameters': num_params,
        'best_val_loss': trainer.best_val_loss,
        'test_metrics': test_metrics,
        'training_config': {
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'batch_size': args.batch_size,
            'augmentation': args.augment
        }
    }

    info_path = args.output / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\nModel info saved to {info_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: {args.output / 'best_model.pth'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

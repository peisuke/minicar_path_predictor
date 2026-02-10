"""Neural network models for path prediction."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathPredictorMLP(nn.Module):
    """MLP-based path predictor."""

    def __init__(
        self,
        input_dim: int = 360,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 10,
        dropout: float = 0.1
    ):
        """Initialize MLP model.

        Args:
            input_dim: Number of LiDAR rays
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (num_waypoints * 2)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LiDAR ranges, shape (batch, num_rays)

        Returns:
            Predicted waypoints, shape (batch, output_dim)
        """
        return self.network(x)


class PathPredictorCNN(nn.Module):
    """1D CNN-based path predictor for LiDAR data."""

    def __init__(
        self,
        input_dim: int = 360,
        output_dim: int = 10,
        dropout: float = 0.1
    ):
        """Initialize CNN model.

        Args:
            input_dim: Number of LiDAR rays
            output_dim: Output dimension (num_waypoints * 2)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 1D CNN layers (treat LiDAR as 1D signal with circular padding)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        # Calculate flattened size after conv layers
        # 360 -> 180 -> 90 -> 45
        conv_output_size = (input_dim // 8) * 128

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LiDAR ranges, shape (batch, num_rays)

        Returns:
            Predicted waypoints, shape (batch, output_dim)
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch, 1, num_rays)

        # CNN layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class PathPredictorTransformer(nn.Module):
    """Transformer-based path predictor."""

    def __init__(
        self,
        input_dim: int = 360,
        output_dim: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize Transformer model.

        Args:
            input_dim: Number of LiDAR rays
            output_dim: Output dimension (num_waypoints * 2)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Project each ray to d_model dimensions
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim * d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LiDAR ranges, shape (batch, num_rays)

        Returns:
            Predicted waypoints, shape (batch, output_dim)
        """
        batch_size = x.size(0)

        # Project each ray
        x = x.unsqueeze(-1)  # (batch, num_rays, 1)
        x = self.input_projection(x)  # (batch, num_rays, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer
        x = self.transformer(x)  # (batch, num_rays, d_model)

        # Flatten and project to output
        x = x.view(batch_size, -1)
        x = self.output_projection(x)

        return x


def create_model(
    model_type: str = "mlp",
    input_dim: int = 360,
    output_dim: int = 10,
    **kwargs
) -> nn.Module:
    """Create model by type.

    Args:
        model_type: One of "mlp", "cnn", "transformer"
        input_dim: Number of LiDAR rays
        output_dim: Output dimension
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    if model_type == "mlp":
        return PathPredictorMLP(input_dim, output_dim=output_dim, **kwargs)
    elif model_type == "cnn":
        return PathPredictorCNN(input_dim, output_dim=output_dim, **kwargs)
    elif model_type == "transformer":
        return PathPredictorTransformer(input_dim, output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

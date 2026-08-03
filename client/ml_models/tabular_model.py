"""Tabular Feature Encoder Neural Network Model for MedShield FL.

Processes patient numerical and categorical lifestyle metrics (e.g., age, blood pressure,
cholesterol, fasting blood sugar, ST slope) into feature embeddings for multimodal fusion.
"""

from __future__ import annotations

try:
    # pyrefly: ignore [missing-import]
    import torch
    import torch.nn as nn
    _TorchBase = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

    class _TorchBase:  # type: ignore[no-redef]
        """Sentinel base class used when PyTorch is not installed."""


class TabularEncoder(_TorchBase):
    """Feed-forward neural network encoder for patient lifestyle & clinical tabular data.

    Attributes:
        net: Sequential neural network layers for feature encoding.
        cat_embeddings: Optional ModuleDict of Embedding layers for discrete categorical features.
    """

    def __init__(
        self,
        num_features: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.2,
        cat_cardinalities: dict[str, int] | None = None,
        cat_embed_dim: int = 8,
    ) -> None:
        """Initialize the TabularEncoder.

        Args:
            num_features: Number of numerical/pre-encoded tabular input features.
            hidden_dim: Dimension of intermediate hidden layer.
            output_dim: Dimension of final tabular feature embedding output.
            dropout: Dropout probability for regularization.
            cat_cardinalities: Optional dictionary mapping categorical feature names
                to vocabulary size for categorical embedding lookup layers.
            cat_embed_dim: Embedding dimension for each categorical feature if cat_cardinalities is supplied.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.cat_cardinalities = cat_cardinalities or {}

        total_input_dim = num_features
        self.cat_embeddings = nn.ModuleDict()
        if self.cat_cardinalities:
            for cat_name, num_classes in self.cat_cardinalities.items():
                self.cat_embeddings[cat_name] = nn.Embedding(
                    num_embeddings=num_classes, embedding_dim=cat_embed_dim
                )
                total_input_dim += cat_embed_dim

        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        cat_x: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass for tabular feature data.

        Args:
            x: Input feature tensor of shape (batch_size, num_features).
            cat_x: Optional dictionary of categorical feature integer tensors,
                each of shape (batch_size,).

        Returns:
            torch.Tensor: Encoded tabular embedding tensor of shape (batch_size, output_dim).
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor of shape (batch_size, num_features), "
                f"got tensor with dimension {x.dim()} and shape {tuple(x.shape)}."
            )
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} input features, but received shape {tuple(x.shape)}."
            )

        inputs = [x]

        if self.cat_cardinalities and cat_x is not None:
            for cat_name in self.cat_cardinalities:
                if cat_name in cat_x:
                    cat_idx = cat_x[cat_name]
                    embed_layer = self.cat_embeddings[cat_name]
                    cat_emb = embed_layer(cat_idx)
                    inputs.append(cat_emb)

        if len(inputs) > 1:
            combined_x = torch.cat(inputs, dim=1)
        else:
            combined_x = x

        out: torch.Tensor = self.net(combined_x)
        return out

"""Multimodal Neural Fusion Network Model for MedShield FL.

Fuses ECG time-series representations, clinical text transformer embeddings,
and lifestyle tabular metrics into a diagnostic risk probability score for heart disease.
"""

from __future__ import annotations

try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    import torch.nn as nn
    _TorchBase = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

    class _TorchBase:  # type: ignore[no-redef]
        """Sentinel base class used when PyTorch is not installed."""


class GNNMultimodalFusion(_TorchBase):
    """Concatenation & Deep Neural Fusion Network for multimodal diagnostic features.

    Attributes:
        ecg_dim: Input dimension of ECG signal embeddings (default 128).
        text_dim: Input dimension of clinical text embeddings (default 128).
        tab_dim: Input dimension of tabular lifestyle feature embeddings (default 64).
        fusion_head: Sequential feed-forward neural network mapping concatenated features to diagnostic logits.
    """

    def __init__(
        self,
        ecg_dim: int = 128,
        text_dim: int = 128,
        tab_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Initialize GNNMultimodalFusion network.

        Args:
            ecg_dim: Feature vector dimension of ECG embeddings.
            text_dim: Feature vector dimension of clinical text embeddings.
            tab_dim: Feature vector dimension of tabular feature embeddings.
            num_classes: Output target classes (default 2 for Low Risk vs. High Risk).
            dropout: Regularization dropout probability.
        """
        super().__init__()
        self.ecg_dim = ecg_dim
        self.text_dim = text_dim
        self.tab_dim = tab_dim
        self.num_classes = num_classes

        total_dim = ecg_dim + text_dim + tab_dim  # 320d by default

        self.fusion_head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        ecg_embed: torch.Tensor,
        text_embed: torch.Tensor,
        tab_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to compute classification logits from multimodal embeddings.

        Args:
            ecg_embed: ECG feature tensor of shape (batch_size, ecg_dim).
            text_embed: Text feature tensor of shape (batch_size, text_dim).
            tab_embed: Tabular feature tensor of shape (batch_size, tab_dim).

        Returns:
            torch.Tensor: Classification logits tensor of shape (batch_size, num_classes).
        """
        if ecg_embed.dim() != 2 or ecg_embed.size(1) != self.ecg_dim:
            raise ValueError(
                f"Expected ecg_embed shape (batch_size, {self.ecg_dim}), got {tuple(ecg_embed.shape)}."
            )
        if text_embed.dim() != 2 or text_embed.size(1) != self.text_dim:
            raise ValueError(
                f"Expected text_embed shape (batch_size, {self.text_dim}), got {tuple(text_embed.shape)}."
            )
        if tab_embed.dim() != 2 or tab_embed.size(1) != self.tab_dim:
            raise ValueError(
                f"Expected tab_embed shape (batch_size, {self.tab_dim}), got {tuple(tab_embed.shape)}."
            )
        if not (ecg_embed.size(0) == text_embed.size(0) == tab_embed.size(0)):
            raise ValueError(
                f"Batch size mismatch across modalities: ecg={ecg_embed.size(0)}, "
                f"text={text_embed.size(0)}, tab={tab_embed.size(0)}."
            )

        # Concatenate modal embeddings: (batch_size, ecg_dim + text_dim + tab_dim)
        fused = torch.cat([ecg_embed, text_embed, tab_embed], dim=1)
        logits: torch.Tensor = self.fusion_head(fused)
        return logits

    def predict_proba(
        self,
        ecg_embed: torch.Tensor,
        text_embed: torch.Tensor,
        tab_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized diagnostic risk probability scores.

        Args:
            ecg_embed: ECG embedding tensor.
            text_embed: Text embedding tensor.
            tab_embed: Tabular embedding tensor.

        Returns:
            torch.Tensor: Normalized class probabilities of shape (batch_size, num_classes).
        """
        logits = self.forward(ecg_embed, text_embed, tab_embed)
        proba: torch.Tensor = torch.softmax(logits, dim=1)
        return proba

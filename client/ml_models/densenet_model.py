"""1D DenseNet Model Architecture for 12-lead ECG Signal Feature Extraction."""

from __future__ import annotations

try:
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


class DenseLayer1D(_TorchBase):
    """Single 1D Dense Layer with BatchNorm, ReLU, and 1D Convolution."""

    def __init__(self, in_channels: int, growth_rate: int = 32) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)


class DenseBlock1D(_TorchBase):
    """Dense Block containing multiple 1D dense layers."""

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int = 32) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer1D(channels, growth_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition1D(_TorchBase):
    """Transition layer between Dense Blocks for spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.bn(x)))
        return self.pool(out)


class DenseNet1D(_TorchBase):
    """1D DenseNet Model for ECG Waveform Classification and Feature Embedding.

    Attributes:
        in_channels: Number of input ECG leads (default 12).
        embedding_dim: Output dimension of feature embedding (default 128).
        num_classes: Number of diagnostic target classes (default 2).
    """

    def __init__(
        self,
        in_channels: int = 12,
        embedding_dim: int = 128,
        num_classes: int = 2,
        growth_rate: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Initial Conv
        self.conv1 = nn.Conv1d(
            in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Dense Blocks
        self.block1 = DenseBlock1D(num_layers=3, in_channels=32, growth_rate=growth_rate)
        self.trans1 = Transition1D(self.block1.out_channels, 48)

        self.block2 = DenseBlock1D(num_layers=3, in_channels=48, growth_rate=growth_rate)
        self.trans2 = Transition1D(self.block2.out_channels, 64)

        # Global Pooling and Embedding Projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(64, embedding_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract classification logits from ECG waveform signals of shape (batch, 12, length)."""
        out = self.pool1(self.relu(self.bn1(self.conv1(x))))
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.global_pool(out).squeeze(-1)
        embed = self.fc_embed(out)
        logits = self.classifier(embed)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128d ECG feature embedding vector."""
        out = self.pool1(self.relu(self.bn1(self.conv1(x))))
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.global_pool(out).squeeze(-1)
        embed = self.fc_embed(out)
        return embed

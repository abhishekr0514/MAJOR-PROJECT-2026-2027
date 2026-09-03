"""ECG Bidirectional LSTM Neural Network Model for MedShield FL.

Processes 12-lead ECG time-series signals to extract temporal representations
for multimodal heart disease diagnostic prediction.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    _TorchBase = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

    class _TorchBase:  # type: ignore[no-redef]
        """Sentinel base class used when PyTorch is not installed."""


class TemporalAttentionPooling(nn.Module):
    """Temporal self-attention pooling layer for sequential ECG features."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, feature_dim).
        Returns:
            Tensor of shape (batch_size, feature_dim).
        """
        # (batch_size, seq_len, 1)
        weights = self.attn(x)
        weights = F.softmax(weights, dim=1)
        # Weighted sum over seq_len
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class ECGBiLSTM(_TorchBase):
    """Multi-scale 1D Convolutional + Bidirectional LSTM + Attention Pooling network for 12-lead ECG signals.

    Attributes:
        conv_block: Multi-layer 1D convolutional network for local multi-scale temporal feature extraction.
        lstm: Bidirectional LSTM layers for long-range sequential dynamics.
        attn_pool: Temporal self-attention pooling mechanism.
        fc: Linear projection layer to map sequence representations to output embedding space.
    """

    def __init__(
        self,
        in_channels: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 256,
        conv_out_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the ECGBiLSTM module.

        Args:
            in_channels: Number of input signal leads/channels (default 12 for 12-lead ECG).
            hidden_dim: Hidden dimension size for each direction of the LSTM.
            num_layers: Number of stacked LSTM layers.
            embedding_dim: Output dimension of the ECG feature embedding vector (default 256).
            conv_out_channels: Number of output feature channels for 1D Conv layers.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.conv_out_channels = conv_out_channels
        self.dropout_rate = dropout

        # Multi-scale 1D Convolutional Feature Extractor
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_out_channels // 2,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(conv_out_channels // 2),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv_out_channels // 2,
                out_channels=conv_out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm1d(conv_out_channels),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(conv_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Bidirectional LSTM / Recurrent Layer
        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Temporal Attention Pooling
        self.attn_pool = TemporalAttentionPooling(feature_dim=hidden_dim * 2)

        # Feature Projection Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Forward pass for 12-lead ECG signals.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length).
            verbose: Enable diagnostic logging.

        Returns:
            torch.Tensor: ECG feature embedding tensor of shape (batch_size, embedding_dim).
        """
        if verbose:
            print(f"[ECGBiLSTM] Input shape: {tuple(x.shape)}")

        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor of shape (batch_size, in_channels, sequence_length), "
                f"got tensor with dimension {x.dim()} and shape {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but received shape {tuple(x.shape)}."
            )
        if x.size(2) != 1000:
            raise ValueError(
                f"Expected sequence length 1000, but received shape {tuple(x.shape)}."
            )

        # 1D Multi-scale Conv feature extraction
        c1 = self.conv_block1(x)
        c2 = self.conv_block2(c1)
        x_conv = self.conv_block3(c2)  # (batch_size, conv_out_channels, seq_len_conv)

        if verbose:
            print(f"[ECGBiLSTM] Multi-scale Conv output shape: {tuple(x_conv.shape)}")

        # Transpose to (batch_size, seq_len_conv, conv_out_channels) for LSTM
        x_lstm_in = x_conv.transpose(1, 2)

        # LSTM forward pass: output shape (batch_size, seq_len_conv, hidden_dim * 2)
        lstm_out, _ = self.lstm(x_lstm_in)

        if verbose:
            print(f"[ECGBiLSTM] BiLSTM output shape: {tuple(lstm_out.shape)}")

        # Temporal Attention Pooling over sequence length: (batch_size, hidden_dim * 2)
        pooled_out = self.attn_pool(lstm_out)

        if verbose:
            print(
                f"[ECGBiLSTM] Temporal Attention Pooled shape: {tuple(pooled_out.shape)}"
            )

        # Projection to output embedding dimension: (batch_size, embedding_dim)
        # Bypass BatchNorm if batch_size == 1 during evaluation to avoid errors
        if x.size(0) == 1 and self.training:
            self.fc[1].eval()
            embed = self.fc(pooled_out)
            self.fc[1].train()
        elif x.size(0) == 1:
            embed = self.fc(pooled_out)
        else:
            embed = self.fc(pooled_out)

        if verbose:
            print(f"[ECGBiLSTM] Final embedding output shape: {tuple(embed.shape)}")
        return embed

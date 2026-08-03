"""ECG Bidirectional LSTM Neural Network Model for MedShield FL.

Processes 12-lead ECG time-series signals to extract temporal representations
for multimodal heart disease diagnostic prediction.
"""

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


class ECGBiLSTM(_TorchBase):
    """1D-Convolutional + Bidirectional LSTM network for 12-lead ECG signals.

    Attributes:
        conv1d: Sequential 1D convolutional layers for local temporal feature extraction.
        lstm: Bidirectional LSTM layers for sequential temporal dynamics.
        fc: Linear projection layer to map sequence representations to output embedding space.
    """

    def __init__(
        self,
        in_channels: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 128,
        conv_out_channels: int = 32,
    ) -> None:
        """Initialize the ECGBiLSTM module.

        Args:
            in_channels: Number of input signal leads/channels (default 12 for 12-lead ECG).
            hidden_dim: Hidden dimension size for each direction of the LSTM.
            num_layers: Number of stacked LSTM layers.
            embedding_dim: Output dimension of the ECG feature embedding vector.
            conv_out_channels: Number of output feature maps for the initial 1D Conv layer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 12-lead ECG signals.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: ECG feature embedding tensor of shape (batch_size, embedding_dim).
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor of shape (batch_size, in_channels, sequence_length), "
                f"got tensor with dimension {x.dim()} and shape {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but received shape {tuple(x.shape)}."
            )

        # 1D Conv extraction: (batch_size, 32, seq_len_conv)
        x_conv = self.conv1d(x)

        # Transpose to (batch_size, seq_len_conv, 32) for PyTorch LSTM
        x_lstm_in = x_conv.transpose(1, 2)

        # LSTM forward pass: output shape (batch_size, seq_len_conv, hidden_dim * 2)
        out, _ = self.lstm(x_lstm_in)

        # Extract last time-step hidden representation: (batch_size, hidden_dim * 2)
        last_timestep_out = out[:, -1, :]

        # Project to output embedding dimension: (batch_size, embedding_dim)
        embed: torch.Tensor = self.fc(last_timestep_out)
        return embed

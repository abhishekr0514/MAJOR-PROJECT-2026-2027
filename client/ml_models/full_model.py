"""End-to-End Multimodal Diagnostic Neural Network for MedShield FL.

Combines ECG BiLSTM, Clinical Text BERT, Tabular Encoder, and GNN Neural Fusion Head
into a unified diagnostic prediction architecture for heart disease risk.
"""

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

from client.ml_models.gnn_fusion import GNNMultimodalFusion
from client.ml_models.lstm_model import ECGBiLSTM
from client.ml_models.tabular_model import TabularEncoder
from client.ml_models.text_model import BioClinicalBERTFeatureExtractor


class MedShieldDiagnosticNet(_TorchBase):
    """Unified MedShield Diagnostic Network combining all 3 clinical data modalities.

    Attributes:
        ecg_net: ECGBiLSTM extractor for 12-lead time-series signals.
        text_net: BioClinicalBERTFeatureExtractor for clinical text notes.
        tab_net: TabularEncoder for numerical and lifestyle metrics.
        fusion_net: GNNMultimodalFusion head for diagnostic risk prediction.
    """

    def __init__(
        self,
        ecg_channels: int = 12,
        tab_features: int = 10,
        text_output_dim: int = 128,
        ecg_output_dim: int = 128,
        tab_output_dim: int = 64,
        num_classes: int = 2,
    ) -> None:
        """Initialize MedShieldDiagnosticNet.

        Args:
            ecg_channels: Number of ECG channels (default 12).
            tab_features: Number of tabular lifestyle metrics (default 10).
            text_output_dim: Output dimension of text embedding (default 128).
            ecg_output_dim: Output dimension of ECG embedding (default 128).
            tab_output_dim: Output dimension of tabular embedding (default 64).
            num_classes: Number of diagnostic target risk classes (default 2).
        """
        super().__init__()
        self.ecg_net = ECGBiLSTM(
            in_channels=ecg_channels,
            embedding_dim=ecg_output_dim,
        )
        self.text_net = BioClinicalBERTFeatureExtractor(
            output_dim=text_output_dim,
        )
        self.tab_net = TabularEncoder(
            num_features=tab_features,
            output_dim=tab_output_dim,
        )
        self.fusion_net = GNNMultimodalFusion(
            ecg_dim=ecg_output_dim,
            text_dim=text_output_dim,
            tab_dim=tab_output_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_data: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for multimodal patient data.

        Args:
            ecg_signal: ECG signal tensor of shape (batch_size, 12, seq_len).
            input_ids: Text token ID tensor of shape (batch_size, text_seq_len).
            attention_mask: Text attention mask tensor of shape (batch_size, text_seq_len).
            tabular_data: Tabular feature matrix of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Binary risk logits tensor of shape (batch_size, num_classes).
        """
        ecg_emb = self.ecg_net(ecg_signal)
        text_emb = self.text_net(input_ids, attention_mask)
        tab_emb = self.tab_net(tabular_data)

        logits: torch.Tensor = self.fusion_net(ecg_emb, text_emb, tab_emb)
        return logits

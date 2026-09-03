"""End-to-End Multimodal Diagnostic Neural Network for MedShield FL.

Combines ECGBiLSTM, BioClinicalBERT, TabularEncoder, and GNNFusion into a unified
diagnostic prediction model, with optimization for missing modalities.
"""

import torch
from torch import nn

from client.ml_models.gnn_fusion import GNNMultimodalFusion, SimpleMultimodalFusion
from client.ml_models.lstm_model import ECGBiLSTM
from client.ml_models.tabular_model import TabularEncoder
from client.ml_models.text_model import BioClinicalBERTFeatureExtractor


class MedShieldDiagnosticNet(nn.Module):
    """Unified MedShield Diagnostic Network combining all 3 clinical data modalities."""

    def __init__(
        self,
        ecg_channels: int = 12,
        tab_features: int = 2,
        text_output_dim: int = 128,
        ecg_output_dim: int = 256,
        tab_output_dim: int = 64,
        num_classes: int = 2,
        fusion_type: str = "gat",
    ) -> None:
        """Initialize MedShieldDiagnosticNet.

        Args:
            ecg_channels: Number of ECG channels (default 12 for 12-lead ECG).
            tab_features: Number of numerical tabular features (default 2 for Age, Sex).
            text_output_dim: Output dimension of text features projection.
            ecg_output_dim: Output dimension of ECG features projection (default 256).
            tab_output_dim: Output dimension of tabular features projection.
            num_classes: Target classification output nodes.
            fusion_type: Type of multimodal fusion ("gat" or "simple").
        """
        super().__init__()
        self.ecg_channels = ecg_channels
        self.tab_features = tab_features
        self.text_output_dim = text_output_dim
        self.ecg_output_dim = ecg_output_dim
        self.tab_output_dim = tab_output_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type.lower()

        # Modality networks
        self.ecg_net = ECGBiLSTM(
            in_channels=ecg_channels,
            embedding_dim=ecg_output_dim,
        )
        self.text_net = BioClinicalBERTFeatureExtractor(
            output_dim=text_output_dim,
            lazy_load=True,
        )
        self.tab_net = TabularEncoder(
            num_features=tab_features,
            output_dim=tab_output_dim,
        )

        # Multimodal Fusion Layer
        if self.fusion_type == "simple":
            self.fusion_net: nn.Module = SimpleMultimodalFusion(
                ecg_dim=ecg_output_dim,
                text_dim=text_output_dim,
                tab_dim=tab_output_dim,
                num_classes=num_classes,
            )
        else:
            self.fusion_net = GNNMultimodalFusion(
                ecg_dim=ecg_output_dim,
                text_dim=text_output_dim,
                tab_dim=tab_output_dim,
                num_classes=num_classes,
                common_dim=128,
            )

    def forward(
        self,
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_data: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Forward pass for multimodal patient data.

        Args:
            ecg_signal: ECG signal tensor of shape (batch_size, 12, 1000).
            input_ids: Tokenized ID tensor of shape (batch_size, seq_len).
            attention_mask: Attention masks tensor.
            tabular_data: Tabular feature matrix shape (batch_size, 2).
            modality_mask: Optional mask list/tensor of shape (3,) or (batch_size, 3).
                           [1.0, 0.0, 1.0] implies ECG and Tabular active.
            verbose: Enables diagnostic logs printing.

        Returns:
            torch.Tensor: Combined target class logits (batch_size, num_classes).
        """
        # ECG Forward
        ecg_emb = self.ecg_net(ecg_signal, verbose=verbose)

        # Retrieve text mask state to bypass heavy BERT operations when masked
        is_text_masked = False
        if modality_mask is not None:
            if modality_mask.dim() == 1:
                is_text_masked = modality_mask[1] == 0.0
            elif modality_mask.dim() == 2:
                is_text_masked = (modality_mask[:, 1] == 0.0).all()

        if is_text_masked:
            # Bypass BERT completely
            text_emb = torch.zeros(
                ecg_signal.size(0), self.text_output_dim, device=ecg_signal.device
            )
        else:
            text_emb = self.text_net(input_ids, attention_mask, verbose=verbose)

        # Tabular Forward
        tab_emb = self.tab_net(tabular_data, verbose=verbose)

        if verbose:
            print("\n====================================================")
            print("MODEL PIPELINE")
            print("====================================================")
            print("REAL DATA\n")
            print("ECG:")
            print("Input:", tuple(ecg_signal.shape))
            print("Output:", tuple(ecg_emb.shape))
            print("\nTABULAR:")
            print("Input:", tuple(tabular_data.shape))
            print("Output:", tuple(tab_emb.shape))
            print("\nTEXT:")
            if is_text_masked:
                print("Unavailable for PTB-XL (Masked)")
            else:
                print("Input:", tuple(input_ids.shape))
                print("Output:", tuple(text_emb.shape))
            print("====================================================\n")

        # GNN Fusion Forward
        logits: torch.Tensor = self.fusion_net(
            ecg_emb, text_emb, tab_emb, modality_mask=modality_mask, verbose=verbose
        )
        return logits

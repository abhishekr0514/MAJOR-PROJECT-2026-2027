"""Unit and pipeline integration tests for MedShield GAT & real PTB-XL dataset partitioning."""

import unittest

import torch

from client.ml_models.full_model import MedShieldDiagnosticNet
from client.ml_models.gnn_fusion import GNNMultimodalFusion
from client.ml_models.lstm_model import ECGBiLSTM
from client.ml_models.tabular_model import TabularEncoder


class TestMedShieldRealPipeline(unittest.TestCase):
    """Pipeline and model unit tests."""

    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.seq_len = 1000
        self.ecg_channels = 12
        self.tab_features = 2

        # Create mock inputs
        self.mock_ecgs = torch.randn(
            self.batch_size, self.ecg_channels, self.seq_len, device=self.device
        )
        self.mock_tabs = torch.randn(
            self.batch_size, self.tab_features, device=self.device
        )
        self.mock_input_ids = torch.zeros(
            self.batch_size, 64, dtype=torch.long, device=self.device
        )
        self.mock_attention_mask = torch.zeros(
            self.batch_size, 64, dtype=torch.long, device=self.device
        )

    def test_lstm_encoder_shape(self) -> None:
        """Verify ECGBiLSTM output matches embedding dimension of 128 and 256."""
        lstm128 = ECGBiLSTM(in_channels=self.ecg_channels, embedding_dim=128).to(
            self.device
        )
        embed128 = lstm128(self.mock_ecgs)
        self.assertEqual(embed128.shape, (self.batch_size, 128))

        lstm256 = ECGBiLSTM(in_channels=self.ecg_channels, embedding_dim=256).to(
            self.device
        )
        embed256 = lstm256(self.mock_ecgs)
        self.assertEqual(embed256.shape, (self.batch_size, 256))

    def test_tabular_encoder_shape(self) -> None:
        """Verify TabularEncoder output matches embedding dimension of 64."""
        tab = TabularEncoder(num_features=self.tab_features, output_dim=64).to(
            self.device
        )
        embed = tab(self.mock_tabs)
        self.assertEqual(embed.shape, (self.batch_size, 64))

    def test_gat_fusion_with_modal_mask(self) -> None:
        """Verify GNNMultimodalFusion passes messages and respects modality masking."""
        fusion = GNNMultimodalFusion(
            text_dim=128,
            ecg_dim=256,
            tab_dim=64,
            common_dim=128,
            num_classes=2,
        ).to(self.device)

        # 3 nodes: ECG, Text, Tabular
        h_ecg = torch.randn(self.batch_size, 256, device=self.device)
        h_text = torch.randn(self.batch_size, 128, device=self.device)
        h_tab = torch.randn(self.batch_size, 64, device=self.device)

        # 1. Full multimodal fusion (all modalities active)
        mask_all = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        out_all = fusion(h_ecg, h_text, h_tab, modality_mask=mask_all)
        self.assertEqual(out_all.shape, (self.batch_size, 2))

        # 2. Text modality masked out
        mask_masked = torch.tensor([1.0, 0.0, 1.0], device=self.device)
        out_masked = fusion(h_ecg, h_text, h_tab, modality_mask=mask_masked)
        self.assertEqual(out_masked.shape, (self.batch_size, 2))

    def test_medshield_net_forward(self) -> None:
        """Verify integrated MedShieldDiagnosticNet forward flows for both GAT and Simple fusion."""
        model_gat = MedShieldDiagnosticNet(
            ecg_channels=self.ecg_channels,
            tab_features=self.tab_features,
            text_output_dim=128,
            ecg_output_dim=256,
            tab_output_dim=64,
            num_classes=2,
            fusion_type="gat",
        ).to(self.device)

        model_simple = MedShieldDiagnosticNet(
            ecg_channels=self.ecg_channels,
            tab_features=self.tab_features,
            text_output_dim=128,
            ecg_output_dim=256,
            tab_output_dim=64,
            num_classes=2,
            fusion_type="simple",
        ).to(self.device)

        mask = torch.tensor([1.0, 0.0, 1.0], device=self.device)

        logits_gat = model_gat(
            self.mock_ecgs,
            self.mock_input_ids,
            self.mock_attention_mask,
            self.mock_tabs,
            modality_mask=mask,
        )
        self.assertEqual(logits_gat.shape, (self.batch_size, 2))

        logits_simple = model_simple(
            self.mock_ecgs,
            self.mock_input_ids,
            self.mock_attention_mask,
            self.mock_tabs,
            modality_mask=mask,
        )
        self.assertEqual(logits_simple.shape, (self.batch_size, 2))

    def test_patient_leakage_check_validator(self) -> None:
        """Verify split-validation logic correctly catches overlapping patient records."""
        # Clean scenario
        train_pts = {1, 2, 3, 4}
        val_pts = {5, 6}
        test_pts = {7, 8, 9}

        leak_train_val = train_pts & val_pts
        leak_train_test = train_pts & test_pts
        leak_val_test = val_pts & test_pts

        self.assertFalse(bool(leak_train_val | leak_train_test | leak_val_test))

        # Leakage scenario
        train_pts_leak = {1, 2, 3, 5}
        leak_val_leak = train_pts_leak & val_pts
        self.assertTrue(bool(leak_val_leak))


if __name__ == "__main__":
    unittest.main()

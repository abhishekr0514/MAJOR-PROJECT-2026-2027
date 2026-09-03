"""Unit tests for PyTorch ML models in client/ml_models/."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed in this environment")

from client.ml_models.full_model import MedShieldDiagnosticNet
from client.ml_models.gnn_fusion import GNNMultimodalFusion
from client.ml_models.lstm_model import ECGBiLSTM
from client.ml_models.tabular_model import TabularEncoder
from client.ml_models.text_model import (
    BioClinicalBERTFeatureExtractor,
    ClinicalTextBERT,
)


def test_ecg_bilstm_forward_shape():
    """Test ECGBiLSTM forward pass output shape for 12-lead ECG signal inputs."""
    model = ECGBiLSTM(in_channels=12, hidden_dim=64, embedding_dim=128)
    model.eval()

    x = torch.randn(4, 12, 1000)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 128), f"Expected output shape (4, 128), got {out.shape}"


def test_ecg_bilstm_single_sample_shape():
    """Test ECGBiLSTM with a single sample batch (batch_size=1)."""
    model = ECGBiLSTM(in_channels=12, hidden_dim=64, embedding_dim=128)
    model.eval()

    x = torch.randn(1, 12, 1000)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 128), f"Expected output shape (1, 128), got {out.shape}"


def test_ecg_bilstm_invalid_input_dim():
    """Test ECGBiLSTM raises ValueError on invalid input dimensions."""
    model = ECGBiLSTM(in_channels=12)

    x = torch.randn(12, 1000)
    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        model(x)

    x_wrong_ch = torch.randn(2, 8, 1000)
    with pytest.raises(ValueError, match="Expected 12 input channels"):
        model(x_wrong_ch)


def test_tabular_encoder_forward_shape():
    """Test TabularEncoder forward pass output shape for tabular feature matrix."""
    model = TabularEncoder(num_features=10, hidden_dim=64, output_dim=64)
    model.eval()

    x = torch.randn(4, 10)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 64), f"Expected output shape (4, 64), got {out.shape}"


def test_tabular_encoder_single_sample():
    """Test TabularEncoder with batch_size=1."""
    model = TabularEncoder(num_features=10, hidden_dim=64, output_dim=64)
    model.eval()

    x = torch.randn(1, 10)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 64), f"Expected output shape (1, 64), got {out.shape}"


def test_tabular_encoder_categorical_embeddings():
    """Test TabularEncoder with optional categorical embedding lookups."""
    cat_specs = {"gender": 2, "cp_type": 4}
    model = TabularEncoder(
        num_features=8,
        hidden_dim=64,
        output_dim=64,
        cat_cardinalities=cat_specs,
        cat_embed_dim=4,
    )
    model.eval()

    x_num = torch.randn(4, 8)
    cat_x = {
        "gender": torch.tensor([0, 1, 0, 1]),
        "cp_type": torch.tensor([0, 2, 1, 3]),
    }
    with torch.no_grad():
        out = model(x_num, cat_x=cat_x)

    assert out.shape == (4, 64), f"Expected output shape (4, 64), got {out.shape}"


def test_tabular_encoder_invalid_input_dim():
    """Test TabularEncoder raises ValueError on invalid input dimensions."""
    model = TabularEncoder(num_features=10)

    x_3d = torch.randn(4, 10, 1)
    with pytest.raises(ValueError, match="Expected 2D input tensor"):
        model(x_3d)

    x_wrong_features = torch.randn(4, 5)
    with pytest.raises(ValueError, match="Expected 10 input features"):
        model(x_wrong_features)


def test_bio_clinical_bert_default_shape():
    """Test BioClinicalBERTFeatureExtractor forward pass with default 768d embedding."""
    model = BioClinicalBERTFeatureExtractor(output_dim=768)
    model.eval()

    input_ids = torch.randint(0, 1000, (3, 32))
    attn_mask = torch.ones(3, 32)
    with torch.no_grad():
        out = model(input_ids, attn_mask)

    assert out.shape == (3, 768), f"Expected output shape (3, 768), got {out.shape}"


def test_bio_clinical_bert_projected_shape():
    """Test BioClinicalBERTFeatureExtractor with 128d projection head."""
    model = ClinicalTextBERT(output_dim=128)
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 16))
    attn_mask = torch.ones(2, 16)
    with torch.no_grad():
        out = model(input_ids, attn_mask)

    assert out.shape == (2, 128), f"Expected output shape (2, 128), got {out.shape}"


def test_gnn_fusion_forward_shape():
    """Test GNNMultimodalFusion forward pass output logits and probabilities."""
    fusion_model = GNNMultimodalFusion(
        ecg_dim=128, text_dim=128, tab_dim=64, num_classes=2
    )
    fusion_model.eval()

    ecg_emb = torch.randn(4, 128)
    text_emb = torch.randn(4, 128)
    tab_emb = torch.randn(4, 64)

    with torch.no_grad():
        logits = fusion_model(ecg_emb, text_emb, tab_emb)
        proba = fusion_model.predict_proba(ecg_emb, text_emb, tab_emb)

    assert logits.shape == (4, 2), f"Expected logits shape (4, 2), got {logits.shape}"
    assert proba.shape == (4, 2), f"Expected proba shape (4, 2), got {proba.shape}"
    assert torch.allclose(proba.sum(dim=1), torch.ones(4)), (
        "Probabilities must sum to 1.0"
    )


def test_gnn_fusion_invalid_shapes():
    """Test GNNMultimodalFusion input dimension validation."""
    fusion_model = GNNMultimodalFusion(ecg_dim=128, text_dim=128, tab_dim=64)

    ecg_wrong = torch.randn(4, 64)  # 64 instead of 128
    text_emb = torch.randn(4, 128)
    tab_emb = torch.randn(4, 64)

    with pytest.raises(ValueError, match="Expected ecg_embed shape"):
        fusion_model(ecg_wrong, text_emb, tab_emb)


def test_full_diagnostic_net_forward_pass():
    """Test end-to-end MedShieldDiagnosticNet forward pass with all 3 modalities."""
    model = MedShieldDiagnosticNet()
    model.eval()

    ecg = torch.randn(2, 12, 1000)
    input_ids = torch.randint(0, 1000, (2, 32))
    attn_mask = torch.ones(2, 32)
    tab = torch.randn(2, 2)

    with torch.no_grad():
        logits = model(ecg, input_ids, attn_mask, tab)

    assert logits.shape == (2, 2), (
        f"Expected binary logits shape (2, 2), got {logits.shape}"
    )

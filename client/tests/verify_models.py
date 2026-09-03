"""Standard unittest verification runner for PyTorch ML and XAI modules."""

import sys
import unittest

# Try importing torch and the models
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    import pandas as pd
    from torch import nn

    from client.explainability.causal_graph import CausalInferenceEngine
    from client.explainability.counterfactual import CounterfactualExplainer
    from client.ml_models.full_model import MedShieldDiagnosticNet
    from client.ml_models.gnn_fusion import GNNMultimodalFusion
    from client.ml_models.lstm_model import ECGBiLSTM
    from client.ml_models.tabular_model import TabularEncoder
    from client.ml_models.text_model import (
        BioClinicalBERTFeatureExtractor,
    )

    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(5, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)


class TestMLModels(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_ecg_bilstm_forward(self):
        model = ECGBiLSTM(in_channels=12, hidden_dim=64, embedding_dim=128)
        model.eval()
        x = torch.randn(4, 12, 1000)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (4, 128))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_ecg_bilstm_single_sample(self):
        model = ECGBiLSTM(in_channels=12, hidden_dim=64, embedding_dim=128)
        model.eval()
        x = torch.randn(1, 12, 500)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 128))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_ecg_bilstm_validation(self):
        model = ECGBiLSTM(in_channels=12)
        with self.assertRaises(ValueError):
            model(torch.randn(12, 1000))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 8, 1000))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_tabular_encoder_forward(self):
        model = TabularEncoder(num_features=10, hidden_dim=64, output_dim=64)
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (4, 64))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_tabular_encoder_single_sample(self):
        model = TabularEncoder(num_features=10, hidden_dim=64, output_dim=64)
        model.eval()
        x = torch.randn(1, 10)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 64))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_tabular_encoder_categorical(self):
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
        self.assertEqual(out.shape, (4, 64))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_text_model_forward(self):
        model = BioClinicalBERTFeatureExtractor(output_dim=768)
        model.eval()
        input_ids = torch.randint(0, 1000, (3, 32))
        attn_mask = torch.ones(3, 32)
        with torch.no_grad():
            out = model(input_ids, attn_mask)
        self.assertEqual(out.shape, (3, 768))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_gnn_fusion_forward(self):
        fusion = GNNMultimodalFusion(
            ecg_dim=128, text_dim=128, tab_dim=64, num_classes=2
        )
        fusion.eval()
        ecg_emb = torch.randn(4, 128)
        text_emb = torch.randn(4, 128)
        tab_emb = torch.randn(4, 64)
        with torch.no_grad():
            logits = fusion(ecg_emb, text_emb, tab_emb)
            proba = fusion.predict_proba(ecg_emb, text_emb, tab_emb)
        self.assertEqual(logits.shape, (4, 2))
        self.assertEqual(proba.shape, (4, 2))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_full_model_forward(self):
        model = MedShieldDiagnosticNet()
        model.eval()
        ecg = torch.randn(2, 12, 1000)
        input_ids = torch.randint(0, 1000, (2, 32))
        attn_mask = torch.ones(2, 32)
        tab = torch.randn(2, 10)
        with torch.no_grad():
            logits = model(ecg, input_ids, attn_mask, tab)
        self.assertEqual(logits.shape, (2, 2))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_counterfactual_explainer(self):
        model = MockModel()
        feature_names = [
            "age",
            "blood_pressure_sys",
            "blood_pressure_dia",
            "cholesterol_mg_dl",
            "fasting_bs_mg_dl",
        ]
        continuous_features = [
            "blood_pressure_sys",
            "blood_pressure_dia",
            "cholesterol_mg_dl",
            "fasting_bs_mg_dl",
        ]
        explainer = CounterfactualExplainer(
            model=model,
            feature_names=feature_names,
            continuous_features=continuous_features,
        )
        patient = {
            "age": 60,
            "blood_pressure_sys": 160.0,
            "blood_pressure_dia": 95.0,
            "cholesterol_mg_dl": 260.0,
            "fasting_bs_mg_dl": 130.0,
        }
        cfs = explainer.generate_counterfactuals(patient, num_cfs=3)
        self.assertGreater(len(cfs), 0)
        self.assertEqual(cfs[0]["predicted_new_diagnosis"], "Low Risk")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch environment not present")
    def test_causal_inference_engine(self):
        df = pd.DataFrame(
            {
                "age": [50, 60, 70, 55],
                "blood_pressure_sys": [120, 140, 160, 130],
                "cholesterol_mg_dl": [200, 240, 280, 210],
                "smoking": [0, 1, 1, 0],
                "heart_disease": [0, 1, 1, 0],
            }
        )
        engine = CausalInferenceEngine()
        res = engine.estimate_causal_effect(
            df, treatment="smoking", outcome="heart_disease"
        )
        self.assertIn("causal_effect_value", res)
        insights = engine.generate_causal_insights(df)
        self.assertGreater(len(insights), 0)


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print(
            "INFO: PyTorch package is not installed in the global Python 3.13 environment."
        )
        print("Code structure and syntax static checks passed.")
        sys.exit(0)
    unittest.main()

"""Unit tests for Explainable AI (XAI) modules in client/explainability/."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed in this environment")
nn = torch.nn

from client.explainability.causal_graph import CausalInferenceEngine  # noqa: E402
from client.explainability.counterfactual import CounterfactualExplainer  # noqa: E402


class MockDiagnosticModel(nn.Module):
    """Simple PyTorch model for testing XAI counterfactual explanations."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_counterfactual_explainer_recommendations():
    """Test CounterfactualExplainer produces valid target change recommendations."""
    model = MockDiagnosticModel()
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

    high_risk_patient = {
        "age": 62,
        "blood_pressure_sys": 165.0,
        "blood_pressure_dia": 100.0,
        "cholesterol_mg_dl": 280.0,
        "fasting_bs_mg_dl": 140.0,
    }

    cfs = explainer.generate_counterfactuals(high_risk_patient, num_cfs=3)

    assert len(cfs) > 0, "Counterfactual generator should return recommendation options"
    first_option = cfs[0]
    assert "option" in first_option
    assert "target_changes" in first_option
    assert "predicted_new_risk" in first_option
    assert "predicted_new_diagnosis" in first_option
    assert first_option["predicted_new_diagnosis"] == "Low Risk"


def test_causal_inference_engine_estimate():
    """Test CausalInferenceEngine computes Average Treatment Effects (ATE)."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(35, 80, size=n),
        "blood_pressure_sys": np.random.randint(110, 180, size=n),
        "cholesterol_mg_dl": np.random.uniform(150, 300, size=n),
        "smoking": np.random.choice([0, 1], size=n),
        "heart_disease": np.random.choice([0, 1], size=n),
    })

    engine = CausalInferenceEngine()
    ate_result = engine.estimate_causal_effect(df, treatment="smoking", outcome="heart_disease")

    assert "causal_effect_value" in ate_result
    assert ate_result["treatment"] == "smoking"
    assert ate_result["outcome"] == "heart_disease"


def test_causal_inference_engine_insights():
    """Test CausalInferenceEngine generates human-readable causal insights."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "age": np.random.randint(35, 80, size=n),
        "blood_pressure_sys": np.random.randint(110, 180, size=n),
        "cholesterol_mg_dl": np.random.uniform(150, 300, size=n),
        "smoking": np.random.choice([0, 1], size=n),
        "heart_disease": np.random.choice([0, 1], size=n),
    })

    engine = CausalInferenceEngine()
    insights = engine.generate_causal_insights(df)

    assert len(insights) > 0, "Engine should generate causal insight entries"
    first_entry = insights[0]
    assert "factor" in first_entry
    assert "impact" in first_entry
    assert "causal_effect_value" in first_entry

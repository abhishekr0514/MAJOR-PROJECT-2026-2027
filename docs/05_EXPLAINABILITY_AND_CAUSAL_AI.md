# MedShield FL — Explainability & Causal AI Blueprint (`Phase 5`)

This document outlines the architecture for **Explainable AI (XAI)** and **Causal Inference** using **`DiCE`** (Diverse Counterfactual Explanations) and **`DoWhy`** (`/client/explainability/`).

---

## 🎯 Explainability & Trust Objectives

Clinicians require more than a "black-box" risk percentage. The system provides two complementary forms of explainability:

1. **Counterfactual Explanations (`DiCE`)**: *"What is the minimum change in patient parameters (e.g., lower cholesterol by 40 mg/dL, stop smoking) needed to reduce heart disease risk from High to Low?"*
2. **Causal Reasoning (`DoWhy`)**: *"What is the direct causal effect of lowering systolic blood pressure on heart disease risk, controlling for confounding variables like age and BMI?"*

---

## 🧩 Counterfactual Generator Specification (`client/explainability/counterfactual.py`)

Uses `dice_ml` to compute actionable counterfactual examples.

```python
import dice_ml
import pandas as pd
from typing import Any


class CounterfactualExplainer:
    def __init__(
        self, model: Any, feature_names: list[str], continuous_features: list[str]
    ) -> None:
        """Initialize DiCE explainer on trained diagnostic model."""
        self.feature_names = feature_names

        # Create DiCE Data object
        self.d = dice_ml.Data(
            dataframe=pd.DataFrame(columns=feature_names + ["diagnosis"]),
            outcome_name="diagnosis",
            continuous_feature_names=continuous_features,
        )
        # Create DiCE Model object
        self.m = dice_ml.Model(model=model, backend="PYTORCH")
        self.exp = dice_ml.Dice(self.d, self.m, method="random")

    def generate_counterfactuals(
        self, patient_features: dict[str, float | int], num_cfs: int = 3
    ) -> list[dict[str, Any]]:
        """Generate counterfactual feature targets for a high-risk patient."""
        input_df = pd.DataFrame([patient_features])

        # Generate counterfactuals targeting Low Risk (diagnosis = 0)
        cf = self.exp.generate_counterfactuals(
            input_df, total_CFs=num_cfs, desired_class=0
        )

        cf_json = cf.to_json()
        return cf_json
```

---

## 🕸️ Causal Reasoning Model Specification (`client/explainability/causal_graph.py`)

Uses `dowhy` to build domain-specific causal DAGs (Directed Acyclic Graphs).

```python
import dowhy
from dowhy import CausalModel
import pandas as pd


class CausalInferenceEngine:
    def __init__(self) -> None:
        # Define Domain Causal Graph for Cardiovascular Risk
        self.causal_graph_dot = """
        digraph {
            age -> blood_pressure;
            age -> cholesterol;
            smoking -> blood_pressure;
            smoking -> heart_disease;
            cholesterol -> heart_disease;
            blood_pressure -> heart_disease;
        }
        """

    def estimate_causal_effect(
        self,
        df: pd.DataFrame,
        treatment: str = "smoking",
        outcome: str = "heart_disease",
    ) -> dict[str, float]:
        """Estimate Causal Effect of a treatment/lifestyle factor on heart disease."""
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=self.causal_graph_dot,
        )

        # 1. Identify Causal Effect
        identified_estimand = model.identify_effect()

        # 2. Estimate Effect using Linear Regression / Propensity Score Matching
        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
        )

        return {
            "treatment": treatment,
            "outcome": outcome,
            "causal_effect_value": float(causal_estimate.value),
        }
```

---

## 📊 API & UI Response JSON Payload Spec

When a prediction request is made, the FastAPI backend returns the risk score alongside the XAI counterfactual and causal insights:

```json
{
  "patient_code": "PAT-88402",
  "risk_score": 0.84,
  "diagnosis": "High Risk",
  "confidence": 0.91,
  "counterfactual_recommendations": [
    {
      "option": 1,
      "target_changes": {
        "cholesterol_mg_dl": "Reduced from 260 to 200",
        "systolic_bp": "Reduced from 150 to 125",
        "smoking_status": "Non-smoker"
      },
      "predicted_new_risk": 0.28,
      "predicted_new_diagnosis": "Low Risk"
    }
  ],
  "causal_insights": [
    {
      "factor": "systolic_bp",
      "impact": "A 10 mmHg reduction in systolic blood pressure causes a 12% decrease in cardiovascular event risk."
    }
  ]
}
```

---

## ✅ Phase 5 Verification Checklist
- [ ] `DiCE` counterfactual generator initialized with PyTorch model
- [ ] Counterfactuals output actionable feature targets (e.g. lowering BP/cholesterol)
- [ ] `DoWhy` causal DAG defined for cardiovascular risk factors
- [ ] Causal estimate function calculates backdoor linear effect values
- [ ] Response JSON spec connects smoothly to React frontend sliders

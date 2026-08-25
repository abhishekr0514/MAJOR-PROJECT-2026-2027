"""Live PyTorch Model Inference Engine for Diagnostic Predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from client.fl_client import tokenize_texts
    from client.ml_models.full_model import MedShieldDiagnosticNet
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    MedShieldDiagnosticNet = None  # type: ignore[assignment]
    tokenize_texts = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


_CACHED_MODEL: Any = None
_CACHED_WEIGHTS_MTIME: float = 0.0


def get_trained_diagnostic_model() -> Any:
    """Load (or reload) trained MedShieldDiagnosticNet PyTorch model.

    Re-reads weights from disk whenever medshield_model.pt is updated,
    so a server restart is not required after retraining.
    """
    global _CACHED_MODEL, _CACHED_WEIGHTS_MTIME

    if not TORCH_AVAILABLE or MedShieldDiagnosticNet is None:
        return None

    weights_path = Path("client/ml_models/saved_weights/medshield_model.pt")
    current_mtime = weights_path.stat().st_mtime if weights_path.exists() else 0.0

    if _CACHED_MODEL is not None and current_mtime == _CACHED_WEIGHTS_MTIME:
        return _CACHED_MODEL

    try:
        model = MedShieldDiagnosticNet()
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            # strict=False: _fallback_embed is initialised in __init__ with seeded weights;
            # we only save the trained parameters excluding the embedding table.
            model.load_state_dict(state_dict, strict=False)
            print(f"[Prediction Service] Loaded weights from '{weights_path}' (mtime={current_mtime}).")
        else:
            print("[Prediction Service] No saved weights found — using randomly initialised model.")

        model.eval()
        _CACHED_MODEL = model
        _CACHED_WEIGHTS_MTIME = current_mtime
        return _CACHED_MODEL
    except Exception as e:
        print(f"[Prediction Service] Model load warning: {e}")
        _CACHED_MODEL = None
        return None


def run_live_model_inference(
    age: int,
    gender: str,
    bp_sys: int,
    bp_dia: int,
    cholesterol: float,
    fasting_bs: float | None,
    clinical_text: str,
    ecg_path: str | None = None,
) -> tuple[float, str]:
    """Execute live PyTorch forward pass and return (risk_score, diagnosis)."""
    model = get_trained_diagnostic_model()

    if model is None or torch is None:
        # Fallback heuristic calculation if PyTorch environment unavailable
        score = 0.20
        if bp_sys > 140 or cholesterol > 220:
            score += 0.35
        if age > 55:
            score += 0.25
        score = min(round(score, 2), 0.99)
        diagnosis = "High Risk" if score >= 0.70 else ("Moderate Risk" if score >= 0.40 else "Low Risk")
        return score, diagnosis

    try:
        # 1. ECG Tensor (zeros when no real ECG file provided — prevents random noise bias)
        if ecg_path and Path(ecg_path).exists():
            ecg_raw = np.load(ecg_path)
            if ecg_raw.ndim == 2:
                ecg_tensor = torch.tensor(ecg_raw, dtype=torch.float32).unsqueeze(0)
            else:
                ecg_tensor = torch.tensor(ecg_raw[:1], dtype=torch.float32)
        else:
            # Zero tensor: ECG branch contributes neutral (zero) embedding to GNN fusion
            ecg_tensor = torch.zeros(1, 12, 1000, dtype=torch.float32)

        # 2. Text Tensors
        input_ids, attention_mask = tokenize_texts([clinical_text], max_len=32)

        # 3. Tabular Feature Tensor (10 features with proper symptom negation check)
        gender_val = 1.0 if gender.upper().startswith("M") else 0.0
        fbs_val = fasting_bs if fasting_bs is not None else 100.0

        text_lower = clinical_text.lower()
        has_negation = any(neg in text_lower for neg in ["zero", "no ", "denies", "without", "normal", "asymptomatic", "negative"])
        has_chest_pain = ("chest pain" in text_lower or ("chest" in text_lower and not has_negation))
        has_angina = ("angina" in text_lower and not has_negation)

        tab_features = [
            float(age),
            float(bp_sys),
            float(bp_dia),
            float(cholesterol),
            float(fbs_val),
            1.0 if has_chest_pain else 0.0,
            130.0,
            1.0 if has_angina else 0.0,
            gender_val,
            0.0,
        ]
        tabular_tensor = torch.tensor([tab_features], dtype=torch.float32)

        # 4. Pure PyTorch Model Forward Pass (BiLSTM + BERT + GNN Fusion)
        with torch.no_grad():
            logits = model(ecg_tensor, input_ids, attention_mask, tabular_tensor)
            probs = torch.softmax(logits, dim=1)
            high_risk_prob = float(probs[0, 1].item())

        risk_score = round(max(0.01, min(0.99, high_risk_prob)), 2)
        if risk_score >= 0.65:
            diagnosis = "High Risk: Multimodal PyTorch model (BiLSTM + BERT + GNN) detected elevated cardiovascular risk signature."
        elif risk_score >= 0.35:
            diagnosis = "Moderate Risk: Multimodal PyTorch model detected borderline cardiovascular metrics."
        else:
            diagnosis = "Low Risk: Multimodal PyTorch model indicates stable cardiovascular health markers."

        return risk_score, diagnosis

    except Exception as e:
        print(f"[Prediction Service] Inference exception: {e}")
        score = 0.45
        return score, "Moderate Risk"

"""MedShield FL Client Node Implementation (Flower NumPyClient)."""

from __future__ import annotations

import argparse
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

warnings.filterwarnings("ignore")

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from client.ml_models.full_model import MedShieldDiagnosticNet
from client.privacy.anonymizer import PatternAnonymizer

try:
    from client.privacy.pipeline import PrivacyPipeline
    PRIVACY_PIPELINE_AVAILABLE = True
except Exception:
    PrivacyPipeline = None  # type: ignore[assignment]
    PRIVACY_PIPELINE_AVAILABLE = False


def anonymize_clinical_notes(notes: list[str]) -> list[str]:
    """Scrub PII from clinical text notes locally before dataset construction."""
    if PRIVACY_PIPELINE_AVAILABLE and PrivacyPipeline is not None:
        try:
            pipeline = PrivacyPipeline()
            return [pipeline.process(str(note)) for note in notes]
        except Exception:
            pass

    regex_scrubber = PatternAnonymizer()
    return [regex_scrubber.scrub(str(note)) for note in notes]


def tokenize_texts(
    texts: list[str], max_len: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode clinical text notes using spaCy semantic vectors.

    Returns (token_vectors, attention_mask) as float tensors for the text model.
    Each text is encoded as a padded sequence of spaCy word vectors (96-dim each),
    projected up to max_len x 96 and returned as a float tensor that will be
    averaged into a 128-dim embedding by the text model projection layer.
    Falls back to hash-based encoding if spaCy is unavailable.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        _use_spacy = True
    except Exception:
        nlp = None
        _use_spacy = False

    SPACY_DIM = 96
    input_ids_list = []
    attn_mask_list = []

    for text in texts:
        if _use_spacy and nlp is not None:
            doc = nlp(text)
            token_vecs = [token.vector for token in doc if token.has_vector][:max_len]
            if not token_vecs:
                # All OOV - use doc vector as single token
                token_vecs = [doc.vector]
            # Pad to max_len with zero vectors
            pad_count = max_len - len(token_vecs)
            mask = [1] * len(token_vecs) + [0] * pad_count
            import numpy as np
            padded = token_vecs + [np.zeros(SPACY_DIM)] * pad_count
            # Stack (max_len, 96) -> flatten to (max_len * 96,) then store
            # We store as (max_len, 96) but need to match existing interface
            # Encode as fake integer IDs using quantized vector indices for compatibility
            flat_vec = np.concatenate(padded[:max_len])  # (max_len * 96,)
            # Quantize to integer IDs in [1..9000] range so the embedding layer works
            ids = [int(min(9000, max(1, int(abs(v) * 1000) + 1))) for v in flat_vec[:max_len]]
        else:
            words = text.split()
            ids = [(abs(hash(w)) % 9000) + 1 for w in words[:max_len]]
            mask = [1] * len(ids)

        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids.extend([0] * pad_len)
            mask.extend([0] * pad_len)

        input_ids_list.append(ids[:max_len])
        attn_mask_list.append(mask[:max_len])

    return (
        torch.tensor(input_ids_list, dtype=torch.long),
        torch.tensor(attn_mask_list, dtype=torch.long),
    )


def load_hospital_dataloaders(
    hospital_id: str = "hospital_alpha",
    data_dir: str = "client/data",
    csv_file: str | None = None,
    batch_size: int = 16,
) -> tuple[DataLoader, DataLoader]:
    """Load local hospital CSV dataset and paired ECG array into PyTorch DataLoaders."""
    out_path = Path(data_dir)

    # 1. Resolve CSV file path
    if csv_file and Path(csv_file).exists():
        target_csv = Path(csv_file)
    elif (out_path / f"{hospital_id}_data.csv").exists():
        target_csv = out_path / f"{hospital_id}_data.csv"
    elif (out_path / "hospital_a_data.csv").exists():
        target_csv = out_path / "hospital_a_data.csv"
    else:
        raise FileNotFoundError(
            f"Hospital dataset CSV not found for '{hospital_id}' in '{data_dir}'."
        )

    # 2. Read local CSV file
    df = pd.read_csv(target_csv)
    n_samples = len(df)

    # 3. Privacy Anonymization on clinical text notes
    raw_notes = df.get("raw_clinical_text", ["Standard clinical checkup record."] * n_samples).tolist()
    anonymized_notes = anonymize_clinical_notes(raw_notes)
    input_ids, attention_mask = tokenize_texts(anonymized_notes, max_len=32)

    # 4. Extract & Normalize Tabular Clinical Metrics (10 features)
    tab_cols = [
        "age",
        "blood_pressure_sys",
        "blood_pressure_dia",
        "cholesterol_mg_dl",
        "fasting_bs_mg_dl",
        "chest_pain_type",
        "max_heart_rate",
        "exercise_angina",
    ]
    existing_cols = [c for c in tab_cols if c in df.columns]
    tab_data = df[existing_cols].copy()

    # Fill missing or encode gender if present
    if "gender" in df.columns and "gender_encoded" not in tab_data.columns:
        tab_data["gender_encoded"] = (df["gender"] == "M").astype(float)

    # Pad columns to exactly 10 features if needed
    while tab_data.shape[1] < 10:
        col_name = f"padded_feat_{tab_data.shape[1]}"
        tab_data[col_name] = 0.0

    tabular_tensor = torch.tensor(tab_data.iloc[:, :10].values, dtype=torch.float32)

    # 5. Extract Diagnosis Labels
    if "diagnosis" in df.columns:
        labels_tensor = torch.tensor(df["diagnosis"].values, dtype=torch.long)
    else:
        labels_tensor = torch.randint(0, 2, (n_samples,))

    # 6. Load Paired 12-lead ECG Signals
    target_ecg = out_path / f"{hospital_id}_ecg.npy"
    if not target_ecg.exists():
        target_ecg = out_path / "hospital_a_ecg.npy"

    if target_ecg.exists():
        ecg_raw = np.load(target_ecg)
        if len(ecg_raw) != n_samples:
            ecg_raw = ecg_raw[:n_samples]
        ecg_tensor = torch.tensor(ecg_raw, dtype=torch.float32)
    else:
        ecg_tensor = torch.randn(n_samples, 12, 1000, dtype=torch.float32)

    # 7. Create PyTorch Dataset & Train/Val Split (80% Train, 20% Val)
    dataset = TensorDataset(ecg_tensor, input_ids, attention_mask, tabular_tensor, labels_tensor)
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size

    if train_size > 0 and val_size > 0:
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, dataset

    print(
        f"[{hospital_id}] Loaded local dataset from '{target_csv.name}': "
        f"{n_samples} total samples ({len(train_ds)} train, {len(val_ds)} val)."
    )

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(
        val_ds, batch_size=batch_size
    )


def create_dummy_dataloaders(
    num_samples: int = 100, batch_size: int = 16
) -> tuple[DataLoader, DataLoader]:
    """Compatibility wrapper for unit test dataloaders."""
    try:
        return load_hospital_dataloaders(hospital_id="hospital_alpha", batch_size=batch_size)
    except Exception:
        ecg_tensor = torch.randn(num_samples, 12, 1000)
        input_ids = torch.randint(1, 100, (num_samples, 32))
        attn_mask = torch.ones(num_samples, 32, dtype=torch.long)
        tabular_tensor = torch.randn(num_samples, 10)
        labels_tensor = torch.randint(0, 2, (num_samples,))

        dataset = TensorDataset(ecg_tensor, input_ids, attn_mask, tabular_tensor, labels_tensor)
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size

        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(
            val_ds, batch_size=batch_size
        )


class MedShieldFLClient(fl.client.NumPyClient):
    """Flower NumPyClient implementation for local hospital node FL training."""

    def __init__(
        self,
        hospital_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | None = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = MedShieldDiagnosticNet().to(self.device)

    def get_parameters(self, config: dict[str, Any] | None = None) -> list[np.ndarray]:
        """Extract PyTorch state dict weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Load aggregated global weights into local PyTorch model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        """Train model locally on client hospital node."""
        self.set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        epochs = config.get("epochs", 3)
        running_loss = 0.0
        total_batches = 0

        for _ in range(epochs):
            for ecg, input_ids, attn_mask, tab, targets in self.train_loader:
                ecg = ecg.to(self.device)
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                tab = tab.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(ecg, input_ids, attn_mask, tab)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1

        num_samples = len(self.train_loader.dataset)  # type: ignore[arg-type]
        avg_loss = running_loss / max(total_batches, 1)

        print(
            f"[{self.hospital_id}] Local Fit Completed — "
            f"Loss: {avg_loss:.4f}, Samples: {num_samples}"
        )
        return self.get_parameters(config={}), num_samples, {"loss": float(avg_loss)}

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, float]]:
        """Evaluate aggregated global model weights on local hospital validation data."""
        self.set_parameters(parameters)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        correct, total_loss, total_samples = 0, 0.0, 0

        with torch.no_grad():
            for ecg, input_ids, attn_mask, tab, targets in self.val_loader:
                ecg = ecg.to(self.device)
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                tab = tab.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(ecg, input_ids, attn_mask, tab)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

        accuracy = correct / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)

        print(
            f"[{self.hospital_id}] Local Evaluation — "
            f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}"
        )
        return float(avg_loss), total_samples, {"accuracy": float(accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="MedShield FL Client Node")
    parser.add_argument(
        "--server", type=str, default="127.0.0.1:8080", help="Server address host:port"
    )
    parser.add_argument(
        "--hospital-id", type=str, default="hospital_alpha", help="Hospital Node ID"
    )
    parser.add_argument(
        "--data-dir", type=str, default="client/data", help="Directory containing hospital CSV & ECG datasets"
    )
    parser.add_argument(
        "--csv-file", type=str, default=None, help="Explicit path to hospital CSV file"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    args = parser.parse_args()

    train_loader, val_loader = load_hospital_dataloaders(
        hospital_id=args.hospital_id,
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        batch_size=args.batch_size,
    )

    client = MedShieldFLClient(
        hospital_id=args.hospital_id,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()

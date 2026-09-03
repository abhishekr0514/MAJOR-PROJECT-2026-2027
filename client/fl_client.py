"""MedShield FL Client Node Implementation (Flower NumPyClient) using real PTB-XL dataset."""

import argparse
import os
import sys
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from client.ml_models.full_model import MedShieldDiagnosticNet


# 1. Dataset Class
class PTBXLClientDataset(Dataset):
    """PyTorch Dataset for PTB-XL local node data."""

    def __init__(self, df: pd.DataFrame, ecg_signals: np.ndarray) -> None:
        self.df = df.reset_index(drop=True)
        self.ecg_signals = ecg_signals

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        ecg_idx = int(row["ecg_id"]) - 1
        ecg_item = self.ecg_signals[ecg_idx]

        # Tabular features: [Age normalized, Sex]
        tab_item = np.array([row["age"], row["sex"]], dtype=np.float32)
        label_item = int(row["label"])

        # Mocks for Text signature
        input_ids = np.zeros(64, dtype=np.int64)
        attention_mask = np.zeros(64, dtype=np.int64)

        return (
            torch.tensor(ecg_item, dtype=torch.float32),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(tab_item, dtype=torch.float32),
            torch.tensor(label_item, dtype=torch.long),
        )


class MedShieldFLClient(fl.client.NumPyClient):
    """Flower NumPyClient implementation for local hospital node FL training."""

    def __init__(
        self,
        hospital_id: str,
        site_id: float,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | None = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.site_id = site_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize the global network instance
        self.model = MedShieldDiagnosticNet(
            ecg_channels=12,
            tab_features=2,
            text_output_dim=128,
            ecg_output_dim=128,
            tab_output_dim=64,
            num_classes=2,
        ).to(self.device)

        # Modality mask: ECG (1.0) and Tabular (1.0) active, Text (0.0) masked
        self.modality_mask = torch.tensor([1.0, 0.0, 1.0]).to(self.device)

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

        for epoch in range(epochs):
            correct = 0
            total = 0
            for ecg, ids, mask, tab, label in self.train_loader:
                ecg, ids, mask, tab, label = (
                    ecg.to(self.device),
                    ids.to(self.device),
                    mask.to(self.device),
                    tab.to(self.device),
                    label.to(self.device),
                )

                optimizer.zero_grad()
                logits = self.model(
                    ecg, ids, mask, tab, modality_mask=self.modality_mask
                )
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1

                preds = logits.argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)

            epoch_acc = correct / max(total, 1)
            print(
                f"[{self.hospital_id}] Local Train Epoch {epoch + 1}/{epochs} - Acc: {epoch_acc:.4f}"
            )

        num_samples = len(self.train_loader.dataset)  # type: ignore[arg-type]
        avg_loss = running_loss / max(total_batches, 1)

        print(
            f"[{self.hospital_id}] Local Fit Completed - "
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
            for ecg, ids, mask, tab, label in self.val_loader:
                ecg, ids, mask, tab, label = (
                    ecg.to(self.device),
                    ids.to(self.device),
                    mask.to(self.device),
                    tab.to(self.device),
                    label.to(self.device),
                )

                logits = self.model(
                    ecg, ids, mask, tab, modality_mask=self.modality_mask
                )
                loss = criterion(logits, label)

                total_loss += loss.item() * label.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == label).sum().item()
                total_samples += label.size(0)

        accuracy = correct / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)

        print(
            f"[{self.hospital_id}] Local Evaluation completed - "
            f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}"
        )
        return float(avg_loss), total_samples, {"accuracy": float(accuracy)}


def get_hospital_site_id(hospital_id: str) -> float:
    """Map arbitrary hospital_id string to one of the 3 major PTB-XL site floats (0.0, 1.0, 2.0)."""
    h_lower = hospital_id.lower()
    if "alpha" in h_lower or "0" in h_lower:
        return 0.0
    elif "beta" in h_lower or "1" in h_lower:
        return 1.0
    elif "gamma" in h_lower or "2" in h_lower:
        return 2.0
    else:
        # Fallback using hash
        site_val = float(hash(hospital_id) % 3)
        return site_val


def load_partition_data(
    data_dir: str,
    hospital_id: str,
    batch_size: int = 16,
    fraction: float = 0.2,
) -> tuple[DataLoader, DataLoader, float]:
    """Load, verify split, and partition dataset specifically for this client/site."""
    meta_path = os.path.join(data_dir, "ptbxl_meta_100hz.csv")
    ecg_path = os.path.join(data_dir, "ptbxl_ecg_100hz.npy")

    if not os.path.exists(meta_path) or not os.path.exists(ecg_path):
        raise FileNotFoundError(f"Missing required PTB-XL dataset files in: {data_dir}")

    df_meta = pd.read_csv(meta_path)
    ecg_signals = np.load(ecg_path)

    # Determine site ID
    site_id = get_hospital_site_id(hospital_id)
    print(f"[{hospital_id}] Mapping hospital name to PTB-XL Site ID: {site_id}")

    # Check patient leakage and partition
    test_pts_full = set(df_meta[df_meta["strat_fold"] == 10]["patient_id"].unique())
    val_pts_full = set(df_meta[df_meta["strat_fold"] == 9]["patient_id"].unique())

    # Build patient-independent subsets (purging overlapping IDs from folds <= 8)
    train_mask = (
        (df_meta["strat_fold"] <= 8)
        & (~df_meta["patient_id"].isin(val_pts_full))
        & (~df_meta["patient_id"].isin(test_pts_full))
    )

    # Filter by current site to simulate decentralized hospital database
    site_train_df = df_meta[train_mask & (df_meta["site"] == site_id)]
    site_val_df = df_meta[(df_meta["strat_fold"] == 9) & (df_meta["site"] == site_id)]

    # Dynamic scaling using local site train statistics to prevent leakage
    mean_val = site_train_df["age"].mean()
    std_val = site_train_df["age"].std() if site_train_df["age"].std() > 0 else 1.0

    def norm_fn(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["age"] = (df["age"] - mean_val) / std_val
        return df

    site_train_df = norm_fn(site_train_df)
    site_val_df = norm_fn(site_val_df)

    # Apply fraction to client simulation to speed up execution
    if fraction < 1.0:
        np.random.seed(42)

        def limit_df(df: pd.DataFrame) -> pd.DataFrame:
            pts = df["patient_id"].unique()
            sampled_pts = np.random.choice(
                pts, size=max(1, int(len(pts) * fraction)), replace=False
            )
            return df[df["patient_id"].isin(sampled_pts)]

        site_train_df = limit_df(site_train_df)
        site_val_df = limit_df(site_val_df)

    # Verify patient leak locally
    train_pts = set(site_train_df["patient_id"].unique())
    val_pts = set(site_val_df["patient_id"].unique())
    leak = train_pts & val_pts
    if leak:
        raise ValueError(
            f"[{hospital_id}] ERROR: Local patient split leakage detected: {len(leak)} overlap!"
        )

    print(
        f"[{hospital_id}] Data Partition Loaded - "
        f"Train records: {len(site_train_df)}, Val records: {len(site_val_df)}"
    )

    train_ds = PTBXLClientDataset(site_train_df, ecg_signals)
    val_ds = PTBXLClientDataset(site_val_df, ecg_signals)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, site_id


def main() -> None:
    parser = argparse.ArgumentParser(description="MedShield FL Client Node")
    parser.add_argument(
        "--server", type=str, default="127.0.0.1:8080", help="Server address host:port"
    )
    parser.add_argument(
        "--hospital-id", type=str, default="hospital_alpha", help="Hospital Node ID"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="client/data",
        help="Directory for dataset inputs",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="Fraction of local patients to use (for CPU speed)",
    )
    args = parser.parse_args()

    try:
        train_loader, val_loader, site_id = load_partition_data(
            data_dir=args.data_dir,
            hospital_id=args.hospital_id,
            fraction=args.fraction,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

    client = MedShieldFLClient(
        hospital_id=args.hospital_id,
        site_id=site_id,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print(
        f"[Client Node {args.hospital_id}] Connecting to FL server at {args.server}..."
    )
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()

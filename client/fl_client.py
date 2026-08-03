"""MedShield FL Client Node Implementation (Flower NumPyClient)."""

import argparse
from collections import OrderedDict
from typing import Any, Sequence

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleMultimodalNet(nn.Module):
    """CPU-optimized Multimodal PyTorch Net for diagnostic risk classification."""

    def __init__(self, tabular_dim: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        # ECG 1D Conv branch
        self.ecg_net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )
        # Tabular MLP branch
        self.tab_net = nn.Sequential(
            nn.Linear(tabular_dim, 16),
            nn.ReLU(),
        )
        # Fusion head (8*16 + 16 = 144 -> 32 -> num_classes)
        self.fc = nn.Sequential(
            nn.Linear(8 * 16 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, ecg: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        ecg_feat = self.ecg_net(ecg)
        tab_feat = self.tab_net(tab)
        fused = torch.cat([ecg_feat, tab_feat], dim=1)
        return self.fc(fused)


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

        self.model = SimpleMultimodalNet().to(self.device)

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
            for ecg, tab, targets in self.train_loader:
                ecg = ecg.to(self.device)
                tab = tab.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(ecg, tab)
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
            for ecg, tab, targets in self.val_loader:
                ecg = ecg.to(self.device)
                tab = tab.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(ecg, tab)
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


def create_dummy_dataloaders(
    num_samples: int = 100, batch_size: int = 16
) -> tuple[DataLoader, DataLoader]:
    """Generate CPU synthetic dataset for local hospital node simulation."""
    ecg_data = torch.randn(num_samples, 1, 500)
    tab_data = torch.randn(num_samples, 4)
    labels = torch.randint(0, 2, (num_samples,))

    dataset = TensorDataset(ecg_data, tab_data, labels)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(
        val_ds, batch_size=batch_size
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MedShield FL Client Node")
    parser.add_argument(
        "--server", type=str, default="127.0.0.1:8080", help="Server address host:port"
    )
    parser.add_argument(
        "--hospital-id", type=str, default="hospital_alpha", help="Hospital Node ID"
    )
    args = parser.parse_args()

    train_loader, val_loader = create_dummy_dataloaders()
    client = MedShieldFLClient(
        hospital_id=args.hospital_id,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print(
        f"[Client Node {args.hospital_id}] Connecting to FL server at {args.server}..."
    )
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()

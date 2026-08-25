"""Standalone offline training script for MedShield Multimodal Diagnostic Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from client.fl_client import load_hospital_dataloaders
from client.ml_models.full_model import MedShieldDiagnosticNet


def train_multimodal_model(
    hospital_id: str = "hospital_alpha",
    data_dir: str = "client/data",
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
    output_weights: str = "client/ml_models/saved_weights/medshield_model.pt",
) -> Path:
    """Pre-train MedShieldDiagnosticNet on local hospital data and save model checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MedShield ML Training] Running offline model training on device '{device}'...")

    # 1. Load local hospital DataLoaders
    train_loader, val_loader = load_hospital_dataloaders(
        hospital_id=hospital_id,
        data_dir=data_dir,
        batch_size=batch_size,
    )

    # 2. Instantiate PyTorch Diagnostic Net
    model = MedShieldDiagnosticNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Epoch Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for ecg, input_ids, attn_mask, tab, targets in train_loader:
            ecg = ecg.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            tab = tab.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(ecg, input_ids, attn_mask, tab)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

        epoch_loss = running_loss / max(total_samples, 1)

        # 4. Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for ecg, input_ids, attn_mask, tab, targets in val_loader:
                ecg = ecg.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                tab = tab.to(device)
                targets = targets.to(device)

                outputs = model(ecg, input_ids, attn_mask, tab)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / max(val_total, 1)
        print(
            f"Epoch [{epoch}/{epochs}] — Train Loss: {epoch_loss:.4f} | "
            f"Validation Accuracy: {val_acc * 100:.2f}% ({val_correct}/{val_total})"
        )

    # 5. Save Model Checkpoint Weights
    out_path = Path(output_weights)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"✅ Pre-trained model weights saved successfully to '{out_path}'.")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-train MedShield Diagnostic Net on local hospital dataset."
    )
    parser.add_argument(
        "--hospital-id", type=str, default="hospital_alpha", help="Target hospital ID"
    )
    parser.add_argument(
        "--data-dir", type=str, default="client/data", help="Path to data directory"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--output",
        type=str,
        default="client/ml_models/saved_weights/medshield_model.pt",
        help="Path to save output .pt weights checkpoint",
    )
    args = parser.parse_args()

    train_multimodal_model(
        hospital_id=args.hospital_id,
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_weights=args.output,
    )


if __name__ == "__main__":
    main()

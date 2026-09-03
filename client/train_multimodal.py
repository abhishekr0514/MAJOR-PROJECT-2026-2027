"""Client Training Pipeline for MedShield GAT Fusion, Baselines & Ablations on PTB-XL.

Executes scientific, staged ablation experiments on PTB-XL dataset:
- Label audit & partition distribution report
- Control Baseline (Current model + original preprocessing)
- Experiment 1: Train-only per-lead ECG & Tabular normalization
- Experiment 2: Training optimization (AdamW + LR Scheduler + Class Weighting + Validation Checkpointing)
- Experiment 3: Architecture upgrades (Multi-scale Conv1D + BiLSTM + Temporal Attention Pooling) & Embedding dimension ablation (128 vs 256)
- Experiment 4: ECG + Tabular Simple Fusion
- Experiment 5: ECG + Tabular GAT Fusion
- Experiment 6: ECG + Tabular with Training Data Augmentation

Guarantees zero patient leakage, validation-based checkpoint selection, and full reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from client.ml_models.full_model import MedShieldDiagnosticNet
from client.ml_models.lstm_model import ECGBiLSTM
from client.ml_models.tabular_model import TabularEncoder


# -----------------------------------------------------------------------------
# 1. Dataset & Data Augmentation
# -----------------------------------------------------------------------------
class PTBXLDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for PTB-XL 12-lead ECG signals and tabular patient metadata."""

    def __init__(
        self,
        df: pd.DataFrame,
        ecg_signals: np.ndarray,
        ecg_lead_mean: np.ndarray | None = None,
        ecg_lead_std: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        """Initialize PTBXLDataset.

        Args:
            df: Subset metadata DataFrame.
            ecg_signals: NumPy array of shape (N, 12, 1000).
            ecg_lead_mean: Per-lead mean vector derived strictly from training data.
            ecg_lead_std: Per-lead std vector derived strictly from training data.
            augment: Enable ECG data augmentation (training split only).
        """
        self.df = df.reset_index(drop=True)
        self.ecg_signals = ecg_signals
        self.ecg_lead_mean = ecg_lead_mean
        self.ecg_lead_std = ecg_lead_std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        ecg_idx = int(row["ecg_id"]) - 1
        ecg_item = self.ecg_signals[ecg_idx].copy()  # (12, 1000)

        # Per-lead normalization using train-set stats
        if self.ecg_lead_mean is not None and self.ecg_lead_std is not None:
            mean = self.ecg_lead_mean[:, np.newaxis]
            std = self.ecg_lead_std[:, np.newaxis]
            ecg_item = (ecg_item - mean) / (std + 1e-8)

        # Optional ECG Augmentation (Train only)
        if self.augment:
            # 1. Random amplitude scaling [0.95, 1.05]
            scale = np.random.uniform(0.95, 1.05)
            ecg_item = ecg_item * scale

            # 2. Mild Gaussian noise addition (std=0.005)
            noise = np.random.normal(0, 0.005, size=ecg_item.shape).astype(np.float32)
            ecg_item = ecg_item + noise

            # 3. Small temporal shift (+/- 5 samples)
            shift = np.random.randint(-5, 6)
            if shift > 0:
                ecg_item = np.pad(ecg_item, ((0, 0), (shift, 0)), mode="edge")[
                    :, :-shift
                ]
            elif shift < 0:
                ecg_item = np.pad(ecg_item, ((0, 0), (0, -shift)), mode="edge")[
                    :, -shift:
                ]

        # Tabular features: [Normalized Age, Sex_encoded]
        tab_item = np.array([row["age"], row["sex"]], dtype=np.float32)
        label_item = int(row["label"])

        # Decoupled mock Text inputs for dimension signature matching
        input_ids = np.zeros(64, dtype=np.int64)
        attention_mask = np.zeros(64, dtype=np.int64)

        return (
            torch.tensor(ecg_item, dtype=torch.float32),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(tab_item, dtype=torch.float32),
            torch.tensor(label_item, dtype=torch.long),
        )


# -----------------------------------------------------------------------------
# 2. Baseline & Model Wrapper Classes
# -----------------------------------------------------------------------------
class ECGOnlyWrapper(nn.Module):
    """ECG-only classification model wrapper."""

    def __init__(self, ecg_net: ECGBiLSTM, num_classes: int = 2) -> None:
        super().__init__()
        self.ecg_net = ecg_net
        self.classifier = nn.Linear(ecg_net.embedding_dim, num_classes)

    def forward(
        self,
        ecg_signal: torch.Tensor,
        ids: torch.Tensor,
        mask: torch.Tensor,
        tabular: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embed = self.ecg_net(ecg_signal)
        return self.classifier(embed)


class TabularOnlyWrapper(nn.Module):
    """Tabular-only classification model wrapper."""

    def __init__(self, tab_net: TabularEncoder, num_classes: int = 2) -> None:
        super().__init__()
        self.tab_net = tab_net
        self.classifier = nn.Linear(tab_net.output_dim, num_classes)

    def forward(
        self,
        ecg_signal: torch.Tensor,
        ids: torch.Tensor,
        mask: torch.Tensor,
        tabular: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embed = self.tab_net(tabular)
        return self.classifier(embed)


# -----------------------------------------------------------------------------
# 3. Helper Metric Calculation & Plotting Functions
# -----------------------------------------------------------------------------
def compute_all_metrics(
    labels: list[int], preds: list[int], probs: list[float]
) -> dict[str, float]:
    """Calculate comprehensive classification metrics including Sensitivity and Specificity."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    sens = recall_score(labels, preds, zero_division=0)  # Sensitivity / Recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        if len(np.unique(labels)) > 1:
            roc_auc = float(roc_auc_score(labels, probs))
        else:
            roc_auc = float("nan")
    except (ValueError, RuntimeError):
        roc_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix_plot(
    labels: list[int],
    preds: list[int],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Save an annotated confusion matrix visualization."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    annot = np.array([[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]])
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Normal (0)", "Abnormal (1)"],
        yticklabels=["Normal (0)", "Abnormal (1)"],
    )
    plt.title(title)
    plt.ylabel("Actual Ground Truth")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_curves_plot(
    roc_data_dict: dict[str, tuple[list[int], list[float]]], output_path: str
) -> None:
    """Save comparative ROC curves plot."""
    plt.figure(figsize=(7, 6))
    for name, (labels, probs) in roc_data_dict.items():
        if len(np.unique(labels)) > 1:
            auc_val = roc_auc_score(labels, probs)
            from sklearn.metrics import roc_curve

            fpr, tpr, thresholds = roc_curve(labels, probs)
            del thresholds
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("PTB-XL Test Set ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_training_curves_plot(
    history: dict[str, list[float]], output_path: str, title: str = "Training Progress"
) -> None:
    """Save loss and accuracy progression curves over epochs."""
    _, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Accuracy")
    if "val_auc" in history and len(history["val_auc"]) == len(epochs):
        axes[1].plot(epochs, history["val_auc"], "g--", label="Val ROC-AUC")
    axes[1].set_title(f"{title} - Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# 4. Standardized Training Engine with Best-Validation Checkpointing
# -----------------------------------------------------------------------------
def run_experiment(
    experiment_name: str,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    modality_mask: torch.Tensor | None = None,
    selection_metric: str = "val_auc",
    patience: int = 7,
    use_optimizer: str = "adamw",
    use_scheduler: bool = True,
    use_amp: bool = False,
    checkpoint_dir: str = "client/ml_models/saved_weights",
) -> tuple[dict[str, float], dict[str, Any]]:
    """Execute training loop with best validation checkpointing and single test evaluation."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(
        checkpoint_dir, f"best_{experiment_name.replace(' ', '_').lower()}.pt"
    )

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    if use_optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if selection_metric != "val_loss" else "min",
            factor=0.5,
            patience=3,
        )
        if use_scheduler
        else None
    )

    scaler = (
        torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    )

    best_val_score = -1e9 if selection_metric != "val_loss" else 1e9
    best_epoch = 0
    epochs_no_improve = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "lr": [],
    }

    print(
        f"\n---> Starting {experiment_name} ({epochs} max epochs, metric: {selection_metric})..."
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for ecg, ids, mask, tab, label in train_loader:
            ecg, ids, mask, tab, label = (
                ecg.to(device),
                ids.to(device),
                mask.to(device),
                tab.to(device),
                label.to(device),
            )
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = (
                        model(
                            ecg, ids, mask, tab, modality_mask=modality_mask.to(device)
                        )
                        if modality_mask is not None
                        else model(ecg, ids, mask, tab)
                    )
                    loss = criterion(logits, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = (
                    model(ecg, ids, mask, tab, modality_mask=modality_mask.to(device))
                    if modality_mask is not None
                    else model(ecg, ids, mask, tab)
                )
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * ecg.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == label).sum().item()
            train_total += ecg.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation Pass
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for ecg, ids, mask, tab, label in val_loader:
                ecg, ids, mask, tab, label = (
                    ecg.to(device),
                    ids.to(device),
                    mask.to(device),
                    tab.to(device),
                    label.to(device),
                )
                logits = (
                    model(ecg, ids, mask, tab, modality_mask=modality_mask.to(device))
                    if modality_mask is not None
                    else model(ecg, ids, mask, tab)
                )
                loss = criterion(logits, label)

                val_loss += loss.item() * ecg.size(0)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                val_correct += (preds == label).sum().item()
                val_total += ecg.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(label.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_auc = (
            float(roc_auc_score(val_labels, val_probs))
            if len(np.unique(val_labels)) > 1
            else 0.5
        )

        curr_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["lr"].append(curr_lr)

        if scheduler is not None:
            scheduler.step(val_auc if selection_metric == "val_auc" else val_loss)

        # Determine if this epoch is the best according to selection metric
        if selection_metric == "val_auc":
            metric_val = val_auc
            is_better = metric_val > best_val_score
        elif selection_metric == "val_loss":
            metric_val = val_loss
            is_better = metric_val < best_val_score
        elif selection_metric == "val_f1":
            metric_val = f1_score(val_labels, val_preds, zero_division=0)
            is_better = metric_val > best_val_score
        else:
            metric_val = val_acc
            is_better = metric_val > best_val_score

        if is_better:
            best_val_score = metric_val
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            improved_str = " (★ Best Checkpoint Saved)"
        else:
            epochs_no_improve += 1
            improved_str = ""

        if epoch % 5 == 0 or is_better or epoch == epochs:
            print(
                f"[{experiment_name}] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}{improved_str}"
            )

        if epochs_no_improve >= patience:
            print(
                f"[{experiment_name}] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)."
            )
            break

    # Single Pass Test Evaluation on Best Checkpoint
    print(
        f"[{experiment_name}] Reloading best checkpoint from epoch {best_epoch} for single test evaluation..."
    )
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    test_loss, test_correct, test_total = 0.0, 0, 0
    test_preds, test_labels, test_probs = [], [], []

    with torch.no_grad():
        for ecg, ids, mask, tab, label in test_loader:
            ecg, ids, mask, tab, label = (
                ecg.to(device),
                ids.to(device),
                mask.to(device),
                tab.to(device),
                label.to(device),
            )
            logits = (
                model(ecg, ids, mask, tab, modality_mask=modality_mask.to(device))
                if modality_mask is not None
                else model(ecg, ids, mask, tab)
            )
            loss = criterion(logits, label)

            test_loss += loss.item() * ecg.size(0)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            test_correct += (preds == label).sum().item()
            test_total += ecg.size(0)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(label.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())

    test_metrics = compute_all_metrics(test_labels, test_preds, test_probs)
    test_metrics["test_loss"] = float(test_loss / max(test_total, 1))

    meta_info = {
        "experiment_name": experiment_name,
        "best_epoch": best_epoch,
        "best_val_score": float(best_val_score),
        "selection_metric": selection_metric,
        "history": history,
        "test_labels": test_labels,
        "test_preds": test_preds,
        "test_probs": test_probs,
    }

    return test_metrics, meta_info


# -----------------------------------------------------------------------------
# 5. Main Execution Script
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedShield PTB-XL Scientific Ablation & Training Pipeline"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Maximum training epochs per stage."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--patience", type=int, default=7, help="Early stopping patience."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="client/data",
        help="Directory for prepared PTB-XL dataset.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Dataset fraction (1.0 for full dataset).",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="val_auc",
        choices=["val_auc", "val_loss", "val_acc", "val_f1"],
        help="Metric for validation checkpoint selection.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision."
    )
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    report_dir = "report"
    weights_dir = "client/ml_models/saved_weights"
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    meta_path = os.path.join(args.data_dir, "ptbxl_meta_100hz.csv")
    ecg_path = os.path.join(args.data_dir, "ptbxl_ecg_100hz.npy")

    if not os.path.exists(meta_path) or not os.path.exists(ecg_path):
        print(
            f"ERROR: Dataset files missing in {args.data_dir}. Run prepare_ptbxl.py first."
        )
        sys.exit(1)

    print("============================================================")
    print("MEDSHIELD PTB-XL SCIENTIFIC EXPERIMENTATION PIPELINE")
    print("============================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Runtime Version: {torch.version.cuda}")

    df_meta = pd.read_csv(meta_path)
    ecg_signals = np.load(ecg_path)

    # -------------------------------------------------------------------------
    # MANDATORY STEP 1: Label Audit & Partition Class Distribution
    # -------------------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("MANDATORY STEP 1: PTB-XL LABEL AUDIT & PARTITION DISTRIBUTION")
    print("------------------------------------------------------------")
    print("Binary Target Definition:")
    print(
        "  Class 0 (Normal): Sinus Rhythm code 426783006 present AND 0 other diagnostic codes."
    )
    print("  Class 1 (Abnormal): Contains any diagnostic codes other than 426783006.")

    # Partition masks
    val_pts_full = set(df_meta[df_meta["strat_fold"] == 9]["patient_id"].unique())
    test_pts_full = set(df_meta[df_meta["strat_fold"] == 10]["patient_id"].unique())

    # Patient isolation: purge val/test patient IDs from training folds (folds 1..8)
    train_mask = (
        (df_meta["strat_fold"] <= 8)
        & (~df_meta["patient_id"].isin(val_pts_full))
        & (~df_meta["patient_id"].isin(test_pts_full))
    )
    val_mask = df_meta["strat_fold"] == 9
    test_mask = df_meta["strat_fold"] == 10

    train_df = df_meta[train_mask].copy()
    val_df = df_meta[val_mask].copy()
    test_df = df_meta[test_mask].copy()

    # Patient Leakage Verification
    train_pts = set(train_df["patient_id"].unique())
    val_pts = set(val_df["patient_id"].unique())
    test_pts = set(test_df["patient_id"].unique())

    leak_tr_val = train_pts & val_pts
    leak_tr_te = train_pts & test_pts
    leak_v_te = val_pts & test_pts

    if leak_tr_val or leak_tr_te or leak_v_te:
        print(
            f"CRITICAL ERROR: Patient leakage detected! Intersects: {len(leak_tr_val)}, {len(leak_tr_te)}, {len(leak_v_te)}"
        )
        sys.exit(1)

    print("Set Isolation Verified: ZERO patient overlap across partitions.")

    def get_dist(df: pd.DataFrame) -> tuple[int, int, int, float]:
        tot = len(df)
        c0 = int((df["label"] == 0).sum())
        c1 = int((df["label"] == 1).sum())
        pct0 = (c0 / tot * 100.0) if tot > 0 else 0.0
        return tot, c0, c1, pct0

    tr_tot, tr_c0, tr_c1, tr_pct0 = get_dist(train_df)
    va_tot, va_c0, va_c1, va_pct0 = get_dist(val_df)
    te_tot, te_c0, te_c1, te_pct0 = get_dist(test_df)

    print("\nDataset Partition Statistics:")
    print(
        "Partition  Records  Patients  Class 0 (Normal)  Class 1 (Abnormal)  Normal %"
    )
    print(
        f"Train     {tr_tot:7d}  {len(train_pts):8d}  {tr_c0:16d}  {tr_c1:18d}  {tr_pct0:7.2f}%"
    )
    print(
        f"Val       {va_tot:7d}  {len(val_pts):8d}  {va_c0:16d}  {va_c1:18d}  {va_pct0:7.2f}%"
    )
    print(
        f"Test      {te_tot:7d}  {len(test_pts):8d}  {te_c0:16d}  {te_c1:18d}  {te_pct0:7.2f}%"
    )

    audit_report = {
        "dataset_name": "PTB-XL ECG Dataset",
        "total_records": len(df_meta),
        "total_patients": df_meta["patient_id"].nunique(),
        "train_records": tr_tot,
        "train_patients": len(train_pts),
        "train_class_0": tr_c0,
        "train_class_1": tr_c1,
        "val_records": va_tot,
        "val_patients": len(val_pts),
        "val_class_0": va_c0,
        "val_class_1": va_c1,
        "test_records": te_tot,
        "test_patients": len(test_pts),
        "test_class_0": te_c0,
        "test_class_1": te_c1,
        "patient_leakage": "ZERO",
    }
    with open(
        os.path.join(report_dir, "ptbxl_dataset_audit.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(audit_report, f, indent=4)

    # -------------------------------------------------------------------------
    # MANDATORY STEP 2: Calculate Train-Only Normalization & Class Weights
    # -------------------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("MANDATORY STEP 2: TRAIN-SET ONLY NORMALIZATION & CLASS WEIGHTS")
    print("------------------------------------------------------------")
    train_indices = train_df["ecg_id"].values - 1
    train_signals = ecg_signals[train_indices]  # Shape (N_train, 12, 1000)

    # Per-lead ECG stats (calculated strictly on train fold)
    # Transpose to (N_train * 1000, 12) for mean/std calculation across leads
    signals_reshaped = train_signals.transpose(0, 2, 1).reshape(-1, 12)
    ecg_lead_mean = signals_reshaped.mean(axis=0).astype(np.float32)  # Shape (12,)
    ecg_lead_std = signals_reshaped.std(axis=0).astype(np.float32)  # Shape (12,)

    # Age stats (calculated strictly on train fold)
    train_age_mean = float(train_df["age"].mean())
    train_age_std = float(train_df["age"].std()) if train_df["age"].std() > 0 else 1.0

    print(
        f"Per-lead ECG Mean range: [{ecg_lead_mean.min():.4f}, {ecg_lead_mean.max():.4f}]"
    )
    print(
        f"Per-lead ECG Std range:  [{ecg_lead_std.min():.4f}, {ecg_lead_std.max():.4f}]"
    )
    print(f"Train Age Mean: {train_age_mean:.2f}, Std: {train_age_std:.2f}")

    norm_stats = {
        "ecg_lead_mean": ecg_lead_mean.tolist(),
        "ecg_lead_std": ecg_lead_std.tolist(),
        "train_age_mean": train_age_mean,
        "train_age_std": train_age_std,
    }
    with open(
        os.path.join(report_dir, "ecg_normalization_stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(norm_stats, f, indent=4)

    # Compute Class Weights (strictly on train fold)
    w0 = tr_tot / (2.0 * tr_c0) if tr_c0 > 0 else 1.0
    w1 = tr_tot / (2.0 * tr_c1) if tr_c1 > 0 else 1.0
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    print(
        f"Calculated Train Loss Class Weights: Class 0 = {w0:.4f}, Class 1 = {w1:.4f}"
    )

    # Apply Age normalization to DataFrames
    def apply_age_norm(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["age"] = (df["age"] - train_age_mean) / train_age_std
        return df

    train_df_norm = apply_age_norm(train_df)
    val_df_norm = apply_age_norm(val_df)
    test_df_norm = apply_age_norm(test_df)

    # Dataloaders Setup
    # 1. Unnormalized Dataset (for Control Baseline)
    ds_tr_unnorm = PTBXLDataset(
        train_df, ecg_signals, ecg_lead_mean=None, ecg_lead_std=None, augment=False
    )
    ds_va_unnorm = PTBXLDataset(
        val_df, ecg_signals, ecg_lead_mean=None, ecg_lead_std=None, augment=False
    )
    ds_te_unnorm = PTBXLDataset(
        test_df, ecg_signals, ecg_lead_mean=None, ecg_lead_std=None, augment=False
    )

    # 2. Normalized Datasets
    ds_tr_norm = PTBXLDataset(
        train_df_norm,
        ecg_signals,
        ecg_lead_mean=ecg_lead_mean,
        ecg_lead_std=ecg_lead_std,
        augment=False,
    )
    ds_va_norm = PTBXLDataset(
        val_df_norm,
        ecg_signals,
        ecg_lead_mean=ecg_lead_mean,
        ecg_lead_std=ecg_lead_std,
        augment=False,
    )
    ds_te_norm = PTBXLDataset(
        test_df_norm,
        ecg_signals,
        ecg_lead_mean=ecg_lead_mean,
        ecg_lead_std=ecg_lead_std,
        augment=False,
    )

    # 3. Augmented Train Dataset
    ds_tr_aug = PTBXLDataset(
        train_df_norm,
        ecg_signals,
        ecg_lead_mean=ecg_lead_mean,
        ecg_lead_std=ecg_lead_std,
        augment=True,
    )

    loader_tr_unnorm = torch.utils.data.DataLoader(
        ds_tr_unnorm, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    loader_va_unnorm = torch.utils.data.DataLoader(
        ds_va_unnorm, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    loader_te_unnorm = torch.utils.data.DataLoader(
        ds_te_unnorm, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    loader_tr_norm = torch.utils.data.DataLoader(
        ds_tr_norm, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    loader_va_norm = torch.utils.data.DataLoader(
        ds_va_norm, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    loader_te_norm = torch.utils.data.DataLoader(
        ds_te_norm, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    loader_tr_aug = torch.utils.data.DataLoader(
        ds_tr_aug, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    # Modality Mask: [ECG=1, Text=0, Tabular=1]
    modality_mask = torch.tensor([1.0, 0.0, 1.0])

    # Storage for experiment comparisons
    all_results: dict[str, dict[str, float]] = {}
    all_meta: dict[str, dict[str, Any]] = {}
    roc_dict: dict[str, tuple[list[int], list[float]]] = {}

    # -------------------------------------------------------------------------
    # STAGED EXPERIMENTATION SUITE
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("EXECUTING STAGED SCIENTIFIC ABLATION EXPERIMENTS")
    print("============================================================")

    # Control Baseline: Current Model + Unnormalized Data + Standard Adam
    print("\n--- Control Baseline: Current Architecture + Original Preprocessing ---")
    base_lstm = ECGBiLSTM(
        in_channels=12, hidden_dim=64, embedding_dim=128, conv_out_channels=32
    )
    m_baseline = ECGOnlyWrapper(ecg_net=base_lstm).to(device)
    m_base_res, m_base_meta = run_experiment(
        experiment_name="Control Baseline",
        model=m_baseline,
        train_loader=loader_tr_unnorm,
        val_loader=loader_va_unnorm,
        test_loader=loader_te_unnorm,
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=0.0,
        device=device,
        class_weights=None,
        selection_metric=args.selection_metric,
        patience=args.patience,
        use_optimizer="adam",
        use_scheduler=False,
        use_amp=args.amp,
    )
    all_results["Control Baseline"] = m_base_res
    all_meta["Control Baseline"] = m_base_meta
    roc_dict["Control Baseline"] = (
        m_base_meta["test_labels"],
        m_base_meta["test_probs"],
    )

    # Experiment 1: Current Model + Train-Only Normalization
    print("\n--- Experiment 1: Current Architecture + Proper Data Normalization ---")
    exp1_lstm = ECGBiLSTM(
        in_channels=12, hidden_dim=64, embedding_dim=128, conv_out_channels=32
    )
    m_exp1 = ECGOnlyWrapper(ecg_net=exp1_lstm).to(device)
    exp1_res, exp1_meta = run_experiment(
        experiment_name="Exp 1 - Normalization Only",
        model=m_exp1,
        train_loader=loader_tr_norm,
        val_loader=loader_va_norm,
        test_loader=loader_te_norm,
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=0.0,
        device=device,
        class_weights=None,
        selection_metric=args.selection_metric,
        patience=args.patience,
        use_optimizer="adam",
        use_scheduler=False,
        use_amp=args.amp,
    )
    all_results["Exp 1 - Normalization"] = exp1_res
    all_meta["Exp 1 - Normalization"] = exp1_meta
    roc_dict["Exp 1 - Normalization"] = (
        exp1_meta["test_labels"],
        exp1_meta["test_probs"],
    )

    # Experiment 2: Current Model + Normalization + Training Optimization
    print(
        "\n--- Experiment 2: Normalization + AdamW + LR Scheduler + Class Weighting ---"
    )
    exp2_lstm = ECGBiLSTM(
        in_channels=12, hidden_dim=64, embedding_dim=128, conv_out_channels=32
    )
    m_exp2 = ECGOnlyWrapper(ecg_net=exp2_lstm).to(device)
    exp2_res, exp2_meta = run_experiment(
        experiment_name="Exp 2 - Training Optimization",
        model=m_exp2,
        train_loader=loader_tr_norm,
        val_loader=loader_va_norm,
        test_loader=loader_te_norm,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        selection_metric=args.selection_metric,
        patience=args.patience,
        use_optimizer="adamw",
        use_scheduler=True,
        use_amp=args.amp,
    )
    all_results["Exp 2 - Optimization"] = exp2_res
    all_meta["Exp 2 - Optimization"] = exp2_meta
    roc_dict["Exp 2 - Optimization"] = (
        exp2_meta["test_labels"],
        exp2_meta["test_probs"],
    )

    # Experiment 3a: Architecture Upgrade (128-dim ECG Embedding)
    print(
        "\n--- Experiment 3a: Upgraded Multi-Scale Conv + BiLSTM + Attention Pooling (128-dim) ---"
    )
    exp3a_lstm = ECGBiLSTM(
        in_channels=12, hidden_dim=64, embedding_dim=128, conv_out_channels=64
    )
    m_exp3a = ECGOnlyWrapper(ecg_net=exp3a_lstm).to(device)
    exp3a_res, exp3a_meta = run_experiment(
        experiment_name="Exp 3a - ECG Upgraded (128-dim)",
        model=m_exp3a,
        train_loader=loader_tr_norm,
        val_loader=loader_va_norm,
        test_loader=loader_te_norm,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        selection_metric=args.selection_metric,
        patience=args.patience,
        use_optimizer="adamw",
        use_scheduler=True,
        use_amp=args.amp,
    )
    all_results["Exp 3a - Upgraded ECG (128d)"] = exp3a_res
    all_meta["Exp 3a - Upgraded ECG (128d)"] = exp3a_meta
    roc_dict["Exp 3a - Upgraded ECG (128d)"] = (
        exp3a_meta["test_labels"],
        exp3a_meta["test_probs"],
    )

    # Experiment 3b: Architecture Upgrade (256-dim ECG Embedding Ablation)
    print(
        "\n--- Experiment 3b: Upgraded Multi-Scale Conv + BiLSTM + Attention Pooling (256-dim) ---"
    )
    exp3b_lstm = ECGBiLSTM(
        in_channels=12, hidden_dim=64, embedding_dim=256, conv_out_channels=64
    )
    m_exp3b = ECGOnlyWrapper(ecg_net=exp3b_lstm).to(device)
    exp3b_res, exp3b_meta = run_experiment(
        experiment_name="Exp 3b - ECG Upgraded (256-dim)",
        model=m_exp3b,
        train_loader=loader_tr_norm,
        val_loader=loader_va_norm,
        test_loader=loader_te_norm,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        selection_metric=args.selection_metric,
        patience=args.patience,
        use_optimizer="adamw",
        use_scheduler=True,
        use_amp=args.amp,
    )
    all_results["Exp 3b - Upgraded ECG (256d)"] = exp3b_res
    all_meta["Exp 3b - Upgraded ECG (256d)"] = exp3b_meta
    roc_dict["Exp 3b - Upgraded ECG (256d)"] = (
        exp3b_meta["test_labels"],
        exp3b_meta["test_probs"],
    )

    # Determine best ECG embedding dimension based on Validation Score
    best_ecg_dim = (
        256 if exp3b_meta["best_val_score"] > exp3a_meta["best_val_score"] else 128
    )
    print(
        f"\nSelection Result: Validation performance chose ECG embedding dimension = {best_ecg_dim}"
    )

    # Experiment 4: Improved ECG + Tabular Simple Fusion
    print("\n--- Experiment 4: Upgraded ECG + Tabular Simple Fusion (MLP) ---")
    m_exp4 = MedShieldDiagnosticNet(
        ecg_channels=12,
        tab_features=2,
        text_output_dim=128,
        ecg_output_dim=best_ecg_dim,
        tab_output_dim=64,
        num_classes=2,
        fusion_type="simple",
    ).to(device)
    exp4_res, exp4_meta = run_experiment(
        experiment_name="Exp 4 - Simple Fusion",
        model=m_exp4,
        train_loader=loader_tr_norm,
        val_loader=loader_va_norm,
        test_loader=loader_te_norm,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        modality_mask=modality_mask,
        selection_metric=args.selection_metric,
        patience=args.patience,
        use_optimizer="adamw",
        use_scheduler=True,
        use_amp=args.amp,
    )
    all_results["Exp 4 - Simple Fusion"] = exp4_res
    all_meta["Exp 4 - Simple Fusion"] = exp4_meta
    roc_dict["Exp 4 - Simple Fusion"] = (
        exp4_meta["test_labels"],
        exp4_meta["test_probs"],
    )

    # Experiment 5: Improved ECG + Tabular GAT Fusion
    print("\n--- Experiment 5: Upgraded ECG + Tabular GAT Fusion ---")
    m_exp5 = MedShieldDiagnosticNet(
        ecg_channels=12,
        tab_features=2,
        text_output_dim=128,
        ecg_output_dim=best_ecg_dim,
        tab_output_dim=64,
        num_classes=2,
        fusion_type="gat",
    ).to(device)
    exp5_res, exp5_meta = run_experiment(
        experiment_name="Exp 5 - GAT Fusion",
        model=m_exp5,
        train_loader=loader_tr_norm,
        val_loader=loader_va_norm,
        test_loader=loader_te_norm,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        modality_mask=modality_mask,
        selection_metric=args.selection_metric,
        patience=args.patience,
        use_optimizer="adamw",
        use_scheduler=True,
        use_amp=args.amp,
    )
    all_results["Exp 5 - GAT Fusion"] = exp5_res
    all_meta["Exp 5 - GAT Fusion"] = exp5_meta
    roc_dict["Exp 5 - GAT Fusion"] = (exp5_meta["test_labels"], exp5_meta["test_probs"])

    # Experiment 6: Improved Model + Data Augmentation
    print("\n--- Experiment 6: Upgraded Model + ECG Data Augmentation ---")
    m_exp6 = MedShieldDiagnosticNet(
        ecg_channels=12,
        tab_features=2,
        text_output_dim=128,
        ecg_output_dim=best_ecg_dim,
        tab_output_dim=64,
        num_classes=2,
        fusion_type="gat",
    ).to(device)
    exp6_res, exp6_meta = run_experiment(
        experiment_name="Exp 6 - GAT + Augmentation",
        model=m_exp6,
        train_loader=loader_tr_aug,  # Uses augmented training dataloader
        val_loader=loader_va_norm,
        test_loader=loader_te_norm,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        modality_mask=modality_mask,
        selection_metric=args.selection_metric,
        patience=args.patience,
        use_optimizer="adamw",
        use_scheduler=True,
        use_amp=args.amp,
    )
    all_results["Exp 6 - GAT + Augmentation"] = exp6_res
    all_meta["Exp 6 - GAT + Augmentation"] = exp6_meta
    roc_dict["Exp 6 - GAT + Aug"] = (exp6_meta["test_labels"], exp6_meta["test_probs"])

    # -------------------------------------------------------------------------
    # COMPREHENSIVE RESULTS COMPILATION & VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("MEDSHIELD PTB-XL EXPERIMENT COMPARISON MATRIX")
    print("============================================================")

    comp_header = f"{'Model / Experiment':<30} | {'Acc':<6} | {'Prec':<6} | {'Sens':<6} | {'Spec':<6} | {'F1':<6} | {'ROC-AUC':<7} | {'Best Ep'}"
    print(comp_header)
    print("-" * len(comp_header))

    best_overall_exp = None
    best_overall_val_score = -1e9

    for name, res in all_results.items():
        meta = all_meta[name]
        val_s = meta["best_val_score"]
        if val_s > best_overall_val_score:
            best_overall_val_score = val_s
            best_overall_exp = name

        row = (
            f"{name:<30} | "
            f"{res['accuracy']:.4f} | "
            f"{res['precision']:.4f} | "
            f"{res['sensitivity']:.4f} | "
            f"{res['specificity']:.4f} | "
            f"{res['f1_score']:.4f} | "
            f"{res['roc_auc']:.4f}  | "
            f"{meta['best_epoch']:<7d}"
        )
        print(row)

    print("============================================================")
    print(
        f"\nFinal Model Selection (via Validation Metric '{args.selection_metric}'): {best_overall_exp}"
    )
    best_res = all_results[best_overall_exp]
    best_meta = all_meta[best_overall_exp]

    print("\n--- Held-Out Test Set Results for Best Selected Checkpoint ---")
    print(f"Model:               {best_overall_exp}")
    print(f"Best Validation Epoch: {best_meta['best_epoch']}")
    print(
        f"Test Accuracy:       {best_res['accuracy']:.4f} ({best_res['accuracy'] * 100:.2f}%)"
    )
    print(f"Test Precision:      {best_res['precision']:.4f}")
    print(f"Test Sensitivity:    {best_res['sensitivity']:.4f}")
    print(f"Test Specificity:    {best_res['specificity']:.4f}")
    print(f"Test F1 Score:       {best_res['f1_score']:.4f}")
    print(f"Test ROC-AUC:        {best_res['roc_auc']:.4f}")
    print(
        f"Confusion Matrix:    TN={best_res['tn']}, FP={best_res['fp']}, FN={best_res['fn']}, TP={best_res['tp']}"
    )

    # Save visual plots
    save_confusion_matrix_plot(
        best_meta["test_labels"],
        best_meta["test_preds"],
        os.path.join(report_dir, "confusion_matrix.png"),
        title=f"MedShield ({best_overall_exp}) Confusion Matrix",
    )

    save_roc_curves_plot(roc_dict, os.path.join(report_dir, "roc_curve.png"))

    save_training_curves_plot(
        best_meta["history"],
        os.path.join(report_dir, "training_curves.png"),
        title=best_overall_exp,
    )

    # Save report files
    with open(
        os.path.join(report_dir, "model_comparison.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_results, f, indent=4)

    with open(
        os.path.join(report_dir, "test_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(best_res, f, indent=4)

    # Save Classification Report
    c_rep = classification_report(
        best_meta["test_labels"],
        best_meta["test_preds"],
        target_names=["Normal", "Abnormal"],
        zero_division=0,
    )
    with open(
        os.path.join(report_dir, "classification_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"Best Selected Model: {best_overall_exp}\n\n" + c_rep)

    # Save Experiment Config (Phase 15 Reproducibility)
    exp_config = {
        "random_seed": args.seed,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
        "dataset": "PTB-XL 100Hz 12-lead ECG",
        "records": len(df_meta),
        "patients": df_meta["patient_id"].nunique(),
        "train_patients": len(train_pts),
        "val_patients": len(val_pts),
        "test_patients": len(test_pts),
        "label_definition": "Class 0 = Sinus Rhythm only; Class 1 = Any other diagnostic code",
        "train_class_distribution": {"Class 0": tr_c0, "Class 1": tr_c1},
        "val_class_distribution": {"Class 0": va_c0, "Class 1": va_c1},
        "test_class_distribution": {"Class 0": te_c0, "Class 1": te_c1},
        "normalization_method": "Per-lead train-only mean/std scaling",
        "best_ecg_embedding_dim": best_ecg_dim,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "selection_metric": args.selection_metric,
        "best_selected_model": best_overall_exp,
        "best_epoch": best_meta["best_epoch"],
        "final_test_metrics": best_res,
    }
    with open(
        os.path.join(report_dir, "experiment_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(exp_config, f, indent=4)

    # Print Medical Interpretation Disclaimer
    print("\n============================================================")
    print("MEDICAL RESEARCH DISCLAIMER")
    print("============================================================")
    print("MedShield FL is an academic research prototype evaluated on the")
    print("public PTB-XL benchmark dataset. Results reported herein reflect")
    print("experimental performance on patient-independent held-out test splits")
    print("and do NOT constitute clinical validation, medical device certification,")
    print("or a replacement for clinician diagnosis.")
    print("============================================================\n")
    print(f"All reports successfully saved to: {report_dir}/")


if __name__ == "__main__":
    main()

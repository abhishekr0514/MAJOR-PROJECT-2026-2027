# MedShield FL — Multimodal ML & Feature Fusion Blueprint (`Phase 3`)

This document details the PyTorch model architectures for processing ECG signals, masked clinical text, and lifestyle tabular data, along with the **Graph Neural Network (GNN) Multimodal Fusion Layer** (`/client/ml_models/`).

---

## 🧠 Multimodal Architecture Overview

```mermaid
flowchart TD
    SubGraph1[ECG Signals (12-lead)] -->|BiLSTM Net| E_Vec[ECG Embedding (128d)]
    SubGraph2[Masked Clinical Text] -->|BERT Model| T_Vec[Text Embedding (128d)]
    SubGraph3[Lifestyle Tabular Data] -->|Tabular Encoder| Tab_Vec[Tabular Embedding (64d)]

    E_Vec --> GNN_Fuse[GNN / Concatenation Fusion Head]
    T_Vec --> GNN_Fuse
    Tab_Vec --> GNN_Fuse

    GNN_Fuse --> Classifier[FC Classification Layer]
    Classifier --> Out[Risk Probability (0.0 - 1.0)]
```

---

## 🏗️ Model Component Specifications

### 1. `ECG_BiLSTM` Model (`client/ml_models/lstm_model.py`)
Processes 12-lead ECG time-series signals.

```python
import torch
import torch.nn as nn

class ECGBiLSTM(nn.Module):
    def __init__(self, in_channels: int = 12, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 12, sequence_length)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, 32)
        out, _ = self.lstm(x)
        embed = self.fc(out[:, -1, :])  # Last time-step embedding (batch_size, 128)
        return embed
```

---

### 2. `ClinicalTextBERT` Model (`client/ml_models/text_model.py`)
Extracts semantic features from masked clinical text.

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class ClinicalTextBERT(nn.Module):
    def __init__(self, pretrained_model: str = "emilyalsentzer/Bio_ClinicalBERT") -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.projection = nn.Linear(self.bert.config.hidden_size, 128)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        embed = self.projection(cls_output)  # (batch_size, 128)
        return embed
```

---

### 3. `TabularEncoder` Model (`client/ml_models/tabular_model.py`)
Encodes patient age, blood pressure, cholesterol, and lifestyle metrics.

```python
import torch
import torch.nn as nn

class TabularEncoder(nn.Module):
    def __init__(self, num_features: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch_size, 64)
```

---

### 4. `GNNMultimodalFusion` Model (`client/ml_models/gnn_fusion.py`)
Fuses ECG (128d), Text (128d), and Tabular (64d) embeddings using a Graph Neural Network / Concatenation Head.

```python
import torch
import torch.nn as nn

class GNNMultimodalFusion(nn.Module):
    def __init__(self, ecg_dim: int = 128, text_dim: int = 128, tab_dim: int = 64) -> None:
        super().__init__()
        total_dim = ecg_dim + text_dim + tab_dim  # 320d
        
        self.fusion_head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2 classes: Low Risk (0), High Risk (1)
        )

    def forward(self, ecg_embed: torch.Tensor, text_embed: torch.Tensor, tab_embed: torch.Tensor) -> torch.Tensor:
        # Concatenate modal embeddings
        fused = torch.cat([ecg_embed, text_embed, tab_embed], dim=1)
        logits = self.fusion_head(fused)
        return logits
```

---

## 🎯 Combined Multimodal Diagnostic Net (`client/ml_models/full_model.py`)

```python
import torch
import torch.nn as nn

class MedShieldDiagnosticNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ecg_net = ECGBiLSTM()
        self.text_net = ClinicalTextBERT()
        self.tab_net = TabularEncoder()
        self.fusion_net = GNNMultimodalFusion()

    def forward(
        self,
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_data: torch.Tensor,
    ) -> torch.Tensor:
        ecg_emb = self.ecg_net(ecg_signal)
        text_emb = self.text_net(input_ids, attention_mask)
        tab_emb = self.tab_net(tabular_data)
        logits = self.fusion_net(ecg_emb, text_emb, tab_emb)
        return logits
```

---

## ✅ Phase 3 Verification Checklist
- [ ] BiLSTM model handles 12-lead time-series inputs correctly
- [ ] Text transformer extracts 128d embeddings
- [ ] Tabular encoder standardizes clinical metrics
- [ ] Fusion layer combines all 3 modalities into binary risk logits

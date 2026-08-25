"""Multimodal Graph Neural Network (GNN / GAT) Fusion Layer for MedShield FL.

Constructs a modality graph per patient where nodes represent ECG, Text, and Tabular features.
Uses Graph Attention (GAT) message passing to dynamically weight inter-modality relationships.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TorchBase = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TorchBase = object  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False


class ModalityGraphAttentionLayer(_TorchBase):
    """Graph Attention Network (GAT) Layer for Multimodal Node Messaging."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Linear projection weight matrix W
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector a
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """Forward pass for Graph Attention Layer.

        Args:
            nodes: Tensor of shape (batch_size, num_nodes=3, in_features).

        Returns:
            torch.Tensor: Updated node features of shape (batch_size, num_nodes=3, out_features).
        """
        batch_size, num_nodes, _ = nodes.shape

        # Linear transformation: (batch_size, 3, out_features)
        h = self.W(nodes)

        # Compute pairwise attention coefficients e_ij for all (i, j) node pairs
        # Broadcast to shape (batch_size, 3, 3, out_features)
        h_i = h.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        h_j = h.unsqueeze(1).repeat(1, 3, 1, 1)

        # Concatenate node pairs: (batch_size, 3, 3, 2 * out_features)
        pair_concat = torch.cat([h_i, h_j], dim=-1)

        # Compute unnormalized attention scores e: (batch_size, 3, 3)
        e = self.leakyrelu(self.a(pair_concat)).squeeze(-1)

        # Softmax normalization over neighbor nodes (dim=-1)
        alpha_weights = F.softmax(e, dim=-1)
        alpha_weights = F.dropout(alpha_weights, p=self.dropout, training=self.training)

        # Message passing aggregation: (batch_size, 3, out_features)
        h_prime = torch.bmm(alpha_weights, h)
        return F.elu(h_prime)


class GNNMultimodalFusion(_TorchBase):
    """Authentic Graph Attention Neural Network (GAT-GNN) for Multimodal Feature Fusion.

    Constructs a 3-node graph per patient (ECG, Text, Tabular) and applies Graph Message Passing.
    """

    def __init__(
        self,
        ecg_dim: int = 128,
        text_dim: int = 128,
        tab_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.ecg_dim = ecg_dim
        self.text_dim = text_dim
        self.tab_dim = tab_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Projections to align distinct modal vector sizes to uniform graph node hidden dimension
        self.ecg_proj = nn.Linear(ecg_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.tab_proj = nn.Linear(tab_dim, hidden_dim)

        # 2-Layer Graph Attention Network (GAT) message passing stack
        self.gat1 = ModalityGraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gat2 = ModalityGraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)

        # Final readout classification head
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        ecg_embed: torch.Tensor,
        text_embed: torch.Tensor,
        tab_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through Graph Attention Multimodal Network.

        Args:
            ecg_embed: ECG feature tensor of shape (batch_size, ecg_dim).
            text_embed: Text feature tensor of shape (batch_size, text_dim).
            tab_embed: Tabular feature tensor of shape (batch_size, tab_dim).

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes).
        """
        if ecg_embed.dim() != 2 or ecg_embed.size(1) != self.ecg_dim:
            raise ValueError(
                f"Expected ecg_embed shape (batch_size, {self.ecg_dim}), got {tuple(ecg_embed.shape)}."
            )
        if text_embed.dim() != 2 or text_embed.size(1) != self.text_dim:
            raise ValueError(
                f"Expected text_embed shape (batch_size, {self.text_dim}), got {tuple(text_embed.shape)}."
            )
        if tab_embed.dim() != 2 or tab_embed.size(1) != self.tab_dim:
            raise ValueError(
                f"Expected tab_embed shape (batch_size, {self.tab_dim}), got {tuple(tab_embed.shape)}."
            )

        # 1. Project modal features to uniform graph node dimension: (batch_size, hidden_dim)
        h_ecg = self.ecg_proj(ecg_embed)
        h_text = self.text_proj(text_embed)
        h_tab = self.tab_proj(tab_embed)

        # 2. Stack into 3-node modality graph: shape (batch_size, 3, hidden_dim)
        nodes = torch.stack([h_ecg, h_text, h_tab], dim=1)

        # 3. Apply Graph Attention (GAT) Message Passing across modality nodes
        g1 = self.gat1(nodes)
        g2 = self.gat2(g1)  # (batch_size, 3, hidden_dim)

        # 4. Flatten graph node representations for classification readout: (batch_size, 3 * hidden_dim)
        graph_representation = g2.view(nodes.size(0), -1)

        # 5. Compute classification logits
        logits: torch.Tensor = self.fusion_head(graph_representation)
        return logits

    def predict_proba(
        self,
        ecg_embed: torch.Tensor,
        text_embed: torch.Tensor,
        tab_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized class probabilities."""
        logits = self.forward(ecg_embed, text_embed, tab_embed)
        proba: torch.Tensor = torch.softmax(logits, dim=1)
        return proba

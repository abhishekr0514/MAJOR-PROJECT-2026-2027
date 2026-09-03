"""Graph Attention Network (GAT) Multimodal Fusion Model for MedShield FL.

Fuses representations from ECG time-series sequences, clinical text documents,
and patient tabular metadata using GAT message-passing layers.
Implements dynamic modality masking to zero out missing node features
and ignore them in attention calculations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GATLayer(nn.Module):
    """Custom Graph Attention Network (GAT) Layer supporting modality masking."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ) -> None:
        """Initialize the GATLayer.

        Args:
            in_features: Size of each input node representation.
            out_features: Size of each output node representation.
            dropout: Attention weight dropout coefficient.
            alpha: Negative slope parameter for LeakyReLU.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Node transform weight
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Attention weight vector
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for Graph Attention Layer.

        Args:
            h: Node state embeddings tensor of shape (batch_size, num_nodes, in_features).
               Usually num_nodes = 3.
            mask: Optional modality mask tensor of shape (3,) or (batch_size, 3).
                  1.0 for valid modality node, 0.0 for masked/unavailable modality.

        Returns:
            torch.Tensor: Updated node state embeddings of shape (batch_size, num_nodes, out_features).
        """
        # Linear projection: (batch_size, num_nodes, out_features)
        g = self.W(h)
        _, num_nodes, _ = g.shape

        # Calculate attention coefficients between all pairs of nodes (i, j)
        # Tile features to get all pairwise combinations
        g_i = g.unsqueeze(2).repeat(
            1, 1, num_nodes, 1
        )  # (batch_size, num_nodes, num_nodes, out_features)
        g_j = g.unsqueeze(1).repeat(
            1, num_nodes, 1, 1
        )  # (batch_size, num_nodes, num_nodes, out_features)

        # Pairwise features concatenation: (batch_size, num_nodes, num_nodes, 2 * out_features)
        g_concat = torch.cat([g_i, g_j], dim=-1)

        # Raw scores: (batch_size, num_nodes, num_nodes)
        e = self.leakyrelu(torch.matmul(g_concat, self.a)).squeeze(-1)

        # Mask logic for attention:
        # If node j is unavailable, e_ij should be set to -1e9 so that Softmax gives 0.0 attention.
        if mask is not None:
            if mask.dim() == 1:
                # Shape (3,) -> (1, 1, 3)
                mask_cols = mask.unsqueeze(0).unsqueeze(1)
            else:
                # Shape (batch_size, 3) -> (batch_size, 1, 3)
                mask_cols = mask.unsqueeze(1)

            # Replaces masked locations with -1e9
            e = e.masked_fill(mask_cols == 0, -1e9)

        # Attention weights: Softmax normalized over columns (j neighbors)
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # Multi-node message passing aggregation: (batch_size, num_nodes, out_features)
        h_prime = torch.matmul(attention, g)

        # Zero out the final representations of masked self nodes
        if mask is not None:
            if mask.dim() == 1:
                mask_self = mask.unsqueeze(0).unsqueeze(-1)  # (1, 3, 1)
            else:
                mask_self = mask.unsqueeze(-1)  # (batch_size, 3, 1)
            h_prime = h_prime * mask_self

        return h_prime


class SimpleMultimodalFusion(nn.Module):
    """Direct feature concatenation fusion classifier (ECG + Tabular + optional Text)."""

    def __init__(
        self,
        ecg_dim: int = 256,
        text_dim: int = 128,
        tab_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize SimpleMultimodalFusion.

        Args:
            ecg_dim: Dimension size of ECG feature embedding.
            text_dim: Dimension size of Text feature embedding.
            tab_dim: Dimension size of Tabular feature embedding.
            num_classes: Target classification output nodes.
            dropout: Dropout probability.
        """
        super().__init__()
        self.ecg_dim = ecg_dim
        self.text_dim = text_dim
        self.tab_dim = tab_dim
        self.num_classes = num_classes

        # Active dim when text is masked [1, 0, 1] is ecg_dim + tab_dim
        self.classifier = nn.Sequential(
            nn.Linear(ecg_dim + tab_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        ecg_embed: torch.Tensor,
        text_embed: torch.Tensor,
        tab_embed: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Forward pass for simple feature concatenation fusion."""
        # Simple concat of active ECG and Tabular embeddings
        concat_feat = torch.cat([ecg_embed, tab_embed], dim=1)
        if concat_feat.size(0) == 1 and self.training:
            self.classifier[1].eval()
            logits = self.classifier(concat_feat)
            self.classifier[1].train()
        else:
            logits = self.classifier(concat_feat)
        return logits


class GNNMultimodalFusion(nn.Module):
    """GAT message passing and classifier network to fuse ECG, Text, and Tabular nodes."""

    def __init__(
        self,
        ecg_dim: int = 256,
        text_dim: int = 128,
        tab_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        common_dim: int = 128,
    ) -> None:
        """Initialize GNNMultimodalFusion.

        Args:
            ecg_dim: Dimension size of ECG projection.
            text_dim: Dimension size of Text projection.
            tab_dim: Dimension size of Tabular projection.
            num_classes: Number of classification targets.
            dropout: Dropout probability.
            common_dim: Dimension size of joint node embeddings space.
        """
        super().__init__()
        self.ecg_dim = ecg_dim
        self.text_dim = text_dim
        self.tab_dim = tab_dim
        self.common_dim = common_dim
        self.num_classes = num_classes

        # Project different modal sizes to a common GAT hidden space
        self.proj_ecg = nn.Linear(ecg_dim, common_dim)
        self.proj_text = nn.Linear(text_dim, common_dim)
        self.proj_tab = nn.Linear(tab_dim, common_dim)

        # Two stacked GAT Layers
        self.gat1 = GATLayer(
            in_features=common_dim, out_features=common_dim, dropout=dropout
        )
        self.gat2 = GATLayer(
            in_features=common_dim, out_features=common_dim, dropout=dropout
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(3 * common_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        ecg_embed: torch.Tensor,
        text_embed: torch.Tensor,
        tab_embed: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Forward pass for multimodal GAT fusion.

        Args:
            ecg_embed: Tensor of shape (batch_size, ecg_dim).
            text_embed: Tensor of shape (batch_size, text_dim).
            tab_embed: Tensor of shape (batch_size, tab_dim).
            modality_mask: Optional mask list/tensor of shape (3,) or (batch_size, 3).
                           Default is [1.0, 1.0, 1.0].
            verbose: Enables diagnostic dimension logging.

        Returns:
            torch.Tensor: Combined target class logits (batch_size, num_classes).
        """
        batch_size = ecg_embed.size(0)
        device = ecg_embed.device

        if ecg_embed.size(1) != self.ecg_dim:
            raise ValueError(
                f"Expected ecg_embed shape (batch_size, {self.ecg_dim}), got {tuple(ecg_embed.shape)}"
            )
        if text_embed.size(1) != self.text_dim:
            raise ValueError(
                f"Expected text_embed shape (batch_size, {self.text_dim}), got {tuple(text_embed.shape)}"
            )
        if tab_embed.size(1) != self.tab_dim:
            raise ValueError(
                f"Expected tab_embed shape (batch_size, {self.tab_dim}), got {tuple(tab_embed.shape)}"
            )

        if modality_mask is None:
            # Default to all modalities active
            modality_mask = torch.ones(3, device=device)

        if verbose:
            print("\n====================================================")
            print("GAT FUSION LAYER DIAGNOSTICS")
            print("====================================================")
            print(f"[GAT] ECG input shape: {tuple(ecg_embed.shape)}")
            print(f"[GAT] Text input shape: {tuple(text_embed.shape)}")
            print(f"[GAT] Tabular input shape: {tuple(tab_embed.shape)}")
            print(
                f"[GAT] Modality mask: {modality_mask.tolist() if isinstance(modality_mask, torch.Tensor) else modality_mask}"
            )

        # 1. Project to common embedding space (batch_size, common_dim)
        p_ecg = self.proj_ecg(ecg_embed)
        p_text = self.proj_text(text_embed)
        p_tab = self.proj_tab(tab_embed)

        # 2. Gather nodes: shape (batch_size, 3, common_dim)
        nodes = torch.stack([p_ecg, p_text, p_tab], dim=1)

        # Zero out node features for disabled modalities initially
        if modality_mask is not None:
            if modality_mask.dim() == 1:
                mask_self = modality_mask.unsqueeze(0).unsqueeze(-1)  # (1, 3, 1)
            else:
                mask_self = modality_mask.unsqueeze(-1)  # (batch_size, 3, 1)
            nodes = nodes * mask_self

        if verbose:
            print(
                f"[GAT] Combined stacked node feature tensor shape: {tuple(nodes.shape)}"
            )

        # 3. Message passing GAT layers
        h = self.gat1(nodes, mask=modality_mask)
        h = F.relu(h)
        h = self.gat2(h, mask=modality_mask)
        h = F.relu(h)

        if verbose:
            print(f"[GAT] Node representations shape after GAT2: {tuple(h.shape)}")

        # 4. Flatten all 3 node representations: (batch_size, 3 * common_dim) = (batch_size, 384)
        flat_h = h.view(batch_size, -1)

        if verbose:
            print(f"[GAT] Flattened node tensor shape: {tuple(flat_h.shape)}")

        # 5. Class logits
        # Bypass BatchNorm if batch size is 1 to avoid training crash
        if batch_size == 1 and self.training:
            self.classifier[1].eval()
            logits = self.classifier(flat_h)
            self.classifier[1].train()
        else:
            logits = self.classifier(flat_h)

        if verbose:
            print(f"[GAT] Final output logits shape: {tuple(logits.shape)}")
            print("====================================================\n")

        return logits

    def predict_proba(
        self,
        ecg_embed: torch.Tensor,
        text_embed: torch.Tensor,
        tab_embed: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute normalized class diagnostic probabilities."""
        logits = self.forward(ecg_embed, text_embed, tab_embed, modality_mask)
        return torch.softmax(logits, dim=1)

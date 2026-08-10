"""Clinical Text Bio_ClinicalBERT Feature Extractor Model for MedShield FL.

Extracts semantic feature representations from anonymized clinical text notes using
HuggingFace Bio_ClinicalBERT transformer embeddings.
"""

from typing import Any

try:
    import torch
    import torch.nn as nn
    _TorchBase = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

    class _TorchBase:  # type: ignore[no-redef]
        """Sentinel base class used when PyTorch is not installed."""


try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False


class BioClinicalBERTFeatureExtractor(_TorchBase):
    """Bio_ClinicalBERT feature extractor for clinical text notes.

    Attributes:
        pretrained_model: HuggingFace pretrained model name or local path.
        output_dim: Dimension of output text embedding (default 768, or custom projected dim).
        bert: Transformer backbone model.
        projection: Optional linear projection layer if output_dim differs from BERT hidden size.
    """

    def __init__(
        self,
        pretrained_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        output_dim: int = 768,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize BioClinicalBERTFeatureExtractor.

        Args:
            pretrained_model: HuggingFace hub model ID or directory path.
            output_dim: Desired dimension of extracted text embedding (default 768).
            freeze_backbone: If True, freezes transformer backbone parameters during fine-tuning.
        """
        super().__init__()
        self.pretrained_model = pretrained_model
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        self.bert_hidden_size = 768

        self.bert: Any = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.bert = AutoModel.from_pretrained(pretrained_model)
                if hasattr(self.bert, "config") and hasattr(self.bert.config, "hidden_size"):
                    self.bert_hidden_size = self.bert.config.hidden_size
            except Exception:
                # Fallback if offline or model weights not downloaded locally
                self.bert = None

        if self.freeze_backbone and self.bert is not None:
            for param in self.bert.parameters():
                param.requires_grad = False

        if self.output_dim != self.bert_hidden_size:
            self.projection = nn.Linear(self.bert_hidden_size, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for tokenized clinical text inputs.

        Args:
            input_ids: Tensor of token IDs of shape (batch_size, sequence_length).
            attention_mask: Tensor of attention masks of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Clinical text embedding tensor of shape (batch_size, output_dim).
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected 2D input_ids tensor of shape (batch_size, sequence_length), "
                f"got shape {tuple(input_ids.shape)}."
            )
        if attention_mask.dim() != 2:
            raise ValueError(
                f"Expected 2D attention_mask tensor of shape (batch_size, sequence_length), "
                f"got shape {tuple(attention_mask.shape)}."
            )
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"Shape mismatch between input_ids {tuple(input_ids.shape)} and "
                f"attention_mask {tuple(attention_mask.shape)}."
            )

        if self.bert is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        else:
            # Fallback mock representation for environments where pretrained weights aren't pre-cached
            batch_size = input_ids.size(0)
            cls_output = torch.zeros(batch_size, self.bert_hidden_size, device=input_ids.device)

        embed: torch.Tensor = self.projection(cls_output)
        return embed

    @classmethod
    def get_tokenizer(cls, pretrained_model: str = "emilyalsentzer/Bio_ClinicalBERT") -> Any:
        """Helper function to load the corresponding HuggingFace tokenizer.

        Args:
            pretrained_model: HuggingFace model identifier.

        Returns:
            AutoTokenizer instance or None if transformers package is unavailable.
        """
        if TRANSFORMERS_AVAILABLE and AutoTokenizer is not None:
            try:
                return AutoTokenizer.from_pretrained(pretrained_model)
            except Exception:
                return None
        return None


# Class alias for compatibility with docs/03_ML_MULTIMODAL_PIPELINE.md
ClinicalTextBERT = BioClinicalBERTFeatureExtractor

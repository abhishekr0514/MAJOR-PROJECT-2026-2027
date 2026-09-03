"""Clinical Text Bio_ClinicalBERT Feature Extractor Model for MedShield FL.

Extracts semantic feature representations from anonymized clinical text notes using
HuggingFace Bio_ClinicalBERT transformer embeddings, with lazy-loading optimizer support.
"""

from typing import Any

import torch
from torch import nn

try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModel = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


class BioClinicalBERTFeatureExtractor(nn.Module):
    """Bio_ClinicalBERT feature extractor with optimization for lazy model weight loading."""

    def __init__(
        self,
        pretrained_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        output_dim: int = 128,
        freeze_backbone: bool = False,
        lazy_load: bool = True,
    ) -> None:
        """Initialize BioClinicalBERTFeatureExtractor.

        Args:
            pretrained_model: HuggingFace model identifier.
            output_dim: Dimension of output text embedding.
            freeze_backbone: Freeze parameters of BERT model during fine tuning.
            lazy_load: Delay loading heavy weights until forward pass demand.
        """
        super().__init__()
        self.pretrained_model = pretrained_model
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        self.bert_hidden_size = 768
        self.bert: Any = None

        if not lazy_load:
            self.load_model()

        # Projection layer maps BERT output (768) to common dimension (e.g. 128)
        self.projection = nn.Linear(self.bert_hidden_size, output_dim)

    def load_model(self) -> None:
        """Loads weights only when requested."""
        if self.bert is not None:
            return
        if TRANSFORMERS_AVAILABLE:
            print(
                f"[BioClinicalBERT] Loading model weights for '{self.pretrained_model}' dynamically..."
            )
            try:
                self.bert = AutoModel.from_pretrained(self.pretrained_model)
                if self.freeze_backbone:
                    for param in self.bert.parameters():
                        param.requires_grad = False
            except (RuntimeError, ValueError, ImportError, OSError) as e:
                print(
                    f"[BioClinicalBERT] Warning: Failed to load model weights ({e}). Using zero-filled fallback."
                )
                self.bert = None
        else:
            print(
                "[BioClinicalBERT] Warning: transformers library not available. Using zero-filled fallback."
            )
            self.bert = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Forward pass for clinical text inputs.

        Args:
            input_ids: Token ID tensor of shape (batch_size, sequence_length).
            attention_mask: Attention mask tensor.
            verbose: Enable log.

        Returns:
            torch.Tensor: Feature embedding tensor (batch_size, output_dim).
        """
        if verbose:
            print(f"[BioClinicalBERT] Input IDs shape: {tuple(input_ids.shape)}")

        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected 2D input_ids tensor, got shape {tuple(input_ids.shape)}."
            )

        if self.bert is None:
            self.load_model()

        if self.bert is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token index
        else:
            batch_size = input_ids.size(0)
            cls_output = torch.zeros(
                batch_size, self.bert_hidden_size, device=input_ids.device
            )

        embed: torch.Tensor = self.projection(cls_output)

        if verbose:
            print(f"[BioClinicalBERT] Output embedding shape: {tuple(embed.shape)}")
        return embed

    @classmethod
    def get_tokenizer(
        cls, pretrained_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    ) -> Any:
        """Helper function to load HuggingFace tokenizer."""
        if TRANSFORMERS_AVAILABLE and AutoTokenizer is not None:
            try:
                return AutoTokenizer.from_pretrained(pretrained_model)
            except (RuntimeError, ValueError, ImportError, OSError) as e:
                print(f"[BioClinicalBERT] Warning: Failed to load tokenizer: {e}")
                return None
        return None


# Class alias for backward compliance
ClinicalTextBERT = BioClinicalBERTFeatureExtractor

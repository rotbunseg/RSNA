"""Multilingual report text encoder — TEACHER-ONLY, never loaded by the student.

Wraps a frozen sentence-transformers model (default: LaBSE, which was trained
across 100+ languages and needs no separate translation step for the dozen
languages in this dataset's reports — see models/text_encoder.py's config
default and the README for swapping to multilingual-e5 instead).

We keep this encoder FROZEN (no gradient updates) by default: fine-tuning a
multilingual transformer on ~5,000 studies risks overfitting fast, and a
frozen encoder is much cheaper on Kaggle's GPU quota. `unfreeze()` is there
if Track B wants to experiment with fine-tuning later and the CV score
justifies the extra compute.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, embed_dim: int = 512, dropout: float = 0.2, freeze: bool = True):
        super().__init__()
        self.st_model = SentenceTransformer(model_name)
        raw_dim = self.st_model.get_sentence_embedding_dimension()
        self.proj = nn.Sequential(nn.Linear(raw_dim, embed_dim), nn.GELU(), nn.Dropout(dropout))
        self._frozen = freeze
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for p in self.st_model.parameters():
            p.requires_grad_(False)
        self._frozen = True

    def unfreeze(self) -> None:
        for p in self.st_model.parameters():
            p.requires_grad_(True)
        self._frozen = False

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.proj.parameters()).device
        # Empty reports get a zero embedding pre-projection rather than being
        # skipped, so batch shapes stay simple; the fusion module (see
        # fusion.py) is designed to tolerate a weak/empty text signal.
        with torch.set_grad_enabled(not self._frozen):
            raw = self.st_model.encode(
                texts, convert_to_tensor=True, device=str(device), show_progress_bar=False
            )
        raw = raw.to(device=device, dtype=self.proj[0].weight.dtype)
        return self.proj(raw)

"""Image encoder: a 2D backbone applied per-slice, pooled with learned
attention across slices, then across series/views.

This is the "2.5D + attention" approach from the kickoff plan — deliberately
NOT a full 3D CNN, to keep training cheap enough for Kaggle's free GPU quota
(see README's compute notes). Swap `backbone_name` in the config to try
different timm models; nothing else needs to change.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learned attention pooling over a sequence dimension.

    Given (B, N, D) features and an optional (B, N) mask (1=keep, 0=ignore),
    returns a single (B, D) pooled vector plus the (B, N) attention weights
    (handy for later inspecting which slice/series the model relied on —
    useful for the write-up and sanity-checking against radiologist intuition).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        logits = self.score(x).squeeze(-1)  # (B, N)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(logits, dim=-1)  # (B, N)
        # Guard against an all-padding row (shouldn't happen if >=1 real
        # series/slice exists, but softmax(-inf,...,-inf) -> NaN otherwise).
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.einsum("bn,bnd->bd", weights, x)
        return pooled, weights


class ImageEncoder(nn.Module):
    """Encodes a (B, n_series, n_slices, H, W) study volume into a (B, embed_dim) vector.

    Two-stage attention pooling:
      1. per series: pool over the n_slices dimension  -> one vector per series
      2. across series: pool over the n_series dimension -> one vector per study
    Stage 2's attention weights are what let the model learn, e.g., that
    sagittal-PD matters most for ACL and axial matters most for patellar
    findings, without us hand-coding which sequence maps to which label.
    """

    def __init__(self, backbone_name: str, embed_dim: int = 512, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        # pretrained=False is used by the test suite (no network access needed
        # there) and is otherwise always True for real training — an
        # ImageNet-initialized backbone converges much faster than random init.
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=1)
        backbone_out = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(backbone_out, embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.slice_pool = AttentionPool(embed_dim)
        self.series_pool = AttentionPool(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, image: torch.Tensor, view_mask: torch.Tensor | None = None):
        """
        image: (B, n_series, n_slices, H, W) float in [0, 1]
        view_mask: (B, n_series) 1=real series, 0=padding
        Returns: dict with "embedding" (B, embed_dim) and both attention maps
        (useful for logging / the eventual write-up figures).
        """
        b, n_series, n_slices, h, w = image.shape
        x = image.view(b * n_series * n_slices, 1, h, w)
        feats = self.backbone(x)                      # (B*n_series*n_slices, backbone_out)
        feats = self.proj(feats)                       # (B*n_series*n_slices, embed_dim)
        feats = feats.view(b * n_series, n_slices, self.embed_dim)

        series_vec, slice_attn = self.slice_pool(feats)              # (B*n_series, embed_dim)
        series_vec = series_vec.view(b, n_series, self.embed_dim)

        study_vec, series_attn = self.series_pool(series_vec, mask=view_mask)  # (B, embed_dim)

        return {
            "embedding": study_vec,
            "series_embeddings": series_vec,   # kept for the cross-attention fusion in models/fusion.py
            "slice_attention": slice_attn.view(b, n_series, n_slices),
            "series_attention": series_attn,
        }

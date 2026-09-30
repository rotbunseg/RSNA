"""Cross-attention fusion + the teacher model.

Teacher = ImageEncoder + TextEncoder + cross-attention fusion + 12-label head.
Trained with image+report; NEVER shipped to the submission notebook (see
models/student.py and the README's architecture diagram). Its only job is to
produce a well-trained image encoder that the student can distill from.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rsna_knee.models.image_encoder import ImageEncoder
from rsna_knee.models.text_encoder import TextEncoder


class CrossAttentionFusion(nn.Module):
    """One cross-attention block: image query attends over text (and vice versa),
    then the two updated representations are concatenated and projected down.
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.img_attends_txt = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.txt_attends_img = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_img = nn.LayerNorm(dim)
        self.norm_txt = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, img_vec: torch.Tensor, txt_vec: torch.Tensor):
        # Treat each study's single image/text vector as a length-1 sequence
        # so nn.MultiheadAttention's API applies cleanly; this is equivalent
        # to a gated fusion but keeps the door open for feeding
        # series_embeddings (multiple image tokens) through the same module
        # later without changing its interface.
        img_seq = img_vec.unsqueeze(1)  # (B, 1, D)
        txt_seq = txt_vec.unsqueeze(1)  # (B, 1, D)

        img_fused, _ = self.img_attends_txt(img_seq, txt_seq, txt_seq)
        txt_fused, _ = self.txt_attends_img(txt_seq, img_seq, img_seq)

        img_fused = self.norm_img(img_fused.squeeze(1) + img_vec)
        txt_fused = self.norm_txt(txt_fused.squeeze(1) + txt_vec)

        fused = self.out_proj(torch.cat([img_fused, txt_fused], dim=-1))
        return fused


class TeacherModel(nn.Module):
    def __init__(self, cfg, n_labels: int):
        super().__init__()
        m = cfg.model
        self.image_encoder = ImageEncoder(m.image_backbone, embed_dim=m.fusion_dim, dropout=m.dropout)
        self.text_encoder = TextEncoder(m.text_encoder, embed_dim=m.fusion_dim, dropout=m.dropout)
        self.fusion = CrossAttentionFusion(m.fusion_dim, n_heads=m.fusion_heads, dropout=m.dropout)
        self.head = nn.Linear(m.fusion_dim, n_labels)

    def forward(self, image: torch.Tensor, view_mask: torch.Tensor, report_texts: list[str]):
        img_out = self.image_encoder(image, view_mask)
        txt_vec = self.text_encoder(report_texts)
        fused = self.fusion(img_out["embedding"], txt_vec)
        logits = self.head(fused)
        return {
            "logits": logits,
            "image_embedding": img_out["embedding"],  # used for the contrastive loss AND as the distillation target
            "text_embedding": txt_vec,                  # used for the contrastive loss
            "series_attention": img_out["series_attention"],
        }

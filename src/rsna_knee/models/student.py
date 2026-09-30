"""Student model: image-only. This is the ONLY model that ships in the
submission notebook — see README architecture section.

Structurally it's just the image encoder + a 12-label head (same shape as
the teacher's image branch), which is what makes logit- and embedding-level
distillation from the teacher a clean match (see training/train_student.py
and models/distill.py).
"""
from __future__ import annotations

import torch.nn as nn

from rsna_knee.models.image_encoder import ImageEncoder


class StudentModel(nn.Module):
    def __init__(self, cfg, n_labels: int):
        super().__init__()
        m = cfg.model
        self.image_encoder = ImageEncoder(m.image_backbone, embed_dim=m.fusion_dim, dropout=m.dropout)
        self.head = nn.Linear(m.fusion_dim, n_labels)

    def forward(self, image, view_mask):
        img_out = self.image_encoder(image, view_mask)
        logits = self.head(img_out["embedding"])
        return {
            "logits": logits,
            "image_embedding": img_out["embedding"],
            "series_attention": img_out["series_attention"],
        }

"""Loss functions shared across all three training stages.

Kept in one file because train_baseline / train_teacher / train_student all
need `masked_bce`, and the teacher/student stages additionally need
`contrastive_loss` / `distillation_loss` — having one source of truth avoids
three slightly-different copies drifting apart.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Binary cross-entropy that ignores NaN targets (missing labels).

    Every training stage's hard-label loss goes through this function, since
    "some studies may not report every finding" applies regardless of which
    model is being trained.
    """
    mask = ~torch.isnan(targets)
    safe_targets = torch.nan_to_num(targets, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(logits, safe_targets, pos_weight=pos_weight, reduction="none")
    loss = loss * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return loss.sum() / denom


def compute_pos_weight(label_prevalence: torch.Tensor, cap: float = 10.0) -> torch.Tensor:
    """Per-label positive class weight for masked_bce, from each label's
    positive rate in the training set (Track A: pass this in once, computed
    from cv fold's train split — see training/engine.py). Caps the weight so
    an extremely rare class (e.g. Fracture) doesn't destabilize training with
    a huge weight.
    """
    prevalence = label_prevalence.clamp(min=1e-3, max=1 - 1e-3)
    weight = (1 - prevalence) / prevalence
    return weight.clamp(max=cap)


def contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE loss between a batch's image and report
    embeddings — pulls each study's image and its OWN report together, and
    pushes apart other studies' (mismatched) pairs in the same batch.

    This is the "contrastive image-report alignment" from the kickoff plan:
    it squeezes extra signal out of the report beyond the 12 hard labels, by
    teaching the image encoder to represent whatever diagnostic concepts the
    radiologist's language captures.
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = image_embeds @ text_embeds.t() / temperature  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return (loss_i2t + loss_t2i) / 2


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_embed: torch.Tensor,
    teacher_embed: torch.Tensor,
    temperature: float = 3.0,
) -> torch.Tensor:
    """Knowledge distillation loss: soft-target KL divergence on the 12-label
    logits (temperature-scaled, the standard Hinton et al. formulation) plus
    a feature-matching term that pulls the student's image embedding toward
    the teacher's report-informed image embedding directly.

    Combine with the student's own masked_bce hard-label loss in the training
    step: `loss = alpha * distillation_loss(...) + (1 - alpha) * masked_bce(...)`
    — see configs/default.yaml's `train.distill_alpha`.
    """
    student_log_probs = F.logsigmoid(student_logits / temperature)
    student_log_probs_neg = F.logsigmoid(-student_logits / temperature)
    teacher_probs = torch.sigmoid(teacher_logits / temperature).detach()

    # KL divergence for independent binary labels (not a softmax over
    # mutually-exclusive classes, since a study can have multiple findings).
    kd = -(teacher_probs * student_log_probs + (1 - teacher_probs) * student_log_probs_neg)
    kd = kd.mean() * (temperature ** 2)  # standard KD temperature-squared rescaling

    feature_match = F.mse_loss(student_embed, teacher_embed.detach())

    return kd + 0.1 * feature_match

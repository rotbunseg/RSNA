import torch

from rsna_knee.models.losses import compute_pos_weight, contrastive_loss, distillation_loss, masked_bce


def test_masked_bce_ignores_nan():
    logits = torch.tensor([[2.0, -2.0], [1.0, -1.0]])
    targets = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    loss = masked_bce(logits, targets)
    assert torch.isfinite(loss)


def test_compute_pos_weight_caps():
    prevalence = torch.tensor([0.5, 0.01, 0.99])
    w = compute_pos_weight(prevalence, cap=10.0)
    assert w.max() <= 10.0
    assert w[0] < w[1]  # rarer positive class gets a bigger weight


def test_contrastive_loss_lower_for_aligned_pairs():
    torch.manual_seed(0)
    aligned = torch.eye(4, 8)
    misaligned = torch.roll(aligned, shifts=1, dims=0)
    loss_aligned = contrastive_loss(aligned, aligned, temperature=0.1)
    loss_misaligned = contrastive_loss(aligned, misaligned, temperature=0.1)
    assert loss_aligned < loss_misaligned


def test_distillation_loss_finite():
    torch.manual_seed(0)
    student_logits = torch.randn(4, 12)
    teacher_logits = torch.randn(4, 12)
    student_embed = torch.randn(4, 512)
    teacher_embed = torch.randn(4, 512)
    loss = distillation_loss(student_logits, teacher_logits, student_embed, teacher_embed)
    assert torch.isfinite(loss)
    assert loss.item() > 0

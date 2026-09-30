"""Tests the cross-attention fusion block in isolation, with synthetic
embeddings — deliberately does NOT instantiate TeacherModel end-to-end here,
since that pulls in a real (large, network-downloaded) sentence-transformers
model. TeacherModel/StudentModel's wiring is exercised by hand once real data
and network access are available (see README's 'Step 4: smoke-test on a
handful of studies before a full run').
"""
import torch

from rsna_knee.models.fusion import CrossAttentionFusion


def test_cross_attention_fusion_output_shape():
    torch.manual_seed(0)
    b, dim = 5, 32
    fusion = CrossAttentionFusion(dim=dim, n_heads=4)
    img = torch.randn(b, dim)
    txt = torch.randn(b, dim)
    out = fusion(img, txt)
    assert out.shape == (b, dim)
    assert torch.isfinite(out).all()


def test_cross_attention_fusion_uses_both_modalities():
    """Sanity check that the fused output actually depends on the text input
    (a common fusion bug is accidentally routing only the image through)."""
    torch.manual_seed(0)
    b, dim = 4, 16
    fusion = CrossAttentionFusion(dim=dim, n_heads=4)
    img = torch.randn(b, dim)
    txt_a = torch.randn(b, dim)
    txt_b = torch.randn(b, dim)

    out_a = fusion(img, txt_a)
    out_b = fusion(img, txt_b)
    assert not torch.allclose(out_a, out_b)

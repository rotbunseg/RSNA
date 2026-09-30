"""Shape/sanity test for the image encoder, using pretrained=False so it runs
offline with no network access (no downloading real ImageNet weights) — fast
enough to run on every commit. This is NOT a substitute for actually training
on real data; it only catches shape/wiring bugs before you burn GPU time on them.
"""
import torch

from rsna_knee.models.image_encoder import ImageEncoder


def test_image_encoder_output_shape():
    torch.manual_seed(0)
    b, n_series, n_slices, hw = 2, 3, 4, 64
    encoder = ImageEncoder("resnet18", embed_dim=32, pretrained=False)
    image = torch.rand(b, n_series, n_slices, hw, hw)
    view_mask = torch.ones(b, n_series)
    view_mask[0, -1] = 0  # simulate one padded (missing) series for the first study

    out = encoder(image, view_mask)
    assert out["embedding"].shape == (b, 32)
    assert out["series_attention"].shape == (b, n_series)
    # Padded series should get ~zero attention weight once masked.
    assert out["series_attention"][0, -1].item() < 1e-4
    # Attention weights over real series should sum to ~1 per study.
    assert torch.allclose(out["series_attention"].sum(dim=-1), torch.ones(b), atol=1e-4)

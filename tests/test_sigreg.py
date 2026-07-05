"""Tests for SIGReg."""

import torch

from ewjepa import latent_diagnostics, sigreg


def _low_rank_unit_std(batch: int, dim: int, rank: int, seed: int) -> torch.Tensor:
    """Fake collapsed latents: low rank but unit std per dim."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(batch, rank, generator=gen)
    u, _, _ = torch.linalg.svd(torch.randn(dim, rank, generator=gen), full_matrices=False)
    z = w @ u.T
    z = z / z.std(dim=0, unbiased=False).clamp_min(1e-6)
    return z


def test_sigreg_small_for_gaussian():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4096, 32, generator=g)
    val = sigreg(z, generator=torch.Generator().manual_seed(1)).item()
    assert val < 1.0


def test_sigreg_large_for_collapse():
    g = torch.Generator().manual_seed(1)
    gaussian = sigreg(torch.randn(4096, 32, generator=g), generator=torch.Generator().manual_seed(2)).item()
    collapsed = sigreg(torch.zeros(4096, 32), generator=torch.Generator().manual_seed(2)).item()
    assert collapsed > 0.1
    assert collapsed > 10 * gaussian


def test_sigreg_large_for_low_rank_unit_std():
    g = torch.Generator().manual_seed(3)
    gaussian = sigreg(torch.randn(4096, 64, generator=g), generator=torch.Generator().manual_seed(4)).item()
    low_rank = sigreg(
        _low_rank_unit_std(4096, 64, rank=2, seed=5),
        generator=torch.Generator().manual_seed(6),
    ).item()
    assert low_rank > 10 * gaussian


def test_sigreg_is_differentiable():
    z = torch.randn(256, 16, requires_grad=True)
    sigreg(z).backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_diagnostics_detect_collapse():
    healthy = latent_diagnostics(torch.randn(2048, 32))
    collapsed = latent_diagnostics(torch.zeros(2048, 32) + 0.01 * torch.randn(1, 32))
    assert healthy["std"] > collapsed["std"]
    assert healthy["effective_rank"] > collapsed["effective_rank"]


def test_diagnostics_flag_low_rank():
    healthy = latent_diagnostics(torch.randn(2048, 64))
    low_rank = latent_diagnostics(_low_rank_unit_std(2048, 64, rank=2, seed=7))
    assert healthy["effective_rank"] > low_rank["effective_rank"]
    assert low_rank["sigreg"] > healthy["sigreg"]

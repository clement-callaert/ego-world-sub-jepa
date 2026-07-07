"""Tests for covariance decorrelation (anti low-rank collapse)."""

import torch

from ewjepa import EgoWorldConfig, EgoWorldJEPA, cov_decorrelation_loss


def _low_rank_unit_std(batch: int, dim: int, rank: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(batch, rank, generator=gen)
    u, _, _ = torch.linalg.svd(torch.randn(dim, rank, generator=gen), full_matrices=False)
    z = w @ u.T
    z = z / z.std(dim=0, unbiased=False).clamp_min(1e-6)
    return z


def test_cov_loss_positive_on_correlated_batch():
    low_rank = _low_rank_unit_std(512, 64, rank=2, seed=0)
    assert cov_decorrelation_loss(low_rank).item() > 0.0


def test_cov_loss_lower_for_isotropic_than_low_rank():
    iso = torch.randn(512, 64)
    low_rank = _low_rank_unit_std(512, 64, rank=2, seed=1)
    assert cov_decorrelation_loss(low_rank).item() > cov_decorrelation_loss(iso).item()


def test_cov_weight_zero_disables_term():
    cfg = EgoWorldConfig(mode="factored", cov_weight=0.0, img_size=64, proprio_dim=4)
    model = EgoWorldJEPA(cfg)
    out = model.compute_loss(
        torch.randn(4, 3, 3, 64, 64),
        torch.randn(4, 3, 4),
        torch.randn(4, 2, 2),
    )
    assert out["cov_loss"].item() == 0.0


def test_cov_weight_positive_in_compute_loss():
    cfg = EgoWorldConfig(mode="factored", cov_weight=1.0, img_size=64, proprio_dim=4)
    model = EgoWorldJEPA(cfg)
    out = model.compute_loss(
        torch.randn(8, 3, 3, 64, 64),
        torch.randn(8, 3, 4),
        torch.randn(8, 2, 2),
    )
    assert out["cov_loss"].item() >= 0.0

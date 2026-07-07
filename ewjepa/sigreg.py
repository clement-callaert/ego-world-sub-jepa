"""SIGReg: push the latents to look like a standard Gaussian N(0, I).

This is the anti collapse loss from LeJEPA. The idea is simple:
if every random 1D projection of the latents looks like N(0, 1),
then the latents are an isotropic Gaussian (Cramer Wold theorem).

We measure "looks like N(0, 1)" with the Epps Pulley test. It compares
the characteristic function of the data to the one of N(0, 1).
"""

from __future__ import annotations

import torch


def epps_pulley_1d(
    x: torch.Tensor,
    t_max: float = 5.0,
    n_points: int = 17,
) -> torch.Tensor:
    """Epps Pulley distance to N(0, 1) for each column of x.

    x has shape (N, M): N samples and M projection directions.
    Returns one score per direction, shape (M,). Lower means closer to N(0, 1).

    The characteristic function of N(0, 1) is real and equals exp(-t^2 / 2).
    The empirical one is the average of exp(i t x) over the samples.
    We integrate the squared distance between them, weighted by exp(-t^2 / 2).

    Note: this is the Epps Pulley statistic WITHOUT the usual factor N.
    That factor is only useful to get p values for a hypothesis test.
    As a training loss it just rescales the term, so we drop it and keep
    the loss on the same scale as the prediction loss.
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if x.dim() != 2:
        raise ValueError(f"epps_pulley_1d expects (N,) or (N, M), got {tuple(x.shape)}.")

    # integration grid, symmetric around 0 like in the paper
    t = torch.linspace(-t_max, t_max, n_points, device=x.device, dtype=x.dtype)  # (T,)
    target_cf = torch.exp(-0.5 * t.pow(2))  # CF of N(0, 1), also used as the weight

    # empirical characteristic function of each projection
    xt = x.unsqueeze(-1) * t  # (N, M, T)
    real_part = xt.cos().mean(dim=0)  # (M, T)
    imag_part = xt.sin().mean(dim=0)  # (M, T)

    # squared distance to the target CF, then weighted and integrated over t
    err = (real_part - target_cf).pow(2) + imag_part.pow(2)  # (M, T)
    weighted = err * target_cf
    return torch.trapezoid(weighted, t, dim=-1)  # (M,)


def sigreg(
    z: torch.Tensor,
    n_directions: int = 256,
    t_max: float = 5.0,
    n_points: int = 17,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SIGReg loss on a batch of latents z with shape (B, D).

    We draw random unit directions, project z on them, and check that each
    projection looks like N(0, 1). Lower is better (closer to N(0, I)).
    """
    if z.dim() != 2:
        raise ValueError(f"sigreg expects (B, D), got shape {tuple(z.shape)}.")

    z = z - z.mean(dim=0, keepdim=True)  # target mean is zero, so center first
    d = z.shape[1]

    # random projection directions, each normalized to unit norm
    dirs = torch.randn(d, n_directions, device=z.device, dtype=z.dtype, generator=generator)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-8)

    proj = z @ dirs  # (B, n_directions)
    stats = epps_pulley_1d(proj, t_max=t_max, n_points=n_points)  # (n_directions,)
    return stats.mean()


def cov_decorrelation_loss(z: torch.Tensor) -> torch.Tensor:
    """Push the latent dimensions to be uncorrelated (VICReg style).

    SIGReg already fights collapse, but this term gives a direct and cheap
    extra signal against low rank latents. We use the correlation matrix
    (not the covariance) so the value stays between 0 and 1 and does not
    depend on the scale of the latents.

    z has shape (B, D). Returns the mean squared off diagonal correlation.
    """
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    z = z / std  # each dimension now has unit variance

    n = max(z.shape[0], 1)
    corr = (z.T @ z) / n  # correlation matrix, (D, D)
    d = corr.shape[0]
    if d < 2:
        return z.new_zeros(())

    off_diag_sq = corr.pow(2).sum() - corr.diagonal().pow(2).sum()
    return off_diag_sq / (d * (d - 1))  # mean over all off diagonal pairs


@torch.no_grad()
def latent_diagnostics(z: torch.Tensor) -> dict[str, float]:
    """Quick health check of a latent batch: std, norm, rank and SIGReg."""
    z = z.detach().float()
    zc = z - z.mean(dim=0, keepdim=True)
    per_dim_std = zc.std(dim=0, unbiased=False)

    # effective rank = participation ratio of the covariance eigenvalues.
    # the eigenvalues are proportional to the squared singular values of zc.
    # value close to D means full rank, value near 0 means collapsed.
    # we do not clamp the eigenvalues up, so a zero latent reads as rank 0.
    eig = torch.linalg.svdvals(zc).pow(2)  # >= 0, proportional to cov eigenvalues
    effective_rank = (eig.sum().pow(2) / eig.pow(2).sum().clamp_min(1e-12)).item()

    return {
        "std": per_dim_std.mean().item(),
        "mean_norm": zc.norm(dim=1).mean().item(),
        "effective_rank": effective_rank,
        "sigreg": sigreg(z).item(),
    }

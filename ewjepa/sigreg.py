"""SIGReg: keep latents close to a standard Gaussian (anti-collapse).

Uses random 1D projections and the Epps-Pulley test (from LeJEPA).
"""

from __future__ import annotations

import torch


def _integration_grid(
    t_max: float,
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights for trapezoid integration."""
    if n_points % 2 == 0:
        raise ValueError(f"n_points must be odd for trapezoid quadrature, got {n_points}.")
    t = torch.linspace(0.0, t_max, n_points, device=device, dtype=dtype)
    dt = t_max / (n_points - 1)
    phi = torch.exp(-0.5 * t.pow(2))
    weights = torch.full((n_points,), 2.0 * dt, device=device, dtype=dtype)
    weights[0] = dt
    weights[-1] = dt
    weights = weights * phi
    return t, weights


def epps_pulley_1d(
    x: torch.Tensor,
    t_max: float = 5.0,
    n_points: int = 17,
) -> torch.Tensor:
    """Compare 1D samples to N(0,1). x is (N,) or (N,M)."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if x.dim() != 2:
        raise ValueError(f"epps_pulley_1d expects (N,) or (N, M); got {tuple(x.shape)}.")

    n_samples = x.shape[0]
    device, dtype = x.device, x.dtype
    t, weights = _integration_grid(t_max, n_points, device, dtype)
    phi = torch.exp(-0.5 * t.pow(2))

    # (N, M, T)
    x_t = x.unsqueeze(-1) * t
    cos_mean = torch.cos(x_t).mean(dim=0)
    sin_mean = torch.sin(x_t).mean(dim=0)

    err = (cos_mean - phi).pow(2) + sin_mean.pow(2)
    stat = (err * weights).sum(dim=-1) * n_samples
    return stat


def sigreg(
    z: torch.Tensor,
    n_directions: int = 256,
    t_max: float = 5.0,
    n_points: int = 17,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SIGReg loss on batch z (B,D). Lower is closer to N(0,I)."""
    if z.dim() != 2:
        raise ValueError(f"sigreg expects (B, D); got shape {tuple(z.shape)}.")

    # center batch (target mean is zero)
    z = z - z.mean(dim=0, keepdim=True)

    _, d = z.shape
    device, dtype = z.device, z.dtype

    dirs = torch.randn(d, n_directions, device=device, dtype=dtype, generator=generator)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-8)
    proj = z @ dirs

    stats = epps_pulley_1d(proj, t_max=t_max, n_points=n_points)
    return stats.mean()


@torch.no_grad()
def latent_diagnostics(z: torch.Tensor) -> dict[str, float]:
    """Check if latents collapsed (std, rank, sigreg)."""
    z = z.detach().float()
    zc = z - z.mean(dim=0, keepdim=True)
    per_dim_std = zc.std(dim=0, unbiased=False)

    cov = (zc.T @ zc) / max(z.shape[0], 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    denom = eig.pow(2).sum().clamp_min(1e-12)
    effective_rank = (eig.sum().pow(2) / denom).item()

    return {
        "std": per_dim_std.mean().item(),
        "mean_norm": zc.norm(dim=1).mean().item(),
        "effective_rank": effective_rank,
        "sigreg": sigreg(z).item(),
    }

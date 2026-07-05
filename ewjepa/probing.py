"""Linear probe on frozen latents.

Fit ridge regression to predict a target (e.g. object pose) from latents.
Used to check if pose info lives in z_world vs z_ego.
"""

from __future__ import annotations

import torch


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge regression with bias. x (N,D), y (N,T) -> W (D,T), b (T,)."""
    n, d = x.shape
    x_aug = torch.cat([x, torch.ones(n, 1, dtype=x.dtype, device=x.device)], dim=1)  # (N, D+1)
    eye = torch.eye(d + 1, dtype=x.dtype, device=x.device)
    eye[-1, -1] = 0.0  # no ridge on bias
    a = x_aug.T @ x_aug + ridge * n * eye
    theta = torch.linalg.solve(a, x_aug.T @ y)  # (D+1, T)
    return theta[:-1], theta[-1]


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """R^2 score, averaged over target dims."""
    ss_res = (y_true - y_pred).pow(2).sum(dim=0)
    ss_tot = (y_true - y_true.mean(dim=0)).pow(2).sum(dim=0).clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).mean().item()


def linear_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    test_frac: float = 0.2,
    ridge: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Train/test split ridge probe. Returns mse, rmse, r2, nrmse."""
    features = features.detach().float()
    targets = targets.detach().float()
    n = features.shape[0]

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(round(n * test_frac)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    # standardize features on train split
    mu, sd = features[train_idx].mean(0), features[train_idx].std(0).clamp_min(1e-6)
    x_tr = (features[train_idx] - mu) / sd
    x_te = (features[test_idx] - mu) / sd

    w, b = fit_ridge(x_tr, targets[train_idx], ridge=ridge)
    pred = x_te @ w + b

    mse = (pred - targets[test_idx]).pow(2).mean().item()
    target_std = targets[train_idx].std(0).clamp_min(1e-6).mean().item()
    rmse = mse ** 0.5
    return {
        "mse": mse,
        "rmse": rmse,
        "nrmse": rmse / target_std,
        "r2": r2_score(targets[test_idx], pred),
    }


def fit_pose_readout(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Fit ridge readout from latents to pose on all data."""
    features = features.detach().float()
    targets = targets.detach().float()
    mu = features.mean(0)
    sd = features.std(0).clamp_min(1e-6)
    x = (features - mu) / sd
    weight, bias = fit_ridge(x, targets, ridge=ridge)
    return {"weight": weight, "bias": bias, "mu": mu, "sd": sd}


def decode_pose(readout: dict[str, torch.Tensor], latents: torch.Tensor) -> torch.Tensor:
    """Apply a fitted pose readout to latents."""
    x = (latents - readout["mu"]) / readout["sd"]
    return x @ readout["weight"] + readout["bias"]

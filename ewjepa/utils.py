"""Helpers: seeding, device, checkpoints."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    cfg: dict,
    step: int,
    **extra,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = extra.pop("model_state", model.state_dict())
    payload = {"model": state, "cfg": cfg, "step": step, **extra}
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Normalizer:
    """Normalize with per-dim mean and std."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.eps = eps

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Normalizer":
        x = x.reshape(-1, x.shape[-1]).float()
        return cls(x.mean(0).cpu().numpy(), x.std(0).cpu().numpy())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        return (x - mean) / (std + self.eps)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(d["mean"], d["std"], d.get("eps", 1e-6))

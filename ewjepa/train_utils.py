"""Training speed helpers (CUDA, compile, prefetch)."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader


def configure_cuda(cudnn_benchmark: bool = True, allow_tf32: bool = True) -> None:
    """Enable cudnn benchmark and TF32 on CUDA."""
    if not torch.cuda.is_available():
        return
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    """Wrap model with torch.compile if enabled."""
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("[train] torch.compile is not available in this PyTorch build.")
        return model
    print("[train] torch.compile enabled (first steps will be slower while compiling).")
    return torch.compile(model)


def make_adamw(params, lr: float, weight_decay: float, device: torch.device) -> torch.optim.AdamW:
    """AdamW with fused CUDA kernel when available."""
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(params, **kwargs)


class BatchPrefetcher:
    """Prefetch next batch on a side CUDA stream."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self._iter: Iterator | None = None
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._next: dict[str, torch.Tensor] | None = None
        self._preload()

    def _next_batch(self) -> dict:
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)

    def _to_device(self, batch: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value.to(self.device, non_blocking=True)
            else:
                out[key] = value
        return out

    def _preload(self) -> None:
        batch = self._next_batch()
        if self._stream is None:
            self._next = self._to_device(batch)
            return
        with torch.cuda.stream(self._stream):
            self._next = self._to_device(batch)

    def next(self) -> dict[str, torch.Tensor]:
        if self._next is None:
            raise RuntimeError("BatchPrefetcher has no batch ready.")
        if self._stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
        batch = self._next
        self._preload()
        return batch

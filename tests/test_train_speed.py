"""Tests for train_utils and a tiny training loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.data import SyntheticPushTDataset
from ewjepa.train_utils import BatchPrefetcher, configure_cuda, maybe_compile


def test_configure_cuda_runs_on_cpu():
    configure_cuda(cudnn_benchmark=True, allow_tf32=True)


def test_batch_prefetcher_runs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticPushTDataset(num_episodes=16, num_steps=3, img_size=64)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    prefetcher = BatchPrefetcher(loader, device)
    batch = prefetcher.next()
    assert batch["pixels"].device.type == device.type
    assert batch["pixels"].shape[0] == 4


def test_few_training_steps():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda()
    model = EgoWorldJEPA(EgoWorldConfig(mode="factored", proprio_dim=4)).to(device)
    model = maybe_compile(model, enabled=False)
    model.train()

    dataset = SyntheticPushTDataset(num_episodes=32, num_steps=3, img_size=64, proprio_dim=4)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    prefetcher = BatchPrefetcher(loader, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for _ in range(3):
        batch = prefetcher.next()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model.compute_loss(batch["pixels"], batch["proprio"], batch["action"])
        scaler.scale(out["loss"]).backward()
        scaler.step(opt)
        scaler.update()

    assert torch.isfinite(out["loss"])

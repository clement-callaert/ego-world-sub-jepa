"""Smoke tests for eval, collapse detection, training."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.data import SyntheticPushTDataset, build_dataloader
from ewjepa.mpc_policy import LatentMPCPolicy
from ewjepa.planning import MPPIPlanner
from ewjepa.sigreg import latent_diagnostics, sigreg


class _FakeEnv:
    num_envs = 1
    action_space = type("S", (), {"shape": (1, 2)})()


def test_collapse_diagnostics_separate_healthy_from_collapsed():
    gen = torch.Generator().manual_seed(0)
    healthy = latent_diagnostics(torch.randn(512, 32, generator=gen))
    collapsed = latent_diagnostics(torch.zeros(512, 32) + 0.01 * torch.randn(1, 32))
    assert healthy["std"] > collapsed["std"]
    assert healthy["effective_rank"] > collapsed["effective_rank"]
    gaussian = sigreg(torch.randn(4096, 32, generator=torch.Generator().manual_seed(1))).item()
    flat = sigreg(torch.zeros(4096, 32), generator=torch.Generator().manual_seed(2)).item()
    assert flat > gaussian


def test_planner_model_smoke():
    device = torch.device("cpu")
    cfg = EgoWorldConfig(mode="factored", img_size=32, proprio_dim=4, action_dim=2)
    model = EgoWorldJEPA(cfg).to(device).eval()
    planner = MPPIPlanner(
        horizon=2,
        action_dim=2,
        n_samples=8,
        n_iters=1,
        device=device,
        generator=torch.Generator(device=device),
    )
    policy = LatentMPCPolicy(model=model, planner=planner, device=device)
    policy.set_env(_FakeEnv())

    info = {
        "pixels": np.random.randint(0, 255, (1, 1, 32, 32, 3), dtype=np.uint8),
        "proprio": np.random.randn(1, 1, 4).astype(np.float32),
        "goal": np.random.randint(0, 255, (1, 1, 32, 32, 3), dtype=np.uint8),
    }
    action = policy.get_action(info)
    assert action.shape == (1, 2)
    assert np.isfinite(action).all()
    assert action.min() >= -1.0 - 1e-5
    assert action.max() <= 1.0 + 1e-5


def _pusht_lance_readable() -> bool:
    for candidate in (Path("data/pusht.lance"), Path.cwd() / "data/pusht.lance"):
        if not candidate.is_dir():
            continue
        for sub in ("_transactions", "_versions", "data"):
            subdir = candidate / sub
            if not subdir.is_dir():
                break
            sample = next((p for p in subdir.iterdir() if p.is_file()), None)
            if sample is None or not os.access(sample, os.R_OK):
                break
        else:
            return True
    return False


@pytest.mark.skipif(not _pusht_lance_readable(), reason="pusht.lance not available or not readable")
def test_real_batch_rollout_and_overfit():
    loader = build_dataloader(
        "data/pusht.lance",
        num_steps=3,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        synthetic_fallback=False,
    )
    batch = next(iter(loader))
    device = torch.device("cpu")
    cfg = EgoWorldConfig(mode="factored", proprio_dim=4, action_dim=2)
    model = EgoWorldJEPA(cfg).to(device).train()

    with torch.no_grad():
        out0 = model.compute_loss(batch["pixels"], batch["proprio"], batch["action"])
    loss0 = out0["pred_loss"].item()
    assert np.isfinite(loss0)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(10):
        out = model.compute_loss(batch["pixels"], batch["proprio"], batch["action"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    loss1 = model.compute_loss(batch["pixels"], batch["proprio"], batch["action"])["pred_loss"].item()
    assert loss1 < loss0


def test_synthetic_overfit_smoke():
    ds = SyntheticPushTDataset(num_episodes=8, num_steps=3, img_size=64, proprio_dim=4)
    batch = default_collate([ds[i] for i in range(4)])
    device = torch.device("cpu")
    model = EgoWorldJEPA(EgoWorldConfig(mode="factored", proprio_dim=4, action_dim=2)).to(device).train()
    loss0 = model.compute_loss(batch["pixels"], batch["proprio"], batch["action"])["pred_loss"].item()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(10):
        out = model.compute_loss(batch["pixels"], batch["proprio"], batch["action"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    loss1 = model.compute_loss(batch["pixels"], batch["proprio"], batch["action"])["pred_loss"].item()
    assert loss1 < loss0


def test_swm_env_steps_without_crash():
    pytest.importorskip("stable_worldmodel")
    import stable_worldmodel as swm

    world = swm.World("swm/PushT-v1", num_envs=1, image_shape=(64, 64), max_episode_steps=10)
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    result = world.evaluate(episodes=3, seed=0)
    assert "success_rate" in result
    assert 0.0 <= float(result["success_rate"]) <= 100.0

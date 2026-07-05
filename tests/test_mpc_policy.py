"""Tests for LatentMPCPolicy."""

from __future__ import annotations

import numpy as np
import torch

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.mpc_policy import LatentMPCPolicy
from ewjepa.planning import MPPIPlanner


class _FakeEnv:
    num_envs = 2
    action_space = type("S", (), {"shape": (2, 2)})()


def test_latent_mpc_policy_get_action():
    device = torch.device("cpu")
    cfg = EgoWorldConfig(mode="factored", img_size=32, proprio_dim=2, action_dim=2)
    model = EgoWorldJEPA(cfg).to(device)
    planner = MPPIPlanner(
        horizon=4,
        action_dim=2,
        n_samples=8,
        n_iters=2,
        device=device,
        generator=torch.Generator(device=device),
    )
    policy = LatentMPCPolicy(model=model, planner=planner, device=device)
    policy.set_env(_FakeEnv())

    info = {
        "pixels": np.random.randint(0, 255, (2, 1, 32, 32, 3), dtype=np.uint8),
        "proprio": np.random.randn(2, 1, 2).astype(np.float32),
        "goal": np.random.randint(0, 255, (2, 1, 32, 32, 3), dtype=np.uint8),
    }
    action = policy.get_action(info)
    assert action.shape == (2, 2)
    assert np.isfinite(action).all()


def test_needs_flush_clears_nominal():
    device = torch.device("cpu")

    class SpyPlanner(MPPIPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.nominals_seen: list[torch.Tensor | None] = []

        def plan(self, cost_fn, nominal=None):
            self.nominals_seen.append(None if nominal is None else nominal.clone())
            return super().plan(cost_fn, nominal=nominal)

    cfg = EgoWorldConfig(mode="factored", img_size=32, proprio_dim=2, action_dim=2)
    model = EgoWorldJEPA(cfg).to(device)
    planner = SpyPlanner(
        horizon=4,
        action_dim=2,
        n_samples=8,
        n_iters=2,
        device=device,
        generator=torch.Generator(device=device),
    )
    policy = LatentMPCPolicy(model=model, planner=planner, device=device, warm_start=True)
    policy.set_env(_FakeEnv())

    stale = torch.ones(4, 2)
    policy._nominal = [stale.clone(), stale.clone()]

    info = {
        "pixels": np.random.randint(0, 255, (2, 1, 32, 32, 3), dtype=np.uint8),
        "proprio": np.random.randn(2, 1, 2).astype(np.float32),
        "goal": np.random.randint(0, 255, (2, 1, 32, 32, 3), dtype=np.uint8),
        "_needs_flush": np.array([True, False]),
    }
    policy.get_action(info)
    assert planner.nominals_seen[0] is None
    assert planner.nominals_seen[1] is not None
    assert torch.allclose(planner.nominals_seen[1], stale)

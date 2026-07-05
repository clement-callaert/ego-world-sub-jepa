"""Planners should minimize a simple known cost (distance to a target action)."""

import torch

from ewjepa import CEMPlanner, HermiteMPPIPlanner, MPPIPlanner

TARGET = 0.5


def _cost_fn(actions: torch.Tensor) -> torch.Tensor:
    return (actions - TARGET).pow(2).mean(dim=(1, 2))


def test_cem_reaches_target():
    planner = CEMPlanner(horizon=6, action_dim=2, n_samples=1024, n_elites=64, n_iters=6)
    _, first = planner.plan(_cost_fn)
    assert torch.allclose(first, torch.full((2,), TARGET), atol=0.1)


def test_mppi_reaches_target():
    planner = MPPIPlanner(
        horizon=6, action_dim=2, n_samples=2048, n_iters=6, temperature=0.05, noise_std=0.4
    )
    _, first = planner.plan(_cost_fn)
    assert torch.allclose(first, torch.full((2,), TARGET), atol=0.15)


def test_hermite_mppi_reaches_target():
    # need enough iters for weighted average to converge
    planner = HermiteMPPIPlanner(
        horizon=8,
        action_dim=2,
        n_nodes=4,
        n_samples=2048,
        n_iters=15,
        temperature=0.05,
        node_noise_std=0.5,
        beta1=2.0,
        beta2=2.0,
        generator=torch.Generator().manual_seed(0),
    )
    _, first = planner.plan(_cost_fn)
    assert torch.allclose(first, torch.full((2,), TARGET), atol=0.15)


def test_actions_respect_bounds():
    planner = MPPIPlanner(horizon=4, action_dim=3, n_samples=256, action_low=-1.0, action_high=1.0)
    seq, _ = planner.plan(_cost_fn)
    assert seq.max().item() <= 1.0 + 1e-6
    assert seq.min().item() >= -1.0 - 1e-6

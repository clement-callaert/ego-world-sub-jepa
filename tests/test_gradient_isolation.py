"""Check ego and world latents have the right gradient dependencies."""

import torch

from ewjepa import EgoWorldConfig, EgoWorldJEPA


def test_ego_dynamics_independent_of_world():
    cfg = EgoWorldConfig(mode="factored")
    model = EgoWorldJEPA(cfg)

    z_world = torch.randn(4, cfg.world_dim, requires_grad=True)
    z_ego = torch.randn(4, cfg.ego_dim, requires_grad=True)
    action = torch.randn(4, cfg.action_dim)

    _, z_ego_next = model.predictor(z_world, z_ego, action)
    grad_world = torch.autograd.grad(
        z_ego_next.sum(), z_world, retain_graph=True, allow_unused=True
    )[0]
    # z_ego_next should not depend on z_world
    assert grad_world is None or torch.allclose(grad_world, torch.zeros_like(grad_world))


def test_world_dynamics_depend_on_ego():
    cfg = EgoWorldConfig(mode="factored")
    model = EgoWorldJEPA(cfg)

    z_world = torch.randn(4, cfg.world_dim, requires_grad=True)
    z_ego = torch.randn(4, cfg.ego_dim, requires_grad=True)
    action = torch.randn(4, cfg.action_dim)

    z_world_next, _ = model.predictor(z_world, z_ego, action)
    grad_ego = torch.autograd.grad(z_world_next.sum(), z_ego, allow_unused=True)[0]
    assert grad_ego is not None
    assert grad_ego.abs().sum() > 0


def test_monolithic_has_no_ego_head():
    model = EgoWorldJEPA(EgoWorldConfig(mode="monolithic"))
    assert model.predictor.ego_head is None

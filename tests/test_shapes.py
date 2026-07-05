"""Model shape and integration tests."""

import torch

from ewjepa import EgoWorldConfig, EgoWorldJEPA


def _dummy_batch(b=4, t=3, cfg=None):
    cfg = cfg or EgoWorldConfig()
    pixels = torch.rand(b, t, cfg.in_chans, cfg.img_size, cfg.img_size)
    proprio = torch.rand(b, t, cfg.proprio_dim)
    action = torch.rand(b, t, cfg.action_dim) * 2 - 1
    return pixels, proprio, action


def test_encode_shapes_factored():
    cfg = EgoWorldConfig(mode="factored")
    model = EgoWorldJEPA(cfg)
    zw, ze = model.encode(torch.rand(5, 3, 64, 64), torch.rand(5, cfg.proprio_dim))
    assert zw.shape == (5, cfg.world_dim)
    assert ze.shape == (5, cfg.ego_dim)


def test_encode_shapes_monolithic():
    cfg = EgoWorldConfig(mode="monolithic")
    model = EgoWorldJEPA(cfg)
    zw, ze = model.encode(torch.rand(5, 3, 64, 64), torch.rand(5, cfg.proprio_dim))
    assert zw.shape == (5, cfg.world_dim)
    assert ze is None


def test_rollout_shapes():
    cfg = EgoWorldConfig(mode="factored")
    model = EgoWorldJEPA(cfg)
    zw = torch.randn(7, cfg.world_dim)
    ze = torch.randn(7, cfg.ego_dim)
    actions = torch.randn(7, 5, cfg.action_dim)
    zf, ef, traj = model.rollout(zw, ze, actions)
    assert zf.shape == (7, cfg.world_dim)
    assert ef.shape == (7, cfg.ego_dim)
    assert traj.shape == (7, 5, cfg.world_dim)


def test_get_cost_shapes():
    cfg = EgoWorldConfig(mode="factored")
    model = EgoWorldJEPA(cfg)
    n, h = 16, 4
    zw = torch.randn(n, cfg.world_dim)
    ze = torch.randn(n, cfg.ego_dim)
    actions = torch.randn(n, h, cfg.action_dim)
    goal = torch.randn(cfg.world_dim)
    cost = model.get_cost(zw, ze, actions, goal)
    assert cost.shape == (n,)
    assert torch.all(cost >= 0)


def test_compute_loss_runs_and_backprops():
    for mode in ("factored", "monolithic"):
        cfg = EgoWorldConfig(mode=mode)
        model = EgoWorldJEPA(cfg)
        pixels, proprio, action = _dummy_batch(cfg=cfg)
        out = model.compute_loss(pixels, proprio, action)
        assert out["loss"].ndim == 0
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


def test_param_budget_is_small():
    # factored model should stay under 20M params
    model = EgoWorldJEPA(EgoWorldConfig(mode="factored"))
    assert model.num_parameters() < 20_000_000

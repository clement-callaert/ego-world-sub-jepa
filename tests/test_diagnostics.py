"""Tests for ewjepa.diagnostics (rollout error and action sensitivity)."""

from __future__ import annotations

import pytest
import torch

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.diagnostics import _wrap_angle, action_sensitivity, rollout_pose_errors
from ewjepa.probing import fit_pose_readout


def _tiny_model(mode: str) -> EgoWorldJEPA:
    cfg = EgoWorldConfig(
        mode=mode,
        img_size=16,
        proprio_dim=4,
        patch_size=8,
        embed_dim=32,
        depth=1,
        num_heads=2,
        world_dim=16,
        ego_dim=8,
        ego_hidden=16,
        ego_depth=1,
        action_emb_dim=8,
        pred_hidden=16,
        pred_depth=1,
    )
    return EgoWorldJEPA(cfg).eval()


def _fake_readout(world_dim: int, out_dim: int = 3) -> dict[str, torch.Tensor]:
    feats = torch.randn(64, world_dim)
    targets = torch.randn(64, out_dim)
    return fit_pose_readout(feats, targets)


def _window(b=3, t=9, img=16):
    return (
        torch.rand(b, t, 3, img, img),
        torch.randn(b, t, 4),
        torch.rand(b, t, 2) * 2 - 1,
        torch.randn(b, t, 7) * 100,
    )


@pytest.mark.parametrize("mode", ["factored", "monolithic"])
def test_rollout_pose_errors_shapes(mode):
    model = _tiny_model(mode)
    readout = _fake_readout(model.cfg.world_dim)
    pixels, proprio, actions, states = _window()
    horizons = (1, 2, 4, 8)
    out = rollout_pose_errors(model, readout, pixels, proprio, actions, states, horizons=horizons)
    assert set(out.keys()) == set(horizons) | {0}
    assert out[0]["sq_xy"].shape == (3,)  # encode+decode reference, no dynamics
    for h in horizons:
        for key in ("sq_xy", "abs_angle", "sq_xy_disp", "abs_angle_disp", "sq_xy_zero"):
            assert out[h][key].shape == (3,)
            assert bool(torch.isfinite(out[h][key]).all())
            assert bool((out[h][key] >= 0).all())
            assert not out[h][key].requires_grad


def test_rollout_pose_errors_too_short_window():
    model = _tiny_model("factored")
    readout = _fake_readout(model.cfg.world_dim)
    pixels, proprio, actions, states = _window(t=4)
    with pytest.raises(ValueError, match="steps"):
        rollout_pose_errors(model, readout, pixels, proprio, actions, states, horizons=(8,))


def test_rollout_pose_errors_perfect_static_case():
    """A zero-weight readout predicts only its bias; if the true pose equals
    that bias at every step, the error is exactly zero."""
    model = _tiny_model("factored")
    d = model.cfg.world_dim
    bias = torch.tensor([10.0, 20.0, 0.5])
    readout = {
        "weight": torch.zeros(d, 3),
        "bias": bias,
        "mu": torch.zeros(d),
        "sd": torch.ones(d),
    }
    pixels, proprio, actions, states = _window(b=2, t=3)
    states[:, :, 2:5] = bias
    out = rollout_pose_errors(model, readout, pixels, proprio, actions, states, horizons=(1, 2))
    for h in (1, 2):
        assert torch.allclose(out[h]["sq_xy"], torch.zeros(2), atol=1e-8)
        assert torch.allclose(out[h]["abs_angle"], torch.zeros(2), atol=1e-6)


def test_wrap_angle():
    import math

    d = torch.tensor([0.0, math.pi, 2 * math.pi - 0.1, -2 * math.pi + 0.1])
    w = _wrap_angle(d)
    assert torch.allclose(w, torch.tensor([0.0, math.pi, 0.1, 0.1]), atol=1e-6)


@pytest.mark.parametrize("mode", ["factored", "monolithic"])
def test_action_sensitivity_shapes(mode):
    model = _tiny_model(mode)
    readout = _fake_readout(model.cfg.world_dim)
    b = 4
    zw = torch.randn(b, model.cfg.world_dim)
    ze = torch.randn(b, model.cfg.ego_dim) if mode == "factored" else None
    out = action_sensitivity(model, readout, zw, ze, horizon=3, n_sequences=8)
    assert out["xy_std"].shape == (2,)
    assert out["angle_std"].dim() == 0
    assert bool(torch.isfinite(out["xy_std"]).all())
    assert bool((out["xy_std"] >= 0).all())
    assert not out["xy_std"].requires_grad


def test_action_sensitivity_zero_for_constant_readout():
    """Degenerate case: a zero-weight readout decodes the same pose whatever
    the latent, so the measured sensitivity must be exactly 0."""
    model = _tiny_model("factored")
    d = model.cfg.world_dim
    readout = {
        "weight": torch.zeros(d, 3),
        "bias": torch.tensor([1.0, 2.0, 3.0]),
        "mu": torch.zeros(d),
        "sd": torch.ones(d),
    }
    zw = torch.randn(2, d)
    ze = torch.randn(2, model.cfg.ego_dim)
    out = action_sensitivity(model, readout, zw, ze, horizon=2, n_sequences=4)
    assert torch.allclose(out["xy_std"], torch.zeros(2), atol=1e-8)
    assert float(out["angle_std"]) == pytest.approx(0.0, abs=1e-8)


def test_action_sensitivity_deterministic_with_generator():
    model = _tiny_model("factored")
    readout = _fake_readout(model.cfg.world_dim)
    zw = torch.randn(2, model.cfg.world_dim)
    ze = torch.randn(2, model.cfg.ego_dim)
    outs = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(123)
        outs.append(action_sensitivity(model, readout, zw, ze, horizon=2, n_sequences=4, generator=gen))
    assert torch.allclose(outs[0]["xy_std"], outs[1]["xy_std"])

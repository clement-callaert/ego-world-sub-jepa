"""Tests for Hermite spline helpers."""

import torch

from ewjepa import clamp_node_velocity, hermite_basis, hermite_interpolate


def test_basis_endpoints():
    h00, h10, h01, h11 = hermite_basis(torch.tensor(0.0))
    assert torch.allclose(torch.tensor([h00, h10, h01, h11]), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    h00, h10, h01, h11 = hermite_basis(torch.tensor(1.0))
    assert torch.allclose(torch.tensor([h00, h10, h01, h11]), torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_interpolation_hits_node_positions():
    # horizon = K: query times match node times
    k, d = 4, 2
    nodes_q = torch.randn(1, k, d)
    nodes_v = torch.randn(1, k, d)
    traj = hermite_interpolate(nodes_q, nodes_v, horizon=k, dt=1.0)
    assert torch.allclose(traj[0, 0], nodes_q[0, 0], atol=1e-5)
    assert torch.allclose(traj[0, -1], nodes_q[0, -1], atol=1e-5)


def test_interpolation_reproduces_linear_ramp():
    # straight line with slope 1 should interpolate exactly
    k = 3
    q = torch.tensor([[0.0], [1.0], [2.0]]).unsqueeze(0)  # (1, K, 1)
    v = torch.ones(1, k, 1)  # slope 1 per unit time (dt = 1)
    traj = hermite_interpolate(q, v, horizon=5, dt=1.0)
    expected = torch.linspace(0.0, 2.0, 5).view(1, 5, 1)
    assert torch.allclose(traj, expected, atol=1e-5)


def test_velocity_clamp_respects_bounds():
    nodes_q = torch.tensor([[1.0], [0.0], [-1.0]]).unsqueeze(0)  # at/inside [-1, 1]
    nodes_v = torch.full((1, 3, 1), 10.0)
    clamped = clamp_node_velocity(nodes_q, nodes_v, dt=1.0, low=-1.0, high=1.0)
    # velocity is 0 at bounds
    assert torch.allclose(clamped[0, 0], torch.zeros(1), atol=1e-6)
    assert torch.allclose(clamped[0, 2], torch.zeros(1), atol=1e-6)
    # interior node can still move
    assert clamped[0, 1].abs().item() > 0


def test_batched_interpolation_shape():
    traj = hermite_interpolate(torch.randn(8, 4, 3), torch.randn(8, 4, 3), horizon=10, dt=1.0)
    assert traj.shape == (8, 10, 3)

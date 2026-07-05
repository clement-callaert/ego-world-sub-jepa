"""Tests for linear_probe."""

import torch

from ewjepa.probing import linear_probe


def test_probe_recovers_linear_signal():
    torch.manual_seed(0)
    x = torch.randn(2000, 16)
    w = torch.randn(16, 2)
    y = x @ w + 0.01 * torch.randn(2000, 2)  # almost linear
    metrics = linear_probe(x, y, ridge=1e-4)
    assert metrics["r2"] > 0.98
    assert metrics["nrmse"] < 0.1


def test_probe_low_r2_for_unrelated_target():
    torch.manual_seed(0)
    x = torch.randn(2000, 16)
    y = torch.randn(2000, 2)  # independent of x
    metrics = linear_probe(x, y)
    assert metrics["r2"] < 0.2

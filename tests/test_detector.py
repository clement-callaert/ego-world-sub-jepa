"""Tests for the supervised block pose detector."""

from __future__ import annotations

import math

import torch

from ewjepa.detector import BlockPoseDetector, load_detector, pose_to_target, save_detector


def test_detector_forward_and_predict_shapes():
    det = BlockPoseDetector(width=16, img_size=96)
    det.set_target_stats(torch.tensor([250.0, 250.0]), torch.tensor([90.0, 90.0]))
    det.eval()
    x = torch.rand(4, 3, 96, 96)

    with torch.no_grad():
        raw = det(x)
        assert raw.shape == (4, 4)  # norm x, norm y, sin, cos
        pose = det.predict(x)
    assert pose.shape == (4, 3)  # x px, y px, angle rad
    # The angle is wrapped into [0, 2 pi).
    assert float(pose[:, 2].min()) >= 0.0
    assert float(pose[:, 2].max()) < 2.0 * math.pi + 1e-4


def test_detector_runs_at_other_resolution():
    # Adaptive pooling should let the same detector accept a 64 px image.
    det = BlockPoseDetector(width=16, img_size=96)
    det.set_target_stats(torch.tensor([250.0, 250.0]), torch.tensor([90.0, 90.0]))
    pose = det.predict(torch.rand(2, 3, 64, 64))
    assert pose.shape == (2, 3)


def test_pose_to_target_roundtrip():
    xy_mean = torch.tensor([250.0, 250.0])
    xy_std = torch.tensor([100.0, 100.0])
    pose = torch.tensor([[350.0, 150.0, 0.5], [150.0, 450.0, 3.0]])
    target = pose_to_target(pose, xy_mean, xy_std)
    assert target.shape == (2, 4)
    # Normalized xy recovers the raw xy.
    xy = target[:, :2] * xy_std + xy_mean
    assert torch.allclose(xy, pose[:, :2], atol=1e-4)
    # sin, cos recover the angle.
    angle = torch.atan2(target[:, 2], target[:, 3]) % (2.0 * math.pi)
    expected = pose[:, 2] % (2.0 * math.pi)
    assert torch.allclose(angle, expected, atol=1e-4)


def test_detector_save_load_roundtrip(tmp_path):
    det = BlockPoseDetector(width=16, img_size=96)
    det.set_target_stats(torch.tensor([200.0, 300.0]), torch.tensor([80.0, 70.0]))
    det.eval()
    path = tmp_path / "detector.pt"
    save_detector(path, det)

    loaded = load_detector(path)
    assert not loaded.training  # loaded in eval mode
    assert torch.allclose(loaded.xy_mean, det.xy_mean)
    x = torch.rand(3, 3, 96, 96)
    with torch.no_grad():
        assert torch.allclose(det(x), loaded(x), atol=1e-5)

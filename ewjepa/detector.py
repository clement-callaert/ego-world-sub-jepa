"""Supervised block pose detector for PushT.

The JEPA world latent only localizes the block to about 45 to 67 px, while the
task success check needs the block within about 14 px. That precision is simply
not in the latent, so a linear or MLP readout cannot recover it (measured).

This module is a small convolutional network that reads the block pose straight
from the image. It is trained on the dataset state labels (which carry the true
block x, y and angle) and reaches a few pixels of error. The MPC uses it as the
block sensor, while the JEPA world model still supplies the push dynamics.

The network predicts x and y in a normalized frame (stored mean and std) and the
angle as a (sin, cos) pair so the wrap around at 2 pi is handled smoothly. The
`predict` method converts back to raw pixels and radians in [0, 2 pi).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


class BlockPoseDetector(nn.Module):
    """Image (B, C, H, W) in [0, 1] to block pose (x px, y px, angle rad)."""

    def __init__(self, in_chans: int = 3, width: int = 32, img_size: int = 96):
        super().__init__()
        self.img_size = img_size

        # A short convolutional stack that halves the resolution four times.
        # GroupNorm is used on purpose: unlike BatchNorm it behaves identically
        # in train and eval mode, so the detector cannot suffer the train/eval
        # mismatch that broke the world model head earlier in this project.
        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=c_out),
                nn.GELU(),
            )

        self.features = nn.Sequential(
            block(in_chans, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 4),
        )
        # Adaptive pooling makes the head independent of the input resolution,
        # so the same detector works at 64 px or 96 px.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(width * 4, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, 4),  # normalized x, normalized y, sin, cos
        )

        # Target normalization for x and y. Filled in during training so the
        # network regresses well scaled values. Registered as buffers so they
        # travel with the state dict.
        self.register_buffer("xy_mean", torch.zeros(2))
        self.register_buffer("xy_std", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw network output (B, 4): norm x, norm y, sin, cos."""
        feats = self.features(x)
        feats = self.pool(feats).flatten(1)
        return self.head(feats)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the block pose in raw units (B, 3): x px, y px, angle rad.

        The angle is wrapped into [0, 2 pi) to match the environment, which
        reports `block.angle % (2 pi)`.
        """
        out = self.forward(x)
        xy = out[:, :2] * self.xy_std + self.xy_mean
        angle = torch.atan2(out[:, 2], out[:, 3]) % (2.0 * math.pi)
        return torch.cat([xy, angle.unsqueeze(-1)], dim=-1)

    def set_target_stats(self, xy_mean: torch.Tensor, xy_std: torch.Tensor) -> None:
        """Store the x, y normalization used during training."""
        self.xy_mean.copy_(xy_mean.detach().to(self.xy_mean))
        self.xy_std.copy_(xy_std.detach().clamp_min(1e-6).to(self.xy_std))


def pose_to_target(block_pose: torch.Tensor, xy_mean: torch.Tensor, xy_std: torch.Tensor) -> torch.Tensor:
    """Turn true block pose (B, 3) into the network target (B, 4).

    x and y are normalized with the given stats, the angle becomes (sin, cos).
    """
    xy = (block_pose[:, :2] - xy_mean) / xy_std.clamp_min(1e-6)
    angle = block_pose[:, 2]
    return torch.cat([xy, torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)], dim=-1)


def save_detector(path: str | Path, detector: BlockPoseDetector) -> None:
    """Save weights and the config needed to rebuild the detector."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": detector.state_dict(),
        "img_size": detector.img_size,
        "in_chans": detector.features[0][0].in_channels,
        "width": detector.features[0][0].out_channels,
    }
    torch.save(payload, path)


def load_detector(path: str | Path, map_location: str | torch.device = "cpu") -> BlockPoseDetector:
    """Rebuild a detector from a saved file and put it in eval mode."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    detector = BlockPoseDetector(
        in_chans=payload.get("in_chans", 3),
        width=payload.get("width", 32),
        img_size=payload.get("img_size", 96),
    )
    detector.load_state_dict(payload["state_dict"])
    detector.to(map_location)
    detector.eval()
    return detector

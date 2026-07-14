"""Action-conditioned rollout diagnostics for a trained world model.

Two metrics that a linear probe R2 does not capture:

1. rollout_pose_errors: open-loop multi-step prediction error. Encode the
   first frame, roll the predictor forward with the REAL action sequence from
   the dataset, decode the block pose at each step with a frozen ridge
   readout (the same readout the planner uses), and compare with the true
   simulator block pose.

2. action_sensitivity: from one initial latent, roll out K random action
   sequences and measure how much the decoded block pose at the final
   horizon spreads. Near zero means the predictor ignores the actions, which
   makes planning impossible no matter how good the probe R2 is.

Both run under torch.no_grad and expect a frozen model and readout.
"""

from __future__ import annotations

import math

import torch

from .probing import decode_pose


def _wrap_angle(d: torch.Tensor) -> torch.Tensor:
    """Smallest absolute angle difference, handling the wrap at 2 pi."""
    d = torch.abs(d) % (2.0 * math.pi)
    return torch.minimum(d, 2.0 * math.pi - d)


@torch.no_grad()
def rollout_pose_errors(
    model,
    readout: dict[str, torch.Tensor],
    pixels: torch.Tensor,
    proprio: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    horizons: tuple[int, ...] = (1, 2, 4, 8),
    block_slice: tuple[int, int] = (2, 5),
) -> dict[int, dict[str, torch.Tensor]]:
    """Per-horizon squared errors of the decoded block pose after a rollout.

    pixels (B,T,C,H,W), proprio (B,T,P) already normalized like at train time,
    actions (B,T,A) raw dataset actions, states (B,T,S) raw simulator states.
    T must be at least max(horizons) + 1.

    Returns {h: {"sq_xy": (B,), "abs_angle": (B,), "sq_xy_disp": (B,),
    "abs_angle_disp": (B,)}} plus an entry for h=0 with the encode+decode
    error only (no dynamics), as a readout-quality reference.

    sq_xy is the squared euclidean error of the absolute decoded block xy in
    state units (px on the 512 board); abs_angle the wrapped absolute angle
    error in radians. The *_disp variants compare predicted vs true block
    DISPLACEMENT from step 0: this cancels the per-frame readout bias and is
    what the planner actually uses (displacement mode anchored on the
    detector, see LatentMPCPolicy). Aggregate with rmse = sqrt(mean(sq_xy)).
    """
    h_max = max(horizons)
    if pixels.shape[1] < h_max + 1:
        raise ValueError(f"need at least {h_max + 1} steps, got {pixels.shape[1]}")

    z_world, z_ego = model.encode(pixels[:, 0], proprio[:, 0])
    _, _, world_traj, _ = model.rollout(z_world, z_ego, actions[:, :h_max])
    decoded = decode_pose(readout, world_traj)  # (B, h_max, 3)
    decoded_0 = decode_pose(readout, z_world)   # (B, 3), current frame

    sl = slice(block_slice[0], block_slice[1])
    true_0 = states[:, 0, sl].to(decoded)
    out: dict[int, dict[str, torch.Tensor]] = {
        0: {
            "sq_xy": (decoded_0[:, :2] - true_0[:, :2]).pow(2).sum(dim=-1),
            "abs_angle": _wrap_angle(decoded_0[:, 2] - true_0[:, 2]),
        }
    }
    for h in horizons:
        true_pose = states[:, h, sl].to(decoded)  # pose after h actions
        pred_pose = decoded[:, h - 1]
        pred_disp = pred_pose - decoded_0
        true_disp = true_pose - true_0
        out[h] = {
            "sq_xy": (pred_pose[:, :2] - true_pose[:, :2]).pow(2).sum(dim=-1),
            "abs_angle": _wrap_angle(pred_pose[:, 2] - true_pose[:, 2]),
            "sq_xy_disp": (pred_disp[:, :2] - true_disp[:, :2]).pow(2).sum(dim=-1),
            "abs_angle_disp": _wrap_angle(pred_disp[:, 2] - true_disp[:, 2]),
            # error of the trivial "block never moves" predictor, as a floor
            "sq_xy_zero": true_disp[:, :2].pow(2).sum(dim=-1),
        }
    return out


@torch.no_grad()
def action_sensitivity(
    model,
    readout: dict[str, torch.Tensor],
    z_world: torch.Tensor,
    z_ego: torch.Tensor | None,
    horizon: int = 8,
    n_sequences: int = 32,
    action_low: float = -1.0,
    action_high: float = 1.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Spread of the predicted block pose under random action sequences.

    z_world (B,D) and z_ego (B,E) or None are the initial latents of B states.
    For each state, roll out n_sequences action sequences drawn uniformly in
    [action_low, action_high] for `horizon` steps and decode the block pose at
    the final step.

    Returns per-state stds averaged over states:
      "xy_std":    (2,) std in state units of the decoded block xy
      "angle_std": scalar std of the decoded angle
    Normalize xy_std by the dataset block xy std to get a unitless number; a
    value near 0 means the predictor is blind to the action.
    """
    b, d = z_world.shape
    k = n_sequences
    action_dim = model.cfg.action_dim
    device = z_world.device

    acts = torch.rand(b * k, horizon, action_dim, device=device, generator=generator)
    acts = action_low + (action_high - action_low) * acts

    zw = z_world.repeat_interleave(k, dim=0)
    ze = z_ego.repeat_interleave(k, dim=0) if z_ego is not None else None
    zw_final, _, _, _ = model.rollout(zw, ze, acts)
    pose = decode_pose(readout, zw_final).reshape(b, k, -1)  # (B, K, 3)

    xy_std = pose[..., :2].std(dim=1, unbiased=False).mean(dim=0)  # (2,)
    angle_std = pose[..., 2].std(dim=1, unbiased=False).mean()
    return {"xy_std": xy_std, "angle_std": angle_std}

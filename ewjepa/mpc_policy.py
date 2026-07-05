"""MPC policy for stable-worldmodel World.evaluate.

Encodes obs and goal to latents, plans with CEM/MPPI/Hermite, minimizes latent distance to goal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ewjepa.probing import decode_pose

try:
    from stable_worldmodel.policy import BasePolicy
except ImportError:  # pragma: no cover - optional at import time
    BasePolicy = object  # type: ignore[misc, assignment]


def _as_tensor(x: Any, device: torch.device) -> torch.Tensor:
    t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    return t.to(device)


def _squeeze_env_time(x: torch.Tensor) -> torch.Tensor:
    """SWM shape (E,1,...) -> (E,...)."""
    if x.dim() >= 2 and x.shape[1] == 1:
        x = x.squeeze(1)
    return x


def _to_nchw_float(x: torch.Tensor) -> torch.Tensor:
    """Convert to (E,C,H,W) float in [0,1]."""
    x = _squeeze_env_time(x)
    if x.dim() == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
        x = x.permute(0, 3, 1, 2)
    elif x.dim() == 3 and x.shape[-1] in (1, 3, 4) and x.shape[0] not in (1, 3, 4):
        x = x.permute(2, 0, 1).unsqueeze(0)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    return x.float()


class LatentMPCPolicy(BasePolicy):
    """Plan in the learned latent world model."""

    def __init__(
        self,
        model,
        planner,
        device: torch.device,
        image_key: str = "pixels",
        proprio_key: str = "proprio",
        goal_image_key: str = "goal",
        goal_proprio_key: str = "goal_proprio",
        proprio_normalizer=None,
        warm_start: bool = True,
        pose_readout: dict[str, torch.Tensor] | None = None,
        pose_cost_weight: float = 1.0,
        pose_scale: float = 512.0,
        goal_pose_key: str = "goal_pose",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model.eval()
        self.planner = planner
        self.device = device
        self.image_key = image_key
        self.proprio_key = proprio_key
        self.goal_image_key = goal_image_key
        self.goal_proprio_key = goal_proprio_key
        self.proprio_normalizer = proprio_normalizer
        self.warm_start = warm_start
        self.pose_readout = pose_readout
        self.pose_cost_weight = pose_cost_weight
        self.pose_scale = pose_scale
        self.goal_pose_key = goal_pose_key
        self._nominal: list[torch.Tensor | None] = []

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        n_envs = getattr(env, "num_envs", 1)
        self.reset(n_envs)

    def reset(self, num_envs: int = 1) -> None:
        self._nominal = [None] * num_envs

    def _proprio(self, obs: dict, key: str, batch: int) -> torch.Tensor:
        if key in obs and obs[key] is not None:
            p = _squeeze_env_time(_as_tensor(obs[key], self.device)).float()
            if p.dim() == 1:
                p = p.unsqueeze(0)
        else:
            p = torch.zeros(batch, self.model.cfg.proprio_dim, device=self.device)
        if self.proprio_normalizer is not None:
            p = self.proprio_normalizer(p)
        return p

    @torch.no_grad()
    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert self.goal_image_key in info_dict or "goal" in info_dict, (
            f"Goal image key {self.goal_image_key!r} missing from info_dict"
        )

        # accept raw numpy or preprocessed tensors from SWM
        if hasattr(self, "transform") and self.transform:
            info_dict = self._prepare_info(info_dict)

        # reset warm-start on episode reset
        needs_flush = info_dict.pop("_needs_flush", None)
        if needs_flush is not None:
            for i, flush in enumerate(needs_flush):
                if flush and i < len(self._nominal):
                    self._nominal[i] = None

        goal_key = self.goal_image_key if self.goal_image_key in info_dict else "goal"
        pixels = _to_nchw_float(_as_tensor(info_dict[self.image_key], self.device))
        e = pixels.shape[0]
        if len(self._nominal) != e:
            self.reset(e)

        proprio = self._proprio(info_dict, self.proprio_key, e)
        z_world, z_ego = self.model.encode(pixels, proprio)

        goal_pixels = _to_nchw_float(_as_tensor(info_dict[goal_key], self.device))
        if goal_pixels.shape[0] == 1 and e > 1:
            goal_pixels = goal_pixels.expand(e, -1, -1, -1)
        goal_proprio = self._proprio(info_dict, self.goal_proprio_key, e)
        goal_world, _ = self.model.encode(goal_pixels, goal_proprio)

        goal_pose = None
        if self.pose_readout is not None and self.goal_pose_key in info_dict:
            raw_pose = _squeeze_env_time(_as_tensor(info_dict[self.goal_pose_key], self.device)).float()
            if raw_pose.dim() == 1:
                raw_pose = raw_pose.unsqueeze(0)
            goal_pose = raw_pose[0, :3]

        actions = []
        for i in range(e):
            zw_i = z_world[i : i + 1]
            ze_i = z_ego[i : i + 1] if z_ego is not None else None
            goal_i = goal_world[i]
            goal_pose_i = goal_pose

            def cost_fn(
                cand: torch.Tensor,
                zw_i=zw_i,
                ze_i=ze_i,
                goal_i=goal_i,
                goal_pose_i=goal_pose_i,
            ) -> torch.Tensor:
                n = cand.shape[0]
                zw = zw_i.expand(n, -1)
                ze = ze_i.expand(n, -1) if ze_i is not None else None
                costs = self.model.get_cost(zw, ze, cand, goal_i)
                if self.pose_readout is not None and goal_pose_i is not None:
                    _, _, traj = self.model.rollout(zw, ze, cand)
                    pred_pose = decode_pose(self.pose_readout, traj)
                    pose_err = ((pred_pose - goal_pose_i) / self.pose_scale).pow(2).mean(dim=(1, 2))
                    costs = costs + self.pose_cost_weight * pose_err
                return costs

            nominal, first = self.planner.plan(cost_fn, nominal=self._nominal[i])
            self._nominal[i] = nominal if self.warm_start else None
            actions.append(first)

        action = torch.stack(actions, dim=0)
        if hasattr(self.env, "action_space"):
            target_shape = self.env.action_space.shape
            if len(target_shape) == 2 and target_shape[1] == 1:
                action = action.unsqueeze(1)

        return action.cpu().numpy()

    get_actions = get_action  # alias for tests

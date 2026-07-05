"""Factorized latent world model (Ego-World JEPA).

Factored mode: separate world (pixels) and ego (proprio) latents.
Monolithic mode: one combined latent (LeWM-style baseline).
Training uses prediction loss plus SIGReg. No pixel reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import EgoMLP, WorldViT
from .predictor import Predictor
from .sigreg import sigreg


@dataclass
class EgoWorldConfig:
    mode: str = "factored"  # "factored" or "monolithic"
    # image and proprio sizes
    img_size: int = 64
    in_chans: int = 3
    proprio_dim: int = 2
    action_dim: int = 2
    # world encoder (small ViT)
    patch_size: int = 8
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 2.0
    world_dim: int = 192
    # world head norm: "batchnorm" (LeWM) or "none" (SIGReg can see collapse)
    world_head_norm: str = "none"
    # ego encoder (MLP)
    ego_dim: int = 32
    ego_hidden: int = 128
    ego_depth: int = 2
    # predictor
    action_emb_dim: int = 64
    pred_hidden: int = 256
    pred_depth: int = 2
    # loss weights
    sigreg_mix: float = 0.1
    ego_loss_weight: float = 0.1
    stop_grad_target: bool = False
    variance_weight: float = 0.0  # optional std floor on z_world
    variance_target: float = 0.5
    sigreg_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("factored", "monolithic"):
            raise ValueError(f"mode must be 'factored' or 'monolithic', got {self.mode!r}.")


class EgoWorldJEPA(nn.Module):
    def __init__(self, cfg: EgoWorldConfig):
        super().__init__()
        self.cfg = cfg
        self.factored = cfg.mode == "factored"

        self.world_encoder = WorldViT(
            img_size=cfg.img_size,
            in_chans=cfg.in_chans,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            out_dim=cfg.world_dim,
            head_norm=cfg.world_head_norm,
        )

        if self.factored:
            self.ego_encoder = EgoMLP(cfg.proprio_dim, cfg.ego_dim, cfg.ego_hidden, cfg.ego_depth)
            self.proprio_proj = None
            ego_dim = cfg.ego_dim
        else:
            # monolithic: add proprio into the single world latent
            self.ego_encoder = None
            self.proprio_proj = nn.Linear(cfg.proprio_dim, cfg.world_dim)
            ego_dim = 0

        self.predictor = Predictor(
            world_dim=cfg.world_dim,
            ego_dim=ego_dim,
            action_dim=cfg.action_dim,
            action_emb_dim=cfg.action_emb_dim,
            hidden_dim=cfg.pred_hidden,
            depth=cfg.pred_depth,
        )

    # encode

    def encode(
        self, pixels: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode one step. pixels (B,C,H,W), proprio (B,D) -> z_world, z_ego (or None)."""
        z_world = self.world_encoder(pixels)
        if self.factored:
            return z_world, self.ego_encoder(proprio)
        return z_world + self.proprio_proj(proprio), None

    def encode_sequence(
        self, pixels: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode a (B, T, ...) sequence -> (B, T, world_dim), (B, T, ego_dim|None)."""
        b, t = pixels.shape[:2]
        pix = pixels.reshape(b * t, *pixels.shape[2:])
        pro = proprio.reshape(b * t, *proprio.shape[2:])
        z_world, z_ego = self.encode(pix, pro)
        z_world = z_world.reshape(b, t, -1)
        z_ego = z_ego.reshape(b, t, -1) if z_ego is not None else None
        return z_world, z_ego

    # rollout

    def rollout(
        self,
        z_world: torch.Tensor,
        z_ego: torch.Tensor | None,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Roll out in latent space. Returns final latents and (B,H,world_dim) traj."""
        traj = []
        for h in range(actions.shape[1]):
            z_world, z_ego = self.predictor(z_world, z_ego, actions[:, h])
            traj.append(z_world)
        return z_world, z_ego, torch.stack(traj, dim=1)

    # planning

    def get_cost(
        self,
        z_world: torch.Tensor,
        z_ego: torch.Tensor | None,
        actions: torch.Tensor,
        goal_world: torch.Tensor,
    ) -> torch.Tensor:
        """Cost = mean squared distance to goal latent over the rollout. Returns (N,)."""
        _, _, z_world_traj = self.rollout(z_world, z_ego, actions)
        if goal_world.dim() == 1:
            goal_world = goal_world.unsqueeze(0)
        goal = goal_world.unsqueeze(1)
        # mean over all rollout steps, not only the last one
        return (z_world_traj - goal).pow(2).mean(dim=(1, 2))

    # loss

    def compute_loss(self, pixels: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor) -> dict:
        """Training loss on a (B,T,...) window. Returns loss dict with parts."""
        cfg = self.cfg
        b, t = pixels.shape[:2]
        if t < 2:
            raise ValueError("compute_loss needs at least 2 time steps per window.")

        z_world_all, z_ego_all = self.encode_sequence(pixels, proprio)

        # roll out from the first frame
        z_world = z_world_all[:, 0]
        z_ego = z_ego_all[:, 0] if z_ego_all is not None else None
        pred_world, pred_ego = [], []
        for h in range(t - 1):
            z_world, z_ego = self.predictor(z_world, z_ego, action[:, h])
            pred_world.append(z_world)
            if z_ego is not None:
                pred_ego.append(z_ego)
        pred_world = torch.stack(pred_world, dim=1)  # (B, T-1, world_dim)

        target_world = z_world_all[:, 1:]
        if cfg.stop_grad_target:
            target_world = target_world.detach()
        pred_loss = F.mse_loss(pred_world, target_world)

        # optional ego rollout loss
        ego_loss = pred_world.new_zeros(())
        if self.factored and cfg.ego_loss_weight > 0 and pred_ego:
            pred_ego_t = torch.stack(pred_ego, dim=1)
            target_ego = z_ego_all[:, 1:]
            if cfg.stop_grad_target:
                target_ego = target_ego.detach()
            ego_loss = F.mse_loss(pred_ego_t, target_ego)

        # SIGReg on encoded latents (anti-collapse)
        sig_world = sigreg(z_world_all.reshape(b * t, -1), **cfg.sigreg_kwargs)
        if self.factored and z_ego_all is not None:
            sig_ego = sigreg(z_ego_all.reshape(b * t, -1), **cfg.sigreg_kwargs)
            # lighter weight on ego stream
            sig = sig_world + 0.5 * sig_ego
        else:
            sig_ego = sig_world.new_zeros(())
            sig = sig_world

        var_loss = pred_world.new_zeros(())
        if cfg.variance_weight > 0:
            per_dim_std = z_world_all.reshape(b * t, -1).std(dim=0, unbiased=False)
            var_loss = F.relu(cfg.variance_target - per_dim_std).mean()

        mix = cfg.sigreg_mix
        total = (
            (1.0 - mix) * pred_loss
            + mix * sig
            + cfg.ego_loss_weight * ego_loss
            + cfg.variance_weight * var_loss
        )
        return {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "ego_loss": ego_loss.detach(),
            "sigreg": sig.detach(),
            "sigreg_world": sig_world.detach(),
            "sigreg_ego": sig_ego.detach(),
            "var_loss": var_loss.detach(),
        }

    # utils

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

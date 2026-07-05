"""Latent dynamics predictor (no pixel decoding).

Predicts next world and ego latents from current latents and action.
Uses residual updates (delta) for stable training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualMLP(nn.Module):
    """MLP that outputs a residual of size out_dim."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """World and ego dynamics heads. Set ego_dim=0 for monolithic mode."""

    def __init__(
        self,
        world_dim: int,
        ego_dim: int,
        action_dim: int,
        action_emb_dim: int = 64,
        hidden_dim: int = 256,
        depth: int = 2,
    ):
        super().__init__()
        self.world_dim = world_dim
        self.ego_dim = ego_dim

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, action_emb_dim),
            nn.GELU(),
            nn.Linear(action_emb_dim, action_emb_dim),
        )

        # world: z_world, z_ego, action -> delta z_world
        self.world_head = _ResidualMLP(
            in_dim=world_dim + ego_dim + action_emb_dim,
            out_dim=world_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

        # ego: z_ego, action -> delta z_ego (missing if ego_dim == 0)
        self.ego_head = (
            _ResidualMLP(ego_dim + action_emb_dim, ego_dim, hidden_dim, depth)
            if ego_dim > 0
            else None
        )

    def forward(
        self,
        z_world: torch.Tensor,
        z_ego: torch.Tensor | None,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One step forward. Returns (z_world_next, z_ego_next or None)."""
        a = self.action_embed(action)

        if self.ego_head is not None:
            if z_ego is None:
                raise ValueError("z_ego must be provided when ego_dim > 0.")
            world_in = torch.cat([z_world, z_ego, a], dim=-1)
            z_ego_next = z_ego + self.ego_head(torch.cat([z_ego, a], dim=-1))
        else:
            world_in = torch.cat([z_world, a], dim=-1)
            z_ego_next = None

        z_world_next = z_world + self.world_head(world_in)
        return z_world_next, z_ego_next

"""Planners for the latent world model (CEM, MPPI, Hermite MPPI).

Each planner samples action sequences, scores them with cost_fn, and returns
the best plan plus the first action for MPC.
"""

from __future__ import annotations

from typing import Callable

import torch

CostFn = Callable[[torch.Tensor], torch.Tensor]


# Hermite


def hermite_basis(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cubic Hermite basis at local time s in [0, 1]."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return h00, h10, h01, h11


def hermite_interpolate(
    nodes_q: torch.Tensor,
    nodes_v: torch.Tensor,
    horizon: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Build a length-H trajectory from K Hermite nodes. Returns (..., H, D)."""
    if nodes_q.shape != nodes_v.shape:
        raise ValueError("nodes_q and nodes_v must share shape.")
    k = nodes_q.shape[-2]
    if k < 2:
        raise ValueError("Hermite interpolation needs at least 2 nodes.")
    device, dtype = nodes_q.device, nodes_q.dtype

    times = torch.linspace(0.0, (k - 1) * dt, horizon, device=device, dtype=dtype)
    seg = torch.clamp((times / dt).floor().long(), 0, k - 2)  # (H,)
    s = (times - seg.to(dtype) * dt) / dt  # local time in [0, 1], shape (H,)
    h00, h10, h01, h11 = hermite_basis(s)  # each (H,)

    # segment endpoints for each output time
    q_k = nodes_q.index_select(-2, seg)  # (..., H, D)
    q_k1 = nodes_q.index_select(-2, seg + 1)
    v_k = nodes_v.index_select(-2, seg)
    v_k1 = nodes_v.index_select(-2, seg + 1)

    # broadcast basis to batch dims
    shape = [1] * (nodes_q.dim() - 2) + [horizon, 1]
    h00 = h00.reshape(shape)
    h10 = h10.reshape(shape)
    h01 = h01.reshape(shape)
    h11 = h11.reshape(shape)

    return h00 * q_k + h10 * (dt * v_k) + h01 * q_k1 + h11 * (dt * v_k1)


def clamp_node_velocity(
    nodes_q: torch.Tensor,
    nodes_v: torch.Tensor,
    dt: float,
    low: float,
    high: float,
) -> torch.Tensor:
    """Clamp node velocity so the spline stays in [low, high]."""
    margin = torch.minimum(high - nodes_q, nodes_q - low).clamp_min(0.0)
    bound = margin / (dt / 2.0)
    return torch.clamp(nodes_v, -bound, bound)


# CEM


class CEMPlanner:
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_samples: int = 512,
        n_elites: int = 64,
        n_iters: int = 4,
        init_std: float = 0.5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ):
        self.h = horizon
        self.a = action_dim
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.init_std = init_std
        self.low = action_low
        self.high = action_high
        self.device = torch.device(device)
        self.gen = generator

    @torch.no_grad()
    def plan(self, cost_fn: CostFn, nominal: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            nominal.clone()
            if nominal is not None
            else torch.zeros(self.h, self.a, device=self.device)
        )
        std = torch.full((self.h, self.a), self.init_std, device=self.device)
        for _ in range(self.n_iters):
            noise = torch.randn(self.n_samples, self.h, self.a, device=self.device, generator=self.gen)
            cand = torch.clamp(mean + std * noise, self.low, self.high)
            costs = cost_fn(cand)
            elite_idx = torch.topk(costs, self.n_elites, largest=False).indices
            elites = cand[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(1e-4)
        return mean, mean[0]


# MPPI


class MPPIPlanner:
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_samples: int = 512,
        n_iters: int = 1,
        temperature: float = 1.0,
        noise_std: float = 0.5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ):
        self.h = horizon
        self.a = action_dim
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.temperature = temperature
        self.noise_std = noise_std
        self.low = action_low
        self.high = action_high
        self.device = torch.device(device)
        self.gen = generator

    @torch.no_grad()
    def plan(self, cost_fn: CostFn, nominal: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        u = nominal.clone() if nominal is not None else torch.zeros(self.h, self.a, device=self.device)

        for _ in range(self.n_iters):
            noise = torch.randn(self.n_samples, self.h, self.a, device=self.device, generator=self.gen)
            cand = torch.clamp(u + self.noise_std * noise, self.low, self.high)
            costs = cost_fn(cand)
            # Soft update: weight each sample by how good it is, then average.
            weights = torch.softmax(-(costs - costs.min()) / self.temperature, dim=0)
            u = torch.clamp((weights.view(-1, 1, 1) * cand).sum(dim=0), self.low, self.high)

        return u, u[0]


# Hermite MPPI


class HermiteMPPIPlanner:
    """MPPI over Hermite spline control points (position + velocity nodes)."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_nodes: int = 4,
        n_samples: int = 64,
        n_iters: int = 3,
        temperature: float = 1.0,
        node_noise_std: float = 0.5,
        beta1: float = 1.0,
        beta2: float = 1.0,
        dt: float = 1.0,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ):
        self.h = horizon
        self.a = action_dim
        self.k = n_nodes
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.temperature = temperature
        self.node_noise_std = node_noise_std
        self.beta1 = beta1
        self.beta2 = beta2
        self.dt = dt
        self.low = action_low
        self.high = action_high
        self.device = torch.device(device)
        self.gen = generator

    def _decode(self, nodes_q: torch.Tensor, nodes_v: torch.Tensor) -> torch.Tensor:
        """(N, K, A) nodes -> (N, H, A) action trajectory."""
        nodes_q = torch.clamp(nodes_q, self.low, self.high)
        nodes_v = clamp_node_velocity(nodes_q, nodes_v, self.dt, self.low, self.high)
        traj = hermite_interpolate(nodes_q, nodes_v, self.h, self.dt)
        return torch.clamp(traj, self.low, self.high)

    @torch.no_grad()
    def plan(self, cost_fn: CostFn, nominal: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # warm-start from nominal nodes (K, 2*A) or zeros
        if nominal is not None:
            q = nominal[:, : self.a].clone()
            v = nominal[:, self.a :].clone()
        else:
            q = torch.zeros(self.k, self.a, device=self.device)
            v = torch.zeros(self.k, self.a, device=self.device)

        # noise scale per node (annealing)
        k_idx = torch.arange(self.k, device=self.device)
        node_scale = torch.exp(-(self.k - 1 - k_idx) / (self.beta2 * self.k))  # (K,)

        for i in range(self.n_iters):
            iter_scale = float(torch.exp(torch.tensor(-(self.n_iters - 1 - i) / (self.beta1 * self.n_iters))))
            std = self.node_noise_std * iter_scale * node_scale.view(self.k, 1)  # (K, 1)

            eps_q = torch.randn(self.n_samples, self.k, self.a, device=self.device, generator=self.gen)
            eps_v = torch.randn(self.n_samples, self.k, self.a, device=self.device, generator=self.gen)
            cand_q = q.unsqueeze(0) + std.unsqueeze(0) * eps_q
            cand_v = v.unsqueeze(0) + std.unsqueeze(0) * eps_v

            actions = self._decode(cand_q, cand_v)  # (N, H, A)
            costs = cost_fn(actions)
            weights = torch.softmax(-(costs - costs.min()) / self.temperature, dim=0)
            q = (weights.view(-1, 1, 1) * cand_q).sum(dim=0)
            v = (weights.view(-1, 1, 1) * cand_v).sum(dim=0)

        actions = self._decode(q.unsqueeze(0), v.unsqueeze(0))[0]  # (H, A)
        nominal_nodes = torch.cat([q, v], dim=-1)  # (K, 2A) for warm-starting
        return nominal_nodes, actions[0]


def build_planner(
    kind: str,
    action_dim: int,
    horizon: int,
    device: str | torch.device = "cpu",
    action_low: float = -1.0,
    action_high: float = 1.0,
    generator: torch.Generator | None = None,
    **cfg,
):
    """Pick a planner: 'cem', 'mppi', or 'hermite'."""
    common = dict(
        horizon=horizon,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        device=device,
        generator=generator,
    )
    kind = kind.lower()
    if kind == "cem":
        return CEMPlanner(
            n_samples=cfg.get("n_samples", 512),
            n_elites=cfg.get("n_elites", 64),
            n_iters=cfg.get("n_iters", 4),
            init_std=cfg.get("init_std", 0.5),
            **common,
        )
    if kind == "mppi":
        return MPPIPlanner(
            n_samples=cfg.get("n_samples", 512),
            n_iters=cfg.get("n_iters", 4),
            temperature=cfg.get("temperature", 0.5),
            noise_std=cfg.get("noise_std", 0.4),
            **common,
        )
    if kind == "hermite":
        return HermiteMPPIPlanner(
            n_nodes=cfg.get("n_nodes", 4),
            n_samples=cfg.get("n_samples", 64),
            n_iters=cfg.get("n_iters", 3),
            temperature=cfg.get("temperature", 0.5),
            node_noise_std=cfg.get("node_noise_std", 0.4),
            beta1=cfg.get("beta1", 1.0),
            beta2=cfg.get("beta2", 1.0),
            **common,
        )
    raise ValueError(f"Unknown planner kind: {kind!r} (use 'cem', 'mppi', or 'hermite').")

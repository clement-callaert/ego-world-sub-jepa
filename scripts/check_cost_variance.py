"""Check that latent MPC cost varies with candidate actions (planning diagnostic)."""

from __future__ import annotations

import argparse

import torch

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.utils import get_device, load_checkpoint, set_seed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MPC cost sensitivity to actions.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--n-random", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    raw_cfg = ckpt["cfg"]["model"]
    model_cfg = EgoWorldConfig(**raw_cfg if isinstance(raw_cfg, dict) else dict(raw_cfg))
    model = EgoWorldJEPA(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    b = 1
    z_world = torch.randn(b, model_cfg.world_dim, device=device)
    z_ego = torch.randn(b, model_cfg.ego_dim, device=device) if model_cfg.mode == "factored" else None
    goal = torch.randn(model_cfg.world_dim, device=device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    n = args.n_random
    actions = torch.randn(n, args.horizon, model_cfg.action_dim, device=device, generator=gen)
    actions = actions.clamp(-1.0, 1.0)

    zw = z_world.expand(n, -1)
    ze = z_ego.expand(n, -1) if z_ego is not None else None
    costs = model.get_cost(zw, ze, actions, goal)
    cost_std = costs.std().item()
    cost_range = (costs.max() - costs.min()).item()

    print(f"[cost] n={args.n_random} std={cost_std:.6f} range={cost_range:.6f}")
    print(f"[cost] min={costs.min().item():.6f} max={costs.max().item():.6f}")
    if cost_std < 1e-6:
        print("[warn] Cost surface is flat, so MPPI cannot tell actions apart.")
        raise SystemExit(1)
    print("[ok] Cost varies with candidate actions.")


if __name__ == "__main__":
    main()

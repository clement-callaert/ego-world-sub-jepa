"""Evaluate latent MPC success rate (optional robustness sweeps).

Examples:
    python scripts/evaluate.py checkpoint=outputs/pusht_factored_seed0/model.pt
    python scripts/evaluate.py checkpoint=outputs/pusht_monolithic_seed0/model.pt planner.kind=cem
    python scripts/evaluate.py checkpoint=... robustness.enabled=true
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.mpc_policy import LatentMPCPolicy
from ewjepa.planning import build_planner
from ewjepa.probing import fit_pose_readout
from ewjepa.utils import Normalizer, get_device, load_checkpoint, set_seed


def _load_model(cfg: DictConfig, device: torch.device) -> tuple[EgoWorldJEPA, Normalizer | None]:
    ckpt = load_checkpoint(cfg.checkpoint, map_location=device)
    raw_cfg = ckpt["cfg"]["model"]
    model_cfg = EgoWorldConfig(**(OmegaConf.to_container(raw_cfg, resolve=True) if not isinstance(raw_cfg, dict) else raw_cfg))
    model = EgoWorldJEPA(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    normalizer = None
    if "proprio_normalizer" in ckpt:
        normalizer = Normalizer.from_state_dict(ckpt["proprio_normalizer"])
    return model, normalizer


def _build_policy(cfg: DictConfig, model: EgoWorldJEPA, normalizer, device: torch.device) -> LatentMPCPolicy:
    planner_cfg = dict(OmegaConf.to_container(cfg.planner, resolve=True))
    kind = planner_cfg.pop("kind")
    planner = build_planner(
        kind=kind,
        action_dim=model.cfg.action_dim,
        horizon=planner_cfg.pop("horizon"),
        device=device,
        action_low=cfg.data.action_low,
        action_high=cfg.data.action_high,
        generator=torch.Generator(device=device).manual_seed(cfg.seed),
        **planner_cfg,
    )
    pose_readout = _fit_pose_readout(cfg, model, normalizer, device)
    return LatentMPCPolicy(
        model=model,
        planner=planner,
        device=device,
        proprio_normalizer=normalizer,
        pose_readout=pose_readout,
        pose_cost_weight=float(cfg.get("pose_cost_weight", 1.0)),
    )


@torch.no_grad()
def _fit_pose_readout(
    cfg: DictConfig,
    model: EgoWorldJEPA,
    normalizer: Normalizer | None,
    device: torch.device,
    max_samples: int = 2048,
) -> dict[str, torch.Tensor] | None:
    """Fit ridge readout from world latents to block pose."""
    from torch.utils.data import DataLoader

    from ewjepa.data import build_dataset

    if not bool(cfg.get("use_pose_cost", True)):
        return None

    dataset = build_dataset(
        cfg.data.dataset,
        num_steps=2,
        image_key=cfg.data.image_key,
        proprio_key=cfg.data.proprio_key,
        action_key=cfg.data.action_key,
        state_key=cfg.data.state_key,
        max_episodes=cfg.data.get("max_episodes"),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    sl = slice(cfg.data.probe_target_slice[0], cfg.data.probe_target_slice[1])
    zw_chunks, y_chunks = [], []
    seen = 0
    for batch in loader:
        pixels = batch["pixels"].to(device)
        proprio = batch["proprio"].to(device)
        if normalizer is not None:
            proprio = normalizer(proprio)
        zw, _ = model.encode_sequence(pixels, proprio)
        zw_chunks.append(zw.reshape(-1, zw.shape[-1]).cpu())
        y_chunks.append(batch["state"].reshape(-1, batch["state"].shape[-1])[:, sl])
        seen += zw.shape[0] * zw.shape[1]
        if seen >= max_samples:
            break

    if not zw_chunks:
        return None

    features = torch.cat(zw_chunks, dim=0)[:max_samples]
    targets = torch.cat(y_chunks, dim=0)[:max_samples]
    readout = fit_pose_readout(features, targets, ridge=float(cfg.get("pose_probe_ridge", 1e-3)))
    return {k: v.to(device) for k, v in readout.items()}


def _run_eval(world, episodes: int, seed: int, variation: list, video_dir: str | Path | None = None) -> dict:
    options = {"variation": list(variation)} if variation else None
    return world.evaluate(episodes=episodes, seed=seed, options=options, video=video_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    import stable_worldmodel as swm

    if cfg.get("fast", False):
        cfg.episodes = 5
        cfg.max_episode_steps = 300
        cfg.planner.n_samples = 128
        cfg.planner.n_iters = 3
        cfg.planner.noise_std = 0.5
        print("[fast] episodes=5 max_episode_steps=300 planner.n_samples=128 planner.n_iters=3")

    model, normalizer = _load_model(cfg, device)
    policy = _build_policy(cfg, model, normalizer, device)

    img_size = model.cfg.img_size
    world = swm.World(
        cfg.data.env,
        num_envs=cfg.num_envs,
        image_shape=(img_size, img_size),
        max_episode_steps=cfg.get("max_episode_steps", 300),
    )
    world.set_policy(policy)

    out_dir = Path(cfg.get("out_dir", "outputs/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    video_dir = cfg.get("video_dir")
    if video_dir:
        video_path = Path(video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        print(f"[video] saving episode mp4 files to {video_path}")
    else:
        video_path = None

    results: dict[str, dict] = {}

    # default eval (no variation)
    clean = _run_eval(world, cfg.episodes, cfg.seed, cfg.variation, video_path)
    results["clean"] = {
        "success_rate": float(clean["success_rate"]),
        "episodes": int(cfg.episodes),
        "variation": list(cfg.variation),
    }
    print(f"[clean] success_rate={clean['success_rate']:.1f}%")

    # optional robustness sweep over env variations
    if cfg.get("robustness", {}).get("enabled", False):
        for var in cfg.robustness.variations:
            name = var if isinstance(var, str) else "+".join(var)
            res = _run_eval(
                world,
                cfg.robustness.episodes,
                cfg.seed + 1,
                [var] if isinstance(var, str) else var,
                None,
            )
            results[f"fov/{name}"] = {
                "success_rate": float(res["success_rate"]),
                "episodes": int(cfg.robustness.episodes),
                "variation": [var] if isinstance(var, str) else list(var),
            }
            print(f"[fov/{name}] success_rate={res['success_rate']:.1f}%")

        clean_sr = results["clean"]["success_rate"]
        drops = {
            k: clean_sr - v["success_rate"]
            for k, v in results.items()
            if k.startswith("fov/")
        }
        results["robustness_drop"] = drops
        if drops:
            results["robustness_mean_drop"] = float(sum(drops.values()) / len(drops))
            print(f"[robustness] mean_drop={results['robustness_mean_drop']:.1f} pp")

    ckpt_path = Path(cfg.checkpoint)
    tag = ckpt_path.parent.name if ckpt_path.parent.name not in ("", ".", "outputs") else ckpt_path.stem
    out_path = out_dir / f"eval_{tag}_{cfg.planner.kind}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results saved to {out_path}")


if __name__ == "__main__":
    main()

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
from ewjepa.train_utils import configure_cuda, maybe_compile
from ewjepa.utils import Normalizer, build_run_manifest, get_device, load_checkpoint, set_seed


def _load_model(
    cfg: DictConfig, device: torch.device
) -> tuple[EgoWorldJEPA, Normalizer | None, dict]:
    ckpt = load_checkpoint(cfg.checkpoint, map_location=device)
    raw_cfg = ckpt["cfg"]["model"]
    checkpoint_model_cfg = (
        OmegaConf.to_container(raw_cfg, resolve=True) if not isinstance(raw_cfg, dict) else raw_cfg
    )
    model_cfg = EgoWorldConfig(**checkpoint_model_cfg)
    model = EgoWorldJEPA(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    normalizer = None
    if "proprio_normalizer" in ckpt:
        normalizer = Normalizer.from_state_dict(ckpt["proprio_normalizer"])
    return model, normalizer, checkpoint_model_cfg


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
    pose_readout, agent_readout = _fit_readouts(cfg, model, normalizer, device)
    approach_weight = float(cfg.get("approach_weight", 0.0))
    if bool(cfg.get("use_pose_cost", True)):
        if pose_readout is not None:
            print("[eval] pose readout fitted, auxiliary pose cost enabled in MPC")
            if agent_readout is not None and approach_weight > 0:
                print(f"[eval] agent readout fitted, approach cost enabled (weight={approach_weight})")
        else:
            print("[warn] pose readout unavailable, planner uses latent cost only")

    # Optional supervised block detector. It is the block sensor the planner
    # reads for the current pose. The JEPA world model still supplies the block
    # dynamics through the predicted displacement in the MPC cost.
    block_detector = None
    detector_path = cfg.get("block_detector")
    if detector_path:
        from ewjepa.detector import load_detector

        block_detector = load_detector(detector_path, map_location=device)
        print(f"[eval] block detector loaded from {detector_path}")

    return LatentMPCPolicy(
        model=model,
        planner=planner,
        device=device,
        proprio_normalizer=normalizer,
        pose_readout=pose_readout,
        pose_cost_weight=float(cfg.get("pose_cost_weight", 1.0)),
        agent_readout=agent_readout,
        approach_weight=approach_weight,
        latent_cost_weight=float(cfg.get("latent_cost_weight", 1.0)),
        angle_cost_weight=float(cfg.get("angle_cost_weight", 0.2)),
        agent_goal_weight=float(cfg.get("agent_goal_weight", 1.0)),
        bounds_weight=float(cfg.get("bounds_weight", 10.0)),
        near_block_thresh=float(cfg.get("near_block_thresh", 15.0)),
        park_angle_thresh=float(cfg.get("park_angle_thresh", 0.30)),
        engage_thresh=float(cfg.get("engage_thresh", 70.0)),
        standoff=float(cfg.get("standoff", 60.0)),
        clearance=float(cfg.get("clearance", 45.0)),
        action_penalty=float(cfg.get("action_penalty", 0.02)),
        block_detector=block_detector,
        action_scale=float(cfg.get("action_scale", 100.0)),
        board_margin=float(cfg.get("board_margin", 40.0)),
        push_weight=float(cfg.get("push_weight", 1.0)),
        push_through=float(cfg.get("push_through", 4.0)),
    )


@torch.no_grad()
def _fit_readouts(
    cfg: DictConfig,
    model: EgoWorldJEPA,
    normalizer: Normalizer | None,
    device: torch.device,
    max_samples: int = 8192,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    """Fit ridge readouts used by the planner cost.

    The block pose is read from the world latent (the part that sees the block in
    pixels). The agent xy is read from the ego latent (the part the actions
    control). We fit both on encoded dataset latents and return
    (block_readout, agent_readout), or (None, None) when the pose cost is off.
    """
    from torch.utils.data import DataLoader

    from ewjepa.data import build_dataset

    if not bool(cfg.get("use_pose_cost", True)):
        return None, None

    dataset = build_dataset(
        cfg.data.dataset,
        num_steps=2,
        image_key=cfg.data.image_key,
        proprio_key=cfg.data.proprio_key,
        action_key=cfg.data.action_key,
        state_key=cfg.data.state_key,
        max_episodes=cfg.data.get("max_episodes"),
    )
    sample = dataset[0]
    height, width = sample["pixels"].shape[-2:]
    if (height, width) != (model.cfg.img_size, model.cfg.img_size):
        raise ValueError(
            f"Checkpoint expects {model.cfg.img_size}x{model.cfg.img_size} images, "
            f"but the dataset provides {height}x{width}. Choose matching data=."
        )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    zw_chunks, ze_chunks, state_chunks = [], [], []
    seen = 0
    for batch in loader:
        pixels = batch["pixels"].to(device)
        proprio = batch["proprio"].to(device)
        if normalizer is not None:
            proprio = normalizer(proprio)
        zw, ze = model.encode_sequence(pixels, proprio)
        zw_chunks.append(zw.reshape(-1, zw.shape[-1]).cpu())
        if ze is not None:
            ze_chunks.append(ze.reshape(-1, ze.shape[-1]).cpu())
        state_chunks.append(batch["state"].reshape(-1, batch["state"].shape[-1]))
        seen += zw.shape[0] * zw.shape[1]
        if seen >= max_samples:
            break

    if not zw_chunks:
        return None, None

    world_feats = torch.cat(zw_chunks, dim=0)[:max_samples]
    ego_feats = torch.cat(ze_chunks, dim=0)[:max_samples] if ze_chunks else None
    states = torch.cat(state_chunks, dim=0)[:max_samples]
    ridge = float(cfg.get("pose_probe_ridge", 1e-3))

    def fit(features, state_slice) -> dict[str, torch.Tensor]:
        sl = slice(state_slice[0], state_slice[1])
        readout = fit_pose_readout(features, states[:, sl], ridge=ridge)
        return {k: v.to(device) for k, v in readout.items()}

    block_readout = fit(world_feats, cfg.data.probe_target_slice)
    # Agent xy comes from the ego latent when we have one, else fall back to world.
    agent_feats = ego_feats if ego_feats is not None else world_feats
    agent_readout = fit(agent_feats, cfg.data.get("agent_state_slice", [0, 2]))
    return block_readout, agent_readout


def _run_eval(world, episodes: int, seed: int, variation: list, video_dir: str | Path | None = None) -> dict:
    options = {"variation": list(variation)} if variation else None
    return world.evaluate(episodes=episodes, seed=seed, options=options, video=video_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    configure_cuda(
        cudnn_benchmark=bool(cfg.get("cudnn_benchmark", True)),
        allow_tf32=bool(cfg.get("allow_tf32", True)),
    )
    import stable_worldmodel as swm

    if cfg.get("fast", False):
        cfg.episodes = 5
        cfg.max_episode_steps = 300
        cfg.planner.n_samples = 128
        cfg.planner.n_iters = 3
        cfg.planner.noise_std = 0.5
        print("[fast] episodes=5 max_episode_steps=300 planner.n_samples=128 planner.n_iters=3")

    model, normalizer, checkpoint_model_cfg = _load_model(cfg, device)
    eval_compile = bool(cfg.get("compile", False))
    if eval_compile:
        # Compile the predictor (MPPI hot path). Full-model compile is heavier
        # and the encode shapes vary less often than rollout batches.
        model.predictor = maybe_compile(model.predictor, True)
    policy_kind = str(cfg.get("policy", "mpc"))
    if policy_kind == "random":
        from ewjepa.mpc_policy import RandomPolicy

        policy = RandomPolicy(
            action_dim=model.cfg.action_dim,
            action_low=float(cfg.data.action_low),
            action_high=float(cfg.data.action_high),
            seed=int(cfg.seed),
        )
        print("[eval] RANDOM policy baseline (uniform actions, no planning)")
    elif policy_kind == "mpc":
        policy = _build_policy(cfg, model, normalizer, device)
    else:
        raise ValueError(f"unknown policy {policy_kind!r}, expected 'mpc' or 'random'")

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

    results["checkpoint"] = str(cfg.checkpoint)
    manifest_cfg = OmegaConf.to_container(cfg, resolve=True)
    manifest_cfg["model"] = checkpoint_model_cfg
    results["manifest"] = build_run_manifest(
        manifest_cfg,
        seed=int(cfg.seed),
    )

    ckpt_path = Path(cfg.checkpoint)
    tag = ckpt_path.parent.name if ckpt_path.parent.name not in ("", ".", "outputs") else ckpt_path.stem
    suffix = "random" if policy_kind == "random" else cfg.planner.kind
    out_path = out_dir / f"eval_{tag}_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results saved to {out_path}")


if __name__ == "__main__":
    main()

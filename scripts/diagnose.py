"""Action-conditioned diagnostics on a trained checkpoint (no training).

1. Multi-step open-loop rollout error: encode the first frame of held-out
   windows, roll the predictor with the real dataset actions, decode the block
   pose with a frozen ridge readout (the planner's readout), compare with the
   true simulator pose. Reported as RMSE in px per horizon.
2. Action sensitivity: spread of the decoded block pose at horizon H under
   K uniform random action sequences, normalized by the dataset block pose
   std. Near 0 means the predictor is blind to the actions.

The split is by episode: the readout is fitted on training episodes only and
all rollouts run on held-out episodes, so nothing leaks.

Examples:
    python scripts/diagnose.py checkpoint=outputs/pusht_hires_seed0/model.pt data=pusht_96
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.data import build_dataset
from ewjepa.diagnostics import action_sensitivity, rollout_pose_errors
from ewjepa.probing import fit_pose_readout
from ewjepa.train_utils import configure_cuda
from ewjepa.utils import Normalizer, build_run_manifest, get_device, load_checkpoint, set_seed


def _load_model(cfg: DictConfig, device: torch.device):
    ckpt = load_checkpoint(cfg.checkpoint, map_location=device)
    raw_cfg = ckpt["cfg"]["model"]
    checkpoint_model_cfg = (
        OmegaConf.to_container(raw_cfg, resolve=True) if not isinstance(raw_cfg, dict) else raw_cfg
    )
    model = EgoWorldJEPA(EgoWorldConfig(**checkpoint_model_cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    normalizer = None
    if "proprio_normalizer" in ckpt:
        normalizer = Normalizer.from_state_dict(ckpt["proprio_normalizer"])
    return model, normalizer, checkpoint_model_cfg


def _episode_split(dataset, holdout_frac: float) -> tuple[Subset, Subset, int, int]:
    """Split windows by episode: last holdout_frac of episode ids held out."""
    if not hasattr(dataset, "clip_indices"):
        raise ValueError("dataset has no clip_indices; cannot split by episode")
    episodes = sorted({ep for ep, _ in dataset.clip_indices})
    n_hold = max(1, int(round(len(episodes) * holdout_frac)))
    holdout_eps = set(episodes[-n_hold:])
    train_idx = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep not in holdout_eps]
    hold_idx = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in holdout_eps]
    return Subset(dataset, train_idx), Subset(dataset, hold_idx), len(episodes) - n_hold, n_hold


@torch.no_grad()
def _fit_readout_and_stats(cfg, model, normalizer, train_set, device, block_slice):
    """Fit the frozen ridge readout on training episodes, planner-style."""
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    max_samples = int(cfg.readout_max_samples)
    zw_chunks, state_chunks, seen = [], [], 0
    for batch in loader:
        pixels = batch["pixels"].to(device)
        proprio = batch["proprio"].to(device)
        if normalizer is not None:
            proprio = normalizer(proprio)
        zw, _ = model.encode_sequence(pixels, proprio)
        zw_chunks.append(zw.reshape(-1, zw.shape[-1]).cpu())
        state_chunks.append(batch["state"].reshape(-1, batch["state"].shape[-1]))
        seen += zw_chunks[-1].shape[0]
        if seen >= max_samples:
            break
    feats = torch.cat(zw_chunks, dim=0)[:max_samples]
    states = torch.cat(state_chunks, dim=0)[:max_samples]
    pose = states[:, block_slice[0] : block_slice[1]]
    readout = fit_pose_readout(feats, pose, ridge=float(cfg.readout_ridge))
    readout = {k: v.to(device) for k, v in readout.items()}
    pose_std = pose.std(dim=0, unbiased=False)  # (3,) dataset block pose std
    return readout, pose_std


@hydra.main(version_base=None, config_path="../configs", config_name="diagnose")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    configure_cuda(
        cudnn_benchmark=bool(cfg.get("cudnn_benchmark", True)),
        allow_tf32=bool(cfg.get("allow_tf32", True)),
    )
    model, normalizer, checkpoint_model_cfg = _load_model(cfg, device)

    horizons = tuple(int(h) for h in cfg.horizons)
    h_max = max(max(horizons), int(cfg.sensitivity_horizon))
    block_slice = tuple(cfg.data.probe_target_slice)

    dataset = build_dataset(
        cfg.data.dataset,
        num_steps=h_max + 1,
        image_key=cfg.data.image_key,
        proprio_key=cfg.data.proprio_key,
        action_key=cfg.data.action_key,
        state_key=cfg.data.state_key,
        max_episodes=cfg.data.get("max_episodes"),
    )
    train_set, hold_set, n_train_eps, n_hold_eps = _episode_split(dataset, float(cfg.holdout_frac))
    print(f"[split] {n_train_eps} train episodes (readout fit), {n_hold_eps} held-out episodes")

    readout, pose_std = _fit_readout_and_stats(cfg, model, normalizer, train_set, device, block_slice)
    print(f"[readout] fitted on {int(cfg.readout_max_samples)} train-episode rows")

    # 1.1 open-loop rollout error on held-out episodes
    loader = DataLoader(hold_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    all_h = (0,) + horizons
    sums = {h: {} for h in all_h}
    first_latents = None
    seen = 0
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixels"].to(device)
            proprio = batch["proprio"].to(device)
            if normalizer is not None:
                proprio = normalizer(proprio)
            actions = batch["action"].to(device)
            states = batch["state"].to(device)
            errs = rollout_pose_errors(
                model, readout, pixels, proprio, actions, states,
                horizons=horizons, block_slice=block_slice,
            )
            for h, e in errs.items():
                s = sums[h]
                for k, v in e.items():
                    s[k] = s.get(k, 0.0) + float(v.sum())
                    if k.startswith("abs_"):
                        s["sq_" + k[4:]] = s.get("sq_" + k[4:], 0.0) + float(v.pow(2).sum())
                s["n"] = s.get("n", 0) + int(e["sq_xy"].shape[0])
            if first_latents is None:
                zw0, ze0 = model.encode(pixels[:, 0], proprio[:, 0])
                first_latents = (zw0, ze0)
            seen += pixels.shape[0]
            if seen >= int(cfg.max_eval_windows):
                break

    def _rmse(s: dict, key: str) -> float | None:
        return math.sqrt(s[key] / s["n"]) if key in s else None

    rollout_rmse = {}
    for h in all_h:
        s = sums[h]
        rollout_rmse[str(h)] = {
            "xy_rmse_px": _rmse(s, "sq_xy"),
            "angle_rmse_rad": _rmse(s, "sq_angle"),
            "angle_mae_rad": s["abs_angle"] / s["n"],
            # displacement-mode errors: what the planner consumes (readout
            # bias cancelled by the current-frame anchor)
            "disp_xy_rmse_px": _rmse(s, "sq_xy_disp"),
            "disp_angle_rmse_rad": _rmse(s, "sq_angle_disp"),
            # trivial "block never moves" predictor, floor for disp_xy_rmse_px
            "zero_motion_xy_rmse_px": _rmse(s, "sq_xy_zero"),
            "n_windows": s["n"],
        }
        r = rollout_rmse[str(h)]
        if h == 0:
            print(f"[readout H=0] abs xy_rmse={r['xy_rmse_px']:.2f} px (encode+decode only)")
        else:
            print(
                f"[rollout H={h}] abs xy_rmse={r['xy_rmse_px']:.2f} px "
                f"disp xy_rmse={r['disp_xy_rmse_px']:.2f} px "
                f"zero-motion={r['zero_motion_xy_rmse_px']:.2f} px "
                f"angle_rmse={r['angle_rmse_rad']:.3f} rad (n={r['n_windows']})"
            )
    mean_xy = sum(rollout_rmse[str(h)]["xy_rmse_px"] for h in horizons) / len(horizons)
    mean_disp = sum(rollout_rmse[str(h)]["disp_xy_rmse_px"] for h in horizons) / len(horizons)
    print(f"[rollout] mean abs xy RMSE={mean_xy:.2f} px, mean disp xy RMSE={mean_disp:.2f} px")

    # 1.2 action sensitivity from held-out initial latents
    n_states = min(int(cfg.sensitivity_states), first_latents[0].shape[0])
    zw0 = first_latents[0][:n_states]
    ze0 = first_latents[1][:n_states] if first_latents[1] is not None else None
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
    sens = action_sensitivity(
        model, readout, zw0, ze0,
        horizon=int(cfg.sensitivity_horizon),
        n_sequences=int(cfg.sensitivity_sequences),
        action_low=float(cfg.data.action_low),
        action_high=float(cfg.data.action_high),
        generator=gen,
    )
    xy_std = sens["xy_std"].cpu()
    dataset_xy_std = pose_std[:2]
    normalized = float((xy_std / dataset_xy_std.clamp_min(1e-6)).mean())
    sensitivity = {
        "horizon": int(cfg.sensitivity_horizon),
        "n_sequences": int(cfg.sensitivity_sequences),
        "n_states": n_states,
        "pred_xy_std_px": [float(v) for v in xy_std],
        "pred_angle_std_rad": float(sens["angle_std"]),
        "dataset_block_xy_std_px": [float(v) for v in dataset_xy_std],
        "dataset_block_angle_std_rad": float(pose_std[2]),
        "normalized_xy_sensitivity": normalized,
        "normalized_angle_sensitivity": float(sens["angle_std"] / pose_std[2].clamp_min(1e-6)),
    }
    print(
        f"[sensitivity H={sensitivity['horizon']}] pred xy std={xy_std.tolist()} px, "
        f"normalized={normalized:.4f} (0 = action-blind)"
    )

    manifest_cfg = OmegaConf.to_container(cfg, resolve=True)
    manifest_cfg["model"] = checkpoint_model_cfg
    results = {
        "checkpoint": str(cfg.checkpoint),
        "split": {
            "by": "episode",
            "train_episodes": n_train_eps,
            "holdout_episodes": n_hold_eps,
            "holdout_frac": float(cfg.holdout_frac),
        },
        "rollout_rmse": rollout_rmse,
        "rollout_mean_xy_rmse_px": mean_xy,
        "rollout_mean_disp_xy_rmse_px": mean_disp,
        "action_sensitivity": sensitivity,
        "manifest": build_run_manifest(manifest_cfg, seed=int(cfg.seed)),
    }

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint)
    tag = ckpt_path.parent.name if ckpt_path.parent.name not in ("", ".", "outputs") else ckpt_path.stem
    out_path = out_dir / f"diagnostics_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results saved to {out_path}")


if __name__ == "__main__":
    main()

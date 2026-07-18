"""Linear probe: predict object pose from frozen world vs ego latents.

Reports absolute pose R2 and displacement R2 (pose_t - pose_0 within each
window). Displacement is closer to what the MPPI planner consumes.

Examples:
    python scripts/probe.py checkpoint=outputs/pusht_factored_seed0/model.pt
    python scripts/probe.py checkpoint=outputs/pusht_monolithic_seed0/model.pt data.dataset=data/pusht.lance
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.data import SyntheticPushTDataset, _resolve_dataset_path, build_dataset
from ewjepa.probing import linear_probe
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


def _build_probe_loader(cfg: DictConfig) -> DataLoader:
    name = _resolve_dataset_path(cfg.data.dataset)
    if name == cfg.data.dataset and not Path(name).exists() and cfg.get("synthetic_fallback", True):
        print(f"[probe] {name} not found, using SyntheticPushTDataset.")
        dataset = SyntheticPushTDataset(
            num_episodes=cfg.probe.synthetic_episodes,
            num_steps=cfg.probe.num_steps,
            img_size=cfg.model.img_size,
            seed=cfg.seed,
        )
    else:
        dataset = build_dataset(
            name,
            num_steps=cfg.probe.num_steps,
            image_key=cfg.data.image_key,
            proprio_key=cfg.data.proprio_key,
            action_key=cfg.data.action_key,
            state_key=cfg.data.state_key,
        )
    return DataLoader(dataset, batch_size=cfg.probe.batch_size, shuffle=False, num_workers=cfg.probe.num_workers)


def _validate_image_size(loader: DataLoader, expected_size: int) -> None:
    sample = next(iter(loader))
    height, width = sample["pixels"].shape[-2:]
    if (height, width) != (expected_size, expected_size):
        raise ValueError(
            f"Checkpoint expects {expected_size}x{expected_size} images, "
            f"but the dataset provides {height}x{width}. Choose matching data=."
        )


@torch.no_grad()
def _collect_latents(
    model: EgoWorldJEPA,
    loader: DataLoader,
    normalizer: Normalizer | None,
    device: torch.device,
    target_slice: list[int],
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Collect absolute rows and displacement rows (anchored at frame 0).

    Absolute: every frame (z_t -> pose_t).
    Displacement: frames t>=1 (z_t -> pose_t - pose_0) within each window.
    """
    zw_abs, ze_abs, y_abs = [], [], []
    zw_disp, ze_disp, y_disp = [], [], []
    sl = slice(target_slice[0], target_slice[1])
    n_abs, n_disp = 0, 0

    for batch in loader:
        pixels = batch["pixels"].to(device)
        proprio = batch["proprio"].to(device)
        if normalizer is not None:
            proprio = normalizer(proprio)

        b, t = pixels.shape[:2]
        zw, ze = model.encode_sequence(pixels, proprio)
        state = batch["state"].float()
        pose = state[:, :, sl]  # (B, T, pose_dim)

        zw_rows = zw.reshape(b * t, -1).cpu()
        ze_rows = ze.reshape(b * t, -1).cpu() if ze is not None else None
        y_rows = pose.reshape(b * t, -1)

        if max_samples is not None and n_abs + zw_rows.shape[0] > max_samples:
            keep = max_samples - n_abs
            zw_rows = zw_rows[:keep]
            if ze_rows is not None:
                ze_rows = ze_rows[:keep]
            y_rows = y_rows[:keep]

        zw_abs.append(zw_rows)
        if ze_rows is not None:
            ze_abs.append(ze_rows)
        y_abs.append(y_rows)
        n_abs += zw_rows.shape[0]

        if t >= 2:
            # Displacement from frame 0 of each window, for t = 1 .. T-1.
            pose0 = pose[:, :1, :]
            disp = (pose - pose0)[:, 1:, :]  # (B, T-1, D)
            zw_d = zw[:, 1:, :].reshape(b * (t - 1), -1).cpu()
            ze_d = ze[:, 1:, :].reshape(b * (t - 1), -1).cpu() if ze is not None else None
            y_d = disp.reshape(b * (t - 1), -1)
            if max_samples is not None and n_disp + zw_d.shape[0] > max_samples:
                keep = max_samples - n_disp
                zw_d = zw_d[:keep]
                if ze_d is not None:
                    ze_d = ze_d[:keep]
                y_d = y_d[:keep]
            zw_disp.append(zw_d)
            if ze_d is not None:
                ze_disp.append(ze_d)
            y_disp.append(y_d)
            n_disp += zw_d.shape[0]

        if max_samples is not None and n_abs >= max_samples and (t < 2 or n_disp >= max_samples):
            break

    z_world = torch.cat(zw_abs, dim=0)
    z_ego = torch.cat(ze_abs, dim=0) if ze_abs else None
    targets = torch.cat(y_abs, dim=0)
    if zw_disp:
        z_world_disp = torch.cat(zw_disp, dim=0)
        z_ego_disp = torch.cat(ze_disp, dim=0) if ze_disp else None
        targets_disp = torch.cat(y_disp, dim=0)
    else:
        z_world_disp = z_world[:0]
        z_ego_disp = z_ego[:0] if z_ego is not None else None
        targets_disp = targets[:0]
    return z_world, z_ego, targets, z_world_disp, z_ego_disp, targets_disp


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    model, normalizer, checkpoint_model_cfg = _load_model(cfg, device)
    loader = _build_probe_loader(cfg)
    _validate_image_size(loader, model.cfg.img_size)
    max_samples = cfg.probe.get("max_samples")
    z_world, z_ego, targets, zw_d, ze_d, y_d = _collect_latents(
        model,
        loader,
        normalizer,
        device,
        list(cfg.data.probe_target_slice),
        max_samples=max_samples,
    )

    probe_kw = dict(
        test_frac=cfg.probe.test_frac,
        ridge=cfg.probe.ridge,
        seed=cfg.seed,
        group_split=True,
    )
    manifest_cfg = OmegaConf.to_container(cfg, resolve=True)
    manifest_cfg["model"] = checkpoint_model_cfg
    results = {
        "checkpoint": str(cfg.checkpoint),
        "n_latent_rows": int(z_world.shape[0]),
        "n_samples": int(z_world.shape[0]),  # alias for backward compatibility
        "n_disp_rows": int(zw_d.shape[0]),
        "probe_max_samples": max_samples,
        "target_dim": int(targets.shape[1]),
        "world_probe": linear_probe(z_world, targets, **probe_kw),
        "manifest": build_run_manifest(
            manifest_cfg,
            seed=int(cfg.seed),
        ),
    }
    if z_ego is not None:
        results["ego_probe"] = linear_probe(z_ego, targets, **probe_kw)
    if zw_d.shape[0] > 10:
        results["world_probe_disp"] = linear_probe(zw_d, y_d, **probe_kw)
        if ze_d is not None and ze_d.shape[0] > 10:
            results["ego_probe_disp"] = linear_probe(ze_d, y_d, **probe_kw)

    print("[probe] world abs:  " + " ".join(f"{k}={v:.4f}" for k, v in results["world_probe"].items()))
    if "world_probe_disp" in results:
        print("[probe] world disp: " + " ".join(f"{k}={v:.4f}" for k, v in results["world_probe_disp"].items()))
    if "ego_probe" in results:
        print("[probe] ego abs:    " + " ".join(f"{k}={v:.4f}" for k, v in results["ego_probe"].items()))
    if "ego_probe_disp" in results:
        print("[probe] ego disp:   " + " ".join(f"{k}={v:.4f}" for k, v in results["ego_probe_disp"].items()))

    out_dir = Path(cfg.get("out_dir", "outputs/probe"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint)
    tag = ckpt_path.parent.name if ckpt_path.parent.name not in ("", ".", "outputs") else ckpt_path.stem
    out_path = out_dir / f"probe_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results saved to {out_path}")


if __name__ == "__main__":
    main()

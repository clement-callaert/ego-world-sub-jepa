"""Linear probe: predict object pose from frozen world vs ego latents.

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
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    zw_chunks, ze_chunks, y_chunks = [], [], []
    sl = slice(target_slice[0], target_slice[1])
    est_latent_rows = 0

    for batch in loader:
        pixels = batch["pixels"].to(device)
        proprio = batch["proprio"].to(device)
        if normalizer is not None:
            proprio = normalizer(proprio)

        b, t = pixels.shape[:2]
        zw, ze = model.encode_sequence(pixels, proprio)
        zw_rows = zw.reshape(b * t, -1).cpu()
        ze_rows = ze.reshape(b * t, -1).cpu() if ze is not None else None
        state = batch["state"].float()
        y_rows = state.reshape(b * t, -1)[:, sl]

        if max_samples is not None and est_latent_rows + zw_rows.shape[0] > max_samples:
            keep = max_samples - est_latent_rows
            zw_rows = zw_rows[:keep]
            if ze_rows is not None:
                ze_rows = ze_rows[:keep]
            y_rows = y_rows[:keep]

        zw_chunks.append(zw_rows)
        if ze_rows is not None:
            ze_chunks.append(ze_rows)
        y_chunks.append(y_rows)
        est_latent_rows += zw_rows.shape[0]

        if max_samples is not None and est_latent_rows >= max_samples:
            break

    z_world = torch.cat(zw_chunks, dim=0)
    z_ego = torch.cat(ze_chunks, dim=0) if ze_chunks else None
    targets = torch.cat(y_chunks, dim=0)
    return z_world, z_ego, targets


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    model, normalizer, checkpoint_model_cfg = _load_model(cfg, device)
    loader = _build_probe_loader(cfg)
    _validate_image_size(loader, model.cfg.img_size)
    max_samples = cfg.probe.get("max_samples")
    z_world, z_ego, targets = _collect_latents(
        model,
        loader,
        normalizer,
        device,
        list(cfg.data.probe_target_slice),
        max_samples=max_samples,
    )

    manifest_cfg = OmegaConf.to_container(cfg, resolve=True)
    manifest_cfg["model"] = checkpoint_model_cfg
    results = {
        "checkpoint": str(cfg.checkpoint),
        "n_latent_rows": int(z_world.shape[0]),
        "n_samples": int(z_world.shape[0]),  # alias for backward compatibility
        "probe_max_samples": max_samples,
        "target_dim": int(targets.shape[1]),
        "world_probe": linear_probe(z_world, targets, test_frac=cfg.probe.test_frac, ridge=cfg.probe.ridge, seed=cfg.seed, group_split=True),
        "manifest": build_run_manifest(
            manifest_cfg,
            seed=int(cfg.seed),
        ),
    }
    if z_ego is not None:
        results["ego_probe"] = linear_probe(z_ego, targets, test_frac=cfg.probe.test_frac, ridge=cfg.probe.ridge, seed=cfg.seed, group_split=True)

    print("[probe] world: " + " ".join(f"{k}={v:.4f}" for k, v in results["world_probe"].items()))
    if "ego_probe" in results:
        print("[probe] ego:   " + " ".join(f"{k}={v:.4f}" for k, v in results["ego_probe"].items()))

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

"""Per-factor linear readouts: which latent encodes absolute pose vs displacement.

Four probes (null for ego when monolithic):
  z_ego   -> absolute block pose
  z_ego   -> displacement (pose_t - pose_0 within window)
  z_world -> absolute block pose
  z_world -> displacement

Uses the same grouped train/eval split as scripts/probe.py (no episode leak
across the contiguous tail). Writes one JSON per checkpoint under
results/diagnostics/perfactor/.

Usage:
    python scripts/probe_perfactor.py
    python scripts/probe_perfactor.py --checkpoint outputs/grid/g1_..._seed1/model.pt --gid g1 --seed 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.data import build_dataset
from ewjepa.probing import linear_probe
from ewjepa.utils import Normalizer, get_device, load_checkpoint, set_seed

# Seed-0 grid checkpoint locations (same as scripts/run_grid.sh).
SEED0_CKPTS = {
    "g1": "outputs/pusht_hires_seed0/model.pt",
    "g2": "outputs/grid/g2_monolithic_sgT_cov025_aux1_seed0/model.pt",
    "g3": "outputs/pusht_monolithic_hires_seed0/model.pt",
    "g4": "outputs/grid/g4_factored_sgF_cov0_aux1_seed0/model.pt",
    "g5": "outputs/grid/g5_factored_sgT_cov0_aux1_seed0/model.pt",
    "g6": "outputs/grid/g6_factored_sgF_cov025_aux1_seed0/model.pt",
    "g7": "outputs/grid/g7_factored_sgT_cov025_aux0_seed0/model.pt",
    "g8": "outputs/grid/g8_monolithic_sgT_cov025_aux0_seed0/model.pt",
}

GRID_NAMES = {
    "g1": "g1_factored_sgT_cov025_aux1",
    "g2": "g2_monolithic_sgT_cov025_aux1",
    "g3": "g3_monolithic_sgF_cov0_aux1",
    "g4": "g4_factored_sgF_cov0_aux1",
    "g5": "g5_factored_sgT_cov0_aux1",
    "g6": "g6_factored_sgF_cov025_aux1",
    "g7": "g7_factored_sgT_cov025_aux0",
    "g8": "g8_monolithic_sgT_cov025_aux0",
}


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    raw_cfg = ckpt["cfg"]["model"]
    model_cfg_dict = (
        OmegaConf.to_container(raw_cfg, resolve=True) if not isinstance(raw_cfg, dict) else raw_cfg
    )
    model = EgoWorldJEPA(EgoWorldConfig(**model_cfg_dict)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    normalizer = None
    if "proprio_normalizer" in ckpt:
        normalizer = Normalizer.from_state_dict(ckpt["proprio_normalizer"])
    return model, normalizer, model_cfg_dict


@torch.no_grad()
def _collect(model, loader, normalizer, device, target_slice, max_samples: int):
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
        pose = batch["state"].float()[:, :, sl]

        zw_r = zw.reshape(b * t, -1).cpu()
        ze_r = ze.reshape(b * t, -1).cpu() if ze is not None else None
        y_r = pose.reshape(b * t, -1)
        if n_abs + zw_r.shape[0] > max_samples:
            keep = max_samples - n_abs
            zw_r, y_r = zw_r[:keep], y_r[:keep]
            if ze_r is not None:
                ze_r = ze_r[:keep]
        zw_abs.append(zw_r)
        if ze_r is not None:
            ze_abs.append(ze_r)
        y_abs.append(y_r)
        n_abs += zw_r.shape[0]

        if t >= 2:
            disp = (pose - pose[:, :1, :])[:, 1:, :]
            zw_d = zw[:, 1:, :].reshape(b * (t - 1), -1).cpu()
            ze_d = ze[:, 1:, :].reshape(b * (t - 1), -1).cpu() if ze is not None else None
            y_d = disp.reshape(b * (t - 1), -1)
            if n_disp + zw_d.shape[0] > max_samples:
                keep = max_samples - n_disp
                zw_d, y_d = zw_d[:keep], y_d[:keep]
                if ze_d is not None:
                    ze_d = ze_d[:keep]
            zw_disp.append(zw_d)
            if ze_d is not None:
                ze_disp.append(ze_d)
            y_disp.append(y_d)
            n_disp += zw_d.shape[0]

        if n_abs >= max_samples and n_disp >= max_samples:
            break

    out = {
        "z_world_abs": torch.cat(zw_abs, dim=0),
        "z_ego_abs": torch.cat(ze_abs, dim=0) if ze_abs else None,
        "y_abs": torch.cat(y_abs, dim=0),
        "z_world_disp": torch.cat(zw_disp, dim=0) if zw_disp else None,
        "z_ego_disp": torch.cat(ze_disp, dim=0) if ze_disp else None,
        "y_disp": torch.cat(y_disp, dim=0) if y_disp else None,
    }
    return out


def _probe_or_none(features, targets, **kw):
    if features is None or targets is None or features.shape[0] < 10:
        return None
    return linear_probe(features, targets, **kw)


def run_one(ckpt: Path, gid: str, seed: int, out_dir: Path, data_path: str, max_samples: int) -> Path | None:
    if not ckpt.is_file():
        print(f"[skip] missing checkpoint {ckpt}")
        return None
    set_seed(0)  # protocol seed for split reproducibility
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, model_cfg = _load_model(ckpt, device)
    dataset = build_dataset(
        data_path,
        num_steps=2,
        image_key="pixels",
        proprio_key="proprio",
        action_key="action",
        state_key="state",
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    data = _collect(model, loader, normalizer, device, [2, 5], max_samples)
    kw = dict(test_frac=0.2, ridge=1e-3, seed=0, group_split=True)
    results = {
        "checkpoint": str(ckpt),
        "config": gid,
        "train_seed": seed,
        "mode": model_cfg.get("mode"),
        "n_abs_rows": int(data["y_abs"].shape[0]),
        "n_disp_rows": int(data["y_disp"].shape[0]) if data["y_disp"] is not None else 0,
        "z_world_abs": _probe_or_none(data["z_world_abs"], data["y_abs"], **kw),
        "z_world_disp": _probe_or_none(data["z_world_disp"], data["y_disp"], **kw),
        "z_ego_abs": _probe_or_none(data["z_ego_abs"], data["y_abs"], **kw),
        "z_ego_disp": _probe_or_none(data["z_ego_disp"], data["y_disp"], **kw),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{gid}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    def _r2(d):
        return None if d is None else round(float(d["r2"]), 4)
    print(
        f"[perfactor] {gid} seed{seed}: "
        f"zw_abs={_r2(results['z_world_abs'])} zw_disp={_r2(results['z_world_disp'])} "
        f"ze_abs={_r2(results['z_ego_abs'])} ze_disp={_r2(results['z_ego_disp'])} -> {out_path}"
    )
    return out_path


def discover_jobs(repo: Path) -> list[tuple[str, int, Path]]:
    jobs = []
    for gid, path in SEED0_CKPTS.items():
        p = repo / path
        if p.is_file():
            jobs.append((gid, 0, p))
    for seed in (1, 2):
        for gid, name in GRID_NAMES.items():
            p = repo / "outputs" / "grid" / f"{name}_seed{seed}" / "model.pt"
            if p.is_file():
                jobs.append((gid, seed, p))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gid", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data", default="data/pusht_96.lance")
    ap.add_argument("--out-dir", default="results/diagnostics/perfactor")
    ap.add_argument("--max-samples", type=int, default=8192)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    if args.checkpoint:
        if not args.gid or args.seed is None:
            raise SystemExit("--gid and --seed required with --checkpoint")
        run_one(Path(args.checkpoint), args.gid, args.seed, out_dir, args.data, args.max_samples)
        return

    jobs = discover_jobs(repo)
    print(f"[perfactor] {len(jobs)} checkpoints")
    for gid, seed, ckpt in jobs:
        out = out_dir / f"{gid}_seed{seed}.json"
        if out.is_file():
            print(f"[skip] {out} exists")
            continue
        run_one(ckpt, gid, seed, out_dir, args.data, args.max_samples)


if __name__ == "__main__":
    main()

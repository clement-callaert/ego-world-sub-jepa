"""Train the supervised block pose detector on a PushT Lance dataset.

The detector reads the block x, y and angle straight from the image. It is the
block sensor that the MPC can use.

Two practical points learned while building this:

1. Some dataset frames have the block knocked far off the board (x or y from
   about -1000 to +1300). The block is not visible there, so it cannot be
   localized, and those extreme targets wreck the normalization and the loss.
   We keep only frames whose block sits on the board, which is the only region
   the task cares about anyway.
2. We load the frames once into an in memory buffer and then sample mini batches
   from it. Cycling a multi worker DataLoader for thousands of steps piled up
   shared memory until it ran out, so a single pass is both simpler and safer.

Example:
    PYTHONPATH=. python3 scripts/train_detector.py \
        --dataset data/pusht_96.lance \
        --out outputs/pusht_hires_seed0/detector.pt \
        --img-size 96 --steps 3000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ewjepa.data import build_dataset
from ewjepa.detector import BlockPoseDetector, pose_to_target, save_detector
from ewjepa.utils import get_device, set_seed


# The block pose lives in columns [2, 5] of the state vector: x, y, angle.
BLOCK_SLICE = slice(2, 5)
# The board is 512 px. Keep frames whose block is on it; off board frames show
# no block and cannot be localized.
BOARD_MIN = 0.0
BOARD_MAX = 512.0


def _split_by_episode(dataset, val_frac: float, seed: int) -> tuple[Subset, Subset]:
    """Split windows into train and val by episode, so no episode leaks across.

    Neighbouring frames of one episode look almost identical, so putting some in
    train and some in val would give an over optimistic error. Splitting whole
    episodes avoids that leak.
    """
    clip_indices = getattr(dataset, "clip_indices", None)
    if clip_indices is None:
        raise ValueError("Dataset has no clip_indices; cannot split by episode.")
    episodes = sorted({ep for ep, _ in clip_indices})
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(episodes), generator=rng).tolist()
    n_val = max(1, int(round(len(episodes) * val_frac)))
    val_eps = {episodes[i] for i in perm[:n_val]}

    train_idx, val_idx = [], []
    for i, (ep, _start) in enumerate(clip_indices):
        (val_idx if ep in val_eps else train_idx).append(i)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _on_board(block: torch.Tensor) -> torch.Tensor:
    """Boolean mask of frames whose block x and y are on the board."""
    xy = block[:, :2]
    return ((xy >= BOARD_MIN) & (xy <= BOARD_MAX)).all(dim=1)


def _collect_buffer(subset: Subset, max_frames: int, num_workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load on board frames into memory as uint8 images and float block poses."""
    loader = DataLoader(
        subset,
        batch_size=256,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=False,
    )
    images, blocks = [], []
    seen = 0
    for batch in loader:
        pixels = batch["pixels"]  # (B, T, C, H, W) float in [0, 1]
        state = batch["state"]  # (B, T, 7)
        img = pixels.reshape(-1, *pixels.shape[2:])
        blk = state.reshape(-1, state.shape[-1])[:, BLOCK_SLICE]
        mask = _on_board(blk)
        if mask.any():
            # Store images as uint8 to keep the buffer small; convert back to
            # float when sampling.
            images.append((img[mask] * 255.0).round().to(torch.uint8))
            blocks.append(blk[mask])
            seen += int(mask.sum())
        if seen >= max_frames:
            break
    return torch.cat(images, dim=0), torch.cat(blocks, dim=0)


@torch.no_grad()
def _evaluate(detector: BlockPoseDetector, images: torch.Tensor, blocks: torch.Tensor, device: torch.device) -> tuple[float, float]:
    """Return held out block xy RMSE (px) and mean wrapped angle error (deg)."""
    detector.eval()
    sq_err = 0.0
    angle_err = 0.0
    count = images.shape[0]
    for start in range(0, count, 512):
        img = images[start : start + 512].to(device).float() / 255.0
        blk = blocks[start : start + 512].to(device)
        pred = detector.predict(img)
        sq_err += (pred[:, :2] - blk[:, :2]).pow(2).sum().item()
        diff = (pred[:, 2] - blk[:, 2]).abs()
        diff = torch.minimum(diff, 2.0 * math.pi - diff)
        angle_err += diff.sum().item()
    rmse = math.sqrt(sq_err / max(count, 1))
    mean_angle_deg = math.degrees(angle_err / max(count, 1))
    return rmse, mean_angle_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/pusht_96.lance")
    parser.add_argument("--out", default="outputs/pusht_hires_seed0/detector.pt")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-train-frames", type=int, default=80000)
    parser.add_argument("--max-val-frames", type=int, default=16000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics-out", default=None, help="Optional JSON path for final validation metrics.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    dataset = build_dataset(
        args.dataset,
        num_steps=2,
        image_key="pixels",
        proprio_key="proprio",
        action_key="action",
        state_key="state",
    )
    train_ds, val_ds = _split_by_episode(dataset, args.val_frac, args.seed)
    print(f"[detector] {len(train_ds)} train windows, {len(val_ds)} val windows")

    print("[detector] loading frames into memory ...")
    train_img, train_blk = _collect_buffer(train_ds, args.max_train_frames, args.num_workers)
    val_img, val_blk = _collect_buffer(val_ds, args.max_val_frames, args.num_workers)
    print(f"[detector] on board frames: {train_img.shape[0]} train, {val_img.shape[0]} val")

    detector = BlockPoseDetector(in_chans=3, width=args.width, img_size=args.img_size).to(device)
    xy_mean = train_blk[:, :2].mean(0).to(device)
    xy_std = train_blk[:, :2].std(0).to(device)
    detector.set_target_stats(xy_mean, xy_std)
    print(f"[detector] block xy mean={xy_mean.tolist()} std={xy_std.tolist()}")

    opt = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    n_train = train_img.shape[0]

    detector.train()
    pbar = tqdm(range(1, args.steps + 1), desc="detector", unit="step", dynamic_ncols=True)
    for step in pbar:
        idx = torch.randint(0, n_train, (args.batch_size,), generator=gen, device=device)
        images = train_img[idx.cpu()].to(device).float() / 255.0
        block = train_blk[idx.cpu()].to(device)
        target = pose_to_target(block, detector.xy_mean, detector.xy_std)

        pred = detector(images)
        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        postfix: dict[str, str] = {"loss": f"{loss.item():.4f}"}
        if step % 500 == 0 or step == 1:
            rmse, angle_deg = _evaluate(detector, val_img, val_blk, device)
            detector.train()
            postfix["val_rmse"] = f"{rmse:.1f}px"
            postfix["val_ang"] = f"{angle_deg:.1f}deg"
            pbar.write(
                f"[detector] step {step:5d} loss={loss.item():.4f} "
                f"val_block_xy_RMSE={rmse:.1f}px val_angle_err={angle_deg:.1f}deg"
            )
        pbar.set_postfix(postfix, refresh=False)

    pbar.close()
    rmse, angle_deg = _evaluate(detector, val_img, val_blk, device)
    print(f"[detector] FINAL val_block_xy_RMSE={rmse:.1f}px val_angle_err={angle_deg:.1f}deg")
    save_detector(args.out, detector)
    print(f"[detector] saved to {args.out}")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "dataset": args.dataset,
            "img_size": args.img_size,
            "steps": args.steps,
            "seed": args.seed,
            "val_block_xy_rmse_px": float(rmse),
            "val_angle_err_deg": float(angle_deg),
            "detector_out": args.out,
        }
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[detector] metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

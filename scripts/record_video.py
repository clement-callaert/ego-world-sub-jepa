"""Record MP4 videos of latent MPC on PushT.

Example:
    python scripts/record_video.py \\
        checkpoint=outputs/pusht_factored_sigreg_fix/model.pt \\
        episodes=3 video_dir=outputs/videos/my_run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import imageio.v3 as iio
from PIL import Image

import stable_worldmodel as swm
from ewjepa.utils import get_device, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import _build_policy, _load_model


def _latest_frame(pixels, env_idx: int = 0) -> np.ndarray:
    """Copy the latest frame (SWM reuses the pixel buffer each step)."""
    batch = np.asarray(pixels[env_idx])
    frame = batch[-1] if batch.ndim > 3 else batch
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame.copy())


def _save_episode_video(path: Path, frames: list[np.ndarray], fps: int = 15, scale: int = 8) -> None:
    """Save frames as mp4. Upscale small renders for viewing."""
    if not frames:
        print(f"[warn] no frames to save for {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    out_frames = []
    for f in frames:
        img = Image.fromarray(f)
        if scale > 1:
            w, h = img.size
            img = img.resize((w * scale, h * scale), Image.NEAREST)
        out_frames.append(np.asarray(img))
    # use imageio v3 (SWM save_video has a pyav bug)
    iio.imwrite(path, out_frames, fps=fps, codec="libx264")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    episodes = int(cfg.episodes)
    video_dir = Path(cfg.get("video_dir") or "outputs/videos/run")
    video_dir.mkdir(parents=True, exist_ok=True)

    model, normalizer = _load_model(cfg, device)
    policy = _build_policy(cfg, model, normalizer, device)

    img_size = model.cfg.img_size
    world = swm.World(
        cfg.data.env,
        num_envs=1,
        image_shape=(img_size, img_size),
        max_episode_steps=int(cfg.get("max_episode_steps", 300)),
    )
    world.set_policy(policy)

    successes = []
    meta = []

    for ep in range(episodes):
        seed = int(cfg.seed) + ep
        frames: list[np.ndarray] = []
        world.reset(seed=seed)
        info = world.infos
        frames.append(_latest_frame(info["pixels"], 0))

        state0 = np.asarray(info["state"]).reshape(-1)
        goal = np.asarray(info.get("goal_state", state0)).reshape(-1)
        block_d0 = float(np.linalg.norm(state0[2:4] - goal[2:4]))

        done = False
        steps = 0
        max_steps = int(cfg.get("max_episode_steps", 300))
        while not done and steps < max_steps:
            action = policy.get_action(info)
            _, world.rewards, world.terminateds, world.truncateds, world.infos = world.envs.step(action)
            info = world.infos
            done = bool(world.terminateds[0] or world.truncateds[0])
            frames.append(_latest_frame(info["pixels"], 0))
            steps += 1

        state_end = np.asarray(info["state"]).reshape(-1)
        block_d1 = float(np.linalg.norm(state_end[2:4] - goal[2:4]))
        success = bool(info.get("terminated", [False])[0])
        successes.append(success)

        out_path = video_dir / f"episode_{ep}.mp4"
        _save_episode_video(out_path, frames, fps=15)
        meta.append(
            {
                "episode": ep,
                "seed": seed,
                "steps": steps,
                "success": success,
                "block_dist_start": block_d0,
                "block_dist_end": block_d1,
                "video": str(out_path),
            }
        )
        print(
            f"[ep {ep}] success={success} steps={steps} "
            f"block_dist {block_d0:.0f} -> {block_d1:.0f} saved {out_path}"
        )

    summary = {
        "checkpoint": str(cfg.checkpoint),
        "episodes": episodes,
        "success_rate": float(sum(successes) / max(len(successes), 1) * 100.0),
        "runs": meta,
    }
    summary_path = video_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] success_rate={summary['success_rate']:.1f}% summary={summary_path}")


if __name__ == "__main__":
    main()

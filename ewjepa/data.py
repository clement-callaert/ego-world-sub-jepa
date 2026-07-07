"""Load training data from stable-worldmodel (SWM).

Reads Lance trajectories, picks columns we need, and converts pixels to float [0,1].
SWM is imported lazily so import ewjepa works without it installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_float_image(x: torch.Tensor) -> torch.Tensor:
    x = x if torch.is_tensor(x) else torch.as_tensor(x)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    return x.float()


class ObsTransform:
    """Transform for DataLoader workers (must be picklable)."""

    def __init__(
        self,
        image_key: str = "pixels",
        proprio_key: str = "proprio",
        action_key: str = "action",
        state_key: str | None = "state",
    ):
        self.image_key = image_key
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.state_key = state_key

    def __call__(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "pixels": _to_float_image(sample[self.image_key]),
            "action": torch.as_tensor(sample[self.action_key]).float(),
        }
        if self.proprio_key in sample:
            out["proprio"] = torch.as_tensor(sample[self.proprio_key]).float()
        if self.state_key is not None and self.state_key in sample:
            out["state"] = torch.as_tensor(sample[self.state_key]).float()
        return out


def make_obs_transform(
    image_key: str = "pixels",
    proprio_key: str = "proprio",
    action_key: str = "action",
    state_key: str | None = "state",
) -> ObsTransform:
    """Build an ObsTransform for SWM samples."""
    return ObsTransform(image_key, proprio_key, action_key, state_key)


class SyntheticPushTDataset(Dataset):
    """Fake PushT windows when no Lance file exists (for smoke tests)."""

    def __init__(
        self,
        num_episodes: int = 512,
        num_steps: int = 3,
        img_size: int = 64,
        proprio_dim: int = 4,
        seed: int = 0,
    ):
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.img_size = img_size
        self.proprio_dim = proprio_dim
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=self.gen).item()))
        t = self.num_steps
        h = w = self.img_size

        agent = torch.randn(t, self.proprio_dim, generator=g) * 0.2
        block = torch.cumsum(torch.randn(t, 3, generator=g) * 0.05, dim=0)
        block[:, :2] += agent[:, :2] * 0.3
        action = torch.randn(t, 2, generator=g).clamp(-1, 1)
        proprio = agent.clone()

        pixels = torch.zeros(t, 3, h, w)
        for step in range(t):
            cx = int((block[step, 0].item() * 0.5 + 0.5) * (w - 12))
            cy = int((block[step, 1].item() * 0.5 + 0.5) * (h - 12))
            cx, cy = max(0, min(w - 12, cx)), max(0, min(h - 12, cy))
            pixels[step, 0, cy : cy + 10, cx : cx + 10] = 0.8
            pixels[step, 1, cy : cy + 10, cx : cx + 10] = 0.3
            pixels[step, 2, cy : cy + 10, cx : cx + 10] = 0.2

        vel = torch.diff(agent[:, :2], dim=0, prepend=agent[:1, :2])
        state = torch.cat([agent[:, :2], block[:, :2], block[:, 2:3], vel], dim=-1)

        return {
            "pixels": pixels.float(),
            "proprio": proprio.float(),
            "action": action.float(),
            "state": state.float(),
        }


def _resolve_dataset_path(name: str) -> str:
    """Turn a dataset path into an absolute local path if it exists."""
    candidates = [Path(name)]
    if not Path(name).is_absolute():
        candidates.append(Path.cwd() / name)
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return name


def _subset_dataset_by_episodes(dataset, max_episodes: int):
    """Keep windows from the first max_episodes episodes only."""
    from torch.utils.data import Subset

    if max_episodes is None or not hasattr(dataset, "clip_indices"):
        return dataset
    keep = [i for i, (ep, _start) in enumerate(dataset.clip_indices) if ep < max_episodes]
    if not keep:
        raise ValueError(f"No training windows in the first {max_episodes} episodes.")
    print(f"[data] limiting to first {max_episodes} episodes ({len(keep)} windows).")
    return Subset(dataset, keep)


def build_dataset(
    name: str,
    num_steps: int,
    frameskip: int = 1,
    image_key: str = "pixels",
    proprio_key: str = "proprio",
    action_key: str = "action",
    state_key: str | None = "state",
    cache_dir: str | None = None,
    max_episodes: int | None = None,
    **load_kwargs: Any,
):
    """Load SWM dataset with sliding windows of length num_steps."""
    try:
        import stable_worldmodel as swm
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "stable-worldmodel is required for data loading. "
            "Install the experiment stack: pip install -e '.[experiments]'."
        ) from e

    keys = [image_key, action_key, proprio_key]
    if state_key is not None:
        keys.append(state_key)

    return _subset_dataset_by_episodes(
        swm.data.load_dataset(
            _resolve_dataset_path(name),
            num_steps=num_steps,
            frameskip=frameskip,
            cache_dir=cache_dir,
            keys_to_load=keys,
            transform=make_obs_transform(image_key, proprio_key, action_key, state_key),
            **load_kwargs,
        ),
        max_episodes,
    )


def build_dataloader(
    name: str,
    num_steps: int,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
    frameskip: int = 1,
    drop_last: bool = True,
    prefetch_factor: int = 4,
    dataset_kwargs: dict | None = None,
    synthetic_fallback: bool = True,
    synthetic_episodes: int = 512,
) -> "torch.utils.data.DataLoader":
    """DataLoader wrapper. Falls back to synthetic data if the path is missing."""
    from torch.utils.data import DataLoader

    dataset_kwargs = dataset_kwargs or {}
    resolved = _resolve_dataset_path(name)
    use_synthetic = (
        synthetic_fallback
        and isinstance(name, str)
        and not name.startswith(("hf://", "http://", "https://"))
        and resolved == name
        and not Path(name).exists()
        and not (Path.cwd() / name).exists()
    )
    if use_synthetic:
        print(f"[data] {name} not found, using SyntheticPushTDataset ({synthetic_episodes} episodes).")
        dataset = SyntheticPushTDataset(
            num_episodes=synthetic_episodes,
            num_steps=num_steps,
            img_size=dataset_kwargs.get("img_size", 64),
            proprio_dim=dataset_kwargs.get("proprio_dim", 4),
        )
    else:
        max_episodes = dataset_kwargs.pop("max_episodes", None)
        dataset = build_dataset(
            resolved if resolved != name else name,
            num_steps=num_steps,
            frameskip=frameskip,
            max_episodes=max_episodes,
            **dataset_kwargs,
        )
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)

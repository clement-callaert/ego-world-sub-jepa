"""Save trajectories to a Lance table with stable-worldmodel.

EnvPool runs envs one by one inside each process. Use --processes for real
parallelism. --num-envs only adds envs per process.

Example:
    python scripts/collect_data.py --episodes 2000 --out data/pusht.lance \
        --processes 16 --num-envs 2 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _build_collection_policy(env: str, seed: int, dist_constraint: float):
    """Pick a policy for data collection.

    On PushT, RandomPolicy rarely touches the block. We use WeakPolicy instead.
    Other envs use RandomPolicy.
    """
    import stable_worldmodel as swm

    if "PushT" in env:
        from stable_worldmodel.envs.pusht.expert_policy import WeakPolicy

        return WeakPolicy(dist_constraint=dist_constraint, seed=seed)
    return swm.policy.RandomPolicy(seed=seed)


def _collect_shard(
    env: str,
    out: str,
    episodes: int,
    num_envs: int,
    image_shape: tuple[int, int],
    max_episode_steps: int,
    seed: int,
    dist_constraint: float,
) -> str:
    # headless pygame for WSL / servers
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import stable_worldmodel as swm

    world = swm.World(
        env,
        num_envs=num_envs,
        image_shape=image_shape,
        max_episode_steps=max_episode_steps,
    )
    world.set_policy(_build_collection_policy(env, seed, dist_constraint))
    world.collect(out, episodes=episodes, seed=seed)
    return out


def _merge_lance_shards(shard_paths: list[Path], dest: Path) -> None:
    from stable_worldmodel.data.format import get_format
    from stable_worldmodel.data.utils import _episode_to_step_lists, load_dataset

    writer_cls = get_format("lance")
    with writer_cls.open_writer(dest, mode="overwrite") as writer:
        for shard in shard_paths:
            ds = load_dataset(str(shard.resolve()))
            for ep_idx in range(len(ds.lengths)):
                ep = ds.load_episode(ep_idx)
                ep_len = int(ds.lengths[ep_idx])
                writer.write_episode(_episode_to_step_lists(ep, ep_len))


def _split_episodes(total: int, n: int) -> list[int]:
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="swm/PushT-v1", help="SWM environment id.")
    parser.add_argument("--out", default="data/pusht.lance", help="Output Lance path.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument(
        "--processes",
        "--num-workers",
        type=int,
        default=16,
        dest="processes",
        help="Number of parallel collector processes (true CPU parallelism).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2,
        help="SWM env instances per process (stepped serially inside each process).",
    )
    parser.add_argument("--image-shape", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-constraint",
        type=float,
        default=60.0,
        help="PushT only: max pixel distance the collection agent stays from the "
        "block, so it interacts often enough to generate block motion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing Lance dataset at --out before collecting.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
        print(f"Removed existing dataset at {out}")

    image_shape = tuple(args.image_shape)
    shard_dir = out.parent / f".{out.name}.shards"
    if args.overwrite and shard_dir.exists():
        shutil.rmtree(shard_dir)

    if args.processes <= 1:
        _collect_shard(
            args.env,
            str(out),
            args.episodes,
            args.num_envs,
            image_shape,
            args.max_episode_steps,
            args.seed,
            args.dist_constraint,
        )
        print(f"Collected {args.episodes} episodes into {out}")
        return

    shard_dir.mkdir(parents=True, exist_ok=True)
    counts = _split_episodes(args.episodes, args.processes)
    shard_paths: list[Path] = []

    jobs = []
    for i, n_eps in enumerate(counts):
        if n_eps == 0:
            continue
        shard = shard_dir / f"shard_{i:03d}.lance"
        shard_paths.append(shard)
        jobs.append(
            (
                args.env,
                str(shard),
                n_eps,
                args.num_envs,
                image_shape,
                args.max_episode_steps,
                args.seed + i * 10_000,
                args.dist_constraint,
            )
        )

    print(
        f"[collect] {args.processes} processes requested, "
        f"{len(jobs)} active shards, {args.num_envs} envs/process"
    )

    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(_collect_shard, *job): job[1] for job in jobs}
        for fut in as_completed(futures):
            path = futures[fut]
            fut.result()
            print(f"[collect] shard done: {path}")

    print(f"[collect] merging {len(shard_paths)} shards → {out}")
    _merge_lance_shards(shard_paths, out)
    shutil.rmtree(shard_dir)
    print(f"Collected {args.episodes} episodes into {out}")


if __name__ == "__main__":
    main()

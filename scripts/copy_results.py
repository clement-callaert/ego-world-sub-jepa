"""Copy selected probe and evaluation outputs into results/ for git commit."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ewjepa.utils import _git_sha


def _latest_probe(out_dir: Path, tag: str) -> Path | None:
    path = out_dir / f"probe_{tag}.json"
    return path if path.exists() else None


def _latest_eval(out_dir: Path, tag: str, kind: str = "mppi") -> Path | None:
    path = out_dir / f"eval_{tag}_{kind}.json"
    return path if path.exists() else None


def _copy_if_exists(source: Path | None, destination: Path, copies: list[tuple[Path, Path]]) -> None:
    if source is None or not source.exists():
        print(f"[skip] missing {source}")
        return
    shutil.copy2(source, destination)
    copies.append((source, destination))
    print(f"[copy] {source} -> {destination}")


def _tag(checkpoint: str) -> str:
    return Path(checkpoint).parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy outputs to results/ for commit.")
    parser.add_argument("--factored-checkpoint", default=None, help="Checkpoint for a factored probe/eval.")
    parser.add_argument("--monolithic-checkpoint", default=None, help="Checkpoint for a monolithic probe/eval.")
    parser.add_argument("--planning-checkpoint", default=None, help="Checkpoint for a planning evaluation.")
    parser.add_argument("--block-detector", default=None, help="Detector used by --planning-checkpoint.")
    parser.add_argument("--probe-dir", default="outputs/probe")
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results = Path(args.results_dir)
    probe_dst = results / "probe"
    eval_dst = results / "eval"
    fig_dst = results / "figures"
    for d in (probe_dst, eval_dst, fig_dst):
        d.mkdir(parents=True, exist_ok=True)

    copies: list[tuple[Path, Path]] = []
    probe_dir = Path(args.probe_dir)
    eval_dir = Path(args.eval_dir)

    for checkpoint in (args.factored_checkpoint, args.monolithic_checkpoint):
        if checkpoint:
            tag = _tag(checkpoint)
            _copy_if_exists(
                _latest_probe(probe_dir, tag),
                probe_dst / f"{tag}.json",
                copies,
            )

    for checkpoint in (args.factored_checkpoint, args.planning_checkpoint):
        if checkpoint:
            tag = _tag(checkpoint)
            _copy_if_exists(
                _latest_eval(eval_dir, tag),
                eval_dst / f"{tag}_mppi.json",
                copies,
            )

    for src in Path("outputs/figures").glob("*.png"):
        dst = fig_dst / src.name
        shutil.copy2(src, dst)
        copies.append((src, dst))

    manifest = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "git_sha": _git_sha(),
        "runs": {
            "factored_probe": args.factored_checkpoint,
            "monolithic_probe": args.monolithic_checkpoint,
            "planning": {
                "checkpoint": args.planning_checkpoint,
                "block_detector": args.block_detector,
            },
        },
        "artifacts": [str(d) for _, d in copies],
    }
    manifest_path = results / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {manifest_path} ({len(copies)} artifacts)")


if __name__ == "__main__":
    main()

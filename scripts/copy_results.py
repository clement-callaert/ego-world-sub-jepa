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


def _artifact_path(destination: Path, copies: list[tuple[Path, Path]]) -> str | None:
    for _, dst in copies:
        if dst == destination:
            return str(dst)
    return str(destination) if destination.exists() else None


def _load_existing_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy outputs to results/ for commit.")
    parser.add_argument("--factored-checkpoint", default=None, help="Checkpoint for a factored probe/eval.")
    parser.add_argument("--monolithic-checkpoint", default=None, help="Checkpoint for a monolithic probe/eval.")
    parser.add_argument("--planning-checkpoint", default=None, help="Legacy alias for a single planning eval.")
    parser.add_argument("--block-detector", default=None, help="Detector shared by planning evals.")
    parser.add_argument("--detector-metrics", default=None, help="JSON with final detector validation metrics.")
    parser.add_argument("--probe-dir", default="outputs/probe")
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results = Path(args.results_dir)
    probe_dst = results / "probe"
    eval_dst = results / "eval"
    detector_dst = results / "detector"
    fig_dst = results / "figures"
    for d in (probe_dst, eval_dst, detector_dst, fig_dst):
        d.mkdir(parents=True, exist_ok=True)

    copies: list[tuple[Path, Path]] = []
    probe_dir = Path(args.probe_dir)
    eval_dir = Path(args.eval_dir)

    probe_checkpoints = [
        ckpt for ckpt in (args.factored_checkpoint, args.monolithic_checkpoint) if ckpt
    ]
    for checkpoint in probe_checkpoints:
        tag = _tag(checkpoint)
        _copy_if_exists(
            _latest_probe(probe_dir, tag),
            probe_dst / f"{tag}.json",
            copies,
        )

    eval_checkpoints = []
    for checkpoint in (args.factored_checkpoint, args.monolithic_checkpoint, args.planning_checkpoint):
        if checkpoint and checkpoint not in eval_checkpoints:
            eval_checkpoints.append(checkpoint)

    for checkpoint in eval_checkpoints:
        tag = _tag(checkpoint)
        _copy_if_exists(
            _latest_eval(eval_dir, tag),
            eval_dst / f"{tag}_mppi.json",
            copies,
        )

    if args.detector_metrics:
        metrics_src = Path(args.detector_metrics)
        metrics_name = metrics_src.parent.name if metrics_src.parent.name else metrics_src.stem
        _copy_if_exists(
            metrics_src,
            detector_dst / f"{metrics_name}.json",
            copies,
        )

    for src in Path("outputs/figures").glob("*.png"):
        dst = fig_dst / src.name
        shutil.copy2(src, dst)
        copies.append((src, dst))

    manifest_path = results / "manifest.json"
    manifest = _load_existing_manifest(manifest_path)

    manifest["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    manifest["git_sha"] = _git_sha()

    if args.factored_checkpoint and args.monolithic_checkpoint:
        factored_tag = _tag(args.factored_checkpoint)
        mono_tag = _tag(args.monolithic_checkpoint)
        factored_probe_dst = probe_dst / f"{factored_tag}.json"
        mono_probe_dst = probe_dst / f"{mono_tag}.json"
        factored_eval_dst = eval_dst / f"{factored_tag}_mppi.json"
        mono_eval_dst = eval_dst / f"{mono_tag}_mppi.json"
        detector_metrics_dst = None
        if args.detector_metrics:
            metrics_name = Path(args.detector_metrics).parent.name
            detector_metrics_dst = detector_dst / f"{metrics_name}.json"

        manifest["controlled_comparison_96px"] = {
            "data": "data/pusht_96.lance",
            "detector": args.block_detector,
            "detector_metrics": _artifact_path(detector_metrics_dst, copies) if detector_metrics_dst else None,
            "factored_hires": {
                "checkpoint": args.factored_checkpoint,
                "probe": _artifact_path(factored_probe_dst, copies),
                "eval": _artifact_path(factored_eval_dst, copies),
            },
            "monolithic_hires": {
                "checkpoint": args.monolithic_checkpoint,
                "probe": _artifact_path(mono_probe_dst, copies),
                "eval": _artifact_path(mono_eval_dst, copies),
            },
        }

    if args.factored_checkpoint or args.monolithic_checkpoint or args.planning_checkpoint:
        manifest["runs"] = manifest.get("runs", {})
        if args.factored_checkpoint:
            manifest["runs"]["factored_probe"] = args.factored_checkpoint
        if args.monolithic_checkpoint:
            manifest["runs"]["monolithic_probe"] = args.monolithic_checkpoint
        if args.planning_checkpoint or args.factored_checkpoint:
            manifest["runs"]["planning"] = {
                "checkpoint": args.planning_checkpoint or args.factored_checkpoint,
                "block_detector": args.block_detector,
            }

    existing_artifacts = set(manifest.get("artifacts", []))
    existing_artifacts.update(str(d) for _, d in copies)
    manifest["artifacts"] = sorted(existing_artifacts)

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {manifest_path} ({len(copies)} artifacts copied this run)")


if __name__ == "__main__":
    main()

"""Copy canonical probe/eval outputs into results/ for git commit."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy outputs to results/ for commit.")
    parser.add_argument("--factored-checkpoint", default="outputs/pusht_factored_cov_seed0/model.pt")
    parser.add_argument("--monolithic-checkpoint", default="outputs/pusht_monolithic_seed0/model.pt")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results = Path(args.results_dir)
    probe_dst = results / "probe"
    eval_dst = results / "eval"
    fig_dst = results / "figures"
    for d in (probe_dst, eval_dst, fig_dst):
        d.mkdir(parents=True, exist_ok=True)

    factored_tag = Path(args.factored_checkpoint).parent.name
    mono_tag = Path(args.monolithic_checkpoint).parent.name

    copies: list[tuple[Path, Path]] = []
    factored_probe = _latest_probe(Path("outputs/probe"), factored_tag)
    mono_probe = _latest_probe(Path("outputs/probe"), mono_tag)
    factored_eval = _latest_eval(Path("outputs/eval"), factored_tag)

    if factored_probe:
        dst = probe_dst / "factored_cov_seed0.json"
        shutil.copy2(factored_probe, dst)
        copies.append((factored_probe, dst))
    if mono_probe:
        dst = probe_dst / "monolithic_seed0.json"
        shutil.copy2(mono_probe, dst)
        copies.append((mono_probe, dst))
    if factored_eval:
        dst = eval_dst / "factored_cov_seed0_mppi.json"
        shutil.copy2(factored_eval, dst)
        copies.append((factored_eval, dst))

    for src in Path("outputs/figures").glob("*.png"):
        dst = fig_dst / src.name
        shutil.copy2(src, dst)
        copies.append((src, dst))

    train_log = Path("outputs/train_factored_cov_20k.log")
    if train_log.exists():
        dst = results / "train_factored_cov_20k.log"
        shutil.copy2(train_log, dst)
        copies.append((train_log, dst))

    manifest = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "git_sha": _git_sha(),
        "factored_checkpoint": args.factored_checkpoint,
        "monolithic_checkpoint": args.monolithic_checkpoint,
        "artifacts": [str(d) for _, d in copies],
    }
    manifest_path = results / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {manifest_path} ({len(copies)} artifacts)")


if __name__ == "__main__":
    main()

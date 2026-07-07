"""Plot eval and probe results from JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", nargs="*", default=[], help="Eval JSON files.")
    parser.add_argument("--probe", nargs="*", default=[], help="Probe JSON files.")
    parser.add_argument("--out", default="outputs/figures", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.probe:
        labels, world_r2, ego_r2 = [], [], []
        for path in args.probe:
            data = json.loads(Path(path).read_text())
            labels.append(Path(path).stem.replace("probe_", ""))
            world_r2.append(data["world_probe"]["r2"])
            ego_r2.append(data.get("ego_probe", {}).get("r2", float("nan")))
        x = range(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - w / 2 for i in x], world_r2, width=w, label="z_world")
        ax.bar([i + w / 2 for i in x], ego_r2, width=w, label="z_ego")
        ax.set_xticks(list(x), labels, rotation=15)
        ax.set_ylabel("R² (block pose)")
        ax.set_title("Linear probing: object pose decode")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out_dir / "probe_r2.png", dpi=150)
        plt.close(fig)

    if args.eval:
        for path in args.eval:
            data = json.loads(Path(path).read_text())
            clean = data.get("clean", {}).get("success_rate", 0.0)
            fov = {k.split("/", 1)[-1]: v["success_rate"] for k, v in data.items() if k.startswith("fov/")}
            if not fov:
                continue
            labels = ["clean", *fov.keys()]
            values = [clean, *fov.values()]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, values)
            ax.set_ylabel("Success rate (%)")
            ax.set_title(f"Planning robustness, {Path(path).stem}")
            ax.set_ylim(0, max(100, max(values) * 1.1 if values else 1))
            fig.tight_layout()
            fig.savefig(out_dir / f"{Path(path).stem}_robustness.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()

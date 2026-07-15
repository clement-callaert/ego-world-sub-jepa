"""Aggregate the screening grid (Step 3): CSV, statistics, and the figure.

Reads results/grid/gN_probe.json, gN_diagnostics.json, gN_mppi.json, writes:
  results/grid/grid.csv       one row per run
  results/grid/stats.json     Wilson intervals, Fisher pair tests, Spearman
  results/figures/grid_scatter.png (+ .pdf)

The figure has two panels sharing the y axis (planning success with 95%
Wilson error bars): left x = probe R2, right x = rollout RMSE at H=8.
Spearman correlation and p-value are printed on each panel.

Usage:
    python scripts/aggregate_grid.py
    python scripts/aggregate_grid.py --grid-dir results/grid --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

# Declared controlled comparisons (one factor at a time, except the
# architecturally imposed ego_loss_weight in mode pairs).
FISHER_PAIRS = [
    ("g1", "g2", "mode (sg=T cov=0.25 aux=1)"),
    ("g4", "g3", "mode (sg=F cov=0 aux=1)"),
    ("g7", "g8", "mode (pure JEPA)"),
    ("g1", "g5", "cov_weight (factored)"),
    ("g1", "g6", "stop_grad_target (factored)"),
    ("g1", "g7", "state_aux_weight (factored)"),
    ("g2", "g8", "state_aux_weight (monolithic)"),
]

# Okabe-Ito pair, colorblind safe in print.
MODE_COLORS = {"factored": "#0072B2", "monolithic": "#D55E00"}


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, in [0, 1]."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_run(grid_dir: Path, gid: str) -> dict | None:
    paths = {k: grid_dir / f"{gid}_{k}.json" for k in ("probe", "diagnostics", "mppi")}
    if not all(p.exists() for p in paths.values()):
        return None
    probe = json.loads(paths["probe"].read_text())
    diag = json.loads(paths["diagnostics"].read_text())
    ev = json.loads(paths["mppi"].read_text())

    model = probe["manifest"]["config"]["model"]
    n = int(ev["clean"]["episodes"])
    rate = float(ev["clean"]["success_rate"]) / 100.0
    successes = round(rate * n)
    lo, hi = wilson_interval(successes, n)
    rr = diag["rollout_rmse"]
    return {
        "config": gid,
        "mode": model["mode"],
        "sg": bool(model["stop_grad_target"]),
        "cov": float(model["cov_weight"]),
        "aux": float(model["state_aux_weight"]),
        "seed": int(probe["manifest"]["seed"]),
        "probe_r2": float(probe["world_probe"]["r2"]),
        "rollout_rmse_h1": float(rr["1"]["xy_rmse_px"]),
        "rollout_rmse_h4": float(rr["4"]["xy_rmse_px"]),
        "rollout_rmse_h8": float(rr["8"]["xy_rmse_px"]),
        "rollout_disp_rmse_h8": float(rr["8"]["disp_xy_rmse_px"]),
        "action_sensitivity": float(diag["action_sensitivity"]["normalized_xy_sensitivity"]),
        "planning_success": rate,
        "n_episodes": n,
        "successes": successes,
        "wilson_lo": lo,
        "wilson_hi": hi,
    }


def fisher_tests(rows: dict[str, dict]) -> list[dict]:
    from scipy.stats import fisher_exact

    out = []
    for a, b, label in FISHER_PAIRS:
        if a not in rows or b not in rows:
            continue
        ra, rb = rows[a], rows[b]
        table = [
            [ra["successes"], ra["n_episodes"] - ra["successes"]],
            [rb["successes"], rb["n_episodes"] - rb["successes"]],
        ]
        odds, p = fisher_exact(table)
        out.append({
            "pair": [a, b],
            "varies": label,
            "success": [f"{ra['successes']}/{ra['n_episodes']}", f"{rb['successes']}/{rb['n_episodes']}"],
            "odds_ratio": None if math.isinf(odds) else float(odds),
            "p_value": float(p),
        })
    return out


def make_figure(rows: list[dict], out_png: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    panels = [
        ("probe_r2", "Linear probe R$^2$ (block pose)", False),
        ("rollout_rmse_h8", "Rollout RMSE at H=8 (px)", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    stats = {}
    for ax, (xkey, xlabel, lower_is_better) in zip(axes, panels):
        xs = [r[xkey] for r in rows]
        ys = [100 * r["planning_success"] for r in rows]
        rho, p = spearmanr(xs, ys)
        stats[xkey] = {"spearman_rho": float(rho), "p_value": float(p)}
        for i, r in enumerate(rows):
            y = 100 * r["planning_success"]
            err_lo = y - 100 * r["wilson_lo"]
            err_hi = 100 * r["wilson_hi"] - y
            c = MODE_COLORS[r["mode"]]
            ax.errorbar(
                r[xkey], y, yerr=[[err_lo], [err_hi]],
                fmt="o", ms=7, color=c, ecolor=c, elinewidth=1.2, capsize=3, alpha=0.9,
            )
            # alternate label offsets so stacked points stay readable
            ax.annotate(
                r["config"], (r[xkey], y), textcoords="offset points",
                xytext=(6, 5 if i % 2 == 0 else -11), fontsize=8, color="#444444",
            )
        ax.set_xlabel(xlabel)
        ax.text(
            0.03, 0.95,
            f"Spearman $\\rho$ = {rho:.2f}, p = {p:.3f}",
            transform=ax.transAxes, va="top", fontsize=9, color="#222222",
        )
        if lower_is_better:
            ax.invert_xaxis()  # better models on the right in both panels
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Planning success (%, 95% Wilson)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=m)
        for m, c in MODE_COLORS.items()
    ]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.85), frameon=False, fontsize=9)
    fig.suptitle("Planning success vs probe R$^2$ and vs rollout error (screening grid, seed 0)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"[figure] {out_png} (+ .pdf)")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid-dir", default="results/grid")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    grid_dir = Path(args.grid_dir)
    out_dir = Path(args.out_dir)

    gids = sorted(
        {m.group(1) for p in grid_dir.glob("g*_probe.json") if (m := re.match(r"(g\d+)_probe", p.stem))},
        key=lambda g: int(g[1:]),
    )
    rows = []
    for gid in gids:
        row = load_run(grid_dir, gid)
        if row is None:
            print(f"[warn] {gid}: incomplete artifacts, skipped")
            continue
        rows.append(row)
    if len(rows) < 2:
        raise SystemExit(f"only {len(rows)} complete runs in {grid_dir}, nothing to aggregate")
    print(f"[grid] {len(rows)} complete runs: {[r['config'] for r in rows]}")

    csv_path = grid_dir / "grid.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] {csv_path}")

    by_gid = {r["config"]: r for r in rows}
    spearman = make_figure(rows, out_dir / "figures" / "grid_scatter.png")
    stats = {
        "n_runs": len(rows),
        "spearman": spearman,
        "fisher_pairs": fisher_tests(by_gid),
        "note": "One seed, one environment (PushT), n=50 episodes per run. "
        "Wilson 95% intervals per run; Fisher exact tests on declared pairs only.",
    }
    stats_path = grid_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] {stats_path}")
    for t in stats["fisher_pairs"]:
        print(f"  {t['pair'][0]} vs {t['pair'][1]} ({t['varies']}): {t['success']} p={t['p_value']:.4f}")


if __name__ == "__main__":
    main()

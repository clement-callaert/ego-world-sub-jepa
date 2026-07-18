# Multi-seed session report (uncommitted)

Date: 2026-07-18. PushT screening grid only. No OGBench, no second env.

## Runs

| Status | Count | Configs |
| --- | ---: | --- |
| OK | 5 | g1, g3, g7, g2, g4 (all seed 1) |
| FAILED | 0 | |
| SKIPPED | 0 after start | file stopped at g5 seed1 (budget exhausted) |

Priority order filled: P1 complete (g1, g3, g7, g2), then first P2 item (g4). Seeds 2 not started.

GPU wall time for the 5 OK runs (train+probe+diagnose+eval): **23775 s (~6.6 h)**.
Session budget was 6 h from crash restart; g4 finished after the soft budget wall, then the queue stopped cleanly.

Smoke g1 seed1: OK (init hash differed from seed 0). Train used `compile=false` (compile was slower on this stack); eval used `compile=true` and skip-goal-encode when `latent_cost_weight=0`.

## Target correlation

At **n=13** (8 seed-0 + 5 seed-1 points), planner-faithful Spearman:

| Metric vs planning success | rho | p |
| --- | ---: | ---: |
| probe_r2 | +0.273 | 0.367 |
| rollout_disp_rmse_h8 | **-0.442** | **0.130** |
| action_sensitivity | -0.405 | 0.170 |

Seed-0 only (n=8, committed `stats_corrected.json`): disp_h8 rho=-0.663, p=0.073.

Verdict in one sentence: the planner-faithful displacement-vs-planning correlation **weakens** at the n reached (smaller |rho|, larger p); it does not cross p<0.05 and is no longer as suggestive as the seed-0-only figure.

Notable seed-1 shifts (same MPPI protocol, train seed only): g1 planning 12% -> 4%; g7 disp_h8 91.7 -> 48.5 px with planning still 0%.

## Per-factor mean R2

Linear probes, same group split as `scripts/probe.py`. Means over available checkpoints (factored n=8, monolithic n=5). Ego is null for monolithic.

| Mode | z_world -> abs | z_world -> disp | z_ego -> abs | z_ego -> disp |
| --- | ---: | ---: | ---: | ---: |
| factored | 0.860 | 0.153 | 0.543 | 0.102 |
| monolithic | 0.916 | 0.211 | n/a | n/a |

Absolute block pose still sits mostly in `z_world` when aux is on; displacement R2 is low for both modes; ego carries moderate absolute R2 in factored runs but weak displacement.

## Files to commit (you commit yourself)

Do **not** overwrite committed seed-0 JSON under `results/grid/g*_*.json` or `stats_corrected.json`.

New / updated:

- `results/grid/seed1/g{1,2,3,4,7}_{probe,diagnostics,mppi}.json`
- `results/diagnostics/grid/seed1/g{1,2,3,4,7}_diagnostics.json`
- `results/diagnostics/perfactor/g*_seed{0,1}.json` (13 files)
- `results/grid/stats_corrected_multiseed.json`
- `results/grid/session_log.txt`
- `results/grid/SESSION_REPORT.md` (this file)
- `README.md` (Multi-seed update section at top)
- Code/scripts from this session: `scripts/run_seeds.sh`, `scripts/probe_perfactor.py`, `scripts/probe.py` (disp), `scripts/aggregate_grid.py` (--multiseed), `ewjepa/mpc_policy.py` (skip dead goal encode), `scripts/evaluate.py` / `scripts/diagnose.py` / `configs/eval.yaml` (TF32 + eval compile flag)

Checkpoints under `outputs/grid/*_seed1/` stay local (not evidence).

# Ego-World JEPA report

## Scope

This project studies a small factored latent world model for PushT. The model
uses two latents — `z_world` from the image and `z_ego` from proprioception —
predicts future latents, and plans with MPPI-based MPC. A monolithic baseline
with one latent is trained and evaluated on the identical stack.

This is NOT Sub-JEPA (Zhao et al.); the names are similar, the methods are
not. "Ego-world" means separate world and agent latents.

**Evidence policy:** only JSON files committed under `results/` are evidence
for the numbers below. Checkpoints, Lance datasets, and detector weights live
in `outputs/` locally and are not committed. Results are reported in three
tiers; Tier A is the main comparison but is confounded (see below).

## Tier A — Main comparison (96px, 2026-07-13, canonical, confounded)

Same data, same shared detector, same MPPI stack. The model configs differ on
FOUR axes, not one: `mode` (intended variable), `ego_loss_weight` 0.1 vs 0.0
(imposed by the architecture), `stop_grad_target` true vs false
(**confound**), and `cov_weight` 0.25 vs 0.0 (**confound**). See README.md
for the full delta table; a controlled one-factor-at-a-time ablation is in
progress. Both models also train with `state_aux_weight=1.0`, which directly
supervises the block pose in `z_world`, so the probe R² is quasi-saturated by
construction and is not an independent measure of representation quality.
Manifest: `results/manifest.json` → `controlled_comparison_96px`
(git SHA `c7debee`).

| | Factored hires | Monolithic hires |
| --- | --- | --- |
| Config | `configs/model/factored_hires.yaml` | `configs/model/monolithic_hires.yaml` |
| Block pose R² (probe, 8192 rows) | **0.9973** | **0.9947** |
| Planning success (50 ep, seed 0, MPPI) | **12.0% (6/50)** | **0.0% (0/50)** |
| Probe artifact | `results/probe/pusht_hires_seed0.json` | `results/probe/pusht_monolithic_hires_seed0.json` |
| Eval artifact | `results/eval/pusht_hires_seed0_mppi.json` | `results/eval/pusht_monolithic_hires_seed0_mppi.json` |

Shared detector (independent of either world model),
`results/detector/shared_pusht96_seed0.json`: val block xy RMSE **7.59 px**,
val angle error **2.07°**.

Protocol:

1. Collect `data/pusht_96.lance` (2000 episodes, 96×96).
2. Train both models: 20,000 steps, batch 256, seed 0.
3. Train the shared block detector: 6000 steps.
4. Probes: ridge regression to block pose `[x, y, angle]`,
   `probe.max_samples=8192`, grouped split.
5. Eval: MPPI `n_samples=512`, `n_iters=6`, `horizon=8`,
   `max_episode_steps=700`, `episodes=50`, shared detector, seed 0.

**Claims policy:** the factored-config model outperforms the
monolithic-config one on planning on this stack (12.0% vs 0.0%) with
near-identical probe R². The comparison is confounded on `stop_grad_target`
and `cov_weight`, is one run at n=50 episodes and a single seed; do not state
"factorization improves planning". Do not mix Tier B/C numbers into this
comparison.

Reproduce:

```bash
pip install -e ".[dev,experiments]"
export PYTHONPATH=.
bash scripts/reproduce_full_comparison.sh
# from scratch: FORCE_COLLECT=1 bash scripts/reproduce_full_comparison.sh
```

A fresh run reproduces the protocol, not bit-identical numbers, because
checkpoints and data are not committed.

## Tier B — Historical 64px probes (archived)

8192 latent rows, ridge regression to block pose, grouped split. Probe-only;
NOT a planning comparison. These predate the bug fixes in POSTMORTEM.md.

| Run | Block pose R² | Artifact |
| --- | ---: | --- |
| Factored cov (archived) | 0.2859 | `results/probe/factored_cov_seed0.json` |
| Monolithic cov (archived) | 0.7795 | `results/probe/monolithic_seed0.json` |

The factored probe records `world_head_norm: none` and
`stop_grad_target: false` (pre-fix). The monolithic probe artifact carries an
incorrect Hydra model label; its checkpoint path identifies the monolithic
checkpoint. Reproduce with `bash scripts/reproduce_probes.sh`.

## Tier C — Historical planning runs (not controlled)

| Run | Success | Artifact |
| --- | ---: | --- |
| Factored cov, 64px, no detector, 20 episodes | 0.0% | `results/eval/factored_cov_seed0_mppi.json` |
| Factored hires, MPPI + detector, 50 episodes | 6.0% (3/50) | `results/archive/eval_pusht_hires_seed0_mppi.json` |

Different checkpoints, resolutions, detector settings, and episode counts —
not comparable to each other or to Tier A, and not a 0% → 6% → 12%
progression. The 6% run's detector accuracy was never published. The
historical 6% artifact lives under `results/archive/` to avoid confusion
with the Tier A 12% artifact (`results/eval/pusht_hires_seed0_mppi.json`).

## Artifact publication

`scripts/reproduce_full_comparison.sh` calls `scripts/copy_results.py`
automatically. Manual copy:

```bash
python3 scripts/copy_results.py \
    --factored-checkpoint=outputs/pusht_hires_seed0/model.pt \
    --monolithic-checkpoint=outputs/pusht_monolithic_hires_seed0/model.pt \
    --block-detector=outputs/shared_pusht96_seed0/detector.pt \
    --detector-metrics=outputs/shared_pusht96_seed0/detector_metrics.json
```

This copies probe/eval/detector JSON files and figures, and updates
`results/manifest.json`.

## Debugging history

[POSTMORTEM.md](../POSTMORTEM.md) records the three main bugs (SIGReg
N-scaling, target collapse, BatchNorm train/eval mismatch), the planning
fixes, and the July 2026 closure recap. Measurements there are debugging
context, not published results, unless they also appear in `results/`.

# Ego-World JEPA

A small latent world model for PushT, built on data and environments from
[stable-worldmodel](https://github.com/facebookresearch/stable-worldmodel).
The model has two latents:

- `z_world` comes from the image.
- `z_ego` comes from proprioception.

It predicts future latents and plans with MPPI-based MPC. A monolithic
baseline with a single latent is trained and evaluated on the identical stack
for comparison.

**Name disclaimer:** this project is NOT Sub-JEPA (Zhao et al.). The names are
similar but the methods are unrelated. Here, "ego-world" simply means separate
latents for the world and the agent.

## Evidence policy

**Only JSON files committed under `results/` count as published evidence.**
Checkpoints, Lance datasets, and detector weights are generated locally into
`outputs/` and are NOT in git. Every number in this README quotes a specific
committed artifact. The artifact JSON schema (including the legacy
`invocation_config` variant and its known pitfall) is documented in
`results/SCHEMA.md` and validated by `tests/test_results_schema.py`.
Results fall in three tiers:

- **Tier A — Main comparison (96px, canonical).** Same data, same shared
  detector, same MPPI stack (`factored_hires` vs `monolithic_hires`).
  **Warning: the two model configs differ on four axes, not one** (see the
  delta table in the Tier A section), so this comparison is confounded on
  `stop_grad_target` and `cov_weight`. A controlled ablation is in progress.
- **Tier B — Historical 64px probes.** Archived configs (`factored_cov`,
  `monolithic_cov`). Probe-only; not a planning comparison.
- **Tier C — Historical planning runs.** Incomplete or different protocols.
  Kept for the record only; not comparable to Tier A or to each other.

In particular, the historical 0% (Tier C), the historical 6% (Tier C), and
the controlled 12% (Tier A) are three different experiments. They must NOT be
read as a 0% → 6% → 12% progression.

## Tier A — Main comparison (96px, 2026-07-13) — confounded

Both models were trained and evaluated on the identical pipeline:
`data/pusht_96.lance`, the same shared block detector, the same probes, and
the same MPPI planner. Manifest: `results/manifest.json`
(`controlled_comparison_96px`, git SHA `c7debee`).

**The two model configs are NOT identical up to `mode`.** They differ on four
axes:

| Config key | `factored_hires` | `monolithic_hires` | Status |
| --- | --- | --- | --- |
| `mode` | `factored` | `monolithic` | intended variable |
| `ego_loss_weight` | 0.1 | 0.0 | imposed by the architecture (no ego stream in monolithic) |
| `stop_grad_target` | `true` | `false` | **confound** (not imposed by the architecture) |
| `cov_weight` | 0.25 | 0.0 | **confound** (not imposed by the architecture) |

The 12% vs 0% result below is therefore confounded on `stop_grad_target` and
`cov_weight`; it cannot be attributed to the factorization alone. A controlled
ablation varying one factor at a time is in progress.

**Probe R² caveat.** Both models are trained with `state_aux_weight=1.0`: a
linear head reads the block pose from `z_world` and is directly supervised on
the true pose during training. The ridge probe below therefore measures almost
exactly what the auxiliary loss optimized — the near-saturated R² is partly
tautological, the probe is not an independent measure of representation
quality, and these models are not pure JEPA.

| | Factored hires | Monolithic hires |
| --- | --- | --- |
| Config | `configs/model/factored_hires.yaml` | `configs/model/monolithic_hires.yaml` |
| Checkpoint (local, not committed) | `outputs/pusht_hires_seed0/model.pt` | `outputs/pusht_monolithic_hires_seed0/model.pt` |
| Block pose R² (probe, 8192 rows) | **0.9973** | **0.9947** |
| Planning success (50 ep, seed 0, MPPI) | **12.0% (6/50)** | **0.0% (0/50)** |
| Probe artifact | `results/probe/pusht_hires_seed0.json` | `results/probe/pusht_monolithic_hires_seed0.json` |
| Eval artifact | `results/eval/pusht_hires_seed0_mppi.json` | `results/eval/pusht_monolithic_hires_seed0_mppi.json` |

Shared block detector (trained on the same data, independent of either world
model), artifact `results/detector/shared_pusht96_seed0.json`:

- Val block xy RMSE: **7.59 px** (at 96×96)
- Val angle error: **2.07°**
- Weights (local, not committed): `outputs/shared_pusht96_seed0/detector.pt`

**Interpretation and limits.** On this stack, the factored-config model
reaches 12.0% planning success while the monolithic-config model reaches
0.0%, with near-identical probe R². Because the configs differ on the four
axes above, this does NOT isolate the factorization: `stop_grad_target`
and/or `cov_weight` could explain part or all of the gap. It is also a single
run: n=50 episodes, one seed (0), one task, no confidence intervals or seed
variance. Do not cite this as "factorization improves planning".

### Protocol

1. Collect `data/pusht_96.lance` (2000 episodes, 96×96).
2. Train both models: 20,000 steps, batch 256, seed 0
   (`factored_hires`, `monolithic_hires`).
3. Train the shared block detector: 6000 steps.
4. Probes: ridge regression from `z_world` to block pose `[x, y, angle]`,
   `probe.max_samples=8192`, grouped train/test split (episodes do not leak
   across the split).
5. Eval: MPPI with `n_samples=512`, `n_iters=6`, `horizon=8`,
   `max_episode_steps=700`, `episodes=50`, shared detector, seed 0.

All hyperparameters are recorded inside each committed JSON under `manifest`.

### Reproduce

```bash
pip install -e ".[dev,experiments]"
export PYTHONPATH=.
bash scripts/reproduce_full_comparison.sh
# from scratch (recollect data):
FORCE_COLLECT=1 bash scripts/reproduce_full_comparison.sh
```

Needs CUDA and `stable-worldmodel`; developed on an RTX 5090 (batch 256 uses
~19 GB at 96×96). Expect several hours. The script collects data, trains both
models and the detector, runs both probes and both MPPI evals, and copies the
JSON artifacts into `results/` via `scripts/copy_results.py`. Because
checkpoints and data are not committed, a fresh run reproduces the protocol,
not bit-identical numbers.

Overrides: `TRAIN_STEPS=50000 EVAL_EPISODES=100 bash scripts/reproduce_full_comparison.sh`

## Tier B — Historical 64px probes (archived)

Archived configs at 64×64, probe-only (8192 rows, ridge regression, grouped
split). These predate the bug fixes below and are NOT a planning comparison.

| Run | Block pose R² | Artifact |
| --- | ---: | --- |
| Factored cov (archived) | 0.2859 | `results/probe/factored_cov_seed0.json` |
| Monolithic cov (archived) | 0.7795 | `results/probe/monolithic_seed0.json` |

Reproduce with `bash scripts/reproduce_probes.sh`.

## Tier C — Historical planning runs (not controlled)

| Run | Success | Artifact |
| --- | ---: | --- |
| Factored cov, 64px, no detector, 20 episodes | 0.0% | `results/eval/factored_cov_seed0_mppi.json` |
| Factored hires, MPPI + detector, 50 episodes | 6.0% (3/50) | `results/archive/eval_pusht_hires_seed0_mppi.json` |

These runs use different checkpoints, resolutions, detector settings, and
episode counts; neither is comparable to the other or to Tier A. The
historical 6% artifact lives under `results/archive/` to avoid confusion with
the Tier A 12% artifact `results/eval/pusht_hires_seed0_mppi.json`.

## Debugging history

Three bugs dominated the project (full detail in
[POSTMORTEM.md](POSTMORTEM.md)):

1. SIGReg was scaled by the number of samples, drowning the prediction loss.
2. The target latent could collapse to zero (`stop_grad_target` fix).
3. BatchNorm in the world head produced different latents in train vs eval
   (LayerNorm fix).

Measurements quoted in the postmortem are debugging context, not published
results, unless they also appear in `results/`.

## Install

```bash
pip install -e ".[dev,experiments]"
export PYTHONPATH=.
```

`requirements-results.txt` pins the Python package versions used for the
Tier A run (install a CUDA-compatible PyTorch wheel first). The committed
artifacts record `torch 2.13.0+cu130`.

## Publishing new artifacts

`scripts/reproduce_full_comparison.sh` copies artifacts automatically. Manual
copy:

```bash
python3 scripts/copy_results.py \
    --factored-checkpoint=outputs/pusht_hires_seed0/model.pt \
    --monolithic-checkpoint=outputs/pusht_monolithic_hires_seed0/model.pt \
    --block-detector=outputs/shared_pusht96_seed0/detector.pt \
    --detector-metrics=outputs/shared_pusht96_seed0/detector_metrics.json
```

This writes the probe/eval/detector JSON files and updates
`results/manifest.json` (`controlled_comparison_96px`).

## Layout

```text
ewjepa/           Model, encoders, detector, planning, and MPC policy
configs/          Hydra configuration files
scripts/          Data collection, training, probes, evaluation, and pipelines
results/          Committed evidence for reported results (JSON only)
docs/REPORT.md    Short report mirroring the tables above
POSTMORTEM.md     Debugging history and project closure recap
tests/            Unit tests
```

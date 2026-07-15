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

- **Tier A: Main comparison (96px, canonical).** Same data, same shared
  detector, same MPPI stack (`factored_hires` vs `monolithic_hires`).
  **Warning: the two model configs differ on four axes, not one** (see the
  delta table in the Tier A section), so this comparison is confounded on
  `stop_grad_target` and `cov_weight`. The screening grid below resolves the
  confounds: the matched comparison is 12% vs 4%, not significant at n=50.
- **Tier B: Historical 64px probes.** Archived configs (`factored_cov`,
  `monolithic_cov`). Probe-only; not a planning comparison.
- **Tier C: Historical planning runs.** Incomplete or different protocols.
  Kept for the record only; not comparable to Tier A or to each other.

In particular, the historical 0% (Tier C), the historical 6% (Tier C), and
the controlled 12% (Tier A) are three different experiments. They must NOT be
read as a 0% → 6% → 12% progression.

## Tier A: Main comparison (96px, 2026-07-13): confounded

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
`cov_weight`; it cannot be attributed to the factorization alone. The
screening grid section below varies one factor at a time: with the confounds
matched the gap is 12% vs 4% (Fisher p=0.269, not significant).

**Probe R² caveat.** Both models are trained with `state_aux_weight=1.0`: a
linear head reads the block pose from `z_world` and is directly supervised on
the true pose during training. The ridge probe below therefore measures almost
exactly what the auxiliary loss optimized: the near-saturated R² is partly
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

## Diagnostics: action-conditioned rollout error and action sensitivity

Probe R² alone does not measure whether the model supports planning. Two
diagnostics computed on the Tier A checkpoints, on 200 held-out episodes
(episode split; the frozen ridge readout is fitted on the 1800 training
episodes only, same procedure as the planner's readout). Run with
`python scripts/diagnose.py checkpoint=... data=pusht_96`. Artifacts:
`results/diagnostics/pusht_hires_seed0.json`,
`results/diagnostics/pusht_monolithic_hires_seed0.json`,
`results/eval/pusht_hires_seed0_random.json`.

**Open-loop rollout RMSE of the decoded block pose** (encode frame 0, roll
the predictor with the real dataset actions, decode with the frozen readout,
compare with the simulator pose). "abs" is the absolute decoded pose; "disp"
compares predicted vs true displacement from frame 0, which cancels the
per-frame readout bias and is what the planner consumes (displacement mode
anchored on the detector). "zero" is the trivial block-never-moves predictor.

| Horizon | Factored abs / disp (px) | Monolithic abs / disp (px) | Zero-motion (px) |
| --- | --- | --- | --- |
| 0 (readout only) | 167.3 / n.a. | 19.7 / n.a. | n.a. |
| 1 | 167.8 / 8.6 | 21.4 / 11.0 | 11.3 |
| 2 | 168.3 / 13.6 | 24.0 / 17.2 | 19.9 |
| 4 | 169.4 / 21.3 | 29.0 / 25.1 | 34.2 |
| 8 | 171.7 / 33.2 | 39.6 / 37.9 | 59.0 |

**Action sensitivity** (std in px of the decoded block xy at H=8 under K=32
uniform random action sequences, normalized by the dataset block xy std;
0 means the predictor is blind to the action):

| | Factored | Monolithic |
| --- | --- | --- |
| Normalized xy sensitivity | 0.184 | 0.315 |

**Random-actions baseline**: uniform random actions, same 50 episodes /
seed 0 / 700-step protocol as the MPPI evals: **0.0% (0/50)**. The 12%
factored MPPI result is therefore above the random floor; the monolithic 0%
is indistinguishable from it.

Readings. (1) Neither predictor is action-blind, so the monolithic 0%
planning is NOT mechanically explained by action blindness. (2) Both models
beat the zero-motion floor on displacement at every horizon, the factored
one by a wider margin. (3) The factored readout has a huge absolute error on
held-out episodes (167 px, vs 20 px monolithic) despite its 0.997 committed
probe R²; the committed probes span only the first ~14 episodes of the
dataset (8192 sequential rows), so the probe R² is not comparable to these
held-out numbers and overstates global readability.

## Screening grid

Eight configs under `configs/model/grid/`, seed 0, varying only `mode`,
`stop_grad_target` (sg), `cov_weight` (cov), and `state_aux_weight` (aux).
Everything else (96px, embed_dim 256, depth 6, patch 8, sigreg_mix, variance
terms, data, shared detector, MPPI planner, 20k steps, batch 256) is
identical across the grid. `ego_loss_weight` is 0.1 in factored mode and 0.0
in monolithic mode: the one architecturally imposed difference that cannot
be controlled.

| Config | mode | sg | cov | aux | Role |
| --- | --- | --- | --- | --- | --- |
| g1 | factored | T | 0.25 | 1.0 | = factored_hires (the Tier A 12%) |
| g2 | monolithic | T | 0.25 | 1.0 | the missing control |
| g3 | monolithic | F | 0.0 | 1.0 | = monolithic_hires (the Tier A 0%) |
| g4 | factored | F | 0.0 | 1.0 | symmetric of g3 |
| g5 | factored | T | 0.0 | 1.0 | isolates cov_weight |
| g6 | factored | F | 0.25 | 1.0 | isolates stop_grad_target |
| g7 | factored | T | 0.25 | 0.0 | pure JEPA, factored (probe R2 variance) |
| g8 | monolithic | T | 0.25 | 0.0 | pure JEPA, monolithic (probe R2 variance) |

g1 to g6 de-confound the Tier A comparison. g7 and g8 exist to create
variance on the probe R2: with aux=1.0 every R2 saturates near 0.995 and the
scatter is degenerate. Run with `bash scripts/run_grid.sh` (resumable, logs
GPU time), aggregate with `python3 scripts/aggregate_grid.py` (CSV, Wilson
95% intervals, Fisher exact tests on declared pairs, two-panel scatter with
Spearman correlations). Artifacts land in `results/grid/`.

### Results (seed 0, 50 episodes per run)

| Config | Probe R² | Rollout RMSE H=8 (px) | Disp RMSE H=8 (px) | Action sens. | Planning success (Wilson 95%) |
| --- | --- | --- | --- | --- | --- |
| g1 | 0.997 | 171.6 | 33.2 | 0.184 | 12% (6/50, 5.6 to 23.8) |
| g2 | 0.992 | 27.4 | 25.8 | 0.354 | 4% (2/50, 1.1 to 13.5) |
| g3 | 0.995 | 39.6 | 37.9 | 0.315 | 0% (0/50, 0 to 7.1) |
| g4 | 0.997 | 172.1 | 35.7 | 0.219 | 2% (1/50, 0.4 to 10.5) |
| g5 | 0.996 | 169.6 | 34.9 | 0.215 | 8% (4/50, 3.2 to 18.8) |
| g6 | 0.998 | 169.8 | 34.6 | 0.159 | 4% (2/50, 1.1 to 13.5) |
| g7 | 0.271 | 228.2 | 91.7 | 0.411 | 0% (0/50, 0 to 7.1) |
| g8 | 0.603 | 50.3 | 48.6 | 0.375 | 4% (2/50, 1.1 to 13.5) |

Fisher exact tests on the declared pairs: no single factor is significant at
n=50 except the state auxiliary loss in factored mode (g1 vs g7, 6/50 vs
0/50, p=0.027). The headline mode comparison g1 vs g2 (6/50 vs 2/50,
p=0.269) is not significant; neither are cov_weight (g1 vs g5, p=0.741) nor
stop_grad_target (g1 vs g6, p=0.269). Spearman correlations with planning
success across the 8 runs: probe R² rho=0.48 (p=0.23), rollout RMSE H=8
rho=-0.12 (p=0.77). Notably g2 has the best rollout displacement RMSE and
one of the highest action sensitivities yet plans at 4%, while g1 with far
worse absolute rollout RMSE plans best: at this sample size, neither probe
R² nor rollout error predicts planning success. Figure:
`results/figures/grid_scatter.png`. Per-run artifacts and `grid.csv` /
`stats.json` are in `results/grid/`.

What the grid does NOT show, by design: it is one
seed (0), one environment (PushT), n=50 episodes per run, and there are no
inter-seed intervals at this stage. The Wilson bars capture per-run binomial
noise only, not seed-to-seed variance.

## Tier B: Historical 64px probes (archived)

Archived configs at 64×64, probe-only (8192 rows, ridge regression, grouped
split). These predate the bug fixes below and are NOT a planning comparison.

| Run | Block pose R² | Artifact |
| --- | ---: | --- |
| Factored cov (archived) | 0.2859 | `results/probe/factored_cov_seed0.json` |
| Monolithic cov (archived) | 0.7795 | `results/probe/monolithic_seed0.json` |

Reproduce with `bash scripts/reproduce_probes.sh`.

## Tier C: Historical planning runs (not controlled)

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

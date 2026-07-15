# Post-mortem

This file records the debugging history of the project and the July 2026
closure recap. It is not a result table. Only JSON files committed under
`results/` are published result evidence; local measurements quoted here are
context for the fixes, nothing more.

## Three important bugs

### 1. SIGReg scaling

The Epps-Pulley statistic was multiplied by the number of samples, which made
the SIGReg regularizer orders of magnitude larger than the prediction loss and
dominated training. The fix removed the extra N-scaling.

The historical training log in `results/train_factored_cov_20k.log` records
the old behavior. It must not be used as evidence about the current training
config.

### 2. Target collapse to zero

The target branch could collapse to a constant latent, giving the predictor a
trivial shortcut. The current factored configuration uses
`stop_grad_target: true` to prevent it.

### 3. Train/eval latent mismatch

The old world head used BatchNorm, so training-mode and evaluation-mode
latents differed. The predictor was trained against training-mode latents but
the planner rolled out with evaluation-mode latents. The current
configurations use LayerNorm in the world head (`world_head_norm: layernorm`,
recorded in the committed probe/eval manifests).

## Planning fixes

Beyond the model bugs, the planning stack was changed to:

- keep the agent on the board (bounds cost),
- use the correct goal state (the env's `info["goal_pose"]` is not the
  success target; the goal block pose comes from `goal_state[2:5]`),
- use explicit approach / push / park phases in the cost,
- use a supervised block detector for the pose readout when one is provided.

## Historical planning runs (Tier C — not controlled)

- **0.0%** — `results/eval/factored_cov_seed0_mppi.json`: 64×64 factored_cov
  model, no detector, 20 episodes, pre-fix planning stack.
- **6.0% (3/50)** — `results/archive/eval_pusht_hires_seed0_mppi.json`: an
  isolated factored_hires run with MPPI and a detector, before the controlled
  comparison existed. Its detector accuracy was never published as an
  artifact.

These two runs use different models, resolutions, detectors, and episode
counts. They are not a controlled comparison with each other, and neither is
part of the Tier A comparison below. Do not present 0% → 6% → 12% as a
progression.

## July 2026 closure recap

The main open item — comparing factored and monolithic models on the same
planning stack — was closed on 2026-07-13 with the controlled 96px comparison
(git SHA `c7debee`, `results/manifest.json` → `controlled_comparison_96px`):

- Same data (`data/pusht_96.lance`, 2000 episodes), same training budget
  (20k steps, batch 256, seed 0), same shared detector (6000 steps; val block
  xy RMSE 7.59 px, val angle error 2.07°,
  `results/detector/shared_pusht96_seed0.json`), same probes (8192 rows,
  grouped split), same MPPI eval (512 samples, 6 iters, horizon 8, 50
  episodes, seed 0).
- Factored hires: probe R² 0.9973, planning success **12.0% (6/50)**
  (`results/probe/pusht_hires_seed0.json`,
  `results/eval/pusht_hires_seed0_mppi.json`).
- Monolithic hires: probe R² 0.9947, planning success **0.0% (0/50)**
  (`results/probe/pusht_monolithic_hires_seed0.json`,
  `results/eval/pusht_monolithic_hires_seed0_mppi.json`).

The detector-accuracy gap was also closed: the shared detector's validation
metrics are now a committed artifact.

Interpretation is limited by n=50 episodes and a single seed. Reproduce with
`bash scripts/reproduce_full_comparison.sh` (see README).

## Screening grid closure (2026-07-15)

The Tier A comparison was confounded on `stop_grad_target` and `cov_weight`.
The 8-config screening grid (seed 0, 50 episodes per run, one factor at a
time; see README "Screening grid" and `results/grid/`) closed that item:

- No single factor is significant at n=50 except the state auxiliary loss in
  factored mode (g1 vs g7: 6/50 vs 0/50, Fisher p=0.027). Without it the
  factored probe R² collapses to 0.27 and planning to 0%.
- The de-confounded mode comparison (g1 factored 6/50 vs g2 monolithic 2/50,
  p=0.269) is NOT significant; the Tier A 12% vs 0% gap does not survive
  matching `stop_grad_target` and `cov_weight`. Do not present the Tier A
  result as evidence that factorization improves planning.
- Neither probe R² (Spearman rho=0.48, p=0.23) nor rollout RMSE (rho=-0.12,
  p=0.77) predicts planning success across the 8 runs. g2 has the best
  rollout displacement RMSE (25.8 px at H=8) yet plans at 4%, while g1 with
  ~170 px absolute rollout RMSE plans best at 12%.

## What remains open

- More seeds and confidence intervals: the grid is one seed, and its
  interesting cells (g1, g2, g5) are within binomial noise of each other.
- Factors-of-variation / robustness evaluations (the eval config supports
  them; `robustness.enabled` was false in the committed runs).
- Absolute success rates are low (12% best); the planner and cost shaping
  remain the bottleneck, not the probe quality.

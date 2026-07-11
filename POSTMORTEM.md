# Post-mortem

This file records debugging work from July 2026. It is not a result table.
Only JSON files in `results/` are reproducible result evidence.

## Three important bugs

### 1. SIGReg scaling

The Epps-Pulley calculation was multiplied by the number of samples. This made
the SIGReg term much larger than the prediction loss. The implementation was
changed to remove this extra scaling.

The historical training log in `results/train_factored_cov_20k.log` records the
old behavior. It must not be used as evidence for the current training config.

### 2. Collapse to zero

The target branch could collapse to a constant latent. The current factored
configuration uses `stop_grad_target: true` to prevent this shortcut.

### 3. Train and evaluation mismatch

The old world head used BatchNorm. It produced different latents during training
and evaluation. The predictor learned with training latents but planned with
evaluation latents. The current factored configuration uses LayerNorm.

## Planning fixes

The planning code was also changed to keep the agent on the board, use the
correct goal state, use explicit approach, push, and park phases, and use a
supervised detector when one is provided.

The committed planning result is 6.0% (3/50) in
`results/eval/eval_pusht_hires_seed0_mppi.json`. It uses a factored hires
checkpoint, MPPI, and a detector. The repository does not commit the detector
weights or a detector accuracy artifact.

The earlier 0% artifact uses a different 64x64 model without a detector and
only 20 episodes. It is not a controlled baseline for the 6.0% result.

## What remains open

- Compare factored and monolithic models with the same planning stack.
- Run more seeds.
- Run factors-of-variation evaluations.
- Commit an explicit detector evaluation artifact before reporting detector
  accuracy.

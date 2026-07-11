# Ego-World JEPA report

## Scope

This project studies a small factored latent world model for PushT. The model
uses two latents:

- `z_world` for the image and the block.
- `z_ego` for agent proprioception.

The model predicts future latents. It plans with MPC. The project also contains
a monolithic baseline with one latent.

This report only treats files in `results/` as evidence for result numbers.
Checkpoints, datasets, and detector weights are not committed.

## Committed results

### Representation probes

The two probes use 8,192 latent rows. They predict block pose `[x, y, angle]`
with ridge regression and a grouped split.

| Run | Block pose R² | Artifact |
| --- | ---: | --- |
| Factored cov baseline | 0.285869 | `results/probe/factored_cov_seed0.json` |
| Monolithic baseline | 0.779477 | `results/probe/monolithic_seed0.json` |

These runs are historical 64x64 configurations. The factored probe records
`world_head_norm: none` and `stop_grad_target: false`. The monolithic probe
artifact has an incorrect Hydra model label, but its checkpoint path identifies
the monolithic checkpoint.

The monolithic model is better on this probe. This is not a planning
comparison.

### Planning

| Run | Success | Artifact |
| --- | ---: | --- |
| Factored cov baseline, no detector, 20 episodes | 0.0% | `results/eval/factored_cov_seed0_mppi.json` |
| Factored hires, MPPI, detector, 50 episodes | 6.0% (3/50) | `results/eval/eval_pusht_hires_seed0_mppi.json` |

The two planning runs are different experiments. They use different model
checkpoints, data resolutions, MPPI settings, detector settings, and episode
counts. Therefore, they are not a controlled 0% to 6% comparison.

The planning artifact records a detector path and a LayerNorm model setting.
It does not record detector accuracy. This repository does not publish a
detector accuracy result.

Monolithic planning has not been evaluated with the same hires detector stack.
The claim that factorization improves planning is open.

## Reproduce

The full runs need CUDA, `stable-worldmodel`, and locally collected Lance data.
They can take several hours. The generated weights stay in `outputs/`.

Install the experiment environment:

```bash
pip install -e ".[dev,experiments]"
export PYTHONPATH=.
```

Reproduce the archived probe configurations:

```bash
bash scripts/reproduce_probes.sh
```

This script uses the archived `factored_cov` and `monolithic_cov` configs. It
uses `probe.num_steps=9` and `probe.max_samples=8192`, as recorded by the
committed probe artifacts.

Reproduce the factored hires planning run:

```bash
bash scripts/reproduce_planning.sh
```

This script collects 96x96 data, trains the factored hires model, trains a
detector, and evaluates 50 episodes with MPPI. A new run may not reproduce the
exact 6.0% score because the original checkpoint and data are not committed.

## Artifact publication

Run `scripts/copy_results.py` after an experiment to copy selected JSON outputs
to `results/`. The script derives artifact names from checkpoint directories and
keeps probe runs separate from planning runs in `results/manifest.json`.

## Historical debugging notes

[POSTMORTEM.md](../POSTMORTEM.md) records debugging work. Its local measurements
are useful context, but they are not published results unless they also appear
in `results/`.

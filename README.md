# Ego-World JEPA

A small latent world model for PushT from
[stable-worldmodel](https://github.com/facebookresearch/stable-worldmodel).

The model has two latents:

- `z_world` comes from the image.
- `z_ego` comes from proprioception.

It predicts future latents and plans with MPC.

## Name

This project is not Sub-JEPA by Zhao et al. The names are similar, but the
methods are different. Here, ego-world means separate latents for the world and
the agent.

## Results

Only files committed in `results/` support the numbers below. Checkpoints,
datasets, and detector weights are not committed.

### Representation probes

Both probes use 8,192 latent rows. They predict block pose `[x, y, angle]` with
ridge regression and a grouped split.

| Run | Block pose R² | Artifact |
| --- | ---: | --- |
| Factored cov baseline | 0.285869 | `results/probe/factored_cov_seed0.json` |
| Monolithic baseline | 0.779477 | `results/probe/monolithic_seed0.json` |

The monolithic baseline is better on this probe. This is not a planning result.

### Planning

| Run | Success | Artifact |
| --- | ---: | --- |
| Factored cov, no detector, 20 episodes | 0.0% | `results/eval/factored_cov_seed0_mppi.json` |
| Factored hires, MPPI, detector, 50 episodes | 6.0% (3/50) | `results/eval/eval_pusht_hires_seed0_mppi.json` |

These are different experiments. They do not form a controlled 0% to 6%
comparison. The 6% run uses a factored hires checkpoint, MPPI, and a detector.
The detector accuracy is not published as a committed artifact.

Monolithic planning has not been evaluated on the same stack. The effect of
factorization on planning is still open.

## Important debugging work

The project diagnosed three important bugs:

1. SIGReg was scaled incorrectly.
2. The target latent could collapse to zero.
3. BatchNorm created different latents in training and evaluation.

The full explanation is in [POSTMORTEM.md](POSTMORTEM.md). Historical
measurements in that file are not published results unless they also appear in
`results/`.

## Install

The full experiment needs CUDA, `stable-worldmodel`, and locally collected
Lance data. It was developed with an RTX 5090.

```bash
pip install -e ".[dev,experiments]"
export PYTHONPATH=.
```

`requirements-results.txt` records the Python versions used for this result
cleanup. Install a CUDA-compatible PyTorch wheel before using that file.

## Reproduce the archived probes

```bash
bash scripts/reproduce_probes.sh
```

This trains the archived `factored_cov` and `monolithic_cov` configurations. It
uses the probe settings recorded in the committed JSON:
`probe.num_steps=9` and `probe.max_samples=8192`.

## Reproduce the planning pipeline

```bash
bash scripts/reproduce_planning.sh
```

This collects 96x96 data, trains the factored hires model, trains a detector,
and evaluates 50 episodes with MPPI. A new run may differ from 6.0% because the
original dataset and weights are not committed.

For a configurable full run, use:

```bash
bash scripts/pipeline_long.sh
```

## Publish new artifacts

After an experiment, copy selected outputs with:

```bash
python3 scripts/copy_results.py \
    --planning-checkpoint=outputs/pusht_hires_seed0/model.pt \
    --block-detector=outputs/pusht_hires_seed0/detector.pt
```

The script uses the checkpoint directory name for the destination file and
writes a separate probe and planning manifest.

## Layout

```text
ewjepa/           Model, encoders, detector, planning, and MPC policy
configs/          Hydra configuration files
scripts/          Data collection, training, probes, evaluation, and pipelines
results/          Committed evidence for reported results
docs/REPORT.md    Short report based on committed evidence
POSTMORTEM.md     Historical debugging notes
tests/            Unit tests
```

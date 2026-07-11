# Ego-World JEPA

This repository is a small latent world model for PushT from
[stable-worldmodel](https://github.com/facebookresearch/stable-worldmodel).

The model has two latents:

- `z_world` comes from the image. It represents the block and the scene.
- `z_ego` comes from proprioception. It represents the agent.

The model predicts the next latent state. It does not reconstruct pixels.
It plans actions with MPC and MPPI.

## Important name note

This project is not Sub-JEPA by Zhao et al. The names are similar, but the
methods are different. Here, "ego-world" means that the world and the agent use
separate latent spaces.

## Results

All numbers below come from JSON files committed in `results/`. They use seed 0.

### Representation probes

The probe predicts block pose `[x, y, angle]` from a frozen latent. It uses a
ridge regression and 8,192 samples.

| Model | Block pose R² | Evidence |
| --- | ---: | --- |
| Factored model | 0.286 | `results/probe/factored_cov_seed0.json` |
| Monolithic baseline | 0.779 | `results/probe/monolithic_seed0.json` |

The monolithic baseline is better on this probe. This is a representation
result only. It is not a planning comparison.

The factorized JSON records `world_head_norm: "none"`. Older documents called
this run BatchNorm. The committed JSON does not support that label, so this
README does not use it.

### Closed-loop planning

Planning improved from 0% to **6.0%**, or **3 successes in 50 episodes**.

This result uses:

- the factored model with a LayerNorm head
- MPPI
- a supervised block pose detector
- a 96x96 PushT dataset

The run is recorded in `results/eval/eval_pusht_hires_seed0_mppi.json`.

This is still a weak result. The detector estimates the block pose at about
8 px error. The JEPA latent is used to predict block motion during planning.
The monolithic baseline has not been evaluated with this same detector and MPC
setup.

## Three important bugs we diagnosed

The project includes a full debugging log in
[POSTMORTEM.md](POSTMORTEM.md). The most important findings are:

1. SIGReg was scaled by the number of samples. Its loss became much too large.
2. The target latent could collapse to zero. The fix was to stop gradients in
   the target branch.
3. BatchNorm gave different latents during training and evaluation. The
   predictor learned from one latent distribution and planned with another.
   The fix was to use LayerNorm.

These bugs explain why early training and planning results were unreliable.

## Status

| Part | Status |
| --- | --- |
| Factored and monolithic models | Implemented |
| SIGReg and covariance loss | Implemented |
| Linear probes | Implemented |
| Supervised block detector | Implemented |
| MPPI evaluation | Implemented |
| Factored versus monolithic planning | Not yet compared |
| Several seeds and robustness tests | Not yet run |

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Collect data.
python3 scripts/collect_data.py --episodes 2000 --out data/pusht.lance \
    --overwrite --processes 16 --num-envs 2

# Train the two models.
python3 scripts/train.py model=factored data=pusht \
    out_dir=outputs/pusht_factored_seed0 train.steps=20000
python3 scripts/train.py model=monolithic data=pusht \
    out_dir=outputs/pusht_monolithic_seed0 train.steps=20000

# Run a probe on each checkpoint.
python3 scripts/probe.py checkpoint=outputs/pusht_factored_seed0/model.pt \
    synthetic_fallback=false
python3 scripts/probe.py checkpoint=outputs/pusht_monolithic_seed0/model.pt \
    synthetic_fallback=false

python3 -m pytest tests/ -q --ignore=tests/test_train_speed.py
```

Without `data/pusht.lance`, training uses synthetic data for smoke tests only.

## Reproduce the 6% planning run

The planning run needs `data/pusht_96.lance`, a trained model, and a trained
block detector.

```bash
export PYTHONPATH=.

# Collect 96x96 data and train the world model.
python3 scripts/collect_data.py --out data/pusht_96.lance --episodes 2000 \
    --processes 32 --image-shape 96 96 --overwrite
python3 scripts/train.py model=factored_hires data=pusht_96 train.steps=20000 \
    train.batch_size=256 train.warmup_steps=1000 \
    out_dir=outputs/pusht_hires_seed0

# Train the supervised block pose detector.
python3 scripts/train_detector.py --dataset data/pusht_96.lance \
    --out outputs/pusht_hires_seed0/detector.pt --img-size 96 --steps 6000

# Evaluate 50 episodes with MPPI.
python3 scripts/evaluate.py checkpoint=outputs/pusht_hires_seed0/model.pt \
    block_detector=outputs/pusht_hires_seed0/detector.pt data=pusht_96 episodes=50
```

## Method

- `WorldViT` encodes RGB images.
- `EgoMLP` encodes proprioception.
- The factored model predicts `z_world` and `z_ego` separately.
- The monolithic baseline uses one latent for both.
- The loss uses prediction loss, SIGReg, a variance loss, covariance loss, and
  optional state supervision.
- Planning uses `LatentMPCPolicy`. It supports CEM, MPPI, and Hermite MPPI.

## Repository layout

```text
ewjepa/           Model, encoders, detector, planning, and MPC policy
configs/          Hydra configuration files
scripts/          Data collection, training, probing, and evaluation
results/          Committed probe and evaluation JSON files
tests/            Unit tests
docs/REPORT.md    Detailed technical report
POSTMORTEM.md     Debugging log and measured results
```

## References

- LeCun, JEPA blueprint, 2022
- Balestriero and LeCun, LeJEPA and SIGReg, 2025
- Maes et al., stable-worldmodel, 2026
- Schramm et al., Hermite MPPI, ICRA 2026

## Hardware

The project was run on WSL2 with an RTX 5090 with 32 GB of memory. Use
`python3`. Training for 20,000 steps takes about 40 minutes per model.

# Ego-World JEPA

Small **factored (ego/world) latent world model** on [stable-worldmodel](https://github.com/facebookresearch/stable-worldmodel) PushT. Two latents: `z_world` from pixels (block and scene), `z_ego` from proprio (agent). Dynamics are learned in latent space (JEPA, no pixel decoding). Planning is latent MPC (MPPI).

**What works today (seed 0, measured):**

- World model trains stably with SIGReg + covariance decorrelation.
- Linear probe: block pose R2 about **0.8** from `z_world`, agent xy R2 about **1.0** from `z_ego` (LayerNorm head, grouped split).
- Predictor responds to actions at eval (about **67%** of a true one-step latent move; was 4.8% with a BatchNorm head).
- Planning success **6.0% (3/50 episodes)** with a supervised block detector + corrected MPC (was 0% for weeks).

**What is still weak:** 6% is low. The JEPA latent alone only localizes the block to about 45 px; we added a small supervised CNN detector (~8 px) for control. Monolithic vs factored planning has not been compared on the same eval stack yet.

> **Name note:** Not related to the [Sub-JEPA paper](https://arxiv.org/abs/2605.09241). Here "ego-world" means factored latents with LeJEPA-style SIGReg training.

We also train a **monolithic** LeWM-style baseline (~1.46M params vs ~1.59M factored) for representation comparison. See `configs/model/monolithic.yaml` and `results/probe/monolithic_seed0.json`.

Full debugging story: [`POSTMORTEM.md`](POSTMORTEM.md). Technical report: [`docs/REPORT.md`](docs/REPORT.md).

---

## Status (2026-07-09, seed 0)

| Component | Status |
| --- | --- |
| Factored + monolithic models, SIGReg, `L_cov`, state supervision | Done, tested |
| LayerNorm head (train/eval match) | Fixed (was BatchNorm) |
| Linear probing (grouped split), collapse diagnostics | Working |
| Unit tests | **44 passed** (`pytest tests/`, skip `test_train_speed`) |
| Block detector + corrected MPC | Done; planning **6.0%** on 50 ep |
| Monolithic planning on detector stack | Not run yet |
| Multi-seed / FoV robustness sweeps | Not done |

---

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Data (WeakPolicy so the block moves)
python3 scripts/collect_data.py --episodes 2000 --out data/pusht.lance \
    --overwrite --processes 16 --num-envs 2

# Train both baselines (~40 min each on RTX 5090)
python3 scripts/train.py model=factored data=pusht \
    out_dir=outputs/pusht_factored_ln_seed0 train.steps=20000
python3 scripts/train.py model=monolithic data=pusht \
    out_dir=outputs/pusht_monolithic_seed0 train.steps=20000

# Probe (grouped split, real Lance data)
python3 scripts/probe.py checkpoint=outputs/pusht_factored_ln_seed0/model.pt \
    synthetic_fallback=false
python3 scripts/probe.py checkpoint=outputs/pusht_monolithic_seed0/model.pt \
    synthetic_fallback=false

python3 -m pytest tests/ -q --ignore=tests/test_train_speed.py
```

Without `data/pusht.lance`, training uses a synthetic dataset for smoke tests only.

---

## Reproduce results

### A. Representation (probe R2 about 0.8)

64x64, LayerNorm head, 20k steps:

```bash
export PYTHONPATH=.
python3 scripts/collect_data.py --out data/pusht.lance --episodes 2000 \
    --processes 16 --num-envs 2 --image-shape 64 64 --overwrite
python3 scripts/train.py model=factored data=pusht train.steps=20000 \
    out_dir=outputs/pusht_factored_ln_seed0
python3 scripts/probe.py checkpoint=outputs/pusht_factored_ln_seed0/model.pt \
    synthetic_fallback=false
```

Older one-command path (cov model, 0% planning): `bash scripts/reproduce.sh`.

### B. Planning (6.0% success)

Needs `data/pusht_96.lance`, `outputs/pusht_hires_seed0/model.pt`, and a trained detector.

```bash
export PYTHONPATH=.

# 96x96 data + world model (batch 256 fits ~19 GB VRAM)
python3 scripts/collect_data.py --out data/pusht_96.lance --episodes 2000 \
    --processes 32 --image-shape 96 96 --overwrite
python3 scripts/train.py model=factored_hires data=pusht_96 train.steps=20000 \
    train.batch_size=256 train.warmup_steps=1000 out_dir=outputs/pusht_hires_seed0

# Block detector (~few minutes)
python3 scripts/train_detector.py --dataset data/pusht_96.lance \
    --out outputs/pusht_hires_seed0/detector.pt --img-size 96 --steps 6000

# Full eval (50 episodes, writes manifest JSON)
python3 scripts/evaluate.py checkpoint=outputs/pusht_hires_seed0/model.pt \
    block_detector=outputs/pusht_hires_seed0/detector.pt data=pusht_96 episodes=50
```

Committed copy of the 6% eval: `results/eval/eval_pusht_hires_seed0_mppi.json`.

---

## Results (PushT, seed 0)

**Probe setup:** 2000 episodes, 20k steps, ridge linear probe on block `[x, y, angle]`, **grouped split** (test tail of each sequence; random split inflates R2).

| Metric | BatchNorm head (old) | LayerNorm head (fixed) |
| --- | --- | --- |
| Probe R2, block from `z_world` | 0.35 | **~0.8** |
| Probe R2, agent xy from `z_ego` | 0.94 | **~1.0** |
| Train vs eval encoding gap | ~100% | **0%** |
| Predictor action response / step | 4.8% | **67%** |

BatchNorm made train and eval latents almost orthogonal. The ridge probe still looked OK, but MPC uses raw latents and the predictor barely moved them at eval.

### Planning

| Metric | Value |
| --- | --- |
| Block xy from JEPA latent (ridge) | ~45 px RMSE (precision wall) |
| Block xy from supervised detector | **~8 px**, ~2.4 deg (held-out) |
| Agent xy | exact from `proprio` |
| **Success (MPPI, 50 ep., hires + detector)** | **6.0% (3/50)** |

The latent cannot resolve pushes to ~14 px accuracy. The detector fixes sensing; the JEPA model still scores how actions move the block in latent space. MPC also clamps the agent on the board, routes behind the block, and parks at the end.

---

## Method (short)

- **Encoders:** `WorldViT` on 64x64 RGB, `EgoMLP` on proprio.
- **Predictor (residual):** factored `f(z_w, z_e, a)`, `g(z_e, a)`; monolithic `f(z, a)`.
- **Loss:** prediction MSE + SIGReg + ego term + variance floor + `L_cov` + optional state supervision (`state_aux_weight`).
- **Planning:** `LatentMPCPolicy` for SWM. Planners: CEM, MPPI, Hermite MPPI. Optional block detector for precise pose.
- **Monolith baseline:** `configs/model/monolithic.yaml`, same param budget, single entangled latent.

---

## Layout

```
ewjepa/           model, encoders, predictor, sigreg, detector, planning, mpc_policy
configs/          Hydra configs (train, eval, probe, model/*, data/*)
scripts/          train, evaluate, probe, train_detector, collect_data, reproduce.sh
results/          committed probe/eval JSON + manifest (checkpoints stay in outputs/)
tests/            unit tests
docs/REPORT.md    full report
POSTMORTEM.md     debugging log with measured numbers
```

---

## References

- LeCun, JEPA blueprint (2022)
- Balestriero & LeCun, LeJEPA / SIGReg (2025)
- Maes et al., stable-worldmodel (2026)
- Schramm et al., Hermite MPPI, ICRA 2026

---

## Hardware

WSL2 + RTX 5090 (32 GB), PyTorch 2.12+cu130. Use `python3`. Full PushT: collect ~1 h, train 20k steps ~40 min per model.

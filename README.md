# Ego-World JEPA

Factorized latent world models for planning on [stable-worldmodel](https://github.com/facebookresearch/stable-worldmodel) (PushT).

The model splits observations into two latents:

- **World** (`z_world`): small ViT on pixels, object and scene dynamics
- **Ego** (`z_ego`): MLP on proprioception, robot kinematics

Dynamics run in latent space only (JEPA, no pixel decoding). Planning is latent MPC: roll out actions, minimize distance to the goal world latent. We compare against a **monolithic** LeWM-style baseline with the same parameter budget (~1.5M).

---

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Collect data (WeakPolicy on PushT so the block moves)
python3 scripts/collect_data.py --episodes 2000 --out data/pusht.lance \
    --overwrite --processes 16 --num-envs 2

# Train
python3 scripts/train.py model=factored   data=pusht out_dir=outputs/pusht_factored_seed0
python3 scripts/train.py model=monolithic data=pusht out_dir=outputs/pusht_monolithic_seed0

# Probe frozen latents (block pose)
python3 scripts/probe.py checkpoint=outputs/pusht_factored_seed0/model.pt synthetic_fallback=false

# Evaluate MPC (MPPI by default)
python3 scripts/evaluate.py checkpoint=outputs/pusht_factored_seed0/model.pt episodes=20
python3 scripts/evaluate.py checkpoint=outputs/pusht_factored_seed0/model.pt robustness.enabled=true

# Record rollout videos
python3 scripts/record_video.py checkpoint=outputs/pusht_factored_seed0/model.pt episodes=3

python3 -m pytest tests/ -q
```

Without `data/pusht.lance`, training falls back to a synthetic dataset for smoke tests.

---

## Results (PushT, seed 0)

**Setup:** 2000 offline episodes, **20k training steps**, ~1.59M (factored) / ~1.46M (monolithic) parameters. Linear probe on block pose `[x, y, angle]` from frozen latents (ridge, 80/20 split, real Lance data).

| Metric | Monolithic | Factored |
|--------|------------|----------|
| Probe R² on block pose (`z_world`) | 0.25 | **0.78** |
| Probe R² on block pose (`z_ego`) | — | 0.04 |
| Planning success (MPPI, 20 ep.) | 0% | 0% |

Object pose is readable from `z_world` in the factored model and almost absent from `z_ego`. The monolithic latent mixes both and decodes pose less well at matched capacity.

**Training diagnostics** (factored, step 19k): world std ≈ 0.98, ego std ≈ 1.15, pred loss ≈ 7×10⁻⁴, SIGReg ≈ 0.06. Collapse metrics are logged every 1k steps (`world/std`, `effective_rank`, `sigreg`).

JSON outputs: `outputs/probe/probe_pusht_*_seed0.json`, `outputs/eval/eval_pusht_*_seed0_mppi.json`.

---

## Method (short)

**Encoders:** `z_world = WorldViT(y)` (64×64 RGB, patch 8, dim 192, depth 4); `z_ego = EgoMLP(x)` (32-D).

**Predictor (residual):**
- Factored: `z_world+ = f(z_world, z_ego, a)`, `z_ego+ = g(z_ego, a)`
- Monolithic: `z+ = f(z, a)` with `z = ViT(y) + proj(x)`

**Loss:** prediction MSE + SIGReg on latents (LeJEPA-style Epps-Pulley sketch) + small ego rollout term.

**Planning:** `LatentMPCPolicy` for SWM `World.evaluate`. Planners: CEM, MPPI, Hermite MPPI (Schramm et al., ICRA 2026). Cost = mean L2 distance of rolled-out `z_world` to goal `z_world`.

**Evaluation:** planning success, FoV robustness (`block.color`, `block.shape`, …), linear probing, collapse diagnostics.

---

## Layout

```
ewjepa/           model, encoders, predictor, sigreg, data, planning, mpc_policy, probing
configs/          Hydra (train, eval, probe, model/*, data/*)
scripts/          train, evaluate, probe, collect_data, record_video, plot_results
tests/            shapes, SIGReg, planning, gradient isolation, …
docs/REPORT.md    full technical report
```

---

## References

- LeCun, JEPA blueprint (2022)
- Balestriero & LeCun, LeJEPA / SIGReg (2025)
- Maes et al., stable-worldmodel (2026)
- Schramm et al., Hermite MPPI, ICRA 2026
- Tiofack et al., Guided Flow Policy, ICLR 2026 (planned offline policy step)

---

## Hardware

Tested on WSL2 + RTX 5090 (32 GB), PyTorch 2.12+cu130. Use `python3`. Full PushT runs: collect ~1 h, train 20k steps ~30 min per model on this GPU.

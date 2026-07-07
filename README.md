# Ego-World JEPA

An exploration of **factored (ego/world) latent world models** on [stable-worldmodel](https://github.com/facebookresearch/stable-worldmodel) (PushT). The latents encode the scene (block pose from `z_world`, agent position from `z_ego`), the latent rollout follows the actions, and the world model predicts which way the block will move. Latent-MPC planning still ends at 0% success at the current budget, but the agent now drives to the block and pushes it, instead of drifting off the board as it did before.

> **Name note:** This repo is *not* related to the published [Sub-JEPA paper](https://arxiv.org/abs/2605.09241) (subspace Gaussian regularization). Here “ego-world” refers to factored ego vs world latents with SIGReg/LeJEPA-style training.

The model splits observations into two latents:

- **World** (`z_world`): small ViT on pixels, object and scene dynamics
- **Ego** (`z_ego`): MLP on proprioception, robot kinematics

Dynamics run in latent space only (JEPA, no pixel decoding). To make the latents carry the block and agent positions the planner needs, training adds a small state-supervision head (`state_aux_weight`): a linear head reads the block pose from `z_world` and the agent xy from `z_ego` and is trained on the true state from the dataset. Planning is latent MPC (roll out actions, minimize distance to a goal world latent plus a pose readout cost). We compare against a **monolithic** LeWM-style baseline with the same parameter budget (~1.5M).

---



## Status (2026-07-07, seed 0)


| Component                                                                | Status                                                                   |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| Encoders, factored/monolithic predictors, SIGReg + `L_cov` anti-collapse | Implemented & tested                                                     |
| State supervision (`state_aux_weight`) so `z_world`/`z_ego` encode pose   | Added; trained 20k steps                                                 |
| Linear probing (grouped split), collapse diagnostics, unit tests         | Working                                                                  |
| Probe: `z_world` -> block pose R2 = **0.39**, `z_ego` -> block pose R2 = **0.63** | Measured, see Results                                             |
| Latent rollout follows the actions (agent xy recovered from `z_ego`, R2 ≈ 1.0) | Verified                                                            |
| Latent MPC planning (MPPI)                                               | **0% success**; agent reaches the block and moves it, but not to the goal |
| Multi-seed runs, robustness sweeps                                       | Not done                                                                 |


**Reproduce:** `bash scripts/reproduce.sh` (requires `data/pusht.lance` and a GPU).

---



## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Collect data (WeakPolicy on PushT so the block moves)
python3 scripts/collect_data.py --episodes 2000 --out data/pusht.lance \
    --overwrite --processes 16 --num-envs 2

# Train (factored, with state supervision on by default in the config)
python3 scripts/train.py model=factored data=pusht out_dir=outputs/pusht_factored_stateaux_seed0 train.steps=20000
python3 scripts/train.py model=monolithic data=pusht out_dir=outputs/pusht_monolithic_seed0 train.steps=20000

# Probe frozen latents (block pose); grouped split so neighbouring frames do not leak
python3 scripts/probe.py checkpoint=outputs/pusht_factored_stateaux_seed0/model.pt synthetic_fallback=false

# Evaluate MPC (MPPI by default)
python3 scripts/evaluate.py checkpoint=outputs/pusht_factored_stateaux_seed0/model.pt episodes=20

# One-command reproduction → results/
bash scripts/reproduce.sh

python3 -m pytest tests/ -q
```

Without `data/pusht.lance`, training falls back to a synthetic dataset for smoke tests.

---



## Results (PushT, seed 0)

**Setup:** 2000 offline episodes, **20k training steps**, ~1.59M (factored) parameters. Linear probe on block pose `[x, y, angle]` from frozen latents (ridge, real Lance data). The probe uses a **grouped split**: the test rows are the last part of the sequence, so neighbouring frames (which look almost the same) do not end up on both sides of the split. An earlier version used a random split; because the frames overlap in time this leaked frames across train and test and gave probe R2 numbers that were too high (for example 0.78 / 0.29 with the random split against 0.39 with the grouped split).


| Metric                                         | Factored (state supervision) |
| ---------------------------------------------- | ---------------------------- |
| Probe R2 on block pose (`z_world`)             | **0.39**                     |
| Probe R2 on block pose (`z_ego`)               | 0.63                         |
| Agent xy recovered from `z_ego`                | R2 ≈ **1.0**                 |
| Latent rollout follows the actions             | yes                          |
| World model predicts block push direction      | cosine ≈ 0.75 with truth     |
| Planning success (MPPI, 20 ep.)                | **0%**                       |


**What the state-supervision head changed.** Before adding it, `z_world` did not encode the block (grouped-split R2 near 0) and the latent rollout was the same in every direction, so the MPC cost was flat and the agent drove straight off the board. After adding it, `z_world` reads the block (R2 = 0.39), `z_ego` reads the agent position almost exactly (R2 ≈ 1.0), and a rollout with action `[+1, 0]` moves the decoded agent right while `[-1, 0]` moves it left. In a traced episode the agent now drives to the block and pushes it a few pixels.

**Why planning is still 0%.** The block reaches contact but does not travel to the goal. Two reasons: the planning horizon (8 steps) is short next to the distance the block must cover, and the block-to-goal term in the cost is weaker and noisier than the agent-to-block term, so the agent parks on the block instead of committing to a directed push. The success test is also strict: agent and block both within ~20 px and block angle within 20 degrees. Tuning the planner cost weights and horizon is the next step.

JSON: `results/probe/factored_stateaux_seed0.json`, `results/eval/factored_stateaux_seed0_mppi.json`.

---



## Method (short)

**Encoders:** `z_world = WorldViT(y)` (64×64 RGB, patch 8, dim 192, depth 4); `z_ego = EgoMLP(x)` (32-D).

**Predictor (residual):**

- Factored: `z_world+ = f(z_world, z_ego, a)`, `z_ego+ = g(z_ego, a)`
- Monolithic: `z+ = f(z, a)` with `z = ViT(y) + proj(x)`

**Loss:** prediction MSE + SIGReg on latents (LeJEPA-style Epps-Pulley sketch) + ego rollout term + variance floor + **covariance decorrelation** (`L_cov`, VICReg-style off-diagonal penalty on `z_world`) + **state supervision** (`state_aux_weight`, a linear head that reads the block pose from `z_world` and the agent xy from `z_ego` and matches the true state). The state term is what makes the latents encode the positions the planner reads. Set `state_aux_weight=0` for pure JEPA.

**Planning:** `LatentMPCPolicy` for SWM `World.evaluate`. Planners: CEM, MPPI, Hermite MPPI (Schramm et al., ICRA 2026). The cost rolls the plan forward, reads the block from `z_world` and the agent from `z_ego`, and adds three terms: a small latent distance to the goal image, a block-to-goal pose cost, and an agent-to-block approach cost.

**Evaluation:** planning success, FoV robustness (`block.color`, `block.shape`, and so on), linear probing, collapse diagnostics.

---



## Layout

```
ewjepa/           model, encoders, predictor, sigreg, data, planning, mpc_policy, probing
configs/          Hydra (train, eval, probe, model/*, data/*)
scripts/          train, evaluate, probe, collect_data, record_video, plot_results, reproduce.sh
results/          committed probe/eval JSON + manifest (generated by reproduce.sh)
tests/            shapes, SIGReg, cov_loss, planning, gradient isolation, …
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

Tested on WSL2 + RTX 5090 (32 GB), PyTorch 2.12+cu130. Use `python3`. Full PushT runs: collect ~1 h, train 20k steps ~40 min per model on this GPU.
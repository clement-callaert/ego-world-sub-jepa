# Post-mortem: fixing the world model and chasing 0% planning on PushT

Date: 2026-07-08, updated 2026-07-09 (see section 8). Repo: ego-world-sub-jepa
(factored Ego-World JEPA, latent MPC on PushT).

> **2026-07-09 update:** section 3's "20x too coarse" story was partly wrong for
> on-policy decode, but the JEPA latent still cannot localize the block finely enough
> for PushT (~45 px vs ~14 px needed). Planning reached **6.0%** after a supervised
> detector and MPC fixes. See sections 8 and 9.

This file records what was found, what was fixed, and every mistake made during the
session, so the same errors are not repeated. Numbers here are measured, not guessed.

---

## 1. Goal of the session

Two asks:
1. Make the results verifiable.
2. Improve the planning success rate, which was 0%.

Outcome: the world model was fixed and is verifiable (LayerNorm head, SIGReg fix,
stop-grad target). Planning went from 0% to **6.0% (3/50)** after a supervised block
detector and MPC control fixes (section 9). That number is low but real. See
`results/eval/eval_pusht_hires_seed0_mppi.json`.

---

## 2. What was actually fixed (verified, keep these)

Three real bugs in the world model, all confirmed by measurement:

1. **SIGReg N-scaling bug.** `epps_pulley_1d` multiplied the statistic by `n_samples`,
   which inflated SIGReg by about 4608x. At step 0 that showed as `sigreg=2773` next to
   `pred_loss=2.72`, so SIGReg completely drowned the prediction loss. Fix: remove the
   factor, use a symmetric grid and `torch.trapezoid`.

2. **Collapse to zero.** Needs `stop_grad_target: true` so the model cannot cheat by
   collapsing the detached target latent.

3. **Train/eval head mismatch (the big one for planning).** The world head used
   `nn.BatchNorm1d`. BatchNorm uses batch statistics in train mode and running
   statistics in eval mode, so the encoder produced a *different latent in each mode*.
   Measured: `||z_train - z_eval|| = 447` vs `||z_eval|| = 446`, i.e. about 100%
   different (nearly orthogonal). The predictor was trained on train-mode latents and
   fed eval-mode latents at inference, so at eval it was worse than doing nothing.
   Fix: switch the head to LayerNorm.

Effect of the fixes, measured with a ridge linear probe on encoded latents:

| checkpoint (64px)           | head       | block pose R2 | agent xy R2 | train/eval gap | predictor action response |
|-----------------------------|------------|---------------|-------------|----------------|---------------------------|
| pusht_factored_fixed_seed0  | batchnorm  | 0.35          | 0.94        | ~100%          | 4.8% of a real step       |
| pusht_factored_ln_seed0     | layernorm  | 0.81          | 1.00        | 0.0%           | 67% of a real step        |

Verifiability: `scripts/evaluate.py` now writes a run manifest (git SHA, config, torch
version, seed) into the eval JSON. 42/42 tests pass after all changes.

---

## 3. The unsolved problem, diagnosed precisely

Planning stays at 0% even with the fixed, eval-consistent world model. The cause is
**block localization precision**, and it is now measured, not assumed.

Facts, all in the environment's resolution-independent world units:

- The block moves about **4.5 px per step** (about 36 to 42 px over the 8-step horizon).
- A ridge readout on the **true encoded latent** localizes the block to only about
  **80 to 100 px RMSE** (block xy R2 about 0.65 to 0.75, block std about 180 px).
- So the readout error is about **20x larger than the per-step block motion**.

The MPC pose cost and latent cost are therefore dominated by localization noise, not by
how the actions move the block. No planner budget fixes this; the planner is fine.

To plan single-step pushes you would need block localization near 5 px, which is
R2 about 0.999. The JEPA latent gives about 100 px, R2 about 0.65. That is a ~20x gap.

**Higher resolution did NOT help (tested).** A bigger encoder at 96x96 (embed 256,
depth 6, 144 patches, checkpoint pusht_hires_seed0) trained cleanly (world std 0.89,
effective rank 21) but block xy RMSE was about 103 px, slightly worse than the 64px
model's 81 px. Open-loop rollout of the block was about 95 to 101 px error at every
step, worse than a static "block never moves" baseline. So the bottleneck is the
representation, not the input resolution and not the predictor.

The one lever left that targets this directly: raise `state_aux_weight` a lot
(1.0 -> 20) to force the world latent to encode block xy almost exactly. This makes the
model more of a supervised block detector and less a pure JEPA, so it is a method
choice. This run was started (outputs/pusht_hires_aux20_seed0) but stopped before it
finished, so its result is unknown.

---

## 4. Mistakes made this session, and the lesson for each

These are the errors to not repeat.

1. **Asserted a wrong mechanism instead of measuring it.**
   Claimed "SIGReg is blind to low-rank collapse." A quick experiment disproved it:
   a rank-3 latent gives SIGReg about 330 vs about 0.59 for an isotropic one, so SIGReg
   clearly sees low-rank structure. The real bug was the N-scaling, nothing about
   blindness. **Lesson: test a claim with a 10-line script before writing it as fact.**

2. **Chased a proxy instead of the real quantity.**
   Hypothesized "BatchNorm blows the eval latent up 30x." Measured the eval latent std
   and it was a healthy 0.77, not blown up. The real problem was that train-mode and
   eval-mode *representations* were different (100% apart), which is a different thing
   from scale. **Lesson: measure the exact quantity that matters (here, the train vs
   eval encoding difference), not a convenient proxy (scale).**

3. **Changed a module's parameters without updating the code that touches them.**
   Switched the head LayerNorm to `elementwise_affine=False`, but `_init_weights` still
   called `nn.init.ones_(module.weight)` on it, and a non-affine LayerNorm has
   `weight = None`, so training crashed with `AttributeError: NoneType ... fill_`.
   **Lesson: when you remove or add module parameters, grep for every place that
   inits, saves, or loads those parameters (init functions, state_dict, checkpoint
   loading).**

4. **Forgot that a deep transformer needs LR warmup.**
   The 96px model (depth 6) collapsed to a single direction from a cold start with a
   constant 3e-4 learning rate. Diagnostics showed `world/std=0.000` and centered
   `mean_norm=0.001` (all samples identical). The 64px depth-4 model tolerated no
   warmup; the deeper one did not. Adding a 1000-step linear LR warmup fixed it
   (`world/std` climbed to 0.89). **Lesson: for any deeper or wider transformer, add
   LR warmup before blaming the loss. Also note: non-affine LayerNorm stops the
   collapse-to-zero mode but NOT directional collapse; anti-collapse still relies on
   SIGReg + variance floor + state supervision + warmup.**

5. **Confused which knob reduces GPU memory.**
   VRAM filled (31.5 GB) and spilled into shared GPU memory at 96x96 with batch 512,
   which makes training crawl over PCIe. The instinct to "lower the patch size" is
   backwards: a smaller patch means more tokens and MORE memory. The right knob is a
   smaller batch. Batch 256 fit in about 19 GB. **Lesson: for VRAM, lower the batch
   size (or image size). Smaller patch size increases memory.**

6. **Double-backgrounded a process and got silent failures.**
   Ran `nohup python ... &` inside a tool call that was already set to run in the
   background. The launcher returned, the child was not tracked, and in one case the
   process never really started (empty log, exit code 1). **Lesson: to background a
   long job, use the harness background flag on the bare command. Do not add `nohup`
   or a trailing `&` on top of it.**

7. **Tripped over `set -e` in the shell.**
   The session shell runs with errexit. `pkill -f ...` returns exit code 1 when no
   process matches, which aborted the whole compound command before the real work ran.
   **Lesson: guard commands that can return nonzero with `|| true`, or `set +e` at the
   top of the script, especially `pkill`, `grep`, and `pgrep`.**

8. **Wrong interpreter name.**
   There is no `python` on this machine and no venv; use `python3`. Dependencies are
   installed at user/site level. **Lesson: use `python3` here.**

---

## 5. Experiments run and their results

| run                          | change                                  | result                                   |
|------------------------------|-----------------------------------------|------------------------------------------|
| pusht_factored_fixed_seed0   | SIGReg fix + stop_grad + batchnorm head | probe block R2 0.35, planning 0%         |
| pusht_factored_ln_seed0      | batchnorm -> layernorm head (64px)      | probe block R2 0.81, train/eval gap 0%, planning 0% |
| pusht_hires_seed0            | 96px, embed 256, depth 6, warmup, batch 256 | world stable, block RMSE ~103 px, planning 0% |
| pusht_hires_aux20_seed0      | state_aux_weight 1.0 -> 20 (96px)       | started, not finished (stopped by user)  |

Cost shaping in the MPC (approach cost pulling the agent toward the block, and a
tunable latent cost weight) was tried and did NOT help, because the underlying issue is
representation precision, not the cost design. Those config knobs were reverted to
honest defaults (latent_cost_weight 1.0, approach_weight 0.0).

---

## 6. Files changed this session

- `ewjepa/sigreg.py`: SIGReg N-scale fix, cleaner diagnostics (from earlier work).
- `ewjepa/encoders.py`: added `layernorm` head option, made it non-affine, guarded
  `_init_weights` against None LayerNorm params.
- `ewjepa/model.py`: head-norm docstring; loss wiring comments.
- `scripts/train.py`: added linear LR warmup (`train.warmup_steps`).
- `scripts/evaluate.py`: fits both block and agent readouts; writes run manifest.
- `configs/model/factored.yaml`: `world_head_norm: layernorm`, stop_grad, cov/var.
- `configs/model/factored_hires.yaml`: new 96px bigger encoder.
- `configs/data/pusht_96.yaml`: new 96x96 dataset path.
- `configs/train.yaml`: `warmup_steps: 1000`.
- `configs/eval.yaml`: honest default cost weights.
- `ewjepa/detector.py`, `scripts/train_detector.py`: supervised block pose CNN (Jul 9).
- `ewjepa/mpc_policy.py`: exact agent from proprio, detector block sensor, on-board clamp,
  go-around / push / park phases (Jul 9).

Datasets: `data/pusht.lance` (64x64, 2000 eps), `data/pusht_96.lance` (96x96, 2000 eps,
collected this session).

---

## 7. Where to go next

1. Run the monolithic baseline through the same detector + MPC stack and report planning
   success side by side with the factored model. That is the comparison the README claims.
2. The JEPA latent alone still localizes the block to about 45 px. A stronger encoder or
   more `state_aux_weight` might shrink that gap without a separate detector.
3. 6% is noisy (8 episodes gave 12.5%). Multi-seed eval would make the headline number
   more trustworthy.
4. Do not oversell planner budget or cost shaping alone. The hard part is sensing the
   block precisely and coordinating x, y and angle with a point pusher.

Method notes for any retrain of the bigger model: keep the non-affine LayerNorm head,
keep LR warmup (about 1000 steps), use batch 256 at 96px (about 19 GB), and use
`python3`. Background long jobs with the harness flag only, no `nohup`/`&`.

---

## 8. Session 2026-07-09: from 0% to working planning

Symptom reported: success always 0.0% and the agent drives straight out of the
frame. Every claim below was measured with instrumented rollouts (true state vs
decoded state vs actions logged every step), not guessed.

### 8.1 The section 3 diagnosis was wrong for planning

Section 3 said the block can only be localized to ~100 px, a 20x gap that no
planner can cross. Measured on-policy, that is not what the planner sees:

- The ridge readout that `evaluate.py` fits on ~2048 encoded samples decodes the
  block within about **5 to 20 px** at eval time, and the agent essentially
  exactly (it comes from proprio through the ego MLP).
- The predictor knows how pushes move the block: over dataset windows where the
  block really moved (>8 px in 8 steps), predicted vs true displacement has
  **median direction cosine 0.976** and magnitude 52 vs 55 px. Predicted vs true
  rotation has **86% sign agreement, correlation 0.79**.

The grouped-split probe RMSE (~100 px) mixes distribution shift across episodes
into one number; the on-policy decode error is what matters for MPC and it is an
order of magnitude smaller. The representation was never the blocker.

### 8.2 The real bugs, in order of discovery

1. **The agent can leave the board and the signal dies.** PushT never clamps the
   agent (the clip in `env.step` is commented out in SWM) and the board has no
   walls for the agent or the block. Once the agent is off-screen the image stops
   changing, the encoder output freezes, and the plan can never recover. That is
   the "robot goes straight out of the frame". Fix: bounds costs on the decoded
   agent and block in the planner cost.

2. **The latent cost dominated and misled.** The raw latent distance to the goal
   image is O(1) while the pose costs were divided by 512^2 (O(0.01 to 0.3)), so
   the noisy whole-scene term drove the plan. Fix: plan on decoded positions,
   latent cost off by default.

3. **The model hallucinates block motion off-distribution.** WeakPolicy collects
   data with the agent within 60 px of the block, so whenever the agent is far
   from the block the predictor happily imagines the block moving with the
   actions. A gradient-free planner exploits exactly these errors. Fix: phase
   structure (approach / push / park) that only trusts predicted block motion
   when the agent is engaged with the block.

4. **THE BIG ONE: the pose cost aimed at the wrong goal.** `info["goal_pose"]`
   in SWM PushT is the rendered goal-zone *variation* and stays at its default
   `(256, 256, pi/4)` in **every** episode. The env success check compares
   against `goal_state` (sampled per episode: agent target = `[:2]`, block
   target = `[2:4]`, block angle = `[4]`), which is also what the goal image
   shows. The MPC had been faithfully delivering the block to (256, 256) in
   every episode and success wanted it somewhere else. Fix: block target =
   `goal_state[2:5]`, agent park target = `goal_proprio[:2]`.

5. **Success needs the agent parked too.** `eval_state` checks
   `||goal_state[:4] - state[:4]|| < 20 px`, i.e. agent AND block position
   error together, plus block angle within pi/9. The old cost had no
   agent-to-goal term at all, so even a perfect block placement could never
   terminate an episode. Fix: park phase.

6. **MPPI temperature was 10x too large.** The pose costs live in (px/512)^2
   units, so good candidates differ by ~0.01 while the softmax temperature was
   0.1: all candidates got nearly equal weight and the averaged action was mush.
   Fix: temperature 0.01.

7. **Full-speed pushes fling the block.** The PD controller reaches high speed
   at |action| near 1, and a fast hit moves the block far more than the model
   ever predicts (worst observed: 300 px in a few steps, block knocked off the
   board). Fix: soft cap on |action| at 0.35 near the block, plus a small
   squared action penalty.

### 8.3 The MPC that works (ewjepa/mpc_policy.py)

Three phases picked from the *decoded current state* each step, with hysteresis
so readout noise does not flip the phase every step:

- **approach**: move the agent to a standoff point ~60 px behind the block,
  opposite the goal (a push can only move the block away from the agent), with
  a clearance penalty (~45 px) so the agent walks around the block instead of
  through it. Only the agent readout is trusted here. The clearance radius must
  stay below the standoff and engage distances or the agent hangs at the
  balance point forever (this deadlock happened at clearance 65 > standoff 60).
- **push**: block position cost + always-on wrapped angle cost (1 - cos), both
  on the readout of the rolled-out world latent, with the gentle-action cap.
- **park**: once the decoded block is within 25 px of its goal (readout noise is
  ~8 px, tighter thresholds stop the park phase from ever triggering), move the
  agent to its own goal position while keeping the block cost active.

One measured full success: block delivered to ~6 px of its goal, angle 0.05 rad,
agent parked, `terminated=True` at t=230. Per-episode success is stochastic
(MPPI seed matters), which is why the eval uses 50 episodes.

### 8.4 Mistakes made this session

1. **Used the predicted block as the approach target.** The planner then "brought
   the block to the agent" in imagination instead of moving the agent. Approach
   targets must come from the current frame, held fixed over the plan.
2. **Added a linear progress bonus along the push direction.** It rewarded
   imagined displacement, the planner exploited the model even harder, and every
   metric got worse. Reverted. Lesson: shaping terms that reward predicted
   motion invite model exploitation; penalize distance to real targets instead.
3. **Tightened near_block_thresh below the readout noise.** With ~8 px decode
   error a 20 px park threshold almost never triggers, the agent never parks,
   and the near-miss distance got *worse*. Thresholds must respect sensor noise.
4. **Left the checkpoint format behind.** The affine-LayerNorm checkpoints from
   before the `elementwise_affine=False` change (e.g. pusht_factored_ln_seed0)
   no longer load with the current code. Only pusht_hires_seed0 among the 20k
   checkpoints matches the current code.

---

## 9. Session 2026-07-09 (part 2): a precise block sensor and a working pusher

Symptom reported again: success always 0.0% and the agent drives straight out of
the frame. Everything below was measured with instrumented rollouts, not guessed.

### 9.1 What was measured

- **The block localization wall is real.** From the JEPA latent the block xy
  decodes to about 44 to 67 px RMSE across checkpoints (ln2 67, hires 44.5,
  aux20 51), and a nonlinear MLP readout is no better than the linear one
  (47.4 vs 48.4 px on hires). The success check needs the block within about
  14 px. So the latent cannot carry the block at control precision and no
  readout fixes it. Section 8.1 was too optimistic about on-policy decode.
- **The agent is available exactly.** `proprio` carries the true agent xy and
  `goal_proprio` the agent goal. The old MPC used the noisy `z_ego` decode.
- **The agent leaves the board.** The env drives the agent as a kinematic body
  with relative actions (target = agent + action * 100); walls do not stop a
  kinematic body. Measured the agent reaching x=544, y=553 on a 512 board.

### 9.2 What was changed

- **Supervised block detector** (`ewjepa/detector.py`, `scripts/train_detector.py`).
  A small CNN reads block x, y, angle from the image, trained on the dataset
  state labels. Only on board frames are kept: some frames have the block
  knocked to x or y about -1000 to 1300, which is unlearnable and wrecks the
  target normalization. Held out block xy RMSE **8.3 px**, angle **2.4 deg**
  (episode grouped split). This is the block sensor the MPC reads.
- **Exact agent from proprio** for the phase logic, approach, clearance and
  park, and the exact agent goal from `goal_proprio`.
- **Hard on board action clamp.** Knowing the exact agent and the action scale,
  the applied action is clamped so the agent target stays inside the board. This
  removes the off frame failure entirely (measured agent range now about
  [40, 472]).
- **World model still supplies the block dynamics.** Candidate actions are
  scored by the block displacement the latent predicts, added to the detector's
  precise current block. The readout's large constant per frame bias cancels in
  the difference, so the predicted absolute block is far better than the raw
  readout.
- **Go-around approach.** When the agent is on the goal side of the block it is
  routed to a lateral waypoint (past the clearance radius, slightly behind)
  before the standoff, so it does not shove the block the wrong way.
- **Guided gentle push.** The agent is pulled toward a point just past the block
  toward the goal, with an action cap that shrinks as the block nears its goal,
  so pushes stay gentle and do not overshoot.
- **Park latching.** Once the block is placed (within `near_block_thresh` px and
  `park_angle_thresh` rad) the plan latches park and moves the agent to its own
  goal without re-mashing the block.

### 9.3 Mistakes made this session (do not repeat)

1. **Park hysteresis on position alone froze the block.** Parking at
   25 * 1.8 = 45 px and latching abandoned the block at about 41 px forever: the
   agent left for its own goal and stopped refining. Measured: block frozen at
   41.4 px for 750 steps. Park must require position AND angle, and be tight.
2. **Applied the gentle action cap in the approach phase too.** That made the
   go-around too weak to ever get behind the block, and the block froze near its
   start. The cap belongs only in the push and park phases; the approach must
   stay fast.
3. **Pushed too far past the block center** (`push_through` = 20). The agent
   overshot to the goal side of the block, lost contact, and the re-approach
   knocked the block away, a limit cycle at about 88 px. Aiming near the block
   center keeps the agent trailing behind it.
4. **Mashed a placed block.** Near the goal the push direction
   (goal - block) / ||.|| is ill defined, so a fresh push only knocks the block
   away. Latching park once the block is placed avoids this.
5. **Under-weighted the block angle.** With the angle term at 0.2 to 0.5 the
   block reached its goal position and its goal angle at different steps but
   never together, so the park phase never triggered and success stayed at 0.
   Raising the angle weight to 1.5 made the rotation get corrected during the
   push, the park phase started triggering, and episodes began to succeed.

### 9.4 Result

Measured on 50 episodes with `pusht_hires_seed0` and the trained detector:
**success_rate = 6.0% (3/50)**, up from a hard 0%. On an 8 episode subset it was
12.5%; per-episode success is stochastic (MPPI seed and how far the block and
agent goals fall from the block), which is why the headline number uses 50
episodes. JSON with the run manifest:
`outputs/eval/eval_pusht_hires_seed0_mppi.json`.

What each fix bought, measured:
- Detector: block xy sensing from about 45 px (latent) to about 8 px.
- On board clamp: the agent no longer leaves the frame (range about [40, 472]).
- Go-around plus gentle push: the block is driven from about 230 px to within a
  few px of its goal instead of being shoved the wrong way or jammed at a wall.
- Angle weight plus park latching: position and angle reach tolerance together
  often enough for the agent to park and the episode to terminate.

Honest limits: 6% is low. The remaining failures are the genuine hard core of
PushT: a point pusher must control block x, y and angle at once, and the greedy
one step replanning still leaves a limit cycle on many episodes where the block
reaches its goal transiently but is knocked out of tolerance before the angle
also lands. A stronger result would need coordinated multi push planning (or a
learned policy), not more sensor precision.

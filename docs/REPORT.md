# Ego-World JEPA: Detailed Technical Report

**Date:** 2026-07-09 (updated after planning fix)
**Author:** Clément Callaert
**Scope:** A factorized latent world model (world dynamics vs ego kinematics) trained as a JEPA, compared against a monolithic latent world model of matched capacity, on the PushT manipulation task from `stable-worldmodel`. This report explains the method, the four papers it builds on, the training objective and why it does not collapse, the planning problem, and the measured results.

---

## 0. The problem this project attacks

### 0.1 The question

Modern generalist robot policies are usually trained by **behavior cloning**: you collect large amounts of teleoperation data and you learn a single neural network that maps raw camera pixels directly to robot actions. This is the *monolithic pixel to action* paradigm. It works well when the test situation looks like the training data, but it has one structural weakness. A single latent vector is forced to represent, all at once and in an entangled way:

1. the **scene semantics** (what objects are present),
2. the **camera viewpoint**,
3. the **environment dynamics** (how objects move and collide),
4. the **robot morphology** (how *this specific body* moves).

Because these four things live inside one undifferentiated vector, the policy becomes **brittle**. If you change the robot body (embodiment shift), move the camera (viewpoint shift), recolor the background (visual out of distribution shift), or ask for a long action sequence (compounding prediction errors), performance collapses. This is exactly the motivation stated in the Inria WILLOW PhD offer 2026-10180, *Structured Ego-World Models for Scaling Dexterous Robot Learning* (team WILLOW, supervisor Cordelia Schmid).

The central research question is therefore:

> **Can we make robot world models more robust by giving them a structural prior that separates "how the world behaves" from "how my body behaves", instead of entangling both in one latent space?**



### 0.2 The proposed answer

This project builds a small, fully testable proof of concept of one of the two structural changes the offer asks for, namely **ego-world factorization**. We split the latent state into two streams:

- a **world stream** `z_world`, which should capture object and scene dynamics (persistent, embodiment agnostic),
- an **ego stream** `z_ego`, which should capture the robot's own kinematics (the proprioceptive state of the body).

We train the model as a **Joint-Embedding Predictive Architecture (JEPA)**, meaning the model predicts the *future latent state* rather than reconstructing future pixels. We regularize the latent space with **SIGReg** (from LeJEPA) so it does not collapse, and we plan actions by **Model Predictive Control (MPC)** directly inside the learned latent space. The scientific claim we want to be able to test is: *factorizing the latent state improves planning robustness under distribution shift, at matched parameter count, compared to a monolithic latent model.*

We deliberately keep everything small (about 1.5 million parameters) so the whole pipeline (data collection, training, planning, probing) runs on a single desktop GPU in a few hours. The task is **PushT** (Section 6), a 2D pushing task that is simple to render but genuinely hard to plan in.

---



## 1. Background and definitions

Before the method, here are the terms used throughout, defined simply.

- **Observation.** What the agent sees at time `t`. Here it is an image `y_t` (64×64 RGB pixels) plus a proprioceptive vector `x_t` (the agent's own low dimensional state) plus an action `a_t`.
- **Latent (or embedding).** A vector `z = f(observation)` produced by an encoder network `f`. It is a compressed code of the observation. "Latent space" is the space these vectors live in.
- **World model.** A model that predicts how the state of the world evolves when actions are applied, so that an agent can *imagine* the consequences of actions without acting in the real environment. Formally it learns a transition `z_{t+1} = P(z_t, a_t)`.
- **JEPA (Joint-Embedding Predictive Architecture).** A world model that makes its predictions *in latent space*: it predicts the embedding of the future, `ẑ_{t+1}`, and compares it to the embedding of the true future, `z_{t+1} = f(observation_{t+1})`. It never decodes pixels. This is the opposite of generative world models (like DreamerV3) that reconstruct future images.
- **Proprioception.** The robot's internal sense of its own body configuration (joint angles, end effector position, velocities). In PushT this is the 4D agent state.
- **Representation collapse.** The failure mode where the encoder maps every input to (nearly) the same vector (*complete collapse*), or to a low dimensional subspace (*dimensional collapse*). A collapsed encoder makes the prediction loss trivially zero (predicting a constant is easy) but the representation is useless.
- **Distribution shift.** The test conditions differ from the training conditions (new colors, new shapes, new viewpoints, new bodies).
- **Linear probe.** A simple linear (or ridge) regression fitted on top of a **frozen** encoder to predict some quantity of interest (here the object pose). If a linear probe reads the quantity well, then the frozen representation encodes it *linearly*, which is the cleanest evidence that the information is present and well organized.
- **R² (coefficient of determination).** A probe quality score. `R² = 1` means the probe explains all the variance of the target; `R² = 0` means it does no better than predicting the mean; negative means worse than the mean.
- **NRMSE.** Root mean squared error of the probe, divided by the standard deviation of the target. A scale free error, roughly in `[0, 1]` for a useful probe.

---



## 2. The four papers this project builds on

This project does not invent every component from scratch. It integrates four recent scientific ideas. This section states clearly, for each paper, **what it contributes**, and then Section 3 states clearly **what is reused as is** versus **what is my own contribution**.

### 2.1 LeJEPA (Balestriero and LeCun, 2025): how to stop a JEPA from collapsing

**The contribution.** LeJEPA gives a principled, heuristics free recipe for training a JEPA. It answers two questions.

**Question 1: what distribution should the embeddings follow?** The paper proves (for both linear probes and nonlinear probes) that the **isotropic Gaussian** `N(0, I)` is the *unique optimal* distribution for the embeddings, in the sense that it minimizes the worst case downstream prediction risk when you do not know the downstream task in advance. The intuition: an isotropic Gaussian spreads information equally in all directions (all covariance eigenvalues equal), so no direction is starved of variance. An *anisotropic* distribution (some directions with much more variance than others) provably increases both the bias and the variance of any downstream linear probe. So "good" embeddings should be zero mean, unit variance per direction, and Gaussian.

**Question 2: how do you enforce that distribution efficiently?** The paper introduces **SIGReg (Sketched Isotropic Gaussian Regularization)**. Matching a high dimensional distribution directly is expensive and unstable. SIGReg avoids this with a *sketching* (random projection) trick justified by the **Cramér-Wold theorem**: a multivariate distribution is standard Gaussian if and only if *all of its 1D projections* are standard Gaussian. So instead of a hard `K`-dimensional test, SIGReg:

1. draws many random unit directions `a` on the sphere,
2. projects the batch of embeddings onto each direction, `u = ⟨a, z⟩`,
3. checks whether each set of 1D projections looks like a standard normal, using the **Epps-Pulley test**, which compares **characteristic functions** (the Fourier transform of a density). For projections `u_1, ..., u_N`, the empirical characteristic function is
  `φ_N(t) = (1/N) Σ_n exp(i·t·u_n)`,
   and the target (standard normal) characteristic function is `φ_0(t) = exp(-t²/2)`. The per direction statistic is the weighted squared distance
   `∫ |φ_N(t) - φ_0(t)|² w(t) dt`,
   and SIGReg averages this over all sampled directions.

The Epps-Pulley choice is important: the paper proves (Theorem 4) that its gradient and curvature are **uniformly bounded** regardless of the input distribution, which is what makes training stable. Moment based tests (matching skewness, kurtosis) have gradients that explode, and CDF based tests need sorting which breaks parallelism. Characteristic functions are differentiable, cheap, and bounded.

The final **LeJEPA loss** is just two terms with a single trade off hyperparameter `λ`:

`L_LeJEPA = (1 - λ)·L_pred + λ·SIGReg`,

with no stop gradient, no teacher-student network, no exponential moving average (EMA), no schedulers. This simplicity is the whole point.

**Why it matters here.** SIGReg is exactly what keeps our factorized latent space from collapsing without any of the heavy machinery normally used in self supervised learning. We implement the sketched Epps-Pulley form in `ewjepa/sigreg.py`.

### 2.2 stable-worldmodel (Maes, Le Lidec, et al., 2026): the experimental substrate

**The contribution.** `stable-worldmodel` (SWM) is a modular, tested, documented library for world model research. Its value is *standardization and reproducibility*: it provides a uniform `World` interface, standardized environments, data collection tools, planning solvers (CEM, MPPI, gradient based), and baseline world models, so that different methods can be compared fairly.

Two features are central to this project.

1. **Goal conditioned evaluation.** SWM measures **success rate**: the fraction of episodes where the agent drives the environment into a specified goal configuration. The policy is a plain Python object with a `get_action(infos)` method; SWM queries it each step. We implement our latent MPC policy against this exact interface (`ewjepa/mpc_policy.py`).
2. **Factors of Variation (FoV).** Each SWM environment exposes controllable knobs (color, shape, size, friction, viewpoint, and so on). For PushT there are 16 such factors (block color, block shape, agent color, background color, and more). This is precisely the tool we need to *measure robustness under distribution shift*: you train under default settings, then evaluate while varying one factor at a time, and you report the drop in success rate.

The SWM paper itself demonstrates this by stress testing DINO-WM: it succeeds 94% on in distribution expert states but drops to 12% on states from a random policy, and stays low (4% to 20%) under FoV shifts. This is the concrete evidence of the "monolithic models are brittle" claim from Section 0.

**Why it matters here.** SWM is the environment, the data format (Lance tables), the PushT task, the CEM/MPPI baseline solvers, and the evaluation harness. It is the ground we stand on.

### 2.3 Reference-Free Sampling-Based MPC (Schramm et al., ICRA 2026): the Hermite spline planner

**The contribution.** This paper improves **MPPI (Model Predictive Path Integral)**, a sampling based planner. Standard MPPI keeps a nominal action sequence, perturbs it with Gaussian noise into many candidate sequences, rolls each one out, scores it, and updates the nominal sequence by an importance weighted average of the candidates. The paper's key idea is to *not* sample raw per step actions. Instead it samples a small number of **cubic Hermite spline** control points, where each control point has a **position** `θ^q_k` and a **velocity** `θ^v_k`. A dense action trajectory is then reconstructed from these control points using the cubic Hermite interpolant. On one segment with normalized local time `s = (t - t_k)/Δt ∈ [0, 1]`:

`q(t) = h00(s)·θ^q_k + h10(s)·Δt·θ^v_k + h01(s)·θ^q_{k+1} + h11(s)·Δt·θ^v_{k+1}`,

with the standard basis functions

`h00(s) = 2s³ - 3s² + 1`, `h10(s) = s³ - 2s² + s`, `h01(s) = -2s³ + 3s²`, `h11(s) = s³ - s²`.

Three benefits follow. First, splines produce **smooth, dynamically consistent** trajectories, because you control velocity directly, not only position. Second, this drastically shrinks the search space (a handful of control points instead of one variable per time step), which is why the method runs in real time on a CPU with as few as 30 to 70 sampled trajectories, with no GPU. Third, a cheap **bound preserving velocity clamp** keeps the interpolation inside the action limits:

`|θ^v_k| ≤ min(q_max - θ^q_k, θ^q_k - q_min) / (Δt/2)`.

The paper also borrows a **diffusion inspired noise annealing** schedule (from DIAL-MPC): the sampling variance shrinks over optimization iterations and is larger for control points far in the horizon (which get refined many times) and smaller near execution (which must be stable):

`Σ^i_{θ_k} = exp( -(I-i)/(β1·I) - (K-k)/(β2·K) )·I`.

**Why it matters here.** This gives a smarter planner than plain MPPI for the latent MPC loop. I implement it as an optional planner, `HermiteMPPIPlanner` in `ewjepa/planning.py`, following the paper's equations exactly (basis functions, velocity clamp, two factor annealing).

### 2.4 Guided Flow Policy (Tiofack et al., ICLR 2026): offline policy extraction (future work)

**The contribution.** GFP is an **offline reinforcement learning** method that extracts a good policy from a fixed dataset without further environment interaction. The core difficulty of offline RL is **distribution shift on actions**: if the learned policy proposes actions outside the dataset, the critic's value estimates for those actions are unreliable and tend to be overestimated. The classic fix (BRAC family) is a behavior cloning term that keeps the policy near the dataset, but that clones *all* dataset actions indiscriminately, including bad ones.

GFP's idea is **value aware behavior cloning (VaBC)** through a bidirectional guidance loop between three parts:

- a **critic** `Q_φ` trained with the standard Bellman loss,
- a **one step actor** `π_θ` that maximizes the critic while being distilled toward the flow policy,
- a **flow matching policy** `π_ω` (VaBC), which learns a velocity field that transports Gaussian noise into dataset actions, but **weighted** so it preferentially clones high value actions.

The weighting is a soft-max between the dataset action value and the actor's proposed action value:

`g_η(s, a) = exp((λ/η)·Q_φ(s, a)) / [ exp((λ/η)·Q_φ(s, a)) + exp((λ/η)·Q_φ(s, μ_θ(s, z))) ]`,

where `η` is a temperature. If the dataset action is better than the actor's proposal, `g_η > 0.5` and it gets cloned more; otherwise it is downweighted. The VaBC flow loss and the actor loss are

`L_VaBC(ω) = E[ g_η(s, a)·‖v_ω(t, s, a_t) - (a - ϵ)‖² ]`,   with `a_t = (1-t)ϵ + t·a`,
`L_A(θ) = E[ -λ·Q_φ(s, μ_θ(s, z)) + α·‖μ_θ(s, z) - μ_ω(s, z)‖² ]`.

**Why it matters here.** GFP is the intended *next step* for turning our world model into a fast, single step control policy learned from the same offline dataset, instead of running an expensive MPC search at every control step. **In the current codebase GFP is not implemented; it is documented as future work only.** I include it because it is part of the intended full system (Var-JEPA-GFP) and because the report should be honest about what exists and what is planned.

---



## 3. What existed before, and what is my contribution

Being explicit about credit is important.

**What existed before (reused as is, or as direct inspiration):**

- The **JEPA** idea of predicting in latent space (LeCun).
- **SIGReg / the isotropic Gaussian principle** (LeJEPA). I reimplement the sketched Epps-Pulley estimator, I did not invent it.
- **LeWorldModel (LeWM)** as the baseline *style*: a compact world model (about 15M parameters in the original) trained stably with only a next embedding prediction loss plus a Gaussian regularizer, without pixel reconstruction. Our monolithic baseline is a small LeWM style model.
- **stable-worldmodel**: the PushT environment, the Lance data format, the factors of variation, the CEM and MPPI solvers, and the goal conditioned evaluation harness.
- **The Hermite spline MPPI equations** (Schramm et al.): basis functions, velocity clamp, noise annealing.
- **GFP** (Tiofack et al.): the offline policy extraction algorithm (not yet implemented here).

**What is my contribution (the actual code and design in this repository):**

1. **The ego-world factorization itself.** Two separate encoders (a small ViT over pixels for `z_world`, an MLP over proprioception for `z_ego`) and a **factorized predictor** whose structure encodes the causal prior "the ego moves on its own, the world is moved by the ego". This is the scientific hypothesis being tested.
2. **The factorized latent dynamics** with residual heads:
  `z_world_{t+1} = z_world_t + f_world(z_world_t, z_ego_t, a_t)` and `z_ego_{t+1} = z_ego_t + f_ego(z_ego_t, a_t)`.
3. **Per stream SIGReg** on `z_world` and `z_ego`, plus practical anti collapse engineering: non-affine **LayerNorm** on the ViT head (train/eval match; BatchNorm broke the predictor at eval), optional variance floor, covariance decorrelation (`L_cov`), and optional **state supervision** so `z_world` encodes block pose and `z_ego` agent xy. See Section 5.
4. **A matched capacity monolithic baseline** so any difference is attributable to factorization and not to parameter count (factored 1.59M vs monolithic 1.46M parameters).
5. **The latent MPC policy wired into SWM** (`LatentMPCPolicy.get_action`), including image and proprioception preprocessing, warm starting of the plan, and a planning cost defined purely in world latent space.
6. **A concrete implementation of the Hermite spline MPPI planner** integrated into the same planning interface as CEM and MPPI.
7. **The evaluation apparatus**: linear probing to test factorization, factors of variation robustness sweeps, and anti collapse diagnostics (per dimension standard deviation, effective rank, SIGReg value).

In one sentence: the *ingredients* are from the literature; the *recipe* (a small, matched, factorized ego-world JEPA that plans in latent space on PushT and is instrumented to measure factorization and robustness) is the contribution.

---



## 4. Architecture: factored versus monolithic

This section explains precisely what "factored" and "monolithic" mean in the code.

### 4.1 The monolithic model (baseline, LeWM style)

There is **one entangled latent**. The image goes through a small Vision Transformer, and the proprioception is linearly projected and *added into the same latent*:

`z = WorldViT(y) + proj(x)`   (dimension 192).

The dynamics predictor sees only this single latent and the action:

`ẑ_{t+1} = z_t + f(z_t, a_t)`.

Everything (object pose, agent pose, dynamics) shares one 192D vector. This is the "monolithic pixel to action" style world model that the offer criticizes, reduced to a small controlled baseline.

### 4.2 The factored model (ours)

There are **two separate latents**, produced by two separate encoders:

`z_world = WorldViT(y)`   (dimension 192, object and scene),
`z_ego  = EgoMLP(x)`      (dimension 32, robot kinematics).

The dynamics predictor is **structured** to reflect a causal prior:

`z_ego_{t+1}   = z_ego_t   + f_ego(z_ego_t, a_t)`               (the body moves on its own),
`z_world_{t+1} = z_world_t + f_world(z_world_t, z_ego_t, a_t)`   (the world is moved through the body).

Read the structure carefully. The **ego head** depends only on the ego latent and the action, not on the world. This says: *how my body moves depends on my body and my commands, not on where the block is.* The **world head** depends on the world latent, the ego latent, and the action. This says: *how the object moves depends on the object, on where my body is, and on what I command.* This asymmetry is the factorization prior. In the monolithic model there is no such structure: `f` just mixes everything.

### 4.3 Component sizes


| Component     | Factored (ours)                                                 | Monolithic (baseline)             |
| ------------- | --------------------------------------------------------------- | --------------------------------- |
| World encoder | `WorldViT(pixels)` to 192D (patch 8, dim 192, depth 4, 6 heads) | same `WorldViT` to 192D           |
| Ego encoder   | `EgoMLP(proprio)` to 32D (2 layers, width 128)                  | `Linear(proprio)` added into 192D |
| Predictor     | `f_world(z_w, z_e, a)` and `f_ego(z_e, a)`, residual            | `f(z, a)`, residual               |
| Params        | 1.59M                                                           | 1.46M                             |


The two encoders being separate is *what makes the latent factorizable*. The residual form (predict `Δz`, not the absolute next latent) keeps the dynamics close to the identity at initialization, which stabilizes training.

---



## 5. The training objective, and why it does not collapse



### 5.1 The objective

The model is trained on short temporal windows of length `T` (window `T = 9`, so the open loop rollout horizon is `T - 1 = 8`, matching the planner horizon). For a window we:

1. encode every frame: `z_world_all[t] = WorldViT(y_t)`, `z_ego_all[t] = EgoMLP(x_t)`;
2. roll the predictor **open loop** from the first frame, applying the recorded actions, to obtain predicted future latents `ẑ_world[t]`, `ẑ_ego[t]`;
3. compare predictions to the encoded true futures, and regularize all encoded latents with SIGReg.

The total loss follows LeJEPA's convex mix plus small auxiliary terms:

`L = (1 − λ)·L_pred + λ·(SIGReg(z_world) + 0.5·SIGReg(z_ego)) + λ_ego·L_ego + λ_var·L_var + λ_cov·L_cov`

with the terms:

- **Prediction loss (the JEPA term):**
`L_pred = MSE(ẑ_world[1:], z_world_all[1:])`.
This is the heart of the world model: the predicted future world latent must match the encoded future world latent. There is **no pixel reconstruction**.
- **Ego consistency (auxiliary):**
`L_ego = MSE(ẑ_ego[1:], z_ego_all[1:])`, weighted by `λ_ego = 0.1`.
This keeps the ego rollout accurate over multiple steps, which matters when the planner propagates `z_ego` forward. It is off in the monolithic model.
- **SIGReg (anti collapse):** the LeJEPA Epps-Pulley sketched test (Algorithm 1), applied per stream. The world stream gets the full weight; the ego stream gets half weight (`0.5`). Mix weight `λ = sigreg_mix` (default 0.1). Embeddings are batch centered before the test.
- **Variance floor (extra safety):**
`L_var = mean_d ReLU(σ_target - σ_d)` where `σ_d` is the per dimension standard deviation of `z_world` in the batch and `σ_target = 0.5`, weighted by `λ_var = 0.5`. This guards per dimension standard deviation only; it does not enforce full rank.
- **Covariance decorrelation (VICReg style, anti low rank collapse):**
`L_cov = (1/D) Σ_{i≠j} Cov(z_world)_{ij}²` where `Cov` is the batch covariance of `z_world` and `D` is the latent dimension, weighted by `λ_cov = cov_weight` (default `0.25` in the factored config). SIGReg tests random 1D projections and can miss **correlated low rank collapse**: a rank-`r` latent with unit per dimension standard deviation still yields nearly standard normal projections along random directions. The off diagonal covariance penalty directly discourages dimensions from co varying in a low dimensional subspace and keeps `effective_rank` healthy for latent MPC.
- **State supervision (makes the latents encode the positions the planner reads):**
`L_aux = MSE(head_block(z_world), block_pose) + MSE(head_agent(z_ego), agent_xy)`, weighted by `λ_aux = state_aux_weight` (default `1.0` in the factored config). The block pose is `state[:, 2:5]` and the agent xy is `state[:, 0:2]`; both targets are standardized per batch so each column has a fair weight. The heads are linear and the gradient flows into the encoders, so `z_world` must encode the block and `z_ego` must encode the agent. Without this term SIGReg and `L_cov` keep the latents spread out and full rank but do not force them to carry the block position, which is exactly what the planner needs. The heads are only used during training; the planner fits its own readouts at eval time. Set `state_aux_weight=0` to recover pure JEPA.

Following LeJEPA, we do **not** use an EMA teacher or a schedule. The factored config sets `stop_grad_target: true` so the model cannot collapse the detached target latent (measured fix during debugging). Monolithic keeps `stop_grad_target: false`.

### 5.2 Why it does not collapse: the intuition and the mechanism

The danger with any JEPA is the **shortcut solution**: the cheapest way to make `L_pred` zero is to make the encoder output a constant. If `z_world = c` for every image, then predicting `ẑ_world = z_world` is trivially perfect and the prediction loss is zero, but the representation carries no information. This is representation collapse. Four mechanisms prevent it here.

1. **SIGReg makes collapse expensive.** SIGReg uses the Epps-Pulley test on random 1D projections (LeJEPA Algorithm 1). A collapsed embedding fails the test. The prediction loss wants constancy; SIGReg wants spread.
2. **LayerNorm head, not BatchNorm.** The factored config uses non-affine LayerNorm on the ViT projector (`world_head_norm: layernorm`). BatchNorm uses different statistics in train vs eval and made train/eval latents ~100% apart, so the predictor was useless at planning time (4.8% action response vs 67% after the fix).
3. **The residual predictor** outputs `Δz` with update `z_{t+1} = z_t + Δz`, so near-identity is a good default and collapse is not the easiest shortcut.
4. **Variance floor** penalizes world dimensions with std below 0.5.
5. **`L_cov`** penalizes off-diagonal batch covariance of `z_world` and keeps `effective_rank` healthy (early runs had SIGReg OK but rank ~3 without it).

---



## 6. Planning and evaluation



### 6.1 The planners

All planners take a cost function that maps a batch of candidate action sequences `(N, H, A)` to per candidate costs `(N,)`, and return the optimized first action (receding horizon MPC).

- **CEM (Cross-Entropy Method):** sample candidates from a Gaussian, keep the best `n_elite`, refit the Gaussian to the elites, repeat.
- **MPPI (Model Predictive Path Integral):** sample candidates around the nominal, then update the nominal by a **soft-max weighted average** of all candidates,
`w_i = softmax( -(S_i - min_j S_j) / temperature )`,   `u ← Σ_i w_i · cand_i`,
where `S_i` is the cost of candidate `i`. Low cost candidates get more weight.
- **Hermite MPPI:** the Schramm et al. planner of Section 2.3, sampling spline control points (position and velocity) instead of raw actions, with the velocity clamp and the two factor noise annealing.



### 6.2 The planning cost (current stack)

The planner rolls each candidate action sequence and scores pose costs in pixel space:

1. **Block to goal:** block pose from the **supervised detector** (about 8 px RMSE) compared to `goal_state[2:5]`.
2. **Agent approach / push / park:** agent xy read exactly from **proprio** (not from `z_ego` decode). Phases: go around the block, gentle push toward goal, latch park when block position and angle are close.
3. **Dynamics term:** block displacement predicted by rolling the **JEPA world model** in latent space, added to the detector's current block. This uses the learned push dynamics without trusting the latent for absolute block position.
4. **On-board clamp:** actions are clipped so the agent (kinematic, unclamped in the env) stays inside the 512 px board.
5. **Latent goal distance:** optional, weight 0 by default (too noisy for this task).

Earlier versions used only ridge readouts on `z_world`/`z_ego` and wrong goal keys; those bugs are documented in `POSTMORTEM.md`.

### 6.3 The evaluation protocol

Four measurements, in increasing order of ambition.

1. **Planning success rate** (SWM `World.evaluate`): the headline task metric.
2. **Robustness under factors of variation:** rerun evaluation while varying block color, block shape, agent color, background color, and report the drop in success rate. This is the distribution shift test.
3. **Linear probing:** freeze the encoder, fit a ridge regression from the latent to the true block pose `[x, y, angle]` (`state[:, 2:5]`), report R2 and NRMSE on a held out split. The split is grouped (the test rows are the last part of the sequence) so that overlapping neighbouring frames do not leak across train and test. This tests *whether the representation encodes the block*, independently of whether planning succeeds.
4. **Anti collapse diagnostics** over training (Section 5.3).

## 7. Implementation status and measured results


| Plan item | Status |
| --- | --- |
| WorldViT + EgoMLP + factorized predictor | done |
| SIGReg (Epps-Pulley) + Hydra training | done |
| Covariance decorrelation (`L_cov`) | done |
| Monolithic LeWM-style baseline | done, probed (`results/probe/monolithic_seed0.json`) |
| Factored model + LayerNorm head | done |
| Latent MPC (CEM / MPPI / Hermite) in SWM | done |
| State supervision (`state_aux_weight`) | done |
| Supervised block detector for control | done |
| Linear probing (grouped split) | done |
| Planning success > 0% | **6.0% (3/50)** on factored hires + detector |
| Monolithic planning on same detector stack | not run |
| FoV robustness sweeps, multi-seed | not done |
| GFP offline policy | future work |

### 7.1 Representation (Jul 8 fix, 64px, seed 0)

LayerNorm head vs old BatchNorm head (same training budget, measured):

| Metric | BatchNorm | LayerNorm |
| --- | --- | --- |
| Probe R2, block from `z_world` | 0.35 | **~0.8** |
| Probe R2, agent xy from `z_ego` | 0.94 | **~1.0** |
| Train vs eval latent gap | ~100% | **0%** |
| Predictor action response / step | 4.8% | **67%** |

### 7.2 Monolithic vs factored (probe only, Jul 6 cov run)

From committed `results/probe/` (grouped split, 8192 latent rows). **Different config** than the LayerNorm run above (`world_head_norm: none`, no LayerNorm fix). Planning was 0% for both.

| Metric | Factored (`factored_cov_seed0`) | Monolithic (`monolithic_seed0`) |
| --- | --- | --- |
| Probe R2, block pose from world latent | 0.29 | **0.78** |
| Planning success (MPPI, 20 ep.) | 0% | not evaluated |

So on this older probe, the monolithic latent actually encoded block pose better. We have **not** yet compared planning with the detector stack. The scientific claim (factorization helps planning) remains **open**, not proven.

### 7.3 Planning (Jul 9, seed 0)

After the precision wall was measured (~45 px from JEPA latent vs ~14 px needed for success), we added a small supervised CNN block detector (~8 px) and fixed MPC control (exact agent from proprio, board clamp, go-around / push / park).

| Metric | Value |
| --- | --- |
| Detector block xy RMSE (held-out) | **8.3 px**, angle **2.4 deg** |
| Planning success, MPPI, 50 episodes | **6.0% (3/50)** |
| Artifact | `results/eval/eval_pusht_hires_seed0_mppi.json` |

**Honest reading:** 0% to 6% shows the pipeline is not broken. 6% is **not** comparable to DINO-WM's ~94% in-distribution or strong baselines in the literature: we use a tiny ViT trained from scratch, WeakPolicy data, heuristic MPC phases, and a **supervised** block sensor. LeJEPA and Hermite MPPI inspired parts of the code but this number does not extend those papers' benchmarks.

Details and mistakes: `POSTMORTEM.md` sections 8 and 9.

---

## 8. Reproducibility

**Representation (probe ~0.8):**

```bash
export PYTHONPATH=.
python3 scripts/collect_data.py --out data/pusht.lance --episodes 2000 \
    --processes 16 --num-envs 2 --image-shape 64 64 --overwrite
python3 scripts/train.py model=factored data=pusht train.steps=20000 \
    out_dir=outputs/pusht_factored_ln_seed0
python3 scripts/probe.py checkpoint=outputs/pusht_factored_ln_seed0/model.pt \
    synthetic_fallback=false
python3 scripts/train.py model=monolithic data=pusht train.steps=20000 \
    out_dir=outputs/pusht_monolithic_seed0
python3 scripts/probe.py checkpoint=outputs/pusht_monolithic_seed0/model.pt \
    synthetic_fallback=false
```

**Planning (6%):** see README section "Reproduce results B". Checkpoints and detector weights live under `outputs/` (gitignored). Committed JSON summaries live in `results/`.

`bash scripts/reproduce.sh` still runs an older cov/stateaux pipeline (useful smoke test, 0% planning on that path).

After local runs: `python3 scripts/copy_results.py` refreshes probe/eval copies into `results/` (does not yet copy the hires detector eval by default).

---



## 9. References

1. LeCun, Y. *A Path Towards Autonomous Machine Intelligence* (JEPA blueprint), 2022.
2. Balestriero, R. and LeCun, Y. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, 2025 (SIGReg, isotropic Gaussian optimality).
3. Maes, L., Le Lidec, Q., et al. *stable-worldmodel-V1: Reproducible World Modeling Research and Evaluation*, 2026 (SWM, PushT, factors of variation, CEM/MPPI, evaluation).
4. Schramm, F., Fabre, P., Perrin-Gilbert, N., Carpentier, J. *Reference-Free Sampling-Based Model Predictive Control*, ICRA 2026 (cubic Hermite spline MPPI, noise annealing).
5. Tiofack, F. N., Le Hellard, T., Schramm, F., Perrin-Gilbert, N., Carpentier, J. *Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning*, ICLR 2026 (VaBC, future work).
6. Inria WILLOW offer 2026-10180, *Structured Ego-World Models for Scaling Dexterous Robot Learning* (Cordelia Schmid), the strategic motivation.

---

*All quantitative results in this report come from scripts in this repository. No external benchmark tables are cited as our results.*
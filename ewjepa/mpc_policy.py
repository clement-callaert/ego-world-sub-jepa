"""MPC policy for stable-worldmodel World.evaluate.

Encodes obs and goal to latents and plans with CEM/MPPI/Hermite in three phases:
approach (move the agent to a standoff point behind the block), push (push the
block toward its goal pose), park (block at its goal, move the agent to its own
goal).

Two sensors feed the cost. The agent position is read exactly from proprio. The
block position comes from a supervised detector when one is given, otherwise
from the latent readout. The JEPA world model supplies
the block dynamics: candidate actions are scored by the block displacement the
latent predicts, added to the precise current block pose.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ewjepa.probing import decode_pose


def _wrapped_angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Smallest absolute angle difference, handling the wrap around at 2 pi."""
    d = torch.abs(a - b) % (2.0 * math.pi)
    return torch.minimum(d, 2.0 * math.pi - d)

try:
    from stable_worldmodel.policy import BasePolicy
except ImportError:  # pragma: no cover - optional at import time
    BasePolicy = object  # type: ignore[misc, assignment]


def _as_tensor(x: Any, device: torch.device) -> torch.Tensor:
    t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    return t.to(device)


def _squeeze_env_time(x: torch.Tensor) -> torch.Tensor:
    """SWM shape (E,1,...) -> (E,...)."""
    if x.dim() >= 2 and x.shape[1] == 1:
        x = x.squeeze(1)
    return x


def _to_nchw_float(x: torch.Tensor) -> torch.Tensor:
    """Convert to (E,C,H,W) float in [0,1]."""
    x = _squeeze_env_time(x)
    if x.dim() == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
        x = x.permute(0, 3, 1, 2)
    elif x.dim() == 3 and x.shape[-1] in (1, 3, 4) and x.shape[0] not in (1, 3, 4):
        x = x.permute(2, 0, 1).unsqueeze(0)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    return x.float()


class RandomPolicy(BasePolicy):
    """Uniform random actions in the action bounds. Baseline for planning evals.

    Anchors the MPC success rate: any planner must beat this to show that the
    world model contributes anything at all.
    """

    def __init__(
        self,
        action_dim: int = 2,
        action_low: float = -1.0,
        action_high: float = 1.0,
        seed: int = 0,
        image_key: str = "pixels",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.image_key = image_key
        self.rng = np.random.default_rng(seed)

    def reset(self, num_envs: int = 1) -> None:
        pass

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        n_envs = np.asarray(info_dict[self.image_key]).shape[0]
        action = self.rng.uniform(
            self.action_low, self.action_high, size=(n_envs, self.action_dim)
        ).astype(np.float32)
        if hasattr(self, "env") and hasattr(self.env, "action_space"):
            target_shape = self.env.action_space.shape
            if len(target_shape) == 2 and target_shape[1] == 1:
                action = action[:, None, :]
        return action

    get_actions = get_action


class LatentMPCPolicy(BasePolicy):
    """Plan in the learned latent world model."""

    def __init__(
        self,
        model,
        planner,
        device: torch.device,
        image_key: str = "pixels",
        proprio_key: str = "proprio",
        goal_image_key: str = "goal",
        goal_proprio_key: str = "goal_proprio",
        proprio_normalizer=None,
        warm_start: bool = True,
        pose_readout: dict[str, torch.Tensor] | None = None,
        pose_cost_weight: float = 1.0,
        pose_scale: float = 512.0,
        goal_pose_key: str = "goal_pose",
        goal_state_key: str = "goal_state",
        agent_readout: dict[str, torch.Tensor] | None = None,
        approach_weight: float = 0.0,
        latent_cost_weight: float = 1.0,
        angle_cost_weight: float = 0.2,
        agent_goal_weight: float = 1.0,
        bounds_weight: float = 10.0,
        near_block_thresh: float = 18.0,
        park_angle_thresh: float = 0.30,
        engage_thresh: float = 70.0,
        standoff: float = 60.0,
        clearance: float = 45.0,
        action_penalty: float = 0.02,
        block_detector: Any = None,
        action_scale: float = 100.0,
        board_margin: float = 40.0,
        push_weight: float = 1.0,
        push_through: float = 4.0,
        gentle_cap: float = 0.35,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model.eval()
        self.planner = planner
        self.device = device
        self.image_key = image_key
        self.proprio_key = proprio_key
        self.goal_image_key = goal_image_key
        self.goal_proprio_key = goal_proprio_key
        self.proprio_normalizer = proprio_normalizer
        self.warm_start = warm_start
        self.pose_readout = pose_readout
        self.pose_cost_weight = pose_cost_weight
        self.pose_scale = pose_scale
        self.goal_pose_key = goal_pose_key
        self.goal_state_key = goal_state_key
        # agent_readout decodes the agent xy from the ego latent. approach_weight
        # then pulls the planned agent toward the block so it can push it. Without
        # this term the agent drifts off the board and never touches the block.
        self.agent_readout = agent_readout
        self.approach_weight = approach_weight
        self.latent_cost_weight = latent_cost_weight
        # PushT success needs the block at its goal pose AND the agent at its
        # own goal position, with the block angle within 20 degrees. These
        # weights cover those parts of the task. bounds_weight keeps the planned
        # agent and block inside the 512 px board; PushT never clamps either, so
        # without it the agent can leave the frame and the image signal goes dead.
        self.angle_cost_weight = angle_cost_weight
        self.agent_goal_weight = agent_goal_weight
        self.bounds_weight = bounds_weight
        # The plan switches from pushing the block to parking the agent only
        # once the block is truly placed: within near_block_thresh px of its
        # goal and within park_angle_thresh rad of its goal angle. Parking on
        # position alone freezes the block short of its goal, because the agent
        # then leaves to its own goal and stops refining the block.
        self.near_block_thresh = near_block_thresh
        self.park_angle_thresh = park_angle_thresh
        # the agent counts as engaged with the block when it is closer than
        # this (in px); beyond it the plan only approaches the block
        self.engage_thresh = engage_thresh
        # how far behind the block (in px, opposite the goal) the agent stands
        # before it starts a push
        self.standoff = standoff
        # minimum agent to block distance (in px) kept during the approach, so
        # the agent walks around the block instead of through it. It must stay
        # clearly below standoff and engage_thresh, or the agent can never
        # reach its approach target and hangs at the balance point forever.
        self.clearance = clearance
        # penalty on squared actions near the block. A full speed agent smashes
        # the block much farther than the model predicts, so pushes stay gentle.
        self.action_penalty = action_penalty
        # Supervised block sensor. When a detector is given the planner reads
        # the current block pose from it instead of from the latent readout.
        # The world model still predicts the block displacement.
        self.block_detector = block_detector.eval() if block_detector is not None else None
        # The PushT action is a relative move: the env sets the agent target to
        # agent_position + action * action_scale. Knowing action_scale lets us
        # clamp the commanded action so the agent target stays on the board.
        self.action_scale = action_scale
        # Margin (px) kept from the board edge when clamping the agent target.
        self.board_margin = board_margin
        # During the push phase the agent is pulled toward a point push_through
        # px past the block toward the goal, so it presses the block forward
        # instead of drifting. Without this the plan relies only on the world
        # model to discover that pushing needs the agent behind the block, which
        # its coarse displacement prediction does not do reliably.
        self.push_weight = push_weight
        self.push_through = push_through
        # Largest action size allowed near the block. A full speed hit flings the
        # block much farther than the model predicts, so the push is capped. The
        # cap shrinks as the block nears its goal so the last nudges are gentle
        # and do not overshoot.
        self.gentle_cap = gentle_cap
        self._nominal: list[torch.Tensor | None] = []
        self._phase: list[str] = []

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        n_envs = getattr(env, "num_envs", 1)
        self.reset(n_envs)

    def reset(self, num_envs: int = 1) -> None:
        self._nominal = [None] * num_envs
        self._phase = ["approach"] * num_envs

    def _proprio(self, obs: dict, key: str, batch: int) -> torch.Tensor:
        if key in obs and obs[key] is not None:
            p = _squeeze_env_time(_as_tensor(obs[key], self.device)).float()
            if p.dim() == 1:
                p = p.unsqueeze(0)
        else:
            p = torch.zeros(batch, self.model.cfg.proprio_dim, device=self.device)
        if self.proprio_normalizer is not None:
            p = self.proprio_normalizer(p)
        return p

    @torch.no_grad()
    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert self.goal_image_key in info_dict or "goal" in info_dict, (
            f"Goal image key {self.goal_image_key!r} missing from info_dict"
        )

        # accept raw numpy or preprocessed tensors from SWM
        if hasattr(self, "transform") and self.transform:
            info_dict = self._prepare_info(info_dict)

        # reset warm-start on episode reset
        needs_flush = info_dict.pop("_needs_flush", None)
        if needs_flush is not None:
            for i, flush in enumerate(needs_flush):
                if flush and i < len(self._nominal):
                    self._nominal[i] = None
                    self._phase[i] = "approach"

        goal_key = self.goal_image_key if self.goal_image_key in info_dict else "goal"
        pixels = _to_nchw_float(_as_tensor(info_dict[self.image_key], self.device))
        e = pixels.shape[0]
        if len(self._nominal) != e or len(self._phase) != e:
            self.reset(e)

        proprio = self._proprio(info_dict, self.proprio_key, e)
        z_world, z_ego = self.model.encode(pixels, proprio)

        # Skip the second ViT encode when the latent goal cost is off (grid
        # default). The pose costs below do not use goal_world.
        goal_world = None
        if self.latent_cost_weight > 0:
            goal_pixels = _to_nchw_float(_as_tensor(info_dict[goal_key], self.device))
            if goal_pixels.shape[0] == 1 and e > 1:
                goal_pixels = goal_pixels.expand(e, -1, -1, -1)
            goal_proprio = self._proprio(info_dict, self.goal_proprio_key, e)
            goal_world, _ = self.model.encode(goal_pixels, goal_proprio)

        # Target pose of the block, in raw pixels. It lives in goal_state
        # (agent xy, block xy, block angle, velocities), which is what the env
        # success check compares against and what the goal image shows. The
        # info dict also has a "goal_pose" key, but that one is the rendered
        # goal zone variation and stays at its default (256, 256, pi/4) in
        # every episode, so it must NOT be used as the target.
        goal_pose = None
        goal_poses = None
        raw_pose = None
        if self.pose_readout is not None:
            if self.goal_state_key in info_dict and info_dict[self.goal_state_key] is not None:
                raw_state = _squeeze_env_time(_as_tensor(info_dict[self.goal_state_key], self.device)).float()
                if raw_state.dim() == 1:
                    raw_state = raw_state.unsqueeze(0)
                raw_pose = raw_state[:, 2:5]
            elif self.goal_pose_key in info_dict:
                raw_pose = _squeeze_env_time(_as_tensor(info_dict[self.goal_pose_key], self.device)).float()
                if raw_pose.dim() == 1:
                    raw_pose = raw_pose.unsqueeze(0)
                raw_pose = raw_pose[:, :3]
        if raw_pose is not None:
            if raw_pose.shape[0] == e:
                goal_poses = raw_pose
            else:
                goal_pose = raw_pose[0]

        # Goal position of the agent itself, in raw pixels. PushT success also
        # checks the agent position, so the plan must park the agent there at
        # the end. We read it straight from the info dict, not through
        # _proprio, because _proprio applies the train time normalizer.
        goal_agents = None
        if self.goal_proprio_key in info_dict and info_dict[self.goal_proprio_key] is not None:
            raw_gp = _squeeze_env_time(_as_tensor(info_dict[self.goal_proprio_key], self.device)).float()
            if raw_gp.dim() == 1:
                raw_gp = raw_gp.unsqueeze(0)
            if raw_gp.shape[0] == e:
                goal_agents = raw_gp[:, :2]

        # Exact agent position from proprio. The proprioception carries the true
        # agent xy (state columns 0 and 1), so we read it straight instead of
        # decoding the noisy ego latent. We read the raw value from the info
        # dict, not through _proprio, because _proprio applies the train time
        # normalizer.
        agent_now = None
        if self.proprio_key in info_dict and info_dict[self.proprio_key] is not None:
            raw_p = _squeeze_env_time(_as_tensor(info_dict[self.proprio_key], self.device)).float()
            if raw_p.dim() == 1:
                raw_p = raw_p.unsqueeze(0)
            agent_now = raw_p[:, :2]  # (E, 2), exact agent xy in px

        # Current block pose. Prefer the supervised detector over the latent
        # readout. The readout is still used for predicted block displacement.
        det_block_now = None
        if self.block_detector is not None:
            det_block_now = self.block_detector.predict(pixels)  # (E, 3) x, y, angle
        dec_block_now = None
        if self.pose_readout is not None:
            dec_block_now = decode_pose(self.pose_readout, z_world)  # (E, 3)
        # The block pose used for phase decisions: detector when available.
        block_now_src = det_block_now if det_block_now is not None else dec_block_now

        actions = []
        for i in range(e):
            zw_i = z_world[i : i + 1]
            ze_i = z_ego[i : i + 1] if z_ego is not None else None
            goal_i = goal_world[i] if goal_world is not None else None
            goal_pose_i = goal_poses[i] if goal_poses is not None else goal_pose
            goal_agent_i = goal_agents[i] if goal_agents is not None else None

            # Pick the phase from the current decoded state. The training data
            # only has the agent close to the block (the collection policy stays
            # within 60 px of it), so the model hallucinates block motion when
            # the agent is far away. The phases keep the model in distribution:
            #   approach: move the agent to the standoff point behind the block
            #   push:     agent behind the block, push it toward its goal pose
            #   park:     block at its goal, move the agent to its own goal
            # A push only moves the block away from the agent, never toward it,
            # so the agent must stand on the side of the block opposite the
            # goal before pushing. The standoff point encodes that.
            phase = "push"
            block_now_i = None
            block_pose_now_i = None
            approach_target_i = None
            push_target_i = None
            gentle_cap_i = self.gentle_cap
            if block_now_src is not None and goal_pose_i is not None:
                block_pose_now_i = block_now_src[i]  # (3,) x, y, angle
                block_now_i = block_pose_now_i[:2]
                to_goal = goal_pose_i[:2] - block_now_i
                block_dist = to_goal.norm().clamp_min(1e-6)
                push_dir = to_goal / block_dist
                # Gentleness factor: full pushes when the block is far, tiny
                # nudges once it is close, so the final placement does not
                # overshoot. Ranges from 0.2 (very close) to 1.0 (far).
                fine = min(1.0, max(0.2, float(block_dist) / 100.0))
                # Approach aims behind the block (anti goal side). Push aims a
                # little past the block toward the goal, so the agent presses
                # the block forward.
                #
                # Getting behind the block is the hard part: if the agent is on
                # the goal side, heading straight for the standoff point drives
                # it through the block and shoves it the wrong way. So when the
                # agent is not yet behind the block, we route it to a lateral
                # waypoint (out to the side, past the clearance radius, and a bit
                # behind) first. Once it is behind, we aim at the true standoff.
                standoff_pt = block_now_i - self.standoff * push_dir
                if agent_now is not None:
                    perp = torch.stack([-push_dir[1], push_dir[0]])
                    rel = agent_now[i] - block_now_i
                    along = float(rel @ push_dir)  # >0 means on the goal side
                    side = float(rel @ perp)
                    if along > -0.3 * self.standoff:
                        s = 1.0 if side >= 0 else -1.0
                        standoff_pt = (
                            block_now_i
                            + perp * (s * self.clearance * 1.5)
                            - push_dir * (0.3 * self.standoff)
                        )
                approach_target_i = standoff_pt.clamp(20.0, self.pose_scale - 20.0)
                push_target_i = (block_now_i + self.push_through * fine * push_dir).clamp(
                    20.0, self.pose_scale - 20.0
                )
                gentle_cap_i = self.gentle_cap * fine
                # Park only once the block is truly placed: close in position
                # and within the success angle tolerance. The agent then leaves
                # to its own goal, so parking early would abandon the block far
                # from where it needs to be. Once placed we latch the park phase
                # through moderate drift, so the agent does not turn back and
                # mash the block it just placed (near the goal the push direction
                # is ill defined, so a fresh push would only knock it away).
                block_angle_err = float(_wrapped_angle_diff(block_pose_now_i[2], goal_pose_i[2]))
                placed = block_dist < self.near_block_thresh and block_angle_err < self.park_angle_thresh
                was_park = self._phase[i] == "park"
                stay_park = (
                    was_park
                    and block_dist < self.near_block_thresh * 2.5
                    and block_angle_err < self.park_angle_thresh * 1.8
                )
                if placed or stay_park:
                    phase = "park"
                elif agent_now is not None:
                    from_block = agent_now[i] - block_now_i
                    agent_dist = from_block.norm().clamp_min(1e-6)
                    # behind the block means roughly opposite the push direction
                    behind_dot = float((from_block / agent_dist) @ (-push_dir))
                    # Hysteresis: entering the push phase needs a clean position
                    # behind the block, staying in it tolerates more. Without it
                    # the readout noise flips the phase every step and the agent
                    # dithers in place.
                    was_pushing = self._phase[i] == "push"
                    min_dot = -0.2 if was_pushing else 0.3
                    max_dist = self.engage_thresh + (40.0 if was_pushing else 0.0)
                    if agent_dist > max_dist or behind_dot < min_dot:
                        phase = "approach"
            self._phase[i] = phase

            def cost_fn(
                cand: torch.Tensor,
                zw_i=zw_i,
                ze_i=ze_i,
                goal_i=goal_i,
                goal_pose_i=goal_pose_i,
                goal_agent_i=goal_agent_i,
                phase=phase,
                block_now_i=block_now_i,
                block_pose_now_i=block_pose_now_i,
                approach_target_i=approach_target_i,
                push_target_i=push_target_i,
                gentle_cap_i=gentle_cap_i,
            ) -> torch.Tensor:
                n = cand.shape[0]
                zw = zw_i.expand(n, -1)
                ze = ze_i.expand(n, -1) if ze_i is not None else None
                # One rollout scores every cost term. We read the block from the
                # world path and the agent from the ego path.
                _, _, world_traj, ego_traj = self.model.rollout(zw, ze, cand)
                costs = world_traj.new_zeros(n)
                scale = self.pose_scale

                # Raw latent distance to the goal image. It mixes agent, block
                # and background, so it is noisy; keep its weight small (or 0)
                # and let the decoded pose costs below drive the plan.
                if self.latent_cost_weight > 0:
                    latent_err = (world_traj - goal_i).pow(2).mean(dim=(1, 2))
                    costs = costs + self.latent_cost_weight * latent_err

                # Keep the motion gentle when working at the block. A full speed
                # agent smashes the block much farther than the model ever
                # predicts, so besides the mild squared penalty there is a hard
                # barrier that caps the action size near the block. The approach
                # phase is away from the block and stays fast.
                # Cap the action size while working at the block, so pushes stay
                # gentle. The cap shrinks as the block nears its goal, so the
                # final nudges do not overshoot. The approach phase is away from
                # the block and stays fast so the agent can get behind it.
                if phase != "approach":
                    if self.action_penalty > 0:
                        costs = costs + self.action_penalty * cand.pow(2).mean(dim=(1, 2))
                    over_cap = F.relu(cand.abs() - gentle_cap_i)
                    costs = costs + 3.0 * over_cap.pow(2).mean(dim=(1, 2))

                use_pose = self.pose_readout is not None and goal_pose_i is not None
                pred_block = None
                if self.pose_readout is not None:
                    pred_block = decode_pose(self.pose_readout, world_traj)  # (n, H, 3)
                    # Displacement mode: when a precise current block pose is
                    # available (from the detector), keep only the change the
                    # latent predicts under the actions and add it to the precise
                    # anchor. The readout has a large constant bias per frame,
                    # and taking the difference cancels it, so the absolute
                    # predicted block is far more accurate than the raw readout.
                    if block_pose_now_i is not None:
                        cur_dec = decode_pose(self.pose_readout, zw_i)  # (1, 3)
                        disp = pred_block - cur_dec.unsqueeze(1)  # (n, H, 3)
                        pred_block = block_pose_now_i.view(1, 1, 3) + disp
                pred_agent = None
                if self.agent_readout is not None:
                    agent_source = ego_traj if ego_traj is not None else world_traj
                    pred_agent = decode_pose(self.agent_readout, agent_source)  # (n, H, 2)

                if use_pose and phase != "approach":
                    # Push the block toward its goal position. Skipped in the
                    # approach phase, where predicted block motion is imaginary.
                    block_err = ((pred_block[..., :2] - goal_pose_i[:2]) / scale).pow(2).sum(-1).mean(-1)
                    costs = costs + self.pose_cost_weight * block_err
                    # Keep the block away from the board edge. The board has no
                    # walls, so a block pushed off the edge is lost for good.
                    block_outside = F.relu(pred_block[..., :2] - (scale - 40.0)) + F.relu(
                        40.0 - pred_block[..., :2]
                    )
                    block_bounds_err = (block_outside / scale).pow(2).sum(-1).mean(-1)
                    costs = costs + self.bounds_weight * block_bounds_err
                    # Rotate the block toward its goal angle. 1 - cos handles the
                    # wrap around at 2 pi, unlike a squared difference. The model
                    # predicts rotation well (sign agreement 86%, correlation
                    # 0.79 on dataset windows), so this runs during the whole
                    # push and the rotation gets corrected on the way.
                    if self.angle_cost_weight > 0:
                        angle_err = (1.0 - torch.cos(pred_block[..., 2] - goal_pose_i[2])).mean(-1)
                        costs = costs + self.angle_cost_weight * angle_err

                if pred_agent is not None:
                    # Keep the planned agent inside the board. The env never
                    # clamps the agent, and once it leaves the frame the image
                    # signal goes dead and the plan cannot recover.
                    if self.bounds_weight > 0:
                        outside = F.relu(pred_agent - scale) + F.relu(-pred_agent)
                        bounds_err = (outside / scale).pow(2).sum(-1).mean(-1)
                        costs = costs + self.bounds_weight * bounds_err
                    # The approach target comes from the block position decoded
                    # from the CURRENT frame, held fixed over the plan. Using the
                    # predicted block here lets the planner "bring the block to
                    # the agent" in imagination instead of moving the agent.
                    if approach_target_i is not None and phase == "approach" and self.approach_weight > 0:
                        # Move the agent behind the block. This is the only signal
                        # in the approach phase and it is reliable everywhere,
                        # since the agent position comes from proprio through z_ego.
                        approach_err = ((pred_agent - approach_target_i) / scale).pow(2).sum(-1).mean(-1)
                        costs = costs + self.approach_weight * approach_err
                        # Go around the block, not through it. Walking through the
                        # block shoves it in a random direction, often into a
                        # corner it cannot be pushed out of.
                        clear_dist = (pred_agent - block_now_i).norm(dim=-1)  # (n, H)
                        clearance_err = (
                            (F.relu(self.clearance - clear_dist) / scale).pow(2).mean(-1)
                        )
                        costs = costs + 20.0 * clearance_err
                    if phase == "push" and push_target_i is not None and self.push_weight > 0:
                        # Press the block toward the goal. The target sits a bit
                        # past the block along the push direction, so chasing it
                        # keeps the agent pushing the block forward rather than
                        # drifting around it.
                        push_err = ((pred_agent - push_target_i) / scale).pow(2).sum(-1).mean(-1)
                        costs = costs + self.push_weight * push_err
                    if phase == "park" and goal_agent_i is not None and self.agent_goal_weight > 0:
                        # The block sits at its goal. Park the agent at its own
                        # goal position to satisfy the success check.
                        agent_err = ((pred_agent - goal_agent_i) / scale).pow(2).sum(-1).mean(-1)
                        costs = costs + self.agent_goal_weight * agent_err
                        # Keep clear of the placed block on the way to the goal,
                        # so the agent walks around it instead of shoving it out
                        # of position while parking.
                        if block_now_i is not None:
                            clear_dist = (pred_agent - block_now_i).norm(dim=-1)
                            clearance_err = (
                                (F.relu(self.clearance - clear_dist) / scale).pow(2).mean(-1)
                            )
                            costs = costs + 20.0 * clearance_err
                return costs

            nominal, first = self.planner.plan(cost_fn, nominal=self._nominal[i])
            self._nominal[i] = nominal if self.warm_start else None

            # Hard on board clamp. The env moves the agent to
            # agent_position + action * action_scale, and the agent is a
            # kinematic body that walls do not stop, so a large action drives it
            # off the frame where the image stops changing and the plan can
            # never recover. Because we know the exact agent position from
            # proprio, we clamp the applied action so its target stays inside
            # the board. This is a hard guarantee, not a soft cost.
            if agent_now is not None and self.action_scale > 0:
                agent_i = agent_now[i]
                lo = ((self.board_margin - agent_i) / self.action_scale).clamp(-1.0, 1.0)
                hi = ((self.pose_scale - self.board_margin - agent_i) / self.action_scale).clamp(-1.0, 1.0)
                first = torch.max(torch.min(first, hi), lo)

            actions.append(first)

        action = torch.stack(actions, dim=0)
        if hasattr(self.env, "action_space"):
            target_shape = self.env.action_space.shape
            if len(target_shape) == 2 and target_shape[1] == 1:
                action = action.unsqueeze(1)

        return action.cpu().numpy()

    get_actions = get_action  # alias for tests

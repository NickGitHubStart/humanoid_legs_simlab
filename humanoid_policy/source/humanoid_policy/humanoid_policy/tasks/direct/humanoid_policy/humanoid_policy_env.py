# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment implementation for Humanoid Leg Robot."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from .humanoid_policy_env_cfg import HumanoidPolicyEnvCfg


class HumanoidPolicyEnv(DirectRLEnv):
    """Humanoid Leg Robot environment for balance and locomotion training."""

    cfg: HumanoidPolicyEnvCfg

    def __init__(self, cfg: HumanoidPolicyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices for all 8 joints
        self._joint_ids, _ = self.robot.find_joints(self.cfg.joint_names)
        self.num_joints = len(self._joint_ids)

        # Get body indices for contact detection
        self._termination_body_ids, _ = self.robot.find_bodies(self.cfg.termination_contact_bodies)
        self._foot_body_ids, _ = self.robot.find_bodies(self.cfg.foot_bodies)

        # Initialize state tensors
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Gravity vector in world frame (for projected gravity calculation)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

        # Previous actions for action rate penalty and observations
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Commands (for future locomotion: vx, vy, wz, heading)
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)

        # Joint limits for penalty calculation
        self.joint_pos_limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids, :]

    # ---- Scene Setup ----
    def _setup_scene(self):
        """Setup the simulation scene with robot, ground, and lights."""
        self.robot = Articulation(self.cfg.robot_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---- Pre-Physics Step ----
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions before physics step."""
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    # ---- Apply Actions ----
    def _apply_action(self) -> None:
        """Apply joint torques to all joints."""
        # Scale actions with per-joint action_scale and apply as effort (torque) targets
        action_scale = torch.tensor(self.cfg.action_scale, device=self.device)
        torques = self.actions * action_scale
        self.robot.set_joint_effort_target(torques, joint_ids=self._joint_ids)

    # ---- Get Observations ----
    def _get_observations(self) -> dict:
        """
        Compute observations for the policy.
        
        Observations (37-dim):
        - Projected gravity in base frame (3) - which way is up
        - Base linear velocity in base frame (3) - how fast moving
        - Base angular velocity in base frame (3) - how fast rotating
        - Joint positions (8)
        - Joint velocities (8)
        - Previous actions (8) - for action smoothness
        - Commands (4) - target velocities for locomotion
        """
        # Get base state
        base_quat = self.robot.data.root_quat_w  # [num_envs, 4]
        base_lin_vel = self.robot.data.root_lin_vel_w  # [num_envs, 3]
        base_ang_vel = self.robot.data.root_ang_vel_w  # [num_envs, 3]

        # Transform to base frame
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        base_lin_vel_b = quat_rotate_inverse(base_quat, base_lin_vel)
        base_ang_vel_b = quat_rotate_inverse(base_quat, base_ang_vel)

        # Get joint states
        joint_pos = self.joint_pos[:, self._joint_ids]
        joint_vel = self.joint_vel[:, self._joint_ids]

        # Concatenate observations
        obs = torch.cat(
            [
                projected_gravity,   # (3,) - tells policy which way is up
                base_lin_vel_b,      # (3,) - base linear velocity in base frame
                base_ang_vel_b,      # (3,) - base angular velocity in base frame
                joint_pos,           # (8,) - current joint positions
                joint_vel,           # (8,) - current joint velocities
                self.prev_actions,   # (8,) - previous actions for smoothness
                self.commands,       # (4,) - target commands for locomotion
            ],
            dim=-1,
        )

        return {"policy": obs}

    # ---- Compute Rewards ----
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards for the current step.
        
        Rewards:
        - Alive: constant reward for not falling
        - Upright: reward for keeping base oriented upward
        - Foot contact: reward for feet touching ground
        - Penalties for: falling, jerky motion, large torques, etc.
        """
        # Get base state
        base_quat = self.robot.data.root_quat_w
        base_pos = self.robot.data.root_pos_w
        base_lin_vel = self.robot.data.root_lin_vel_w
        base_ang_vel = self.robot.data.root_ang_vel_w

        # Compute projected gravity (z-component indicates uprightness)
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        upright = projected_gravity[:, 2]  # -1 = upright, +1 = upside down

        # Joint states
        joint_pos = self.joint_pos[:, self._joint_ids]
        joint_vel = self.joint_vel[:, self._joint_ids]

        # Symmetry: difference between left and right leg joints
        # Indices: [0=HipFlexL, 1=HipAbdL, 2=KneeL, 3=AnkleL, 4=HipFlexR, 5=HipAbdR, 6=KneeR, 7=AnkleR]
        # Compare: HipFlex (0 vs 4), Knee (2 vs 6), Ankle (3 vs 7)
        hip_flex_diff = torch.abs(joint_pos[:, 0] - joint_pos[:, 4])  # left vs right hip flexion
        knee_diff = torch.abs(joint_pos[:, 2] - joint_pos[:, 6])      # left vs right knee
        ankle_diff = torch.abs(joint_pos[:, 3] - joint_pos[:, 7])    # left vs right ankle
        symmetry_error = hip_flex_diff + knee_diff + ankle_diff  # smaller = more symmetric

        # Joint limit violation (distance to limits)
        joint_pos_lower = self.joint_pos_limits[:, :, 0]
        joint_pos_upper = self.joint_pos_limits[:, :, 1]
        joint_limit_violation = torch.sum(
            torch.maximum(joint_pos_lower - joint_pos, torch.zeros_like(joint_pos)) +
            torch.maximum(joint_pos - joint_pos_upper, torch.zeros_like(joint_pos)),
            dim=-1
        )

        # Action rate (change from previous action)
        action_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)

        # Compute rewards
        total_reward = compute_rewards(
            # Scales
            rew_scale_alive=self.cfg.rew_scale_alive,
            rew_scale_terminated=self.cfg.rew_scale_terminated,
            rew_scale_upright=self.cfg.rew_scale_upright,
            rew_scale_joint_vel=self.cfg.rew_scale_joint_vel,
            rew_scale_action=self.cfg.rew_scale_action,
            rew_scale_action_rate=self.cfg.rew_scale_action_rate,
            rew_scale_base_vel=self.cfg.rew_scale_base_vel,
            rew_scale_base_ang_vel=self.cfg.rew_scale_base_ang_vel,
            rew_scale_joint_limit=self.cfg.rew_scale_joint_limit,
            rew_scale_symmetry=self.cfg.rew_scale_symmetry,
            # Values
            upright=upright,
            joint_vel=joint_vel,
            actions=self.actions,
            action_rate=action_rate,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_limit_violation=joint_limit_violation,
            reset_terminated=self.reset_terminated,
            symmetry_error=symmetry_error,
        )

        return total_reward

    # ---- Get Done Flags ----
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Check termination and timeout conditions.
        
        Termination:
        - Non-foot body contact with ground (Hüfte, Oberschenkel, Unterschenkel)
        - Backup: Base height very low (as fallback if contact sensors fail)
        
        NO termination on tilt - robot can lean as much as it wants until ground contact!
        
        Timeout:
        - Episode length exceeded
        """
        # Update state tensors
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Get base state
        base_pos = self.robot.data.root_pos_w

        # --- Primary: Body-Ground Contact Termination ---
        # Terminate if Hüfte, Oberschenkel, or Unterschenkel touch ground
        # This requires contact sensors (activate_contact_sensors=True in USD config)
        fallen_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        if self.cfg.use_contact_termination:
            try:
                # Get contact forces for termination bodies
                # body_contact_forces shape: [num_envs, num_bodies, 3]
                contact_forces = self.robot.data.body_contact_forces
                if contact_forces is not None:
                    termination_contacts = contact_forces[:, self._termination_body_ids, :]
                    contact_forces_mag = torch.norm(termination_contacts, dim=-1)
                    # Terminate if ANY termination body has contact force > threshold
                    fallen_contact = torch.any(contact_forces_mag > 1.0, dim=-1)
            except (AttributeError, IndexError):
                # Contact sensors not available, fall back to height check
                pass

        # --- Backup: Base height too low ---
        base_height = base_pos[:, 2]
        fallen_height = base_height < self.cfg.min_base_height

        # Combine termination conditions
        # Primary: ground contact | Backup: very low height
        terminated = fallen_contact | fallen_height

        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    # ---- Reset Environments ----
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments to initial state with optional noise."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # Get default joint states
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # Add small random noise to joint positions
        joint_pos += sample_uniform(
            -self.cfg.initial_joint_noise,
            self.cfg.initial_joint_noise,
            joint_pos.shape,
            joint_pos.device,
        )

        # Get default root state and add env origins
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Update internal state
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # Write states to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset actions
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # Reset commands (zero for balance, randomize for locomotion)
        self.commands[env_ids] = 0.0


@torch.jit.script
def compute_rewards(
    # Scales
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_joint_vel: float,
    rew_scale_action: float,
    rew_scale_action_rate: float,
    rew_scale_base_vel: float,
    rew_scale_base_ang_vel: float,
    rew_scale_joint_limit: float,
    rew_scale_symmetry: float,
    # Values
    upright: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    action_rate: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_limit_violation: torch.Tensor,
    reset_terminated: torch.Tensor,
    symmetry_error: torch.Tensor,
) -> torch.Tensor:
    """
    Compute total reward for balance task.
    
    Args:
        rew_scale_*: reward scaling factors
        upright: z-component of projected gravity (-1 = upright, +1 = upside down)
        joint_vel: joint velocities [num_envs, num_joints]
        actions: applied actions [num_envs, num_actions]
        action_rate: squared difference between current and previous actions [num_envs]
        base_lin_vel: base linear velocity [num_envs, 3]
        base_ang_vel: base angular velocity [num_envs, 3]
        joint_limit_violation: distance outside joint limits [num_envs]
        reset_terminated: termination flags [num_envs]
    
    Returns:
        total_reward: [num_envs]
    """
    # === Positive Rewards ===
    
    # Alive reward (only if not terminated)
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # Upright reward: +1 when upright (upright ≈ -1), 0 when fallen
    # Transform: upright ∈ [-1, +1] → reward ∈ [0, 1]
    rew_upright = rew_scale_upright * (0.5 - 0.5 * upright)

    # Symmetry reward: reward for symmetric leg positions (prevents lunging)
    # symmetry_error = sum of differences between left/right joints
    # Smaller error = more symmetric = higher reward
    # Normalize: exp(-symmetry_error) gives reward in [0, 1] range
    rew_symmetry = rew_scale_symmetry * torch.exp(-symmetry_error)

    # === Negative Rewards (Penalties) ===

    # Termination penalty
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # Joint velocity penalty (smooth motion)
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=-1)

    # Action penalty (energy efficiency)
    rew_action = rew_scale_action * torch.sum(torch.square(actions), dim=-1)

    # Action rate penalty (smooth actions)
    rew_action_rate = rew_scale_action_rate * action_rate

    # Base linear velocity penalty (stay in place for balance)
    rew_base_vel = rew_scale_base_vel * torch.sum(torch.square(base_lin_vel[:, :2]), dim=-1)  # only x,y

    # Base angular velocity penalty (no spinning)
    rew_base_ang_vel = rew_scale_base_ang_vel * torch.sum(torch.square(base_ang_vel), dim=-1)

    # Joint limit penalty
    rew_joint_limit = rew_scale_joint_limit * joint_limit_violation

    # === Total Reward ===
    total_reward = (
        rew_alive +
        rew_upright +
        rew_symmetry +
        rew_termination +
        rew_joint_vel +
        rew_action +
        rew_action_rate +
        rew_base_vel +
        rew_base_ang_vel +
        rew_joint_limit
    )

    return total_reward

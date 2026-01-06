# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Humanoid Leg Robot Environment."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

##
# Robot ArticulationCfg (USD-Pfad, Joints, Aktuatoren, Startpose)
##

HUMANOID_LEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/2003n/projects/humanoid_tracked_legged_robot/USD/onlytorque.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),  # Starting position [x, y, z]
        # rot=(0.7071, 0.0, -0.7071, 0.0),  # Quaternion (w,x,y,z): -90° rotation around Y-axis
        # Keine Rotation - USD bereits korrekt orientiert
        joint_pos={
            # COM-korrigierte Startpose (deg → rad) - Reihenfolge wie in USD:
            # 1. HuefteFlexionRechts: -30° | 2. HuefteFlexionLinks: -30° | 3. AbduktionLinks: 0° | 4. AbduktionRechts: 0°
            # 5. KnieLinks: 40° | 6. AnkleLinks: -40° | 7. KnieRechts: 40° | 8. AnkleRechts: -40°
            "HuefteFlexionRechts": -0.5236,    # -30°
            "HuefteFlexionLinks": -0.5236,     # -30°
            "AbduktionLinks": 0.0,             # 0°
            "AbduktionRechts": 0.0,            # 0°
            "KnieLinks": 0.6981,               # 40°
            "AnkleLinks": -0.6981,             # -40°
            "KnieRechts": 0.6981,              # 40°
            "AnkleRechts": -0.6981,            # -40°
        },
        joint_vel={
            ".*": 0.0,  # All joints start with zero velocity
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "HuefteFlexionRechts",
                "HuefteFlexionLinks",
                "AbduktionLinks",
                "AbduktionRechts",
                "KnieLinks",
                "AnkleLinks",
                "KnieRechts",
                "AnkleRechts",
            ],
            effort_limit=300.0,   # Max torque [Nm]
            velocity_limit=10.0,  # Max velocity [rad/s]
            stiffness=0.0,        # match Isaac Sim setting
            damping=0.0,          # match Isaac Sim setting
        ),
    },
)


##
# Environment Configuration (Env-Settings, Sim-Params, Scene, Custom Params)
##

@configclass
class HumanoidPolicyEnvCfg(DirectRLEnvCfg):
    """Configuration for the Humanoid Leg Robot balancing/walking environment."""

    # ---- Env Settings ----
    decimation = 2  # env step every 2 sim steps (60 Hz control)
    episode_length_s = 5.0  # shorter episodes for faster learning

    # ---- Spaces Definition ----
    # Actions: 8 joint torques
    action_space = 8
    # Observations (37-dim):
    #   - projected gravity (3)
    #   - base linear velocity (3)
    #   - base angular velocity (3)
    #   - joint positions (8)
    #   - joint velocities (8)
    #   - previous actions (8)
    #   - commands (4: vx, vy, wz, heading) - for future locomotion
    observation_space = 37
    state_space = 0

    # ---- Simulation Parameters ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 120 Hz physics
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ---- Robot ----
    robot_cfg: ArticulationCfg = HUMANOID_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ---- Scene ----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # ---- Joint Names (for reference in Env) ----
    # Reihenfolge muss mit USD-Datei übereinstimmen!
    joint_names = [
        "HuefteFlexionRechts",
        "HuefteFlexionLinks",
        "AbduktionLinks",
        "AbduktionRechts",
        "KnieLinks",
        "AnkleLinks",
        "KnieRechts",
        "AnkleRechts",
    ]

    # ---- Body Names for Contact Detection ----
    # Bodies that should NOT touch ground (terminate if they do)
    termination_contact_bodies = [
        "base_link",           # Hüfte/Becken
        "abduktion_flexion1_1_1",  # Hüfte-Abduktion links
        "abduktion_flexion2_1_1",  # Hüfte-Abduktion rechts
        "upper_leg1_1_1",      # Oberschenkel links
        "upper_leg1_1_2",      # Oberschenkel rechts
        "lower_leg1_1_1",      # Unterschenkel links
        "lower_leg1_1_2",      # Unterschenkel rechts
    ]
    # Bodies that SHOULD touch ground (feet)
    foot_bodies = [
        "foot1_1_1",           # Fuß links
        "foot1_1_2",           # Fuß rechts
    ]

    # ---- Action Scale (per Joint) ----
    # Torque multiplier [Nm] for each joint (Reihenfolge wie in USD):
    # 1. HuefteFlexionRechts: 25 Nm - mehr Kraft für Vorwärts/Rückwärts
    # 2. HuefteFlexionLinks: 25 Nm - mehr Kraft für Vorwärts/Rückwärts
    # 3. AbduktionLinks: 15 Nm
    # 4. AbduktionRechts: 15 Nm
    # 5. KnieLinks: 15 Nm
    # 6. AnkleLinks: 15 Nm
    # 7. KnieRechts: 15 Nm
    # 8. AnkleRechts: 15 Nm
    action_scale = (25.0, 25.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0)

    # ---- Reward Scales ----
    # Positive rewards
    rew_scale_alive = 2.0              # reward for staying alive (not fallen)
    rew_scale_upright = 1.0            # reward for upright orientation
    rew_scale_foot_contact = 0.5       # reward for feet touching ground
    
    # Negative rewards (penalties)
    rew_scale_terminated = -5.0        # penalty for falling (ground contact)
    rew_scale_joint_vel = -0.0005      # reduced: allow small joint movements for balance
    rew_scale_action = -0.0001         # penalty on action magnitude (energy)
    rew_scale_action_rate = -0.001     # penalty on action change (smooth actions)
    rew_scale_base_vel = -0.005        # reduced: allow small balance adjustments
    rew_scale_base_ang_vel = 0.0       # NO penalty for angular velocity (turning is allowed)
    rew_scale_joint_limit = -0.1       # penalty for being near joint limits
    
    # Locomotion rewards (set > 0 when training forward walking)
    rew_scale_forward_vel = 0.0        # reward for forward velocity
    target_forward_vel = 0.5           # target forward velocity [m/s]

    # ---- Termination Conditions ----
    # ONLY terminate on ground contact (Hüfte/Oberschenkel/Unterschenkel touching ground)
    # NO termination on tilt - robot can lean as much as it wants until it touches ground
    min_base_height = 0.05          # backup: terminate if base drops very low [m]
    use_contact_termination = True     # primary: terminate on body-ground contact

    # ---- Reset Randomization ----
    initial_joint_noise = 0.1          # random noise on joint angles at reset [rad]
    initial_base_pos_noise = 0.0       # random noise on base position [m]
    initial_base_rot_noise = 0.0       # random noise on base rotation [rad]

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to debug articulation structure in Isaac Lab."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Debug articulation structure.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Template-Humanoid-Policy-Direct-v0", help="Task name.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import omni.usd
from pxr import UsdPhysics

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import humanoid_policy.tasks  # noqa: F401


def main():
    """Debug articulation structure."""
    print("\n" + "="*80)
    print("DEBUGGING ARTICULATION STRUCTURE")
    print("="*80 + "\n")
    
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # Create environment
    print("Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Get robot
    robot = env.unwrapped.robot
    print(f"\nRobot type: {type(robot)}")
    print(f"Robot device: {robot.device}")
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    robot_prim_path = "/World/envs/env_0/Robot"
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    
    if robot_prim.IsValid():
        print(f"\n✓ Robot prim found: {robot_prim_path}")
        
        # Check for ArticulationRoot
        if robot_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print("✓ ArticulationRootAPI found on robot prim")
        else:
            print("❌ ERROR: ArticulationRootAPI NOT found on robot prim!")
            print("   This is why parts spawn disconnected!")
        
        # Get all children
        print(f"\nRobot prim children:")
        for child in robot_prim.GetChildren():
            print(f"  - {child.GetPath()}")
            
            # Check if child has RigidBodyAPI
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                print(f"    ✓ Has RigidBodyAPI")
            else:
                print(f"    ⚠ No RigidBodyAPI")
            
            # Check for joints
            for grandchild in child.GetChildren():
                if grandchild.IsA(UsdPhysics.RevoluteJoint) or grandchild.IsA(UsdPhysics.PrismaticJoint):
                    print(f"    ✓ Joint found: {grandchild.GetPath()}")
    else:
        print(f"❌ Robot prim NOT found at: {robot_prim_path}")
        print("   Available prims at /World/envs/env_0:")
        env_prim = stage.GetPrimAtPath("/World/envs/env_0")
        if env_prim.IsValid():
            for child in env_prim.GetChildren():
                print(f"     - {child.GetPath()}")
    
    # Get joint information
    print(f"\n{'='*80}")
    print("JOINT INFORMATION")
    print(f"{'='*80}\n")
    
    try:
        joint_names = robot.joint_names
        print(f"Joint names from robot: {joint_names}")
        print(f"Number of joints: {len(joint_names)}")
        
        # Get joint positions
        joint_positions = robot.data.joint_pos
        print(f"\nJoint positions shape: {joint_positions.shape}")
        print(f"Joint positions (first env): {joint_positions[0]}")
        
        # Get body information
        print(f"\n{'='*80}")
        print("BODY INFORMATION")
        print(f"{'='*80}\n")
        
        body_names = robot.body_names
        print(f"Body names: {body_names}")
        print(f"Number of bodies: {len(body_names)}")
        
        # Check body positions
        body_positions = robot.data.root_pos_w
        print(f"\nRoot body position (first env): {body_positions[0]}")
        
    except Exception as e:
        print(f"❌ Error getting robot information: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}\n")
    
    # Reset to see initial state
    print("Resetting environment...")
    env.reset()
    
    print("\n✓ Environment reset complete")
    print("\nIf parts are spawning disconnected:")
    print("  1. Open your USD file in Isaac Sim")
    print("  2. Select the root body (base_link or Robot)")
    print("  3. In Properties panel, add 'Physics > Articulation Root'")
    print("  4. Make sure all joints are properly defined")
    print("  5. Save the USD file")
    print("\nPress Ctrl+C to exit...")
    
    # Keep running
    import time
    try:
        while simulation_app.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    env.close()


if __name__ == "__main__":
    main()


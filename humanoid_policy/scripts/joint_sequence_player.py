# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play joint movements from a text file line by line in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play joint sequence from text file.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Humanoid-Policy-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--file", type=str, default="joint_sequence.txt", help="Path to text file with joint positions."
)
parser.add_argument(
    "--delay", type=float, default=1.0, help="Delay between frames in seconds (default: 1.0)."
)
parser.add_argument(
    "--loop", action="store_true", default=False, help="Loop the sequence continuously."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Force GUI mode (disable headless) - Isaac Sim must stay open
if hasattr(args_cli, 'headless'):
    args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import humanoid_policy.tasks  # noqa: F401


def parse_joint_line(line: str) -> list[float]:
    """
    Parse a line from the joint sequence file.
    
    Expected formats:
    - Space-separated: "0.0 0.0 0.3491 -0.3491 0.0 0.0 -0.3491 0.3491"
    - Comma-separated: "0.0, 0.0, 0.3491, -0.3491, 0.0, 0.0, -0.3491, 0.3491"
    - With comments: "0.0 0.0 0.3491 -0.3491 0.0 0.0 -0.3491 0.3491  # Frame 1"
    
    Returns:
        List of 8 joint positions in radians
    """
    # Remove comments (everything after #)
    if "#" in line:
        line = line.split("#")[0]
    
    # Strip whitespace
    line = line.strip()
    
    # Skip empty lines
    if not line:
        return None
    
    # Try comma-separated first
    if "," in line:
        values = [float(x.strip()) for x in line.split(",")]
    else:
        # Space-separated
        values = [float(x) for x in line.split()]
    
    # Validate we have 8 joints
    if len(values) != 8:
        raise ValueError(
            f"Expected 8 joint values, got {len(values)}. "
            f"Line: {line}"
        )
    
    return values


def load_joint_sequence(file_path: str) -> list[list[float]]:
    """
    Load joint sequence from text file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        List of joint position frames, each frame is a list of 8 joint positions
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Joint sequence file not found: {file_path}")
    
    sequence = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                joint_positions = parse_joint_line(line)
                if joint_positions is not None:
                    sequence.append(joint_positions)
            except ValueError as e:
                print(f"[WARNING] Skipping line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"[ERROR] Error parsing line {line_num}: {e}")
                continue
    
    if not sequence:
        raise ValueError(f"No valid joint positions found in {file_path}")
    
    print(f"[INFO] Loaded {len(sequence)} frames from {file_path}")
    return sequence


def main():
    """Play joint sequence from text file."""
    # Load joint sequence
    try:
        sequence = load_joint_sequence(args_cli.file)
    except Exception as e:
        print(f"[ERROR] Failed to load joint sequence: {e}")
        return
    
    # Create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Get robot and joint information
    robot = env.unwrapped.robot
    joint_names = env.unwrapped.cfg.joint_names
    joint_ids, _ = robot.find_joints(joint_names)
    
    print(f"[INFO] Joint names: {joint_names}")
    print(f"[INFO] Joint IDs: {joint_ids}")
    print(f"[INFO] Playing sequence with {args_cli.delay}s delay between frames")
    if args_cli.loop:
        print(f"[INFO] Looping enabled")
    
    # Reset environment
    env.reset()
    
    # File watching setup
    file_path = Path(args_cli.file)
    if not file_path.is_absolute():
        file_path = Path(os.getcwd()) / file_path
    last_modified = file_path.stat().st_mtime if file_path.exists() else 0
    
    frame_idx = 0
    sequence_complete = False
    steps_per_frame = max(1, int(args_cli.delay * 120))  # 120 Hz simulation, convert delay to steps
    
    # Main simulation loop
    step_count = 0
    frame_updated = False  # Track if we've set the first frame
    reload_sequence = False
    
    print(f"[INFO] Watching file: {file_path}")
    print(f"[INFO] Isaac Sim will stay open. Edit {file_path} and save to replay the sequence!")
    print(f"[INFO] Press Ctrl+C to exit")
    print(f"[INFO] Simulation app is_running: {simulation_app.is_running()}")
    
    try:
        # Wait a bit for the app to fully initialize
        time.sleep(0.5)
        
        print(f"[INFO] Starting simulation loop...")
        while simulation_app.is_running():
            with torch.inference_mode():
                # Check if file was modified
                if file_path.exists():
                    current_modified = file_path.stat().st_mtime
                    if current_modified > last_modified:
                        print(f"[INFO] File changed! Reloading sequence...")
                        try:
                            sequence = load_joint_sequence(str(file_path))
                            last_modified = current_modified
                            frame_idx = 0
                            sequence_complete = False
                            frame_updated = False
                            step_count = 0
                            reload_sequence = True
                            print(f"[INFO] Reloaded {len(sequence)} frames")
                        except Exception as e:
                            print(f"[ERROR] Failed to reload sequence: {e}")
                            reload_sequence = False
                
                # Check if we need to load next frame
                if not sequence_complete:
                    # Get current frame
                    if frame_idx < len(sequence):
                        # Update frame immediately on first iteration, after reload, or when step count matches
                        if not frame_updated or reload_sequence or step_count % steps_per_frame == 0:
                            joint_positions = sequence[frame_idx]
                            
                            # Convert to tensor [num_envs, num_joints]
                            joint_pos_tensor = torch.tensor(
                                joint_positions, device=env.unwrapped.device, dtype=torch.float32
                            ).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                            
                            # Set joint velocities to zero for smooth movement
                            joint_vel_tensor = torch.zeros_like(joint_pos_tensor)
                            
                            # Write joint state to simulation
                            # This directly sets the joint positions for the specified joints only
                            robot.write_joint_state_to_sim(
                                joint_pos_tensor, joint_vel_tensor, joint_ids, env.unwrapped.robot._ALL_INDICES
                            )
                            
                            print(f"[INFO] Frame {frame_idx + 1}/{len(sequence)}: {joint_positions}")
                            
                            frame_idx += 1
                            frame_updated = True
                            reload_sequence = False
                            
                            # Check if sequence is complete
                            if frame_idx >= len(sequence):
                                if args_cli.loop:
                                    # Reset to beginning
                                    frame_idx = 0
                                    frame_updated = False
                                    print(f"[INFO] Looping sequence...")
                                else:
                                    sequence_complete = True
                                    print(f"[INFO] Sequence complete. Edit the file and save to replay, or press Ctrl+C to exit.")
                
                # Step simulation (always keep simulation running, even if sequence is complete)
                # Use zero actions to maintain current pose
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)
                step_count += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the simulator
        print("[INFO] Closing environment...")
        env.close()


if __name__ == "__main__":
    # run the main function
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close sim app
        print("[INFO] Closing Isaac Sim...")
        simulation_app.close()


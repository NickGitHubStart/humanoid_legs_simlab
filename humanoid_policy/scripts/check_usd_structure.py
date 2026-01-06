# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to check USD file structure and articulation setup."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Check USD file structure.")
parser.add_argument(
    "--usd_path", 
    type=str, 
    default="C:/Users/2003n/projects/humanoid_tracked_legged_robot/USD/40grad_kippung_sonst_nichts.usd",
    help="Path to USD file to check."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.usd
from pxr import Usd, UsdPhysics, UsdGeom

def check_usd_structure(usd_path: str):
    """Check USD file structure for articulation and joints."""
    print(f"\n{'='*80}")
    print(f"Checking USD file: {usd_path}")
    print(f"{'='*80}\n")
    
    # Open USD stage
    stage = omni.usd.get_context().get_stage()
    
    # Load the USD file
    root_layer = stage.GetRootLayer()
    root_layer.subLayerPaths.append(usd_path)
    stage.Reload()
    
    print("Stage loaded. Searching for articulation and joints...\n")
    
    # Find all prims
    all_prims = [stage.GetPrimAtPath(path) for path in stage.Traverse()]
    
    # Check for ArticulationRoot
    articulation_roots = []
    joints = []
    rigid_bodies = []
    
    for prim in all_prims:
        # Check for ArticulationRoot
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim.GetPath())
            print(f"✓ Found ArticulationRoot: {prim.GetPath()}")
        
        # Check for Joints
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(UsdPhysics.FixedJoint):
            joints.append(prim.GetPath())
            joint_type = "Revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "Prismatic" if prim.IsA(UsdPhysics.PrismaticJoint) else "Fixed"
            print(f"✓ Found {joint_type} Joint: {prim.GetPath()}")
            
            # Get joint properties
            if prim.IsA(UsdPhysics.RevoluteJoint):
                revolute_joint = UsdPhysics.RevoluteJoint(prim)
                body0 = revolute_joint.GetBody0Rel().GetTargets()
                body1 = revolute_joint.GetBody1Rel().GetTargets()
                print(f"    Body0: {body0}")
                print(f"    Body1: {body1}")
        
        # Check for RigidBody
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim.GetPath())
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"ArticulationRoots found: {len(articulation_roots)}")
    if len(articulation_roots) == 0:
        print("  ❌ ERROR: No ArticulationRoot found! This is required for physics simulation.")
        print("  → Solution: Add an ArticulationRootAPI to the root body (usually base_link)")
    else:
        for root in articulation_roots:
            print(f"  ✓ {root}")
    
    print(f"\nJoints found: {len(joints)}")
    if len(joints) == 0:
        print("  ❌ ERROR: No joints found! Bodies will not be connected.")
        print("  → Solution: Add RevoluteJoint, PrismaticJoint, or FixedJoint prims")
    else:
        for joint in joints:
            print(f"  ✓ {joint}")
    
    print(f"\nRigidBodies found: {len(rigid_bodies)}")
    for body in rigid_bodies[:10]:  # Show first 10
        print(f"  ✓ {body}")
    if len(rigid_bodies) > 10:
        print(f"  ... and {len(rigid_bodies) - 10} more")
    
    print(f"\n{'='*80}\n")
    
    # Check if root has articulation
    root_prim = stage.GetPrimAtPath("/")
    if root_prim.IsValid():
        print("Root prim:", root_prim.GetPath())
        children = root_prim.GetChildren()
        print(f"Root children: {[c.GetPath() for c in children]}\n")
    
    return len(articulation_roots) > 0, len(joints) > 0


def main():
    """Main function."""
    has_articulation, has_joints = check_usd_structure(args_cli.usd_path)
    
    if not has_articulation:
        print("\n❌ CRITICAL: USD file is missing ArticulationRoot!")
        print("   The robot parts will spawn disconnected in the air.")
        print("\n   To fix this in Isaac Sim:")
        print("   1. Open the USD file in Isaac Sim")
        print("   2. Select the root body (usually 'base_link' or 'Robot')")
        print("   3. In Properties, add 'Physics > Articulation Root'")
        print("   4. Save the USD file")
    
    if not has_joints:
        print("\n❌ CRITICAL: USD file has no joints!")
        print("   Bodies are not connected.")
    
    print("\nPress any key to exit...")
    input()
    
    # Keep app running
    import time
    while simulation_app.is_running():
        time.sleep(0.1)


if __name__ == "__main__":
    main()


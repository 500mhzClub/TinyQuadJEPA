import torch
import time
import numpy as np
import genesis as gs

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
URDF_PATH = "assets/mini_pupper/mini_pupper.urdf" 
# Standard "Standing" angles for Dog-like robots (in Radians)
# Adjust signs (+/-) if your legs bend the wrong way
HEURISTIC_STAND = [0.0, 0.6, -1.2] * 4  # (Hip, Upper, Lower) repeated 4 times

def main():
    print("🔎 PROJECT CERBERUS: URDF & Stance Analyzer")
    
    # 1. Setup Genesis (Vulkan for rendering)
    gs.init(backend=gs.vulkan, logging_level="warning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 0.0, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        # Gravity off so we can see the pose without it falling
        rigid_options=gs.options.RigidOptions(gravity=(0, 0, 0), enable_joint_limit=True),
        show_viewer=True
    )

    plane = scene.add_entity(morph=gs.morphs.Plane())
    
    # Spawn robot slightly in the air
    robot = scene.add_entity(
        morph=gs.morphs.URDF(file=URDF_PATH, pos=(0, 0, 0.4)),
        material=gs.materials.Rigid()
    )
    
    scene.build(n_envs=1)

    # 2. Analyze Limits
    # Note: motor_dofs usually 6-17 for floating base robots (0-5 are root)
    motor_dofs = np.arange(6, 18)
    dof_names = [robot.get_dof_name(i) for i in motor_dofs]
    lower, upper = robot.get_dofs_limit(motor_dofs)
    
    # Convert to numpy for cleaner printing
    if isinstance(lower, torch.Tensor): lower = lower.cpu().numpy()
    if isinstance(upper, torch.Tensor): upper = upper.cpu().numpy()

    print(f"\n{'IDX':<5} | {'JOINT NAME':<20} | {'LOW':<8} | {'HIGH':<8} | {'ZERO OK?':<8}")
    print("-" * 65)
    
    zero_valid = True
    for i, name, lo, hi in zip(motor_dofs, dof_names, lower, upper):
        is_ok = (0.0 >= lo) and (0.0 <= hi)
        if not is_ok: zero_valid = False
        print(f"{i:<5} | {name:<20} | {lo:<8.3f} | {hi:<8.3f} | {str(is_ok):<8}")

    print("-" * 65)
    
    if not zero_valid:
        print("⚠️  CRITICAL: The 'Zero Pose' (0.0) violates your joint limits!")
        print("    The physics engine will force-snap the robot to the nearest limit.")
    else:
        print("✅ Joint limits allow the 'Zero Pose'. Checking visual validity...")

    # 3. Visual Test Loop
    print("\n🎥 RENDERING: Switching poses every 3 seconds...")
    
    poses = {
        "ZERO (0.0, 0.0, 0.0)": torch.zeros(12, device=device),
        "HEURISTIC STAND (0, 0.6, -1.2)": torch.tensor(HEURISTIC_STAND, device=device)
    }
    
    pose_names = list(poses.keys())
    current_idx = 0
    last_switch = time.time()

    while True:
        # Switch pose logic
        if time.time() - last_switch > 3.0:
            current_idx = (current_idx + 1) % len(pose_names)
            last_switch = time.time()
            print(f"   -> Showing: {pose_names[current_idx]}")

        # Set Position
        target = poses[pose_names[current_idx]]
        
        # We use hard set_dofs_position to bypass physics/actuators for visualization
        robot.set_dofs_position(target, motor_dofs)
        
        # Step scene to update viewer
        scene.step()
        time.sleep(0.01)

if __name__ == "__main__":
    main()
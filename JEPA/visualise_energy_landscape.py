#!/usr/bin/env python3
"""
System 2 EBM JEPA - 3D Energy Landscape Visualizer
Generates a 3D topographic surface plot of the JEPA's cost function.

Usage:
    python JEPA/visualize_energy_landscape.py --ckpt jepa_checkpoints/jepa_epoch_6_step_1000.pt --device cpu
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import genesis as gs

# -----------------------------------------
# JEPA Architecture
# -----------------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int = 47, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class JointEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(feature_dim=128)
        self.prop_enc = ProprioEncoder(input_dim=47, feature_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim),
        )
    def forward(self, vision: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        v_feat = self.vis_enc(vision)
        p_feat = self.prop_enc(proprio)
        return self.fusion(torch.cat([v_feat, p_feat], dim=-1))

class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + cmd_dim, latent_dim),
            nn.ELU()
        )
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, z_t: torch.Tensor, c_t: torch.Tensor, h_t: torch.Tensor):
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        return self.output_proj(h_next), h_next

class EBM_TinyQuadJEPA(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = JointEncoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim, cmd_dim=3)

# -----------------------------------------
# Simulator Helpers
# -----------------------------------------
def init_genesis_scene(device):
    print("🌍 Booting Genesis Simulator (Headless CPU Mode)...")
    gs.init(backend=gs.cpu) 
    scene = gs.Scene(show_viewer=False) 
    plane = scene.add_entity(gs.morphs.Plane()) 
    
    scene.add_entity(gs.morphs.Box(pos=(0.5, 0.4, 0.075), size=(0.15, 0.15, 0.15), fixed=True))
    scene.add_entity(gs.morphs.Box(pos=(0.9, -0.3, 0.05), size=(0.1, 0.2, 0.1), fixed=True))
    scene.add_entity(gs.morphs.Box(pos=(1.3, 0.2, 0.1), size=(0.2, 0.1, 0.2), fixed=True))
    scene.add_entity(gs.morphs.Box(pos=(0.2, -0.5, 0.06), size=(0.12, 0.12, 0.12), fixed=True))

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/mini_pupper/mini_pupper.urdf",
            pos=(0.0, 0.0, 0.12),
            fixed=False,
            merge_fixed_links=False,
            requires_jac_and_IK=False,
        )
    )
    
    cam_brain = scene.add_camera(
        res=(64, 64), pos=(0.8, -0.8, 0.45), lookat=(0.0, 0.0, 0.12), fov=50
    )
    
    scene.build()
    
    actuated_joints = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    dofs_idx = [robot.get_joint(name).dofs_idx_local[0] for name in actuated_joints]
    
    q0 = np.array([0.06, 0.06, -0.06, -0.06, 0.85, 0.85, 0.85, 0.85, -1.75, -1.75, -1.75, -1.75], dtype=np.float32)
    robot.set_dofs_position(q0, dofs_idx)

    return scene, robot, cam_brain, dofs_idx, torch.tensor(q0, device=device)

def get_jepa_state(robot, cam_brain, device):
    # Genesis 0.3.14 returns the image buffers directly from render()
    render_out = cam_brain.render()
    
    img = None
    if isinstance(render_out, tuple) and len(render_out) > 0:
        img = render_out[0] # rgb is usually the first buffer
    elif isinstance(render_out, dict):
        img = render_out.get('rgb', render_out.get('color'))
    elif hasattr(render_out, 'shape'):
        img = render_out
        
    if img is None:
        raise RuntimeError(f"cam.render() returned unexpected type: {type(render_out)}")

    if hasattr(img, 'cpu'):
        img = img.cpu().numpy()

    if isinstance(img, np.ndarray):
        if img.shape[-1] == 4: # Strip alpha channel
            img = img[:, :, :3]
        if img.shape[-1] == 3: # Convert to (C, H, W)
            img = np.transpose(img, (2, 0, 1))
        vis_tensor = torch.from_numpy(img).float().to(device) / 255.0
    else:
        raise ValueError(f"Image extracted is not an array! Type: {type(img)}")

    # Extract Proprioception
    raw_prop = robot.get_dofs_position().cpu().numpy()
    if raw_prop.ndim == 2: raw_prop = raw_prop[0] 
    prop_array = np.zeros(47, dtype=np.float32)
    prop_array[:min(47, len(raw_prop))] = raw_prop[:min(47, len(raw_prop))]
    prop_tensor = torch.from_numpy(prop_array).float().to(device)
        
    return vis_tensor.unsqueeze(0), prop_tensor.unsqueeze(0)

def move_cameras(robot, cam_brain):
    r_pos = robot.get_pos().cpu().numpy()
    if r_pos.ndim > 1: r_pos = r_pos[0]
    c_pos = r_pos + np.array([0.8, -0.8, 0.33], dtype=np.float32)
    try:
        cam_brain.set_pose(pos=c_pos, lookat=r_pos, up=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    except TypeError:
        cam_brain.set_pose(pos=c_pos, lookat=r_pos)

# -----------------------------------------
# Main Visualization Logic
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to JEPA checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--res", type=int, default=30, help="Resolution of the 3D grid")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Loading JEPA into Genesis on {device}...")

    jepa = EBM_TinyQuadJEPA().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    jepa.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()})
    jepa.eval()

    scene, robot, cam_brain, act_dofs, q0 = init_genesis_scene(device)
    q0_np = q0.cpu().numpy()

    # Settle physics
    for _ in range(10): scene.step()

    # --- 1. Capture the Goal State ---
    print("🎯 Teleporting 0.29m forward to capture goal state...")
    robot.set_pos(np.array([0.29, 0.0, 0.12], dtype=np.float32))
    for _ in range(5): scene.step() 
    move_cameras(robot, cam_brain)
    
    vis_goal, prop_goal = get_jepa_state(robot, cam_brain, device)
    with torch.no_grad():
        z_goal = jepa.encoder(vis_goal, prop_goal).detach()

    # --- 2. Capture the Start State ---
    print("⏪ Resetting to origin to map energy landscape...")
    robot.set_pos(np.array([0.0, 0.0, 0.12], dtype=np.float32))
    robot.set_dofs_position(q0_np, act_dofs)
    robot.set_vel(np.zeros(3, dtype=np.float32))
    robot.set_ang(np.zeros(3, dtype=np.float32))
    for _ in range(10): scene.step()
    move_cameras(robot, cam_brain)

    vis_start, prop_start = get_jepa_state(robot, cam_brain, device)
    with torch.no_grad():
        z_start = jepa.encoder(vis_start, prop_start).detach()

    # --- 3. Grid Sweep (The Brain Scan) ---
    print(f"🧠 Scanning latent space across {args.res}x{args.res} command grid...")
    
    vx_vals = np.linspace(-0.6, 0.6, args.res)   # Forward/Backward
    om_vals = np.linspace(-0.8, 0.8, args.res)   # Yaw Left/Right
    
    VX, OM = np.meshgrid(vx_vals, om_vals)
    COSTS = np.zeros_like(VX)

    with torch.no_grad():
        for i in range(args.res):
            for j in range(args.res):
                # Create the candidate command [vx, vy=0, omega]
                cmd = torch.tensor([[VX[i, j], 0.0, OM[i, j]]], device=device, dtype=torch.float32)
                
                # Rollout the predictor
                z_pred = z_start.clone()
                h_t = torch.zeros(1, 256, device=device)
                cost_sum = 0
                
                for t in range(args.horizon):
                    z_pred, h_t = jepa.predictor(z_pred, cmd, h_t)
                    cost_sum += torch.norm(z_pred - z_goal, dim=-1).item()
                    
                COSTS[i, j] = cost_sum

    # --- 4. Render the 3D Plot ---
    print("🎨 Rendering 3D Topographic Plot...")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    
    surf = ax.plot_surface(VX, OM, COSTS, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.9)
    
    batch_idx = ckpt.get('batch_idx', 'Unknown')
    epoch = ckpt.get('epoch', 'Unknown')
    ax.set_title(f"JEPA Energy Landscape (Epoch {epoch}, Step {batch_idx})", fontsize=14)
    ax.set_xlabel("Forward Velocity (vx)", fontsize=11, labelpad=10)
    ax.set_ylabel("Turn Rate (omega)", fontsize=11, labelpad=10)
    ax.set_zlabel("Predicted Distance to Goal State", fontsize=11, labelpad=10)
    
    # Invert Z so "valleys" (lower cost) look like funnels
    ax.invert_zaxis()
    
    fig.colorbar(surf, shrink=0.5, aspect=0.5, pad=0.1, label="Distance to Goal")
    plt.tight_layout()
    
    out_file = f"jepa_logs/energy_landscape_epoch_{epoch}_step_{batch_idx}.png"
    plt.savefig(out_file, dpi=200)
    print(f"✅ Success! Energy landscape saved to {out_file}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
System 2 EBM JEPA - Mind's Eye (CNN Feature Map Visualizer)
Intercepts and visualizes the internal convolutional activations of the Vision Encoder.

Usage:
    python JEPA/visualize_feature_maps.py --ckpt jepa_checkpoints/jepa_epoch_8_step_1000.pt --device cpu
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import genesis as gs

# -----------------------------------------
# JEPA Architecture
# -----------------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),                                                # Index 1: Early Features
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),                                                # Index 7: Deep Features
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
# Simulator Helpers & Math
# -----------------------------------------
def quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)

def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)

def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    vq = torch.cat([zeros, v], dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(q, vq), quat_conj_wxyz(q))[:, 1:4]

def world_to_body_vec(quat_wxyz: torch.Tensor, vec_world: torch.Tensor) -> torch.Tensor:
    return quat_rotate_wxyz(quat_conj_wxyz(quat_wxyz), vec_world)

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
        res=(64, 64), pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50
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

def get_system1_obs(robot, q0, prev_action, cmd, act_dofs, device):
    pos = robot.get_pos().to(device)
    if pos.dim() == 1: pos = pos.unsqueeze(0)
    quat = robot.get_quat().to(device)
    if quat.dim() == 1: quat = quat.unsqueeze(0)
    vel_w = robot.get_vel().to(device)
    if vel_w.dim() == 1: vel_w = vel_w.unsqueeze(0)
    ang_w = robot.get_ang().to(device)
    if ang_w.dim() == 1: ang_w = ang_w.unsqueeze(0)

    vel_b = world_to_body_vec(quat, vel_w)
    ang_b = world_to_body_vec(quat, ang_w)

    q = robot.get_dofs_position(act_dofs).to(device)
    if q.dim() == 1: q = q.unsqueeze(0)
    dq = robot.get_dofs_velocity(act_dofs).to(device)
    if dq.dim() == 1: dq = dq.unsqueeze(0)

    z = pos[:, 2:3]
    q_rel = q - q0.unsqueeze(0)

    obs = torch.cat([z, quat, vel_b, ang_b, q_rel, dq, prev_action, cmd], dim=1)
    return obs

def get_jepa_state(robot, cam_brain, q0, prev_action, act_dofs, device):
    render_out = cam_brain.render()
    img = None
    if isinstance(render_out, tuple) and len(render_out) > 0:
        img = render_out[0] 
    elif isinstance(render_out, dict):
        img = render_out.get('rgb', render_out.get('color'))
    elif hasattr(render_out, 'shape'):
        img = render_out
        
    if img is None:
        raise RuntimeError(f"cam.render() returned unexpected type: {type(render_out)}")

    if hasattr(img, 'cpu'):
        img = img.cpu().numpy()

    raw_img = img.copy() # Save the raw visual for plotting
    
    if isinstance(img, np.ndarray):
        if img.shape[-1] == 4: 
            img = img[:, :, :3]
        if img.shape[-1] == 3: 
            img = np.transpose(img, (2, 0, 1))
        vis_tensor = torch.from_numpy(img.copy()).float().to(device) / 255.0
    else:
        raise ValueError("Image extracted is not an array!")

    dummy_cmd = torch.zeros((1, 3), device=device)
    sys1_obs = get_system1_obs(robot, q0, prev_action, dummy_cmd, act_dofs, device)
    prop_tensor = sys1_obs[:, :47].clone() 
        
    return vis_tensor.unsqueeze(0), prop_tensor, raw_img

def move_cameras(robot, cam_brain, cam_render=None):
    """Mounts the camera securely to the nose of the robot."""
    r_pos = robot.get_pos().cpu().numpy()
    r_quat = robot.get_quat().cpu().numpy()
    
    if r_pos.ndim > 1: 
        r_pos = r_pos[0]
        r_quat = r_quat[0]
        
    # Genesis quaternions are [w, x, y, z]
    w, x, y, z = r_quat
    
    # Calculate the Forward vector (local X-axis rotated by body quat)
    fx = 1 - 2 * (y**2 + z**2)
    fy = 2 * (x*y + w*z)
    fz = 2 * (x*z - w*y)
    forward = np.array([fx, fy, fz], dtype=np.float32)
    
    # Calculate the Up vector (local Z-axis rotated by body quat)
    ux = 2 * (x*z + w*y)
    uy = 2 * (y*z - w*x)
    uz = 1 - 2 * (x**2 + y**2)
    up = np.array([ux, uy, uz], dtype=np.float32)

    # Mount the camera at the "nose" of the dog
    cam_pos = r_pos + (forward * 0.15) + (up * 0.05) 
    
    # Look straight ahead
    look_target = cam_pos + (forward * 1.0)
    
    cams = [cam_brain]
    if cam_render is not None:
        cams.append(cam_render)
        
    for cam in cams:
        try:
            cam.set_pose(pos=cam_pos, lookat=look_target, up=up)
        except TypeError:
            try:
                cam.set_pose(pos=cam_pos, lookat=look_target, up_vector=up)
            except TypeError:
                cam.set_pose(pos=cam_pos, lookat=look_target)

# -----------------------------------------
# Main Logic
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to JEPA checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Loading JEPA into Genesis on {device}...")

    jepa = EBM_TinyQuadJEPA().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    jepa.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()})
    jepa.eval()

    # --- ATTACH HOOKS TO THE VISION ENCODER ---
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Hook into Layer 1 (Early features, 32 channels) and Layer 4 (Deep features, 256 channels)
    jepa.encoder.vis_enc.net[1].register_forward_hook(get_activation('layer1'))
    jepa.encoder.vis_enc.net[7].register_forward_hook(get_activation('layer4'))

    scene, robot, cam_brain, act_dofs, q0 = init_genesis_scene(device)
    
    # Settle physics and grab an interesting frame
    for _ in range(10): scene.step()
    move_cameras(robot, cam_brain)
    
    prev_action = torch.zeros((1, 12), device=device)
    vis_tensor, prop_tensor, raw_img = get_jepa_state(robot, cam_brain, q0, prev_action, act_dofs, device)

    # --- PUSH IMAGE THROUGH NETWORK ---
    with torch.no_grad():
        _ = jepa.encoder(vis_tensor, prop_tensor)

    # --- PLOTTING ---
    print("🎨 Rendering CNN Feature Maps...")
    
    # Extract the hooked tensors (remove batch dimension)
    layer1_maps = activations['layer1'].squeeze(0).cpu().numpy()
    layer4_maps = activations['layer4'].squeeze(0).cpu().numpy()

    # Create a massive plot
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"JEPA Vision Encoder 'Mind's Eye' (Step {ckpt.get('batch_idx', 'Unknown')})", fontsize=16)

    # 1. Plot the Original Image
    ax_orig = plt.subplot2grid((3, 8), (0, 3), colspan=2)
    if raw_img.shape[-1] == 4: raw_img = raw_img[:, :, :3]
    ax_orig.imshow(raw_img)
    ax_orig.set_title("Egocentric Robot View (64x64 RGB)", pad=10)
    ax_orig.axis('off')

    # 2. Plot 16 channels from Layer 1 (Early Features)
    num_l1 = 16
    for i in range(num_l1):
        ax = plt.subplot2grid((3, 8), (1, i % 8))
        # Use a colormap like 'viridis' or 'gray'
        ax.imshow(layer1_maps[i], cmap='viridis')
        if i == 0:
            ax.set_ylabel("Conv1\n(Low-Level)", fontsize=12, labelpad=15, rotation=0, ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])

    # 3. Plot 16 channels from Layer 4 (Deep Features)
    num_l4 = 16
    for i in range(num_l4):
        ax = plt.subplot2grid((3, 8), (2, i % 8))
        ax.imshow(layer4_maps[i], cmap='viridis')
        if i == 0:
            ax.set_ylabel("Conv4\n(Deep/Abstract)", fontsize=12, labelpad=15, rotation=0, ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    
    epoch = ckpt.get('epoch', 'Unknown')
    step = ckpt.get('batch_idx', 'Unknown')
    out_file = f"jepa_logs/feature_maps_epoch_{epoch}_step_{step}.png"
    plt.savefig(out_file, dpi=200)
    print(f"✅ Success! Mind's Eye visualization saved to {out_file}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
JEPA Latent Energy Visualizer

Builds a reachable goal state, scans a (vx, wz) command grid through the 
latent predictor, and uses the trained GoalEnergyHead to plot the energy landscape.
"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from PIL import Image
import genesis as gs

# -----------------------------------------
# Architectures (Backbone + Energy Head)
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 50, act_dim: int = 12, hid: int = 256):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)
        
    def act_deterministic(self, obs): return torch.tanh(self.actor(obs))

class VisionEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(), nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),
            nn.Flatten(), nn.Linear(256 * 4 * 4, feature_dim), nn.LayerNorm(feature_dim),
        )
    def forward(self, x): return self.net(x)

class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int = 47, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU(), nn.Linear(256, feature_dim), nn.LayerNorm(feature_dim))
    def forward(self, x): return self.net(x)

class JointEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(feature_dim=128)
        self.prop_enc = ProprioEncoder(input_dim=47, feature_dim=128)
        self.fusion = nn.Sequential(nn.Linear(256, 256), nn.ELU(), nn.Linear(256, latent_dim))
    def forward(self, vision, proprio): return self.fusion(torch.cat([self.vis_enc(vision), self.prop_enc(proprio)], dim=-1))

class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(latent_dim + cmd_dim, latent_dim), nn.ELU())
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.output_proj = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ELU(), nn.Linear(latent_dim, latent_dim))
    def forward(self, z_t, c_t, h_t):
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        return self.output_proj(h_next), h_next

class TinyQuadJEPA(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = JointEncoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim, cmd_dim=3)

class GoalEnergyHead(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, z_pred, z_goal):
        x = torch.cat([z_pred, z_goal, z_pred - z_goal, z_pred * z_goal], dim=-1)
        return self.net(x).squeeze(-1)

def clean_state_dict(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# -----------------------------------------
# Simulator Helpers
# -----------------------------------------
def quat_conj_wxyz(q): return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)
def quat_mul_wxyz(a, b):
    w = a[:,0]*b[:,0] - a[:,1]*b[:,1] - a[:,2]*b[:,2] - a[:,3]*b[:,3]
    x = a[:,0]*b[:,1] + a[:,1]*b[:,0] + a[:,2]*b[:,3] - a[:,3]*b[:,2]
    y = a[:,0]*b[:,2] - a[:,1]*b[:,3] + a[:,2]*b[:,0] + a[:,3]*b[:,1]
    z = a[:,0]*b[:,3] + a[:,1]*b[:,2] - a[:,2]*b[:,1] + a[:,3]*b[:,0]
    return torch.stack([w, x, y, z], dim=-1)
def world_to_body_vec(quat, vec):
    vq = torch.cat([torch.zeros((vec.shape[0], 1), device=vec.device), vec], dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(quat_conj_wxyz(quat), vq), quat)[:, 1:4]

def init_genesis_scene(device):
    print("🌍 Booting Genesis Simulator (Headless CPU Mode)...")
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    tex_path = os.path.abspath("checkerboard.png")
    scene.add_entity(morph=gs.morphs.Plane(), surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_path=tex_path)))
    robot = scene.add_entity(gs.morphs.URDF(file="assets/mini_pupper/mini_pupper.urdf", pos=(0.0, 0.0, 0.12), fixed=False))
    cam_brain = scene.add_camera(res=(64, 64), pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50)
    scene.build()
    
    act_joints = ["lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint", "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint", "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint"]
    dofs_idx = [robot.get_joint(name).dofs_idx_local[0] for name in act_joints]
    q0 = np.array([0.06, 0.06, -0.06, -0.06, 0.85, 0.85, 0.85, 0.85, -1.75, -1.75, -1.75, -1.75], dtype=np.float32)
    robot.set_dofs_position(q0, dofs_idx)
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, dofs_idx)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, dofs_idx)
    return scene, robot, cam_brain, dofs_idx, torch.tensor(q0, device=device)

def get_system1_obs(robot, q0, prev_action, cmd, act_dofs, device):
    pos = robot.get_pos().to(device).unsqueeze(0) if robot.get_pos().dim()==1 else robot.get_pos().to(device)
    quat = robot.get_quat().to(device).unsqueeze(0) if robot.get_quat().dim()==1 else robot.get_quat().to(device)
    vel_w = robot.get_vel().to(device).unsqueeze(0) if robot.get_vel().dim()==1 else robot.get_vel().to(device)
    ang_w = robot.get_ang().to(device).unsqueeze(0) if robot.get_ang().dim()==1 else robot.get_ang().to(device)
    q = robot.get_dofs_position(act_dofs).to(device).unsqueeze(0) if robot.get_dofs_position(act_dofs).dim()==1 else robot.get_dofs_position(act_dofs).to(device)
    dq = robot.get_dofs_velocity(act_dofs).to(device).unsqueeze(0) if robot.get_dofs_velocity(act_dofs).dim()==1 else robot.get_dofs_velocity(act_dofs).to(device)
    return torch.cat([pos[:, 2:3], quat, world_to_body_vec(quat, vel_w), world_to_body_vec(quat, ang_w), q - q0.unsqueeze(0), dq, prev_action, cmd], dim=1)

def get_jepa_state(robot, cam_brain, q0, prev_action, act_dofs, device):
    img = cam_brain.render()[0]
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    
    # Adding .copy() resolves the negative stride issue after transposing
    rgb = np.transpose(img[:, :, :3], (2, 0, 1)).copy()
    
    vis_tensor = torch.from_numpy(rgb).float().to(device) / 255.0
    prop_tensor = get_system1_obs(robot, q0, prev_action, torch.zeros((1, 3), device=device), act_dofs, device)[:, :47]
    return vis_tensor.unsqueeze(0), prop_tensor

def move_camera(robot, cam_brain):
    r_pos = robot.get_pos().cpu().numpy()
    r_quat = robot.get_quat().cpu().numpy()
    
    if r_pos.ndim > 1: r_pos = r_pos[0]
    if r_quat.ndim > 1: r_quat = r_quat[0]
    
    w, x, y, z = r_quat
    forward = np.array([1 - 2*(y**2 + z**2), 2*(x*y + w*z), 2*(x*z - w*y)], dtype=np.float32)
    up = np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x**2 + y**2)], dtype=np.float32)
    cam_brain.set_pose(pos=r_pos + forward*0.10 + up*0.05, lookat=r_pos + forward*0.10 + up*0.05 + forward*1.0)

# -----------------------------------------
# Main
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", type=str, required=True)
    parser.add_argument("--head_ckpt", type=str, required=True)
    parser.add_argument("--ppo_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--res", type=int, default=35)
    parser.add_argument("--goal_vx", type=float, default=0.40)
    parser.add_argument("--goal_wz", type=float, default=0.00)
    args = parser.parse_args()

    device = torch.device(args.device)
    
    jepa = TinyQuadJEPA().to(device)
    jepa.load_state_dict(clean_state_dict(torch.load(args.jepa_ckpt, map_location=device)["model_state_dict"]))
    jepa.eval()

    energy_head = GoalEnergyHead().to(device)
    energy_head.load_state_dict(clean_state_dict(torch.load(args.head_ckpt, map_location=device)["energy_head_state_dict"]))
    energy_head.eval()

    ppo = ActorCritic().to(device)
    ppo.load_state_dict(torch.load(args.ppo_ckpt, map_location=device)["model"], strict=False)
    ppo.eval()

    scene, robot, cam_brain, act_dofs, q0 = init_genesis_scene(device)
    for _ in range(10): scene.step()

    print(f"🎬 Driving PPO for {args.horizon} steps to capture goal...")
    prev_a = torch.zeros((1, 12), device=device)
    cmd = torch.tensor([[args.goal_vx, 0.0, args.goal_wz]], device=device)
    for _ in range(args.horizon):
        prev_a = ppo.act_deterministic(get_system1_obs(robot, q0, prev_a, cmd, act_dofs, device))
        robot.control_dofs_position(q0.unsqueeze(0) + 0.30 * prev_a, act_dofs)
        for _ in range(4): scene.step()
        
    move_camera(robot, cam_brain)
    v_goal, p_goal = get_jepa_state(robot, cam_brain, q0, prev_a, act_dofs, device)
    z_goal = jepa.encoder(v_goal, p_goal).detach()
    
    goal_pos_raw = robot.get_pos().cpu().numpy()
    goal_pos = goal_pos_raw[0] if goal_pos_raw.ndim > 1 else goal_pos_raw

    robot.set_pos(np.array([0.0, 0.0, 0.12], dtype=np.float32))
    robot.set_dofs_position(q0.cpu().numpy(), act_dofs)
    for _ in range(20): scene.step()
    move_camera(robot, cam_brain)

    v_start, p_start = get_jepa_state(robot, cam_brain, q0, torch.zeros((1, 12), device=device), act_dofs, device)
    z_start = jepa.encoder(v_start, p_start).detach()

    print("🧠 Scanning latent energy landscape...")
    vx_vals, om_vals = np.linspace(-0.40, 0.40, args.res), np.linspace(-0.60, 0.60, args.res)
    VX, OM = np.meshgrid(vx_vals, om_vals)
    ENERGIES = np.zeros_like(VX, dtype=np.float32)

    with torch.no_grad():
        for i in range(args.res):
            for j in range(args.res):
                z_pred = z_start.clone()
                h_t = torch.zeros(1, 256, device=device)
                for _ in range(args.horizon):
                    # Explicitly cast the command tensor to float32 to match the model weights
                    cmd_t = torch.tensor([[VX[i, j], 0.0, OM[i, j]]], device=device, dtype=torch.float32)
                    z_pred, h_t = jepa.predictor(z_pred, cmd_t, h_t)
                ENERGIES[i, j] = float(energy_head(z_pred, z_goal).item())

    best_flat = int(np.argmin(ENERGIES))
    bi, bj = np.unravel_index(best_flat, ENERGIES.shape)
    best_vx, best_wz, best_eng = float(VX[bi, bj]), float(OM[bi, bj]), float(ENERGIES[bi, bj])

    print("🎨 Rendering figure...")
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(ENERGIES, origin='lower', extent=[vx_vals.min(), vx_vals.max(), om_vals.min(), om_vals.max()], aspect='auto', interpolation='bicubic')
    ax1.scatter([best_vx], [best_wz], marker='x', s=100, c='red')
    ax1.set_title("Energy Landscape Heatmap"); ax1.set_xlabel("vx"); ax1.set_ylabel("wz")
    fig.colorbar(im, ax=ax1).set_label("Energy")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(VX, OM, ENERGIES, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.95)
    ax2.scatter([best_vx], [best_wz], [best_eng], s=50, c='red')
    ax2.set_title("Energy Surface"); ax2.set_xlabel("vx"); ax2.set_ylabel("wz"); ax2.set_zlabel("Energy")

    os.makedirs("jepa_logs", exist_ok=True)
    plt.savefig("jepa_logs/energy_landscape.png", dpi=220, bbox_inches='tight')
    print("✅ Saved to jepa_logs/energy_landscape.png")

if __name__ == "__main__":
    main()
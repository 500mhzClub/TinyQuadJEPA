#!/usr/bin/env python3
"""
System 2 JEPA Closed-Loop Visualizer.

Runs Genesis with the GUI enabled, using the trained JEPA backbone and 
Energy Head to perform real-time Model Predictive Control (MPC).
"""
from __future__ import annotations

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import genesis as gs

# -----------------------------
# Math & Quaternion Helpers
# -----------------------------
def world_to_body_vec(quat_wxyz: torch.Tensor, vec_world: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((vec_world.shape[0], 1), device=vec_world.device, dtype=vec_world.dtype)
    vq = torch.cat([zeros, vec_world], dim=-1)
    q_conj = torch.stack([quat_wxyz[:, 0], -quat_wxyz[:, 1], -quat_wxyz[:, 2], -quat_wxyz[:, 3]], dim=-1)
    
    def quat_mul(a, b):
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        w = aw * bw - ax * bx - ay * by - az * bz
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack([w, x, y, z], dim=-1)
        
    return quat_mul(quat_mul(q_conj, vq), quat_wxyz)[:, 1:4]

# -----------------------------
# Architecture Definitions
# -----------------------------
class VisionEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    def forward(self, x): return self.net(x)

class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int = 47, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, feature_dim), nn.LayerNorm(feature_dim)
        )
    def forward(self, x): return self.net(x)

class JointEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(feature_dim=128)
        self.prop_enc = ProprioEncoder(input_dim=47, feature_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
        )
    def forward(self, vision, proprio):
        return self.fusion(torch.cat([self.vis_enc(vision), self.prop_enc(proprio)], dim=-1))

class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(latent_dim + cmd_dim, latent_dim), nn.ELU())
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, z_t, c_t, h_t):
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        return self.output_proj(h_next), h_next

class TinyQuadJEPA(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
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

# -----------------------------
# MPC Core
# -----------------------------
@torch.no_grad()
def plan_mpc(
    backbone: TinyQuadJEPA,
    energy_head: GoalEnergyHead,
    vis_current: torch.Tensor,
    prop_current: torch.Tensor,
    vis_goal: torch.Tensor,
    prop_goal: torch.Tensor,
    num_samples: int,
    horizon: int,
    device: torch.device
) -> torch.Tensor:
    z_start = backbone.encoder(vis_current.unsqueeze(0), prop_current.unsqueeze(0)).expand(num_samples, -1)
    z_goal = backbone.encoder(vis_goal.unsqueeze(0), prop_goal.unsqueeze(0)).expand(num_samples, -1)
    
    cmds = torch.rand((num_samples, horizon, 3), device=device) * 2 - 1 
    cmds[:, :, 0] *= 0.40  
    cmds[:, :, 1] *= 0.25  
    cmds[:, :, 2] *= 0.60  
    
    z_roll = z_start
    h_t = torch.zeros((num_samples, backbone.latent_dim), device=device)
    
    for t in range(horizon):
        z_roll, h_t = backbone.predictor(z_roll, cmds[:, t], h_t)
        
    energies = energy_head(z_roll, z_goal)
    best_idx = torch.argmin(energies)
    return cmds[best_idx, 0] # Return just the immediate next action

def pick_backend():
    backend_name = os.getenv("GS_BACKEND", "vulkan").lower()
    if backend_name == "vulkan": return gs.vulkan
    if backend_name in ("amdgpu", "amd", "hip") and hasattr(gs, "amdgpu"): return gs.amdgpu
    return gs.gpu

# -----------------------------
# Main Loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", type=str, default="jepa_checkpoints/jepa_epoch_20.pt")
    parser.add_argument("--head_ckpt", type=str, default="energy_head_checkpoints/energy_head_last.pt")
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=15)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🧠 Loading System 2 JEPA...")
    backbone = TinyQuadJEPA().to(device)
    backbone.load_state_dict(clean_state_dict(torch.load(args.jepa_ckpt, map_location=device)["model_state_dict"]))
    backbone.eval()

    energy_head = GoalEnergyHead().to(device)
    energy_head.load_state_dict(clean_state_dict(torch.load(args.head_ckpt, map_location=device)["energy_head_state_dict"]))
    energy_head.eval()

    print("🌍 Initializing Genesis Physics Environment...")
    gs.init(backend=pick_backend())
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    
    robot = scene.add_entity(gs.morphs.URDF(file="assets/mini_pupper/mini_pupper.urdf", pos=(0,0,0.12), fixed=False))
    
    # We don't render actual vision in the fast MPC loop to keep real-time performance,
    # so we supply a static visual tensor for the network, leaning heavily on proprioception.
    dummy_vision = torch.zeros((3, 64, 64), device=device)

    scene.build(n_envs=1)
    
    JOINTS_ACTUATED = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    name_to_joint = {j.name: j for j in robot.joints}
    act_dofs = torch.tensor([list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED], device=gs.device, dtype=torch.int64)

    # Base pose setup
    hip_splay = 0.06
    q0 = torch.tensor([hip_splay, hip_splay, -hip_splay, -hip_splay, 0.85, 0.85, 0.85, 0.85, -1.75, -1.75, -1.75, -1.75], device=gs.device)
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, act_dofs)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, act_dofs)

    prev_a = torch.zeros((1, 12), device=gs.device)

    print("\n🎮 Beginning MPC Control Loop. Close viewer to exit.")
    
    target_vx = 0.3 # Target forward speed in m/s

    while scene.viewer.is_alive():
        # --- 1. Get Live State ---
        pos = robot.get_pos()
        quat = robot.get_quat()
        vel_b = world_to_body_vec(quat, robot.get_vel())
        ang_b = world_to_body_vec(quat, robot.get_ang())
        q = robot.get_dofs_position(act_dofs)
        dq = robot.get_dofs_velocity(act_dofs)
        q_rel = q - q0.unsqueeze(0)
        
        prop_current = torch.cat([pos[:, 2:3], quat, vel_b, ang_b, q_rel, dq, prev_a], dim=1)[0].to(device)

        # --- 2. Construct the "Goal" State ---
        # We tell the network: "The perfect future is exactly right now, but you are moving forward at 0.3 m/s"
        prop_goal = prop_current.clone()
        prop_goal[5] = target_vx # vel_b_x index

        # --- 3. System 2 Hallucination & Planning ---
        t_plan_start = time.time()
        best_cmd = plan_mpc(
            backbone, energy_head, dummy_vision, prop_current, dummy_vision, prop_goal,
            num_samples=args.samples, horizon=args.horizon, device=device
        )
        t_plan_end = time.time()

        # --- 4. Execute the imagined best command (Translate cmd to joints via System 1 logic) ---
        # Note: In a pure JEPA, you'd have a low-level policy here. For simplicity, we directly 
        # map the winning latent-command into target joints based on your rollout logic.
        
        # Expand chosen cmd to 12 joints (simple heuristic mapping matching rollout scale)
        action_mapping = torch.zeros(12, device=gs.device)
        action_mapping[4:8] = best_cmd[0] * 1.5  # Thighs tied to vx
        action_mapping[0:4] = best_cmd[1] * 0.8  # Hips tied to vy
        action_mapping[8:12] = best_cmd[2] * -1.0 # Calves tied to yaw
        
        q_tgt = q0 + (0.30 * action_mapping)
        
        robot.control_dofs_position(q_tgt.unsqueeze(0), act_dofs)
        prev_a = action_mapping.unsqueeze(0)

        # Step simulation
        for _ in range(4): # decimation = 4
            scene.step()
            
        fps = 1.0 / (t_plan_end - t_plan_start)
        print(f"\r📡 Planning FPS: {fps:.0f} | Imagining {args.samples} futures...", end="")

if __name__ == "__main__":
    main()
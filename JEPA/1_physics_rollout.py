#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
import time
from dataclasses import dataclass
from typing import Any

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
# Correlated Exploration Generator
# -----------------------------
class OUNoiseBatched:
    def __init__(self, n_envs: int, dim: int, device: str, theta: float = 0.15, sigma: float = 0.2):
        self.n_envs = n_envs
        self.dim = dim
        self.device = device
        self.theta = theta
        self.sigma = sigma
        self.state = torch.zeros((n_envs, dim), device=device)

    def step(self) -> torch.Tensor:
        noise = torch.randn((self.n_envs, self.dim), device=self.device)
        self.state = self.state - self.theta * self.state + self.sigma * noise
        return self.state

    def reset(self, env_ids: torch.Tensor):
        self.state[env_ids] = 0.0

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    urdf: str = "assets/mini_pupper/mini_pupper.urdf"
    n_envs: int = 2048
    dt: float = 0.01
    substeps: int = 4
    decimation: int = 4
    kp: float = 5.0
    kv: float = 0.5
    action_scale: float = 0.30
    hip_splay: float = 0.06
    thigh0: float = 0.85
    calf0: float = -1.75
    min_z: float = 0.05
    max_tilt: float = 1.0

# -----------------------------
# Actor-Critic
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hid: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(), 
            nn.Linear(hid, hid), nn.Tanh(), 
            nn.Linear(hid, act_dim)
        )
    def act_deterministic(self, obs: torch.Tensor):
        return torch.tanh(self.actor(obs))

def pick_backend() -> Any:
    backend_name = os.getenv("GS_BACKEND", "vulkan").lower()
    if backend_name == "vulkan":
        return gs.vulkan
    if backend_name in ("amdgpu", "amd", "hip") and hasattr(gs, "amdgpu"):
        return gs.amdgpu
    return gs.gpu

# -----------------------------
# Main Rollout Script
# -----------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs("jepa_raw_data", exist_ok=True)
    cfg = CFG()
    
    gs.init(backend=pick_backend())
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=cfg.urdf, pos=(0,0,0.12), fixed=False))
    scene.build(n_envs=cfg.n_envs)

    # STRICTLY define actuated joints
    JOINTS_ACTUATED = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]
    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)

    hip_L = cfg.hip_splay
    hip_R = -cfg.hip_splay
    q0 = torch.tensor(
        [
            hip_L, hip_L, hip_R, hip_R,
            cfg.thigh0, cfg.thigh0, cfg.thigh0, cfg.thigh0,
            cfg.calf0,  cfg.calf0,  cfg.calf0,  cfg.calf0,
        ],
        device=gs.device,
        dtype=torch.float32,
    )

    robot.set_dofs_kp(torch.ones(12, device=gs.device) * cfg.kp, act_dofs)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * cfg.kv, act_dofs)

    # Load frozen System 1 policy
    model = ActorCritic(50, 12).to(gs.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=gs.device)["model"], strict=False)
    model.eval()

    ou_noise = OUNoiseBatched(cfg.n_envs, 3, gs.device)
    latency_buffer = torch.zeros((2, cfg.n_envs, 3), device=gs.device)
    prev_a = torch.zeros((cfg.n_envs, 12), device=gs.device)

    print(f"\n🚀 Mining Physical Trajectories...")
    print(f"Generating {args.chunks} chunks of {args.steps} steps across {cfg.n_envs} environments.")

    for chunk in range(args.chunks):
        t0 = time.time()
        
        # EXPLICITLY SET DEVICE="CPU" SO WE DON'T BLOW UP VRAM OR CRASH NUMPY
        d_proprio   = torch.zeros((cfg.n_envs, args.steps, 47), dtype=torch.float32, device="cpu")
        d_cmds      = torch.zeros((cfg.n_envs, args.steps, 3), dtype=torch.float32, device="cpu")
        d_dones     = torch.zeros((cfg.n_envs, args.steps), dtype=torch.bool, device="cpu")
        d_base_pos  = torch.zeros((cfg.n_envs, args.steps, 3), dtype=torch.float32, device="cpu")
        d_base_quat = torch.zeros((cfg.n_envs, args.steps, 4), dtype=torch.float32, device="cpu")
        d_joint_pos = torch.zeros((cfg.n_envs, args.steps, 12), dtype=torch.float32, device="cpu")

        for step in range(args.steps):
            raw_cmds = torch.tanh(ou_noise.step())
            scaled_cmds = raw_cmds.clone()
            scaled_cmds[:, 0] *= 0.40
            scaled_cmds[:, 1] *= 0.25
            scaled_cmds[:, 2] *= 0.60

            # Command Latency
            latency_buffer = torch.roll(latency_buffer, shifts=-1, dims=0)
            latency_buffer[-1] = scaled_cmds
            active_cmds = latency_buffer[0]

            pos = robot.get_pos()
            quat = robot.get_quat()
            vel_b = world_to_body_vec(quat, robot.get_vel())
            ang_b = world_to_body_vec(quat, robot.get_ang())
            q = robot.get_dofs_position(act_dofs)
            dq = robot.get_dofs_velocity(act_dofs)
            q_rel = q - q0.unsqueeze(0)

            proprio = torch.cat([pos[:, 2:3], quat, vel_b, ang_b, q_rel, dq, prev_a], dim=1)
            
            # S2W: Inject synthetic IMU/Encoder noise
            noise = torch.randn_like(proprio) * 0.01
            noise[:, 1:5] *= 2.0  
            noise[:, 5:11] *= 5.0 
            proprio_noisy = proprio + noise

            obs = torch.cat([proprio_noisy, active_cmds], dim=1)
            actions = model.act_deterministic(obs)
            prev_a = actions.clone()

            # Step Physics
            q_tgt = q0.unsqueeze(0) + cfg.action_scale * actions
            q_tgt[:, 0:4] = torch.clamp(q_tgt[:, 0:4], -0.8, 0.8)
            q_tgt[:, 4:8] = torch.clamp(q_tgt[:, 4:8], -1.5, 1.5)
            q_tgt[:, 8:12] = torch.clamp(q_tgt[:, 8:12], -2.5, -0.5)

            robot.control_dofs_position(q_tgt, act_dofs)
            for _ in range(cfg.decimation): 
                scene.step()

            # Check falls
            z = pos[:, 2]
            fallen = z < cfg.min_z
            done_ids = torch.nonzero(fallen).squeeze(-1)
            if done_ids.numel() > 0:
                ou_noise.reset(done_ids)
                robot.set_dofs_position(q0.unsqueeze(0).repeat(len(done_ids), 1), act_dofs, envs_idx=done_ids)
                robot.set_dofs_velocity(torch.zeros_like(q0).unsqueeze(0).repeat(len(done_ids), 1), act_dofs, envs_idx=done_ids)

            # Record Data (sending to CPU to sit in RAM)
            d_proprio[:, step] = proprio_noisy.cpu()
            d_cmds[:, step]    = scaled_cmds.cpu()
            d_dones[:, step]   = fallen.cpu()
            d_base_pos[:, step] = pos.cpu()
            d_base_quat[:, step] = quat.cpu()
            d_joint_pos[:, step] = q.cpu()

        # Add .cpu() here just to be 100% immune to PyTorch default device overrides
        np.savez_compressed(
            f"jepa_raw_data/chunk_{chunk:04d}.npz",
            proprio=d_proprio.cpu().numpy(), 
            cmds=d_cmds.cpu().numpy(), 
            dones=d_dones.cpu().numpy(),
            base_pos=d_base_pos.cpu().numpy(), 
            base_quat=d_base_quat.cpu().numpy(), 
            joint_pos=d_joint_pos.cpu().numpy()
        )
        fps = (cfg.n_envs * args.steps) / (time.time() - t0)
        print(f"✅ Saved Chunk {chunk+1}/{args.chunks} | FPS: {fps:,.0f}")

    print("\n✅ Physical Trajectory Mining Complete.")

if __name__ == "__main__":
    main()
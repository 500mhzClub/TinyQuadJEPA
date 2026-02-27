#!/usr/bin/env python3
from __future__ import annotations
"""
Sequential Benchmark Script for Omnidirectional Controller.
Usage:
    python benchmark.py --ckpt runs/pupper_omni_.../ckpt_02000.pt
"""
import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import genesis as gs

# -----------------------------
# Small helpers
# -----------------------------
def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)).strip())

def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)).strip())

def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

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

def quat_to_euler_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)

# -----------------------------
# Config (Matches Training)
# -----------------------------
@dataclass
class CFG:
    urdf: str = os.getenv("URDF", "assets/mini_pupper/mini_pupper.urdf")
    n_envs: int = 2 
    env_spacing: float = env_float("ENV_SPACING", 1.0)
    dt: float = env_float("DT", 0.01)
    substeps: int = env_int("SUBSTEPS", 4)
    decimation: int = env_int("DECIMATION", 4)
    kp: float = env_float("KP", 5.0)
    kv: float = env_float("KV", 0.5)
    max_ep_len: int = env_int("MAX_EP_LEN", 800)
    hip_splay: float = env_float("HIP_SPLAY", 0.06)
    thigh0: float = env_float("THIGH0", 0.85)
    calf0: float = env_float("CALF0", -1.75)
    action_scale: float = env_float("ACTION_SCALE", 0.30)
    min_z: float = env_float("MIN_Z", 0.05)
    max_tilt: float = env_float("MAX_TILT", 1.0)
    z_target: float = env_float("Z_TARGET", 0.085)
    cmd_vx_min: float = env_float("CMD_VX_MIN", -0.40)
    cmd_vx_max: float = env_float("CMD_VX_MAX",  0.60)
    cmd_vy_min: float = env_float("CMD_VY_MIN", -0.30)
    cmd_vy_max: float = env_float("CMD_VY_MAX",  0.30)
    cmd_omega_min: float = env_float("CMD_OMEGA_MIN", -0.80)
    cmd_omega_max: float = env_float("CMD_OMEGA_MAX",  0.80)
    cmd_resample_steps: int = env_int("CMD_RESAMPLE_STEPS", 0) # Hardcoded to 0 for benchmark!

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CFG":
        c = CFG()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c

# -----------------------------
# Mini Pupper batched env
# -----------------------------
class MiniPupperBatched:
    JOINTS_ACTUATED = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    JOINT_LIMITS = {
        "hip": (-0.8, 0.8),
        "thigh": (-1.5, 1.5),
        "calf": (-2.5, -0.5),
    }

    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.device = gs.device
        self.n_envs = int(cfg.n_envs)
        self.num_actions = 12
        self.obs_dim = 50

        self.ep_len = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.prev_action = torch.zeros(self.n_envs, self.num_actions, device=self.device)
        self.commands = torch.zeros(self.n_envs, 3, device=self.device, dtype=torch.float32)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=cfg.substeps),
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                plane_reflection=False,
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
            ),
            renderer=gs.renderers.Rasterizer(),
        )
        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=cfg.urdf,
                pos=(0.0, 0.0, 0.12),
                fixed=False,
                merge_fixed_links=False,
                requires_jac_and_IK=False,
            )
        )

        self.scene.build(
            n_envs=self.n_envs,
            env_spacing=(cfg.env_spacing, cfg.env_spacing),
        )

        name_to_joint = {j.name: j for j in self.robot.joints}
        dof_idx = []
        for jn in self.JOINTS_ACTUATED:
            dofs = list(name_to_joint[jn].dofs_idx_local)
            dof_idx.append(dofs[0])
        self.act_dofs = torch.tensor(dof_idx, device=self.device, dtype=torch.int64)

        hip_L = cfg.hip_splay
        hip_R = -cfg.hip_splay
        self.q0 = torch.tensor(
            [
                hip_L, hip_L, hip_R, hip_R,
                cfg.thigh0, cfg.thigh0, cfg.thigh0, cfg.thigh0,
                cfg.calf0,  cfg.calf0,  cfg.calf0,  cfg.calf0,
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.robot.set_dofs_kp(torch.ones(self.num_actions, device=self.device) * cfg.kp, self.act_dofs)
        self.robot.set_dofs_kv(torch.ones(self.num_actions, device=self.device) * cfg.kv, self.act_dofs)

        self.reset(torch.arange(self.n_envs, device=self.device, dtype=torch.int64))

    def _clamp_joint_targets(self, q: torch.Tensor) -> torch.Tensor:
        q = q.clone()
        q[:, 0:4] = torch.clamp(q[:, 0:4], *self.JOINT_LIMITS["hip"])
        q[:, 4:8] = torch.clamp(q[:, 4:8], *self.JOINT_LIMITS["thigh"])
        q[:, 8:12] = torch.clamp(q[:, 8:12], *self.JOINT_LIMITS["calf"])
        return q

    def set_commands_batch(self, cmds: torch.Tensor) -> None:
        self.commands = cmds.to(device=self.device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor):
        self.scene.reset(envs_idx=env_ids)
        n = int(env_ids.shape[0])

        q_init = self.q0.unsqueeze(0).repeat(n, 1)
        q_init = self._clamp_joint_targets(q_init)

        self.robot.set_dofs_position(q_init, self.act_dofs, envs_idx=env_ids)
        self.robot.set_dofs_velocity(torch.zeros_like(q_init), self.act_dofs, envs_idx=env_ids)

        self.ep_len[env_ids] = 0
        self.prev_action[env_ids] = 0.0

    @torch.no_grad()
    def get_obs(self) -> torch.Tensor:
        pos = self.robot.get_pos()
        quat = self.robot.get_quat()
        vel_w = self.robot.get_vel()
        ang_w = self.robot.get_ang()

        vel_b = world_to_body_vec(quat, vel_w)
        ang_b = world_to_body_vec(quat, ang_w)

        q = self.robot.get_dofs_position(self.act_dofs)
        dq = self.robot.get_dofs_velocity(self.act_dofs)

        z = pos[:, 2:3]
        q_rel = q - self.q0.unsqueeze(0)

        return torch.cat([z, quat, vel_b, ang_b, q_rel, dq, self.prev_action, self.commands], dim=1)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        action = torch.clamp(action, -1.0, 1.0)
        self.prev_action = action

        q_tgt = self.q0.unsqueeze(0) + self.cfg.action_scale * action
        q_tgt = self._clamp_joint_targets(q_tgt)
        self.robot.control_dofs_position(q_tgt, self.act_dofs)

        for _ in range(self.cfg.decimation):
            self.scene.step(update_visualizer=False, refresh_visualizer=False)

        pos = self.robot.get_pos()
        quat = self.robot.get_quat()
        vel_w = self.robot.get_vel()
        ang_w = self.robot.get_ang()

        eul = quat_to_euler_wxyz(quat)
        roll = eul[:, 0]
        pitch = eul[:, 1]

        vel_b = world_to_body_vec(quat, vel_w)
        ang_b = world_to_body_vec(quat, ang_w)

        z = pos[:, 2]

        tilted = (torch.abs(roll) > self.cfg.max_tilt) | (torch.abs(pitch) > self.cfg.max_tilt)
        fallen = z < self.cfg.min_z

        self.ep_len += 1
        done = tilted | fallen

        info = {
            "v_fwd": vel_b[:, 0],
            "v_lat": vel_b[:, 1],
            "yaw_rate": ang_b[:, 2],
        }

        obs = self.get_obs()
        return obs, torch.zeros_like(done, dtype=torch.float), done, info

# -----------------------------
# Actor-Critic
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hid: int = 256):
        super().__init__()
        self.act_dim = act_dim
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hid),
            nn.Tanh(),
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, act_dim),
        )

    def act_deterministic(self, obs: torch.Tensor):
        return torch.tanh(self.actor(obs))

def pick_backend() -> Any:
    backend_name = os.getenv("GS_BACKEND", "vulkan").lower()
    if backend_name == "vulkan":
        return gs.vulkan
    return gs.gpu

# -----------------------------
# Sequential Benchmark Routine
# -----------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--phase-steps", type=int, default=120, help="Steps per command phase")
    parser.add_argument("--warmup", type=int, default=20, help="Steps to ignore at the start of EACH phase for MAE")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = CFG.from_dict(ckpt.get("cfg", {}))
    
    cfg.n_envs = 2 
    cfg.cmd_resample_steps = 0 # FORCIBLY DISABLE RANDOM RESAMPLING

    gs.init(backend=pick_backend())
    env = MiniPupperBatched(cfg)
    device = gs.device

    model = ActorCritic(env.obs_dim, env.num_actions).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    demo_sequence = [
        ("Demo Forward",  [0.40, 0.00, 0.00]),
        ("Demo Backward", [-0.30, 0.00, 0.00]),
        ("Demo Left",     [0.00, 0.25, 0.00]),
        ("Demo Right",    [0.00, -0.25, 0.00]),
        ("Demo Pivot",    [0.00, 0.00, 0.60]),
    ]
    
    max_sequence = [
        ("Max Forward",   [cfg.cmd_vx_max, 0.0, 0.0]),
        ("Max Backward",  [cfg.cmd_vx_min, 0.0, 0.0]),
        ("Max Left",      [0.0, cfg.cmd_vy_max, 0.0]),
        ("Max Right",     [0.0, cfg.cmd_vy_min, 0.0]),
        ("Max Spin CCW",  [0.0, 0.0, cfg.cmd_omega_max]),
    ]

    total_phases = len(demo_sequence)

    print(f"\n🚀 Running True Velocity Benchmark on {args.ckpt}")
    print(f"Tracking ACTUAL signed velocities dynamically. {args.phase_steps} steps per phase.\n")

    for _ in range(40):
        env.robot.control_dofs_position(env.q0.unsqueeze(0).repeat(cfg.n_envs, 1), env.act_dofs)
        env.scene.step(update_visualizer=False, refresh_visualizer=False)

    obs = env.get_obs()
    is_alive = torch.ones(cfg.n_envs, dtype=torch.bool, device=device)
    
    results_demo = []
    results_max = []

    for phase_idx in range(total_phases):
        demo_name, demo_cmd = demo_sequence[phase_idx]
        max_name, max_cmd = max_sequence[phase_idx]
        
        cmds = torch.tensor([demo_cmd, max_cmd], device=device, dtype=torch.float32)
        env.set_commands_batch(cmds)
        obs = env.get_obs()
        
        # Track ACTUAL sum instead of Error sum
        act_vx_sum = torch.zeros(cfg.n_envs, dtype=torch.float32, device=device)
        act_vy_sum = torch.zeros(cfg.n_envs, dtype=torch.float32, device=device)
        act_om_sum = torch.zeros(cfg.n_envs, dtype=torch.float32, device=device)
        valid_steps = torch.zeros(cfg.n_envs, dtype=torch.float32, device=device)

        for step in range(args.phase_steps):
            a = model.act_deterministic(obs)
            obs, _, done, info = env.step(a)
            is_alive &= ~done

            if step > args.warmup: 
                for i in range(2):
                    if is_alive[i]:
                        act_vx_sum[i] += info["v_fwd"][i]
                        act_vy_sum[i] += info["v_lat"][i]
                        act_om_sum[i] += info["yaw_rate"][i]
                        valid_steps[i] += 1.0

        def format_res(idx, name):
            if not is_alive[idx]:
                return f"{name:<15} | N/A (Fell down)"
            if valid_steps[idx] > 0:
                ax = act_vx_sum[idx] / valid_steps[idx]
                ay = act_vy_sum[idx] / valid_steps[idx]
                ao = act_om_sum[idx] / valid_steps[idx]
                
                cmd_str = f"CMD [{cmds[idx,0]:>5.2f}, {cmds[idx,1]:>5.2f}, {cmds[idx,2]:>5.2f}]"
                act_str = f"ACT [{ax.item():>5.2f}, {ay.item():>5.2f}, {ao.item():>5.2f}]"
                return f"{name:<15} | {cmd_str}  ->  {act_str}"
            return f"{name:<15} | N/A"

        results_demo.append(format_res(0, demo_name))
        results_max.append(format_res(1, max_name))

    print("="*85)
    print("ENV 0: CONTINUOUS DEMO SEQUENCE")
    print("-" * 85)
    for r in results_demo: print(r)
    print("="*85)
    print("ENV 1: CONTINUOUS MAX SEQUENCE")
    print("-" * 85)
    for r in results_max: print(r)
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

"""
train_pupper_sys1_omni_velocity_survival.py

System-1: low-level omnidirectional locomotion controller.
- Policy π(a | proprio, cmd) where cmd = desired base velocity in BODY frame: (vx, vy, wz).
- System-2 (world model / planner) should output cmd trajectories and feed them to System-1.

This phase ("VELOCITY SURVIVAL") is anti-statue:
- Commands are (by default) non-trivial, and the agent must actually move (relative-stall termination).
- You can switch to CMD_MODE=external for deployment/integration tests (no internal cmd resampling).

Key Fixes vs your current logs:
1) Robust CMD_MIN_NORM enforcement via rejection sampling (not scale-up of near-zero vectors).
2) CMD_MODE: "random" (training) vs "external" (Sys2 drives commands).
3) Optional idle curriculum (CMD_IDLE_PROB) for later phases (teach standing / no-motion commands).
4) Stall grace window (STALL_GRACE) to avoid instantly nuking episodes before any exploration.
5) Extra diagnostics: cmd_norm stats, cmd_active fraction, stalled fraction.
6) Camera now follows the robot during video recording.

"""

import os
import sys

# ---------------------------------------------------------------------
# Video Backend Setup (headless-safe)
# ---------------------------------------------------------------------
if "--record-only" in sys.argv:
    os.environ.setdefault("PYOPENGL_PLATFORM", os.getenv("VIDEO_PYOPENGL_PLATFORM", "egl"))
    os.environ.setdefault("EGL_PLATFORM", os.getenv("VIDEO_EGL_PLATFORM", "surfaceless"))

import time
import argparse
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

import genesis as gs


# -----------------------------
# Env helpers
# -----------------------------
def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)).strip())

def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)).strip())

def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# -----------------------------
# Math Helpers
# -----------------------------
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
# Config
# -----------------------------
@dataclass
class CFG:
    urdf: str = env_str("URDF", "assets/mini_pupper/mini_pupper.urdf")

    n_envs: int = env_int("N_ENVS", 2048)
    env_spacing: float = env_float("ENV_SPACING", 1.0)

    dt: float = env_float("DT", 0.01)
    substeps: int = env_int("SUBSTEPS", 4)
    decimation: int = env_int("DECIMATION", 4)

    # --- PHYSICS: Soft & Stable ---
    kp: float = env_float("KP", 5.0)
    kv: float = env_float("KV", 0.5)
    action_scale: float = env_float("ACTION_SCALE", 0.30)

    max_ep_len: int = env_int("MAX_EP_LEN", 800)

    hip_splay: float = env_float("HIP_SPLAY", 0.06)
    thigh0: float = env_float("THIGH0", 0.85)
    calf0: float = env_float("CALF0", -1.75)

    # --- COMMAND INTERFACE ---
    # Desired base velocity in BODY frame
    cmd_vx_range: Tuple[float, float] = (-0.4, 0.4)
    cmd_vy_range: Tuple[float, float] = (-0.3, 0.3)
    cmd_wz_range: Tuple[float, float] = (-1.0, 1.0)

    # Command mode:
    # - random: internal random cmd resampling (training)
    # - external: caller sets commands (Sys2->Sys1 integration / eval)
    cmd_mode: str = env_str("CMD_MODE", "random")  # random|external

    # Enforce minimum planar command magnitude during training sampling
    cmd_min_norm: float = env_float("CMD_MIN_NORM", 0.30)
    cmd_hold_steps: int = env_int("CMD_HOLD_STEPS", 200)

    # Optional idle curriculum (later phases): sometimes sample near-zero commands
    cmd_idle_prob: float = env_float("CMD_IDLE_PROB", 0.00)  # 0 in survival phase
    cmd_idle_norm: float = env_float("CMD_IDLE_NORM", 0.05)

    # --- REWARDS ---
    w_track_lin: float = env_float("W_TRACK_LIN", 5.0)
    w_track_ang: float = env_float("W_TRACK_ANG", 2.0)
    tracking_sigma: float = env_float("TRACKING_SIGMA", 120.0)

    w_along: float = env_float("W_ALONG", 3.0)
    v_along_clip: float = env_float("V_ALONG_CLIP", 0.60)

    w_upright: float = env_float("W_UPRIGHT", 0.10)

    w_height: float = env_float("W_HEIGHT", 0.10)
    z_target: float = env_float("Z_TARGET", 0.085)
    z_sigma: float = env_float("Z_SIGMA", 200.0)

    w_energy: float = env_float("W_ENERGY", 1e-4)
    w_action: float = env_float("W_ACTION", 1e-3)
    w_smooth: float = env_float("W_SMOOTH", 1e-3)

    # Optional per-step stall penalty (helps early learning)
    w_stall_step: float = env_float("W_STALL_STEP", 0.00)

    fall_penalty: float = env_float("FALL_PENALTY", 0.10)

    # Termination thresholds
    min_z: float = env_float("MIN_Z", 0.05)
    max_tilt: float = env_float("MAX_TILT", 1.0)

    # STALL TERMINATION (relative to cmd magnitude)
    stall_cmd_thresh: float = env_float("STALL_CMD_THRESH", 0.20)
    stall_frac: float = env_float("STALL_FRAC", 0.35)
    stall_timeout_steps: int = env_int("STALL_TIMEOUT", 40)
    stall_grace_steps: int = env_int("STALL_GRACE", 80)

    # Training
    seed: int = env_int("SEED", 1)
    total_updates: int = env_int("UPDATES", 20000)
    horizon: int = env_int("HORIZON", 32)
    gamma: float = env_float("GAMMA", 0.99)
    lam: float = env_float("LAMBDA", 0.95)
    clip: float = env_float("CLIP", 0.2)
    lr: float = env_float("LR", 3e-4)
    vf_coef: float = env_float("VF_COEF", 0.5)
    ent_coef: float = env_float("ENT_COEF", 0.01)
    max_grad_norm: float = env_float("MAX_GRAD_NORM", 1.0)

    ppo_epochs: int = env_int("PPO_EPOCHS", 4)
    minibatch_size: int = env_int("MINIBATCH", 65536)

    out_dir: str = env_str("OUT_DIR", f"runs/sys1_omni_vel_{now_tag()}")
    save_every: int = env_int("SAVE_EVERY", 100)
    video_every: int = env_int("VIDEO_EVERY", 200)
    video_steps: int = env_int("VIDEO_STEPS", 600)
    video_fps: int = env_int("VIDEO_FPS", 30)
    video_w: int = env_int("VIDEO_W", 640)
    video_h: int = env_int("VIDEO_H", 480)
    record_video: bool = env_bool("RECORD_VIDEO", "1")

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CFG":
        c = CFG()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


# -----------------------------
# Mini Pupper Omni Environment
# -----------------------------
class MiniPupperOmniBatched:
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

    TERM_FALL = 0
    TERM_TILT = 1
    TERM_STALL = 2
    TERM_TO = 3

    def __init__(self, cfg: CFG, with_camera: bool = False, auto_reset: bool = True):
        self.cfg = cfg
        self.with_camera = with_camera
        self.auto_reset = auto_reset
        self.device = gs.device

        self.n_envs = int(cfg.n_envs)
        self.num_actions = 12

        self.ep_len = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.stall_counters = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.cmd_timers = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)

        self.prev_action = torch.zeros(self.n_envs, self.num_actions, device=self.device)
        self.commands = torch.zeros(self.n_envs, 3, device=self.device, dtype=torch.float32)

        # termination accounting (since last stats fetch)
        self._term_counts = torch.zeros(4, device=self.device, dtype=torch.int64)
        self._term_total = torch.zeros(1, device=self.device, dtype=torch.int64)

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

        self.cam = None
        if with_camera:
            self.cam = self.scene.add_camera(
                res=(cfg.video_w, cfg.video_h),
                pos=(0.8, -0.8, 0.45),
                lookat=(0.0, 0.0, 0.12),
                fov=50,
                GUI=False,
            )

        self.scene.build(
            n_envs=self.n_envs,
            env_spacing=(cfg.env_spacing, cfg.env_spacing),
        )

        name_to_joint = {j.name: j for j in self.robot.joints}
        dof_idx = []
        for jn in self.JOINTS_ACTUATED:
            j = name_to_joint[jn]
            dofs = list(j.dofs_idx_local)
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

        # cmd normalization for obs only
        self.vx_max = max(abs(cfg.cmd_vx_range[0]), abs(cfg.cmd_vx_range[1]))
        self.vy_max = max(abs(cfg.cmd_vy_range[0]), abs(cfg.cmd_vy_range[1]))
        self.wz_max = max(abs(cfg.cmd_wz_range[0]), abs(cfg.cmd_wz_range[1]))

        self.obs_dim = 50  # unchanged: we include normalized cmds but still 3 dims
        self.reset(torch.arange(self.n_envs, device=self.device, dtype=torch.int64))

    def _clamp_joint_targets(self, q: torch.Tensor) -> torch.Tensor:
        q = q.clone()
        q[:, 0:4] = torch.clamp(q[:, 0:4], *self.JOINT_LIMITS["hip"])
        q[:, 4:8] = torch.clamp(q[:, 4:8], *self.JOINT_LIMITS["thigh"])
        q[:, 8:12] = torch.clamp(q[:, 8:12], *self.JOINT_LIMITS["calf"])
        return q

    def _sample_cmd(self, n: int) -> torch.Tensor:
        """
        Robust command sampling:
        - With probability CMD_IDLE_PROB, sample near-zero planar command (standing cases).
        - Otherwise, rejection-sample (vx,vy) until ||(vx,vy)|| >= CMD_MIN_NORM.
        """
        vx_lo, vx_hi = self.cfg.cmd_vx_range
        vy_lo, vy_hi = self.cfg.cmd_vy_range
        wz_lo, wz_hi = self.cfg.cmd_wz_range

        cmd = torch.empty(n, 3, device=self.device, dtype=torch.float32)

        # yaw-rate always uniform
        cmd[:, 2] = torch.rand(n, device=self.device) * (wz_hi - wz_lo) + wz_lo

        idle = torch.rand(n, device=self.device) < float(self.cfg.cmd_idle_prob)
        if bool(idle.any()):
            # near-zero planar commands
            k = int(idle.sum().item())
            vx = (torch.rand(k, device=self.device) * 2.0 - 1.0) * float(self.cfg.cmd_idle_norm)
            vy = (torch.rand(k, device=self.device) * 2.0 - 1.0) * float(self.cfg.cmd_idle_norm)
            cmd[idle, 0] = torch.clamp(vx, vx_lo, vx_hi)
            cmd[idle, 1] = torch.clamp(vy, vy_lo, vy_hi)

        active = ~idle
        if bool(active.any()):
            ids = torch.nonzero(active).squeeze(-1)
            m = ids.numel()
            min_v = float(self.cfg.cmd_min_norm)

            vx = torch.rand(m, device=self.device) * (vx_hi - vx_lo) + vx_lo
            vy = torch.rand(m, device=self.device) * (vy_hi - vy_lo) + vy_lo

            # rejection loop for low-norm planar commands
            for _ in range(12):
                v = torch.sqrt(vx * vx + vy * vy)
                bad = v < min_v
                if not bool(bad.any()):
                    break
                kb = int(bad.sum().item())
                vx[bad] = torch.rand(kb, device=self.device) * (vx_hi - vx_lo) + vx_lo
                vy[bad] = torch.rand(kb, device=self.device) * (vy_hi - vy_lo) + vy_lo

            # final fallback for any remaining bad (rare): set random direction at min_v and clamp
            v = torch.sqrt(vx * vx + vy * vy)
            bad = v < min_v
            if bool(bad.any()):
                theta = (torch.rand(int(bad.sum().item()), device=self.device) * 2.0 - 1.0) * np.pi
                vx_f = min_v * torch.cos(theta)
                vy_f = min_v * torch.sin(theta)
                vx[bad] = torch.clamp(vx_f, vx_lo, vx_hi)
                vy[bad] = torch.clamp(vy_f, vy_lo, vy_hi)

            cmd[ids, 0] = vx
            cmd[ids, 1] = vy

        return cmd

    @torch.no_grad()
    def set_commands(self, cmd: torch.Tensor, env_ids: Optional[torch.Tensor] = None, enforce_limits: bool = True):
        """
        External command injection (Sys2 -> Sys1):
        cmd: (N,3) or (3,) in BODY frame units (m/s, m/s, rad/s).
        """
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)

        if env_ids is None:
            if cmd.shape[0] == 1:
                cmd = cmd.repeat(self.n_envs, 1)
            elif cmd.shape[0] != self.n_envs:
                raise ValueError(f"cmd batch {cmd.shape[0]} != n_envs {self.n_envs}")
            tgt_ids = None
        else:
            tgt_ids = env_ids
            if cmd.shape[0] == 1:
                cmd = cmd.repeat(int(env_ids.numel()), 1)
            elif cmd.shape[0] != int(env_ids.numel()):
                raise ValueError("cmd batch must match env_ids length (or be single row)")

        if enforce_limits:
            vx_lo, vx_hi = self.cfg.cmd_vx_range
            vy_lo, vy_hi = self.cfg.cmd_vy_range
            wz_lo, wz_hi = self.cfg.cmd_wz_range
            cmd = cmd.clone()
            cmd[:, 0] = torch.clamp(cmd[:, 0], vx_lo, vx_hi)
            cmd[:, 1] = torch.clamp(cmd[:, 1], vy_lo, vy_hi)
            cmd[:, 2] = torch.clamp(cmd[:, 2], wz_lo, wz_hi)

        if tgt_ids is None:
            self.commands[:] = cmd.to(self.device)
        else:
            self.commands[tgt_ids] = cmd.to(self.device)

    def reset(self, env_ids: torch.Tensor):
        self.scene.reset(envs_idx=env_ids)
        n = int(env_ids.shape[0])

        noise = (torch.rand(n, self.num_actions, device=self.device) - 0.5) * 0.08
        q_init = self.q0.unsqueeze(0).repeat(n, 1) + noise
        q_init = self._clamp_joint_targets(q_init)

        self.robot.set_dofs_position(q_init, self.act_dofs, envs_idx=env_ids)
        self.robot.set_dofs_velocity(torch.zeros_like(q_init), self.act_dofs, envs_idx=env_ids)

        self.ep_len[env_ids] = 0
        self.stall_counters[env_ids] = 0
        self.cmd_timers[env_ids] = 0
        self.prev_action[env_ids] = 0.0

        if self.cfg.cmd_mode.lower() == "external":
            self.commands[env_ids] = 0.0
        else:
            self.commands[env_ids] = self._sample_cmd(n)

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

        # normalized command for conditioning (still 3 dims)
        cmd_obs = self.commands.clone()
        cmd_obs[:, 0] /= max(1e-6, self.vx_max)
        cmd_obs[:, 1] /= max(1e-6, self.vy_max)
        cmd_obs[:, 2] /= max(1e-6, self.wz_max)

        return torch.cat([z, quat, vel_b, ang_b, q_rel, dq, self.prev_action, cmd_obs], dim=1)

    @torch.no_grad()
    def fetch_and_reset_term_stats(self) -> Dict[str, float]:
        total = int(self._term_total.item())
        out = {
            "total": float(total),
            "fall": float(self._term_counts[self.TERM_FALL].item()),
            "tilt": float(self._term_counts[self.TERM_TILT].item()),
            "stall": float(self._term_counts[self.TERM_STALL].item()),
            "to": float(self._term_counts[self.TERM_TO].item()),
        }
        self._term_counts.zero_()
        self._term_total.zero_()
        return out

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = torch.clamp(action, -1.0, 1.0)
        prev_a = self.prev_action
        self.prev_action = action

        # Internal cmd resampling only in random mode
        if self.cfg.cmd_mode.lower() != "external":
            self.cmd_timers += 1
            resample_mask = self.cmd_timers >= int(self.cfg.cmd_hold_steps)
            if bool(resample_mask.any()):
                ids = torch.nonzero(resample_mask).squeeze(-1)
                self.commands[ids] = self._sample_cmd(int(ids.numel()))
                self.cmd_timers[ids] = 0

        q_tgt = self.q0.unsqueeze(0) + self.cfg.action_scale * action
        q_tgt = self._clamp_joint_targets(q_tgt)

        self.robot.control_dofs_position(q_tgt, self.act_dofs)

        update_vis = bool(self.with_camera)
        for _ in range(self.cfg.decimation):
            self.scene.step(update_visualizer=update_vis, refresh_visualizer=update_vis)

        pos = self.robot.get_pos()
        quat = self.robot.get_quat()
        vel_w = self.robot.get_vel()
        ang_w = self.robot.get_ang()

        vel_b = world_to_body_vec(quat, vel_w)
        ang_b = world_to_body_vec(quat, ang_w)

        v_xy = vel_b[:, :2]
        w_z = ang_b[:, 2]
        z = pos[:, 2]

        cmd_xy = self.commands[:, :2]
        cmd_wz = self.commands[:, 2]

        eul = quat_to_euler_wxyz(quat)
        roll = eul[:, 0]
        pitch = eul[:, 1]

        # -----------------------------
        # Rewards
        # -----------------------------
        lin_err = torch.sum((v_xy - cmd_xy) ** 2, dim=1)
        r_track_lin = self.cfg.w_track_lin * torch.exp(-self.cfg.tracking_sigma * lin_err)

        ang_err = (w_z - cmd_wz) ** 2
        r_track_ang = self.cfg.w_track_ang * torch.exp(-self.cfg.tracking_sigma * ang_err)

        cmd_norm = torch.norm(cmd_xy, dim=1).clamp_min(1e-6)
        cmd_active = cmd_norm > float(self.cfg.stall_cmd_thresh)

        # Along-command reward only when cmd is active
        cmd_unit = cmd_xy / cmd_norm.unsqueeze(1)
        v_along = torch.sum(v_xy * cmd_unit, dim=1)
        r_along = self.cfg.w_along * torch.clamp(v_along / float(self.cfg.v_along_clip), -1.0, 1.0)
        r_along = torch.where(cmd_active, r_along, torch.zeros_like(r_along))

        upright = torch.exp(-10.0 * (roll * roll + pitch * pitch))
        r_upright = self.cfg.w_upright * upright

        r_height = self.cfg.w_height * torch.exp(-self.cfg.z_sigma * (z - float(self.cfg.z_target)) ** 2)

        dq = self.robot.get_dofs_velocity(self.act_dofs)
        p_energy = self.cfg.w_energy * torch.sum(dq * dq, dim=1)
        p_action = self.cfg.w_action * torch.sum(action * action, dim=1)
        p_smooth = self.cfg.w_smooth * torch.sum((action - prev_a) ** 2, dim=1)

        reward = (r_track_lin + r_track_ang + r_along + r_upright + r_height) - (p_energy + p_action + p_smooth)

        # -----------------------------
        # Terminations (including stall)
        # -----------------------------
        tilted = (torch.abs(roll) > self.cfg.max_tilt) | (torch.abs(pitch) > self.cfg.max_tilt)
        fallen = z < self.cfg.min_z
        time_out = self.ep_len >= self.cfg.max_ep_len

        v_norm = torch.norm(v_xy, dim=1)
        vel_low_rel = v_norm < (float(self.cfg.stall_frac) * cmd_norm)
        stall_candidate = cmd_active & vel_low_rel & (self.ep_len > int(self.cfg.stall_grace_steps))

        self.stall_counters = torch.where(stall_candidate, self.stall_counters + 1, torch.zeros_like(self.stall_counters))
        stalled_out = self.stall_counters > int(self.cfg.stall_timeout_steps)

        if float(self.cfg.w_stall_step) > 0.0:
            reward = reward - float(self.cfg.w_stall_step) * stall_candidate.float()

        done = tilted | fallen | time_out | stalled_out

        death = (tilted | fallen | stalled_out).float()
        reward = reward - self.cfg.fall_penalty * death

        # Count termination types at termination time
        done_ids = torch.nonzero(done).squeeze(-1)
        if done_ids.numel() > 0:
            self._term_total += done_ids.numel()
            self._term_counts[self.TERM_FALL] += fallen[done_ids].to(torch.int64).sum()
            self._term_counts[self.TERM_TILT] += tilted[done_ids].to(torch.int64).sum()
            self._term_counts[self.TERM_STALL] += stalled_out[done_ids].to(torch.int64).sum()
            self._term_counts[self.TERM_TO] += time_out[done_ids].to(torch.int64).sum()

        self.ep_len += 1

        if self.auto_reset and done_ids.numel() > 0:
            self.reset(done_ids)

        obs = self.get_obs()
        return obs, reward, done


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
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hid),
            nn.Tanh(),
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def _dist(self, obs: torch.Tensor):
        mu = self.actor(obs)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).unsqueeze(0)
        return torch.distributions.Normal(mu, std)

    def act(self, obs: torch.Tensor):
        dist = self._dist(obs)
        u = dist.rsample()
        a = torch.tanh(u)
        logp_u = dist.log_prob(u).sum(-1)
        log_det = torch.sum(torch.log(1.0 - a * a + 1e-6), dim=-1)
        logp = logp_u - log_det
        v = self.critic(obs).squeeze(-1)
        ent = dist.entropy().sum(-1)
        return a, logp, v, ent

    def eval_actions(self, obs: torch.Tensor, act: torch.Tensor):
        dist = self._dist(obs)
        u = atanh(act)
        logp_u = dist.log_prob(u).sum(-1)
        log_det = torch.sum(torch.log(1.0 - act * act + 1e-6), dim=-1)
        logp = logp_u - log_det
        ent = dist.entropy().sum(-1)
        v = self.critic(obs).squeeze(-1)
        return logp, ent, v

    def act_deterministic(self, obs: torch.Tensor):
        return torch.tanh(self.actor(obs))


# -----------------------------
# Backend selection
# -----------------------------
def pick_backend() -> Any:
    backend_name = env_str("GS_BACKEND", "vulkan").lower()
    if backend_name == "vulkan":
        return gs.vulkan
    if backend_name in ("amdgpu", "amd", "hip") and hasattr(gs, "amdgpu"):
        return gs.amdgpu
    return gs.gpu


# -----------------------------
# Video recording (record-only subprocess)
# -----------------------------
@torch.no_grad()
def record_video_from_ckpt(ckpt_path: str, out_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = CFG.from_dict(ckpt.get("cfg", {}))
    cfg.n_envs = 1
    cfg.cmd_mode = "external"  # IMPORTANT: do not let internal sampler overwrite test commands

    gs.init(backend=pick_backend())

    env = MiniPupperOmniBatched(cfg, with_camera=True, auto_reset=False)
    device = gs.device

    model = ActorCritic(env.obs_dim, env.num_actions).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("[video] settling physics...")
    for _ in range(40):
        env.robot.control_dofs_position(env.q0.unsqueeze(0), env.act_dofs)
        env.scene.step(update_visualizer=False, refresh_visualizer=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    env.cam.start_recording()
    try:
        print("[video] recording omni behaviors...")
        for i in range(cfg.video_steps):
            # scripted cmd segments (body frame)
            if i < 200:
                env.set_commands(torch.tensor([0.30, 0.00, 0.00], device=device))
            elif i < 400:
                env.set_commands(torch.tensor([0.20, 0.00, 0.80], device=device))
            else:
                env.set_commands(torch.tensor([0.00, -0.20, 0.00], device=device))

            obs = env.get_obs()
            a = model.act_deterministic(obs)
            _, _, done = env.step(a)

            # --- Camera Follow Logic ---
            # Get robot position (index 0)
            base_pos = env.robot.get_pos()[0] # (3,) tensor
            
            # Calculate desired camera position (maintain relative offset)
            # Original static pos was (0.8, -0.8, 0.45) looking at (0,0,0.12)
            # We treat (0.8, -0.8, 0.45) as the offset vector
            cam_offset = torch.tensor([0.8, -0.8, 0.45], device=device)
            lookat_offset = torch.tensor([0.0, 0.0, 0.12], device=device)

            cam_pos = base_pos + cam_offset
            cam_lookat = base_pos + lookat_offset

            env.cam.set_pose(
                pos=cam_pos.cpu().numpy(),
                lookat=cam_lookat.cpu().numpy()
            )
            # ---------------------------

            env.cam.render()
            if bool(done.item()):
                break

        env.cam.stop_recording(save_to_filename=out_path, fps=cfg.video_fps)
        print(f"[video] wrote {out_path}")
        return 0
    except Exception as e:
        print(f"[video] record FAILED ({type(e).__name__}): {e}")
        traceback.print_exc()
        return 2


def spawn_record_video(ckpt_path: str, out_path: str):
    envp = os.environ.copy()
    try_list = envp.get("VIDEO_TRY_PLATFORMS", "egl,glx,osmesa").split(",")
    try_list = [x.strip() for x in try_list if x.strip()]

    for plat in try_list:
        envp["VIDEO_PYOPENGL_PLATFORM"] = plat
        if plat == "egl":
            envp.setdefault("VIDEO_EGL_PLATFORM", "surfaceless")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--record-only",
            "--ckpt",
            str(Path(ckpt_path).resolve()),
            "--out",
            str(Path(out_path).resolve()),
        ]
        p = subprocess.run(cmd, env=envp, check=False)
        if p.returncode == 0:
            return
        print(f"[video] failed with PYOPENGL_PLATFORM={plat} (rc={p.returncode}); trying next...")

    print("[video] all backends failed (training continues).")


# -----------------------------
# PPO training loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--out", type=str, default="rollout.mp4")
    args = parser.parse_args()

    if args.record_only:
        rc = record_video_from_ckpt(args.ckpt, args.out)
        raise SystemExit(rc)

    cfg = CFG()
    os.makedirs(cfg.out_dir, exist_ok=True)

    gs.init(backend=pick_backend())

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"[cfg] CMD_MODE={cfg.cmd_mode} CMD_MIN_NORM={cfg.cmd_min_norm} CMD_IDLE_PROB={cfg.cmd_idle_prob}")
    print(f"[cfg] STALL: thresh={cfg.stall_cmd_thresh} frac={cfg.stall_frac} timeout={cfg.stall_timeout_steps} grace={cfg.stall_grace_steps}")

    env = MiniPupperOmniBatched(cfg, with_camera=False, auto_reset=True)
    device = gs.device

    model = ActorCritic(env.obs_dim, env.num_actions).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    T = int(cfg.horizon)
    N = int(cfg.n_envs)
    obs_dim = int(env.obs_dim)
    act_dim = int(env.num_actions)

    obs_buf = torch.zeros(T, N, obs_dim, device=device)
    act_buf = torch.zeros(T, N, act_dim, device=device)
    logp_buf = torch.zeros(T, N, device=device)
    rew_buf = torch.zeros(T, N, device=device)
    done_buf = torch.zeros(T, N, device=device)
    val_buf = torch.zeros(T, N, device=device)

    obs = env.get_obs()

    global_steps = 0
    t0 = time.time()

    for update in range(1, cfg.total_updates + 1):
        model.train()

        with torch.no_grad():
            for t in range(T):
                a, logp, v, _ = model.act(obs)

                obs_buf[t].copy_(obs)
                act_buf[t].copy_(a)
                logp_buf[t].copy_(logp)
                val_buf[t].copy_(v)

                obs, r, d = env.step(a)
                rew_buf[t].copy_(r)
                done_buf[t].copy_(d.float())

                global_steps += N

            v_last = model.critic(obs).squeeze(-1)

        adv = torch.zeros(T, N, device=device)
        last_gae = torch.zeros(N, device=device)

        for t in reversed(range(T)):
            nonterminal = 1.0 - done_buf[t]
            next_val = v_last if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_val * nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.lam * nonterminal * last_gae
            adv[t] = last_gae

        ret = adv + val_buf

        b_obs = obs_buf.reshape(T * N, obs_dim)
        b_act = act_buf.reshape(T * N, act_dim)
        b_logp = logp_buf.reshape(T * N)
        b_adv = adv.reshape(T * N)
        b_ret = ret.reshape(T * N)
        b_val = val_buf.reshape(T * N)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = T * N
        idx = torch.randperm(batch_size, device=device)

        for _epoch in range(cfg.ppo_epochs):
            for start in range(0, batch_size, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                mb_obs = b_obs[mb]
                mb_act = b_act[mb]
                mb_old_logp = b_logp[mb]
                mb_adv = b_adv[mb]
                mb_ret = b_ret[mb]
                mb_old_val = b_val[mb]

                new_logp, ent, v = model.eval_actions(mb_obs, mb_act)

                ratio = torch.exp(new_logp - mb_old_logp)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
                pg_loss = torch.max(pg1, pg2).mean()

                v_clipped = mb_old_val + torch.clamp(v - mb_old_val, -cfg.clip, cfg.clip)
                vf1 = (v - mb_ret) ** 2
                vf2 = (v_clipped - mb_ret) ** 2
                vf_loss = 0.5 * torch.max(vf1, vf2).mean()

                ent_loss = ent.mean()
                loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent_loss

                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optim.step()

        dt_s = max(1e-6, time.time() - t0)
        env_fps = global_steps / dt_s
        mean_rew = rew_buf.mean().item()

        # diagnostics
        quat = env.robot.get_quat()
        vel_w = env.robot.get_vel()
        vel_b = world_to_body_vec(quat, vel_w)
        v_xy = vel_b[:, :2]
        cmd_xy = env.commands[:, :2]

        err_x = (v_xy[:, 0] - cmd_xy[:, 0]).abs().mean().item()
        err_y = (v_xy[:, 1] - cmd_xy[:, 1]).abs().mean().item()
        mean_z = env.robot.get_pos()[:, 2].mean().item()
        vmag = torch.norm(v_xy, dim=1).mean().item()

        cmd_norm = torch.norm(cmd_xy, dim=1)
        cmag = cmd_norm.mean().item()
        cmin = cmd_norm.min().item()
        cmax = cmd_norm.max().item()
        cmd_active_frac = (cmd_norm > float(cfg.stall_cmd_thresh)).float().mean().item()

        stalled_frac = ((vmag * torch.ones_like(cmd_norm)) < (float(cfg.stall_frac) * cmd_norm)).float().mean().item()

        ep_len_mean = float(env.ep_len.float().mean().item())

        term = env.fetch_and_reset_term_stats()
        ttot = max(1.0, term["total"])
        term_str = f"term(fall={100*term['fall']/ttot:.2f}%, tilt={100*term['tilt']/ttot:.2f}%, stall={100*term['stall']/ttot:.2f}%, to={100*term['to']/ttot:.2f}%)"

        print(
            f"upd={update:05d}  env_fps={env_fps:10.0f}  "
            f"mean_rew={mean_rew:+.3f}  ErrX={err_x:.3f}  ErrY={err_y:.3f}  "
            f"z={mean_z:+.3f}  |v|={vmag:.3f}  |cmd|={cmag:.3f} (min={cmin:.3f} max={cmax:.3f})  "
            f"cmd_act={cmd_active_frac*100:5.1f}%  stalled~={stalled_frac*100:5.1f}%  "
            f"ep_len={ep_len_mean:6.1f}  {term_str}"
        )

        if (update % cfg.save_every) == 0:
            ckpt = {"update": update, "cfg": cfg.__dict__, "model": model.state_dict(), "optim": optim.state_dict()}
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{update:05d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"💾 saved {ckpt_path}")

        if cfg.record_video and (update % cfg.video_every) == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{update:05d}.pt")
            vid_path = os.path.join(cfg.out_dir, f"video_{update:05d}.mp4")

            if not os.path.exists(ckpt_path):
                ckpt = {"update": update, "cfg": cfg.__dict__, "model": model.state_dict(), "optim": optim.state_dict()}
                torch.save(ckpt, ckpt_path)

            spawn_record_video(ckpt_path, vid_path)


if __name__ == "__main__":
    main()
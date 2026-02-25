#!/usr/bin/env python3
from __future__ import annotations

"""
train_blind.py

Fixes applied vs previous version:
1. Spawn height auto-calibration: measures actual resting height before training.
2. Physics settle added to reset(): 20 steps of position-hold per reset.
3. stand_updates=150 (pure standing phase before velocity tracking).
4. ramp_updates=600 (longer curriculum ramp).
5. Video recording: no longer breaks on done; resets and continues.
6. obs updated correctly in video loop.
7. Minor: upright/height reward weights increased (1.0 each) for stronger
   standing signal during curriculum.
"""

import os
import sys
import math
import random
import argparse
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Video Backend Setup (headless-safe)
if "--record-only" in sys.argv:
    os.environ.setdefault("PYOPENGL_PLATFORM", os.getenv("VIDEO_PYOPENGL_PLATFORM", "egl"))
    os.environ.setdefault("EGL_PLATFORM", os.getenv("VIDEO_EGL_PLATFORM", "surfaceless"))

import genesis as gs


# -----------------------------
# Env helpers
# -----------------------------
def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)).strip())

def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)).strip())

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()

def _env_tuple2(name: str, default: Tuple[float, float]) -> Tuple[float, float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    for sep in (",", " "):
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            if len(parts) == 2:
                return (float(parts[0]), float(parts[1]))
    s = float(raw)
    return (s, s)

def now_tag() -> str:
    import time
    return time.strftime("%Y%m%d_%H%M%S")

def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# STS3215 Actuator Model
# -----------------------------
class STS3215_Actuator:
    def __init__(self, num_envs: int, device: torch.device, dt: float = 0.01):
        self.device = device
        self.num_envs = num_envs
        self.stall_torque = 3.0
        self.no_load_speed = 6.0
        self.dt = dt
        self.latency_steps = max(1, int(0.02 / dt))
        self.history_len = self.latency_steps + 1
        self.command_queue = torch.zeros((num_envs, 12, self.history_len), device=device)
        self.kp = 45.0
        self.kd = 1.5

    def step(self, target_pos, current_pos, current_vel, voltage: float = 11.1):
        with torch.no_grad():
            if self.history_len > 1:
                self.command_queue[:, :, :-1] = self.command_queue[:, :, 1:].clone()
            self.command_queue[:, :, -1] = target_pos.detach()
            delayed_target = self.command_queue[:, :, 0]

        torque = self.kp * (delayed_target - current_pos) - self.kd * current_vel
        torque_limit = (self.stall_torque * (voltage / 11.1)) * (1.0 - torch.abs(current_vel) / self.no_load_speed)
        torque_limit = torque_limit.clamp_min(0.0)
        return torch.clamp(torque, -torque_limit, torque_limit)

    def reset(self, env_ids: torch.Tensor | None, default_pos: torch.Tensor):
        if env_ids is None:
            self.command_queue[:] = default_pos.view(1, 12, 1).expand_as(self.command_queue)
        else:
            self.command_queue[env_ids] = default_pos.view(1, 12, 1).expand(
                len(env_ids), 12, self.history_len
            )


# -----------------------------
# Math Helpers
# -----------------------------
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
# PPO network & Normalizer
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, log_std_init: float):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs):
        mu = self.pi(obs)
        log_std = self.log_std.expand(obs.shape[0], -1)
        v = self.v(obs).squeeze(-1)
        return mu, log_std, v

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        mu, log_std, v = self.forward(obs)
        a = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        logp = (
            -0.5 * (
                ((a - mu) ** 2) / (log_std.exp() ** 2 + 1e-8)
                + 2.0 * log_std
                + math.log(2.0 * math.pi)
            )
        ).sum(dim=-1)
        return a, logp, v


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(1e-4, device=device)

    @torch.no_grad()
    def update(self, x):
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        c = x.shape[0]
        delta = m - self.mean
        tot = self.count + c
        self.mean += delta * (c / tot)
        self.var = (
            (self.var * self.count + v * c + delta ** 2 * (self.count * c / tot)) / tot
        ).clamp_min(1e-6)
        self.count = tot

    @torch.no_grad()
    def normalize(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10, 10)


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    envs: int = _env_int("ENVS", 1024)
    dt: float = _env_float("DT", 0.01)
    env_spacing: Tuple[float, float] = _env_tuple2("ENV_SPACING", (2.5, 2.5))

    action_scale: float = _env_float("ACTION_SCALE", 0.65)
    force_limit: float = _env_float("FORCE_LIMIT", 30.0)

    # --- COMMAND INTERFACE ---
    cmd_vx_range: Tuple[float, float] = _env_tuple2("CMD_VX_RANGE", (0.2, 0.6))
    cmd_vy_range: Tuple[float, float] = _env_tuple2("CMD_VY_RANGE", (-0.05, 0.05))
    cmd_wz_range: Tuple[float, float] = _env_tuple2("CMD_WZ_RANGE", (-0.2, 0.2))

    cmd_mode: str = _env_str("CMD_MODE", "random")
    cmd_hold_steps: int = _env_int("CMD_HOLD_STEPS", 200)

    # --- REWARDS & CURRICULUM ---
    # FIX: stand_updates=150 gives a pure standing phase before velocity tracking
    # FIX: ramp_updates=600 for a gentler tracking curriculum ramp
    curriculum: bool = _env_bool("CURRICULUM", "1")
    stand_updates: int = _env_int("STAND_UPDATES", 150)
    ramp_updates: int = _env_int("FWD_RAMP_UPDATES", 600)
    upright_decay: float = _env_float("UPRIGHT_DECAY", 0.3)

    w_track_lin: float = _env_float("W_TRACK_LIN", 5.0)
    w_track_ang: float = _env_float("W_TRACK_ANG", 2.5)
    tracking_sigma_lin: float = _env_float("TRACKING_SIGMA_LIN", 0.25)
    tracking_sigma_ang: float = _env_float("TRACKING_SIGMA_ANG", 0.25)

    # FIX: increased posture reward weights from 0.5 to 1.0 for stronger standing signal
    w_upright: float = _env_float("W_UPRIGHT", 1.0)
    w_height: float = _env_float("W_HEIGHT", 1.0)

    pose_penalty: float = _env_float("POSE_PENALTY", 1e-4)
    alive_bonus: float = _env_float("ALIVE_BONUS", 0.0)

    # --- TERMINATIONS ---
    fall_h: float = _env_float("FALL_H", 0.10)
    max_tilt: float = _env_float("MAX_TILT", 1.0)
    ep_len: int = _env_int("EP_LEN", 800)

    stall_cmd_thresh: float = _env_float("STALL_CMD_THRESH", 0.20)
    stall_frac: float = _env_float("STALL_FRAC", 0.0)
    stall_timeout_steps: int = _env_int("STALL_TIMEOUT", 40)
    stall_grace_steps: int = _env_int("STALL_GRACE", 80)

    # --- RESET SETTLING ---
    # FIX: number of physics steps to run at default pose after each reset
    reset_settle_steps: int = _env_int("RESET_SETTLE_STEPS", 20)

    # --- PPO ---
    hidden: int = _env_int("HIDDEN", 256)
    log_std_init: float = _env_float("LOG_STD_INIT", -1.0)
    lr: float = _env_float("LR", 3e-4)
    gamma: float = _env_float("GAMMA", 0.99)
    lam: float = _env_float("LAMBDA", 0.95)
    clip: float = _env_float("CLIP", 0.2)
    vf_coef: float = _env_float("VF_COEF", 0.5)
    ent_coef: float = _env_float("ENT_COEF", 0.02)
    update_epochs: int = _env_int("UPDATE_EPOCHS", 4)
    rollout_T: int = _env_int("ROLLOUT_T", 256)
    total_updates: int = _env_int("UPDATES", 10000)

    kick_after_updates: int = _env_int("KICK_AFTER_UPDATES", 500)
    clone_rollout: bool = _env_bool("CLONE_ROLLOUT", "1")

    seed: int = _env_int("SEED", 0)
    urdf: str = _env_str("URDF", "./assets/mini_pupper/mini_pupper.urdf")

    # Video output
    out_dir: str = _env_str("OUT_DIR", f"runs/sys1_omni_{now_tag()}")
    save_every: int = _env_int("SAVE_EVERY", 100)
    video_every: int = _env_int("VIDEO_EVERY", 200)
    video_steps: int = _env_int("VIDEO_STEPS", 600)
    video_fps: int = _env_int("VIDEO_FPS", 30)
    video_w: int = _env_int("VIDEO_W", 640)
    video_h: int = _env_int("VIDEO_H", 480)
    record_video: bool = _env_bool("RECORD_VIDEO", "1")

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CFG":
        c = CFG()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


# -----------------------------
# Genesis Env
# -----------------------------
class BlindWalkerOmniEnv:
    TERM_FALL = 0
    TERM_TILT = 1
    TERM_STALL = 2
    TERM_TO = 3

    def __init__(self, cfg: CFG, device: torch.device, with_camera: bool = False):
        self.cfg = cfg
        self.device = device
        self._global_update = 0
        self.with_camera = with_camera

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=2),
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                plane_reflection=False,
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
            ),
            rigid_options=gs.options.RigidOptions(gravity=(0, 0, -9.81)),
            renderer=gs.renderers.Rasterizer() if with_camera else None,
        )
        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=cfg.urdf, pos=(0, 0, 0.18)),
            material=gs.materials.Rigid(),
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

        self.scene.build(n_envs=cfg.envs, env_spacing=cfg.env_spacing)

        self.base_dofs = torch.arange(0, 6, device=device)
        self.motor_dofs = torch.arange(6, 18, device=device)

        self.robot.set_dofs_kp(torch.zeros(12, device=device), self.motor_dofs)
        self.robot.set_dofs_kv(torch.zeros(12, device=device), self.motor_dofs)

        f_lim = torch.ones(12, device=device) * cfg.force_limit
        self.robot.set_dofs_force_range(-f_lim, f_lim, self.motor_dofs)

        self.base_link = self.robot.get_link("base_link")

        self.default_dof_pos = torch.tensor([0.0, 0.6, -1.2] * 4, device=device)

        # -------------------------------------------------------
        # FIX: Auto-calibrate spawn height by measuring where the
        # robot actually settles with default joint angles.
        # -------------------------------------------------------
        self.base_pos0, self.base_quat0 = self._calibrate_spawn_height()

        self.actuator = STS3215_Actuator(cfg.envs, device, dt=cfg.dt)
        self.last_action = torch.zeros((cfg.envs, 12), device=device)

        self.episode_length = torch.zeros(cfg.envs, device=device, dtype=torch.int32)
        self.stall_counters = torch.zeros(cfg.envs, device=device, dtype=torch.int32)

        self.commands = torch.zeros(cfg.envs, 3, device=device, dtype=torch.float32)
        self.cmd_timers = torch.zeros(cfg.envs, device=device, dtype=torch.int32)
        self.vx_max = max(abs(cfg.cmd_vx_range[0]), abs(cfg.cmd_vx_range[1]))
        self.vy_max = max(abs(cfg.cmd_vy_range[0]), abs(cfg.cmd_vy_range[1]))
        self.wz_max = max(abs(cfg.cmd_wz_range[0]), abs(cfg.cmd_wz_range[1]))

        self._term_counts = torch.zeros(4, device=device, dtype=torch.int64)
        self._term_total = torch.zeros(1, device=device, dtype=torch.int64)

        self.reset(torch.arange(cfg.envs, device=device))

    # -------------------------------------------------------
    # FIX: Spawn height calibration
    # -------------------------------------------------------
    def _calibrate_spawn_height(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Set all joints to default_dof_pos, run 200 settle steps, measure
        the resulting base_link height and use that as base_pos0.
        This prevents the 'drop from sky' issue caused by a mismatch between
        the URDF spawn z and the actual resting height.
        """
        print("[calibration] Measuring actual resting height...")

        # Hard-set joints to default pose
        all_envs = torch.arange(self.cfg.envs, device=self.device)
        q_default = self.default_dof_pos.unsqueeze(0).expand(self.cfg.envs, -1)
        self.robot.set_dofs_position(q_default, self.motor_dofs)

        # Zero all velocities
        try:
            self.robot.set_dofs_velocity(
                torch.zeros((self.cfg.envs, 6), device=self.device),
                self.base_dofs,
            )
        except Exception:
            pass
        self.robot.set_dofs_velocity(
            torch.zeros((self.cfg.envs, 12), device=self.device),
            self.motor_dofs,
        )

        # Run settle steps holding joint positions
        for _ in range(200):
            self.robot.control_dofs_position(q_default, self.motor_dofs)
            self.scene.step(update_visualizer=False, refresh_visualizer=False)

        h = self.base_link.get_pos()[:, 2].mean().item()
        # Clamp to a sane range in case calibration misfires
        h = float(max(0.10, min(0.35, h)))
        print(f"[calibration] Actual resting height: {h:.4f} m  (using as base_pos0 z)")

        base_pos0 = torch.tensor([0.0, 0.0, h], device=self.device)
        base_quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        return base_pos0, base_quat0

    def set_update(self, upd: int):
        self._global_update = upd

    @torch.no_grad()
    def fetch_and_reset_term_stats(self) -> Dict[str, float]:
        total = max(1, int(self._term_total.item()))
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

    def _sample_cmd(self, n: int) -> torch.Tensor:
        vx_lo, vx_hi = self.cfg.cmd_vx_range
        vy_lo, vy_hi = self.cfg.cmd_vy_range
        wz_lo, wz_hi = self.cfg.cmd_wz_range

        cmd = torch.empty(n, 3, device=self.device, dtype=torch.float32)
        cmd[:, 0] = torch.rand(n, device=self.device) * (vx_hi - vx_lo) + vx_lo
        cmd[:, 1] = torch.rand(n, device=self.device) * (vy_hi - vy_lo) + vy_lo
        cmd[:, 2] = torch.rand(n, device=self.device) * (wz_hi - wz_lo) + wz_lo
        return cmd

    @torch.no_grad()
    def set_commands(self, cmd: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)

        if env_ids is None:
            if cmd.shape[0] == 1:
                cmd = cmd.repeat(self.cfg.envs, 1)
            self.commands[:] = cmd.to(self.device)
        else:
            if cmd.shape[0] == 1:
                cmd = cmd.repeat(int(env_ids.numel()), 1)
            self.commands[env_ids] = cmd.to(self.device)

    def _set_base_pose(self, env_ids: torch.Tensor):
        pos = self.base_pos0.expand(len(env_ids), 3).clone()
        quat = self.base_quat0.expand(len(env_ids), 4).clone()

        try:
            self.base_link.set_pos(pos, envs_idx=env_ids)
            self.base_link.set_quat(quat, envs_idx=env_ids)
            return
        except Exception:
            pass

        try:
            self.robot.set_pos(pos, envs_idx=env_ids)
            self.robot.set_quat(quat, envs_idx=env_ids)
        except Exception:
            pass

    def _zero_velocities(self, env_ids: torch.Tensor):
        try:
            self.robot.set_dofs_velocity(
                torch.zeros((len(env_ids), 6), device=self.device),
                self.base_dofs,
                envs_idx=env_ids,
            )
        except Exception:
            pass

        self.robot.set_dofs_velocity(
            torch.zeros((len(env_ids), 12), device=self.device),
            self.motor_dofs,
            envs_idx=env_ids,
        )

    def _kick_base_velocity(self, env_ids: torch.Tensor):
        if not hasattr(self.robot, "set_dofs_velocity"):
            return

        base_vel = torch.zeros((len(env_ids), 6), device=self.device)
        base_vel[:, 0] = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * 0.5
        base_vel[:, 5] = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * 0.5
        try:
            self.robot.set_dofs_velocity(base_vel, self.base_dofs, envs_idx=env_ids)
        except Exception:
            pass

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self.episode_length[env_ids] = 0
        self.stall_counters[env_ids] = 0
        self.cmd_timers[env_ids] = 0
        self.last_action[env_ids] = 0.0

        if self.cfg.cmd_mode.lower() != "external":
            self.commands[env_ids] = self._sample_cmd(len(env_ids))

        self._set_base_pose(env_ids)
        self._zero_velocities(env_ids)

        q = self.default_dof_pos.unsqueeze(0) + (
            torch.rand((len(env_ids), 12), device=self.device) * 2 - 1
        ) * 0.10
        self.robot.set_dofs_position(q, self.motor_dofs, envs_idx=env_ids)

        self.actuator.reset(env_ids, self.default_dof_pos)

        # -------------------------------------------------------
        # FIX: Physics settle - hold default pose for N steps so
        # the robot lands properly before the episode begins.
        # -------------------------------------------------------
        if self.cfg.reset_settle_steps > 0:
            settle_q = self.default_dof_pos.unsqueeze(0).expand(len(env_ids), -1)
            for _ in range(self.cfg.reset_settle_steps):
                self.robot.control_dofs_position(
                    settle_q, self.motor_dofs, envs_idx=env_ids
                )
                self.scene.step(update_visualizer=False, refresh_visualizer=False)

        if self._global_update >= self.cfg.kick_after_updates:
            self._kick_base_velocity(env_ids)

    def get_obs(self):
        q = self.robot.get_dofs_position(self.motor_dofs)
        dq = self.robot.get_dofs_velocity(self.motor_dofs)

        pos = self.base_link.get_pos()
        quat = self.base_link.get_quat()

        # World frame velocities
        v_lin_w = self.base_link.get_vel()
        v_ang_w = self.base_link.get_ang()

        # Body frame velocities
        v_lin_b = world_to_body_vec(quat, v_lin_w)
        v_ang_b = world_to_body_vec(quat, v_ang_w)

        up_world = torch.tensor([0, 0, 1], device=self.device, dtype=v_lin_w.dtype).expand_as(v_lin_w)
        up_body = quat_rotate_wxyz(quat, up_world)

        cmd_obs = self.commands.clone()
        cmd_obs[:, 0] /= max(1e-6, self.vx_max)
        cmd_obs[:, 1] /= max(1e-6, self.vy_max)
        cmd_obs[:, 2] /= max(1e-6, self.wz_max)

        # obs shape: 49
        return torch.cat(
            [v_ang_b, up_body, v_lin_b, pos[:, 2:3], q - self.default_dof_pos, dq, self.last_action, cmd_obs],
            dim=-1,
        )

    def step(self, action: torch.Tensor, w_track: float):
        self.last_action = torch.tanh(action)
        target = self.default_dof_pos + self.cfg.action_scale * self.last_action

        if self.cfg.cmd_mode.lower() != "external":
            self.cmd_timers += 1
            resample_mask = self.cmd_timers >= int(self.cfg.cmd_hold_steps)
            if bool(resample_mask.any()):
                ids = torch.nonzero(resample_mask).squeeze(-1)
                self.commands[ids] = self._sample_cmd(int(ids.numel()))
                self.cmd_timers[ids] = 0

        q = self.robot.get_dofs_position(self.motor_dofs)
        dq = self.robot.get_dofs_velocity(self.motor_dofs)

        torque = self.actuator.step(target, q, dq)
        self.robot.control_dofs_force(torque, self.motor_dofs)

        update_vis = bool(self.with_camera)
        self.scene.step(update_visualizer=update_vis, refresh_visualizer=update_vis)

        obs = self.get_obs()
        v_ang_b = obs[:, 0:3]
        up_body = obs[:, 3:6]
        v_lin_b = obs[:, 6:9]
        h = obs[:, 9]

        cmd_xy = self.commands[:, :2]
        cmd_wz = self.commands[:, 2]

        v_xy = v_lin_b[:, :2]
        w_z = v_ang_b[:, 2]

        # Tracking Rewards
        lin_err_sq = torch.sum((v_xy - cmd_xy) ** 2, dim=1)
        sigma_lin_sq = self.cfg.tracking_sigma_lin ** 2
        r_track_lin = self.cfg.w_track_lin * torch.exp(-lin_err_sq / (2.0 * max(1e-8, sigma_lin_sq)))

        ang_err_sq = (w_z - cmd_wz) ** 2
        sigma_ang_sq = self.cfg.tracking_sigma_ang ** 2
        r_track_ang = self.cfg.w_track_ang * torch.exp(-ang_err_sq / (2.0 * max(1e-8, sigma_ang_sq)))

        w_track_eff = (
            0.0
            if (self.cfg.curriculum and self._global_update < self.cfg.stand_updates)
            else w_track
        )

        upright = up_body[:, 2].clamp(-1, 1)
        upright_term = torch.exp(-((1.0 - upright) ** 2) / 0.1)
        height_term = torch.exp(-((h - self.base_pos0[2].item()) ** 2) / 0.004)

        # Posture gate: zeroes tracking reward when the robot is not upright/at height
        tracking_gate = (upright_term * height_term).detach()
        r_track_lin = r_track_lin * tracking_gate
        r_track_ang = r_track_ang * tracking_gate

        pose_cost = (q - self.default_dof_pos).pow(2).mean(dim=-1)

        # FIX: use cfg.w_upright / cfg.w_height (default 1.0 each, up from 0.5)
        reward = (
            self.cfg.alive_bonus
            + (w_track_eff * r_track_lin)
            + (w_track_eff * r_track_ang)
            + self.cfg.w_upright * upright_term
            + self.cfg.w_height * height_term
            - (self.cfg.pose_penalty * pose_cost)
        )

        if self.cfg.upright_decay > 0:
            decay = math.exp(-self.cfg.upright_decay * float(self._global_update))
            reward = reward - (1.0 - decay) * self.cfg.w_upright * (1.0 - upright_term)

        # Terminations
        quat = self.base_link.get_quat()
        eul = quat_to_euler_wxyz(quat)
        roll = eul[:, 0]
        pitch = eul[:, 1]

        tilted = (torch.abs(roll) > self.cfg.max_tilt) | (torch.abs(pitch) > self.cfg.max_tilt)
        fallen = h < self.cfg.fall_h
        time_out = self.episode_length >= self.cfg.ep_len

        v_norm = torch.norm(v_xy, dim=1)
        cmd_norm = torch.norm(cmd_xy, dim=1).clamp_min(1e-6)
        cmd_active = cmd_norm > float(self.cfg.stall_cmd_thresh)
        vel_low_rel = v_norm < (float(self.cfg.stall_frac) * cmd_norm)

        if w_track_eff < 0.99:
            stall_candidate = torch.zeros_like(cmd_active)
        else:
            stall_candidate = (
                cmd_active & vel_low_rel & (self.episode_length > int(self.cfg.stall_grace_steps))
            )

        self.stall_counters = torch.where(
            stall_candidate,
            self.stall_counters + 1,
            torch.zeros_like(self.stall_counters),
        )
        stalled_out = self.stall_counters > int(self.cfg.stall_timeout_steps)

        done = tilted | fallen | time_out | stalled_out

        done_ids = torch.nonzero(done).squeeze(-1)
        if done_ids.numel() > 0:
            self._term_total += done_ids.numel()
            self._term_counts[self.TERM_FALL] += fallen[done_ids].to(torch.int64).sum()
            self._term_counts[self.TERM_TILT] += tilted[done_ids].to(torch.int64).sum()
            self._term_counts[self.TERM_STALL] += stalled_out[done_ids].to(torch.int64).sum()
            self._term_counts[self.TERM_TO] += time_out[done_ids].to(torch.int64).sum()

        self.episode_length += 1

        if done_ids.numel() > 0:
            self.reset(done_ids)

        err_x = (v_xy[:, 0] - cmd_xy[:, 0]).abs()
        err_y = (v_xy[:, 1] - cmd_xy[:, 1]).abs()

        info = {
            "v_norm": v_norm,
            "cmd_norm": cmd_norm,
            "h": h,
            "upright": upright,
            "err_x": err_x,
            "err_y": err_y,
            "cmd_active": cmd_active.float(),
            "stalled": vel_low_rel.float(),
            "ep_len": self.episode_length.float(),
        }
        return obs, reward, done.float(), info


# -----------------------------
# PPO update
# -----------------------------
def ppo_update(
    model: ActorCritic,
    opt: torch.optim.Optimizer,
    rms: RunningMeanStd,
    obs, act, lp0, rew, done, val, last_val,
    cfg: CFG,
):
    B, T = obs.shape[0], obs.shape[1]

    with torch.no_grad():
        last_val = last_val.detach() if hasattr(last_val, "detach") else last_val

        adv = torch.zeros((B, T), device=obs.device)
        ret = torch.zeros((B, T), device=obs.device)
        gae = torch.zeros((B,), device=obs.device)

        for t in reversed(range(T)):
            mask = 1.0 - done[:, t]
            delta = rew[:, t] + cfg.gamma * last_val * mask - val[:, t]
            gae = delta + cfg.gamma * cfg.lam * mask * gae
            adv[:, t] = gae
            ret[:, t] = gae + val[:, t]
            last_val = val[:, t]

        obs_flat = obs.reshape(-1, obs.shape[-1]).detach()
        act_flat = act.reshape(-1, act.shape[-1]).detach()
        lp0_flat = lp0.reshape(-1).detach()
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)

        rms.update(obs_flat)
        obs_n = rms.normalize(obs_flat)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    for _ in range(cfg.update_epochs):
        mu, log_std, v = model(obs_n)
        std = log_std.exp()

        lp = (
            -0.5 * (
                ((act_flat - mu) ** 2) / (std ** 2 + 1e-8)
                + 2.0 * log_std
                + math.log(2.0 * math.pi)
            )
        ).sum(-1)

        ratio = torch.exp(lp - lp0_flat)
        clipped = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)

        pg_loss = -torch.min(ratio * adv_flat, clipped * adv_flat).mean()
        vf_loss = F.mse_loss(v, ret_flat)
        ent = (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1).mean()

        loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


# -----------------------------
# Video Recording
# -----------------------------
@torch.no_grad()
def record_video_from_ckpt(ckpt_path: str, out_path: str, device: torch.device) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = CFG.from_dict(ckpt.get("cfg", {}))
    cfg.envs = 1
    cfg.cmd_mode = "external"
    cfg.reset_settle_steps = 0  # calibration settles for us

    env = BlindWalkerOmniEnv(cfg, device, with_camera=True)

    obs_dim = env.get_obs().shape[-1]
    model = ActorCritic(obs_dim, 12, cfg.hidden, cfg.log_std_init).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rms = RunningMeanStd((obs_dim,), device)
    if "rms_mean" in ckpt:
        rms.mean = ckpt["rms_mean"].to(device)
        rms.var = ckpt["rms_var"].to(device)
        rms.count = ckpt["rms_count"].to(device)

    print("[video] settling physics...")
    for _ in range(40):
        env.robot.control_dofs_position(env.default_dof_pos.unsqueeze(0), env.motor_dofs)
        env.scene.step(update_visualizer=False, refresh_visualizer=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    env.cam.start_recording()
    try:
        print("[video] recording omni behaviors...")
        obs = env.get_obs()

        for i in range(cfg.video_steps):
            if i < 200:
                env.set_commands(torch.tensor([0.40, 0.00, 0.00], device=device))
            elif i < 400:
                env.set_commands(torch.tensor([0.20, 0.00, 1.00], device=device))
            else:
                env.set_commands(torch.tensor([0.00, -0.25, 0.00], device=device))

            a, _, _ = model.act(rms.normalize(obs), deterministic=True)
            next_obs, _, done, _ = env.step(a, 1.0)

            base_pos = env.base_link.get_pos()[0]
            cam_offset = torch.tensor([1.2, -1.2, 0.6], device=device)
            lookat_offset = torch.tensor([0.0, 0.0, 0.12], device=device)

            env.cam.set_pose(
                pos=(base_pos + cam_offset).cpu().numpy(),
                lookat=(base_pos + lookat_offset).cpu().numpy(),
            )
            env.cam.render()

            # FIX: on fall/termination, reset and keep recording instead of breaking
            if bool(done.item()):
                env.reset(torch.tensor([0], device=device))
                obs = env.get_obs()
            else:
                obs = next_obs

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
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--out", type=str, default="rollout.mp4")
    args = parser.parse_args()

    cfg = CFG()

    gs.init(backend=gs.vulkan)
    device = torch.device("cuda")

    if args.record_only:
        rc = record_video_from_ckpt(args.ckpt, args.out, device)
        raise SystemExit(rc)

    seed_all(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    env = BlindWalkerOmniEnv(cfg, device)
    env.set_update(0)

    obs = env.get_obs()
    obs_dim = obs.shape[-1]
    act_dim = 12

    model = ActorCritic(obs_dim, act_dim, cfg.hidden, cfg.log_std_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rms = RunningMeanStd((obs_dim,), device)

    def store(x: torch.Tensor) -> torch.Tensor:
        return x.detach().clone() if cfg.clone_rollout else x.detach()

    print(f"Starting Omni-Tracking training with STS3215 Actuator...")
    print(f"Spawn height: {env.base_pos0[2].item():.4f} m")
    print(f"Tracking Sigmas: Lin={cfg.tracking_sigma_lin}, Ang={cfg.tracking_sigma_ang}")
    print(f"Curriculum: stand_updates={cfg.stand_updates}, ramp_updates={cfg.ramp_updates}")

    import time
    t0 = time.time()
    global_steps = 0

    for upd in range(1, cfg.total_updates + 1):
        env.set_update(upd)

        # Curriculum track weight: 0 during stand phase, then ramp to 1
        if upd < cfg.stand_updates:
            w_track = 0.0
        else:
            w_track = min(1.0, (upd - cfg.stand_updates) / max(1, cfg.ramp_updates))

        obs_b, act_b, lp_b, rew_b, don_b, val_b = [], [], [], [], [], []
        info_dict = {
            "v_norm": [], "cmd_norm": [], "h": [], "err_x": [], "err_y": [],
            "cmd_active": [], "stalled": [], "ep_len": [],
        }

        for _ in range(cfg.rollout_T):
            with torch.no_grad():
                obs_in = rms.normalize(obs)
                a, lp, v = model.act(obs_in)

            next_obs, r, d, info = env.step(a, w_track)

            obs_b.append(store(obs))
            act_b.append(store(a))
            lp_b.append(store(lp))
            rew_b.append(store(r))
            don_b.append(store(d))
            val_b.append(store(v))

            for k in info_dict.keys():
                if k in info:
                    info_dict[k].append(info[k])

            obs = next_obs
            global_steps += cfg.envs

        with torch.no_grad():
            last_val = model(rms.normalize(obs))[2].detach()

        ppo_update(
            model, opt, rms,
            torch.stack(obs_b, 1),
            torch.stack(act_b, 1),
            torch.stack(lp_b, 1),
            torch.stack(rew_b, 1),
            torch.stack(don_b, 1),
            torch.stack(val_b, 1),
            last_val,
            cfg,
        )

        dt_s = max(1e-6, time.time() - t0)
        env_fps = global_steps / dt_s

        if upd % 10 == 0:
            mean_rew = torch.stack(rew_b, 0).mean().item()
            vmag = torch.stack(info_dict["v_norm"]).mean().item()

            cmd_t = torch.stack(info_dict["cmd_norm"])
            cmag = cmd_t.mean().item()
            cmin = cmd_t.min().item()
            cmax = cmd_t.max().item()

            err_x = torch.stack(info_dict["err_x"]).mean().item()
            err_y = torch.stack(info_dict["err_y"]).mean().item()
            mean_h = torch.stack(info_dict["h"]).mean().item()

            cmd_act = torch.stack(info_dict["cmd_active"]).mean().item() * 100
            stalled = torch.stack(info_dict["stalled"]).mean().item() * 100
            ep_len_mean = torch.stack(info_dict["ep_len"]).mean().item()

            term = env.fetch_and_reset_term_stats()
            ttot = max(1.0, term["total"])
            term_str = (
                f"term(fall={100*term['fall']/ttot:.1f}%, "
                f"tilt={100*term['tilt']/ttot:.1f}%, "
                f"stall={100*term['stall']/ttot:.1f}%, "
                f"to={100*term['to']/ttot:.1f}%)"
            )

            phase = "STAND" if upd < cfg.stand_updates else f"RAMP({w_track:.2f})"

            print(
                f"upd={upd:05d}  [{phase}]  env_fps={env_fps:10.0f}  "
                f"mean_rew={mean_rew:+.3f}  ErrX={err_x:.3f}  ErrY={err_y:.3f}  "
                f"h={mean_h:.3f}  |v|={vmag:.3f}  |cmd|={cmag:.3f} (min={cmin:.3f} max={cmax:.3f})  "
                f"cmd_act={cmd_act:5.1f}%  stalled~={stalled:5.1f}%  "
                f"ep_len={ep_len_mean:6.1f}  {term_str}"
            )

        if (upd % cfg.save_every) == 0:
            ckpt = {
                "update": upd,
                "cfg": cfg.__dict__,
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "rms_mean": rms.mean.cpu(),
                "rms_var": rms.var.cpu(),
                "rms_count": rms.count.cpu(),
            }
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{upd:05d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"💾 saved {ckpt_path}")

        if cfg.record_video and (upd % cfg.video_every) == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{upd:05d}.pt")
            vid_path = os.path.join(cfg.out_dir, f"video_{upd:05d}.mp4")

            if not os.path.exists(ckpt_path):
                ckpt = {
                    "update": upd,
                    "cfg": cfg.__dict__,
                    "model": model.state_dict(),
                    "optim": opt.state_dict(),
                    "rms_mean": rms.mean.cpu(),
                    "rms_var": rms.var.cpu(),
                    "rms_count": rms.count.cpu(),
                }
                torch.save(ckpt, ckpt_path)

            spawn_record_video(ckpt_path, vid_path)


if __name__ == "__main__":
    main()
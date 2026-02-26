#!/usr/bin/env python3
from __future__ import annotations
"""
System 1 Omnidirectional Controller (cmd_vx, cmd_vy, cmd_omega in BODY frame).

Key knobs:
- N_ENVS: parallel training envs (e.g. 2048 / 4096 on 32GB VRAM)
- VIDEO_EVERY: record progress video every N PPO UPDATES (default 50)
- VIDEO_CMD_SWITCH: in the demo video, switch command every N SIM steps (default 120)
- VIDEO_ENVS: number of envs in the record-only subprocess (default 1; training still uses N_ENVS)

VIDEO_FOLLOW=1 \
VIDEO_CAM_LOCK_YAW=1 \
VIDEO_CAM_DIST=-0.80 \
VIDEO_CAM_SIDE=-0.80 \
VIDEO_CAM_HEIGHT=0.80 \
VIDEO_CAM_LOOKAHEAD=0.00 \
VIDEO_CAM_LOOK_Z=0.10 \
VIDEO_CAM_SMOOTH=0.80 \
VIDEO_ENVS=1 \
N_ENVS=12288 \
VIDEO_EVERY=50 \
RESUME=runs/pupper_omni_20260225_150134/ckpt_02700.pt \
VIDEO_CMD_SWITCH=120 \
VIDEO_ENVS=1 \
python sim/train_blind.py
"""
import os
import sys

# ---------------------------------------------------------------------
# MUST be set BEFORE importing genesis/pyrender/OpenGL in the record subprocess
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

def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def apply_env_overrides(cfg) -> None:
    # ints
    if "VIDEO_EVERY" in os.environ:      cfg.video_every = int(os.environ["VIDEO_EVERY"])
    if "VIDEO_STEPS" in os.environ:      cfg.video_steps = int(os.environ["VIDEO_STEPS"])
    if "VIDEO_CMD_SWITCH" in os.environ: cfg.video_cmd_switch = int(os.environ["VIDEO_CMD_SWITCH"])
    if "VIDEO_ENVS" in os.environ:       cfg.video_envs = int(os.environ["VIDEO_ENVS"])
    if "VIDEO_W" in os.environ:          cfg.video_w = int(os.environ["VIDEO_W"])
    if "VIDEO_H" in os.environ:          cfg.video_h = int(os.environ["VIDEO_H"])
    if "VIDEO_FPS" in os.environ:        cfg.video_fps = int(os.environ["VIDEO_FPS"])

    # bools
    if "VIDEO_FOLLOW" in os.environ:
        cfg.video_follow = os.environ["VIDEO_FOLLOW"].strip().lower() in ("1", "true", "yes", "y", "on")
    if "VIDEO_CAM_LOCK_YAW" in os.environ:
        cfg.video_cam_lock_yaw = os.environ["VIDEO_CAM_LOCK_YAW"].strip().lower() in ("1", "true", "yes", "y", "on")

    # floats
    float_map = {
        "VIDEO_CAM_DIST": "video_cam_dist",
        "VIDEO_CAM_SIDE": "video_cam_side",
        "VIDEO_CAM_HEIGHT": "video_cam_height",
        "VIDEO_CAM_LOOKAHEAD": "video_cam_lookahead",
        "VIDEO_CAM_LOOK_Z": "video_cam_look_z",
        "VIDEO_CAM_SMOOTH": "video_cam_smooth",
    }
    for envk, attr in float_map.items():
        if envk in os.environ:
            setattr(cfg, attr, float(os.environ[envk]))

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def _follow_cam_update(cam, robot_pos_xyz, robot_quat_wxyz, cfg, state: dict):
    """
    Chase cam: behind robot, aligned to yaw, smoothed.
    state holds 'pos' and 'look' numpy arrays across frames.
    """
    # robot_pos_xyz: (3,) torch on device or cpu
    # robot_quat_wxyz: (4,) torch on device or cpu
    if isinstance(robot_pos_xyz, torch.Tensor):
        rp = robot_pos_xyz.detach().float().cpu().numpy()
    else:
        rp = np.asarray(robot_pos_xyz, dtype=np.float32)

    if isinstance(robot_quat_wxyz, torch.Tensor):
        q = robot_quat_wxyz.detach().float().cpu().unsqueeze(0)
        yaw = float(quat_to_euler_wxyz(q)[:, 2].item())
    else:
        q = torch.tensor(robot_quat_wxyz, dtype=torch.float32).unsqueeze(0)
        yaw = float(quat_to_euler_wxyz(q)[:, 2].item())

    # Choose basis: robot-yaw-relative OR world-locked (straight)
    if getattr(cfg, "video_cam_lock_yaw", False):
        # World axes (X forward, Y left). Camera angle stays stable.
        fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        right = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)

    desired_pos = (
        rp
        - fwd * float(cfg.video_cam_dist)
        + right * float(cfg.video_cam_side)
        + np.array([0.0, 0.0, float(cfg.video_cam_height)], dtype=np.float32)
    )
    desired_look = (
        rp
        + fwd * float(cfg.video_cam_lookahead)
        + np.array([0.0, 0.0, float(cfg.video_cam_look_z)], dtype=np.float32)
    )

    a = float(cfg.video_cam_smooth)
    if state.get("pos") is None:
        state["pos"] = desired_pos
        state["look"] = desired_look
    else:
        state["pos"] = a * state["pos"] + (1.0 - a) * desired_pos
        state["look"] = a * state["look"] + (1.0 - a) * desired_look

    # Keep the horizon level: force world-up
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Genesis versions differ slightly in arg names; try the common ones.
    try:
        cam.set_pose(pos=state["pos"], lookat=state["look"], up=up)
    except TypeError:
        try:
            cam.set_pose(pos=state["pos"], lookat=state["look"], up_vector=up)
        except TypeError:
            # Fallback: at least update pos/lookat (may still roll)
            cam.set_pose(pos=state["pos"], lookat=state["look"])

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
    urdf: str = os.getenv("URDF", "assets/mini_pupper/mini_pupper.urdf")

    # Parallel training envs (THIS is what you scale on 32GB VRAM)
    n_envs: int = env_int("N_ENVS", 2048)
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

    # Omnidirectional command ranges (BODY frame)
    cmd_vx_min: float = env_float("CMD_VX_MIN", -0.40)
    cmd_vx_max: float = env_float("CMD_VX_MAX",  0.60)
    cmd_vy_min: float = env_float("CMD_VY_MIN", -0.30)
    cmd_vy_max: float = env_float("CMD_VY_MAX",  0.30)
    cmd_omega_min: float = env_float("CMD_OMEGA_MIN", -0.80)
    cmd_omega_max: float = env_float("CMD_OMEGA_MAX",  0.80)

    # Resample commands mid-episode (forces responsiveness)
    cmd_resample_steps: int = env_int("CMD_RESAMPLE_STEPS", 200)  # 0 disables

    # reward weights
    w_tracking: float = env_float("W_TRACKING", 3.0)
    w_upright: float = env_float("W_UPRIGHT", 0.25)
    w_height: float = env_float("W_HEIGHT", 0.10)

    # penalties
    w_energy: float = env_float("W_ENERGY", 2e-4)
    w_action: float = env_float("W_ACTION", 1e-3)
    w_smooth: float = env_float("W_SMOOTH", 2e-3)

    # anti-stall (command-aware)
    stall_grace: int = env_int("STALL_GRACE", 20)
    w_stall: float = env_float("W_STALL", 0.5)
    stall_terminate: int = env_int("STALL_TERMINATE", 200)

    cmd_lin_thresh: float = env_float("CMD_LIN_THRESH", 0.20)
    cmd_yaw_thresh: float = env_float("CMD_YAW_THRESH", 0.35)
    lin_speed_thresh: float = env_float("LIN_SPEED_THRESH", 0.08)
    yaw_rate_thresh: float = env_float("YAW_RATE_THRESH", 0.20)

    fall_penalty: float = env_float("FALL_PENALTY", 5.0)

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

    out_dir: str = os.getenv("OUT_DIR", f"runs/pupper_omni_{now_tag()}")
    save_every: int = env_int("SAVE_EVERY", 100)

    # IMPORTANT: progress video frequency in TRAINING ITERATIONS (PPO updates)
    video_every: int = env_int("VIDEO_EVERY", 50)

    # record-only subprocess config
    video_envs: int = env_int("VIDEO_ENVS", 1)      # rendering envs (usually 1)
    video_steps: int = env_int("VIDEO_STEPS", 600)  # sim steps inside the video
    video_cmd_switch: int = env_int("VIDEO_CMD_SWITCH", 120)  # switch demo command every N sim steps

    video_fps: int = env_int("VIDEO_FPS", 30)
    video_w: int = env_int("VIDEO_W", 640)
    video_h: int = env_int("VIDEO_H", 480)
    record_video: bool = env_bool("RECORD_VIDEO", "1")

        # --- video camera follow ---
    video_follow: bool = env_bool("VIDEO_FOLLOW", "1")  # enable chase cam
    video_cam_lock_yaw: bool = env_bool("VIDEO_CAM_LOCK_YAW", "1")      
    video_cam_dist: float = env_float("VIDEO_CAM_DIST", 0.9)       # behind robot (m)
    video_cam_height: float = env_float("VIDEO_CAM_HEIGHT", 0.45)  # above robot (m)
    video_cam_side: float = env_float("VIDEO_CAM_SIDE", -0.20)     # lateral offset (m) (+right, -left)
    video_cam_lookahead: float = env_float("VIDEO_CAM_LOOKAHEAD", 0.35)  # look ahead in heading dir (m)
    video_cam_look_z: float = env_float("VIDEO_CAM_LOOK_Z", 0.12)         # look-at height (m)
    video_cam_smooth: float = env_float("VIDEO_CAM_SMOOTH", 0.85)   # 0=no smooth, 0.9=very smooth

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

    def __init__(self, cfg: CFG, with_camera: bool = False, auto_reset: bool = True):
        self.cfg = cfg
        self.with_camera = with_camera
        self.auto_reset = auto_reset
        self.device = gs.device
        self.n_envs = int(cfg.n_envs)
        self.num_actions = 12

        self.ep_len = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.prev_action = torch.zeros(self.n_envs, self.num_actions, device=self.device)
        self.stall_steps = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)

        # commands = [vx, vy, omega]
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
            if len(dofs) != 1:
                raise RuntimeError(f"Expected 1 dof for {jn}, got {dofs}")
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

        # obs = [z(1), quat(4), vel_body(3), ang_body(3), q_rel(12), dq(12), prev_a(12), cmd(3)] = 50
        self.obs_dim = 50
        self.reset(torch.arange(self.n_envs, device=self.device, dtype=torch.int64))

    def _clamp_joint_targets(self, q: torch.Tensor) -> torch.Tensor:
        q = q.clone()
        q[:, 0:4] = torch.clamp(q[:, 0:4], *self.JOINT_LIMITS["hip"])
        q[:, 4:8] = torch.clamp(q[:, 4:8], *self.JOINT_LIMITS["thigh"])
        q[:, 8:12] = torch.clamp(q[:, 8:12], *self.JOINT_LIMITS["calf"])
        return q

    def set_command(self, env_ids: torch.Tensor, cmd: torch.Tensor) -> None:
        # System2 -> System1 injection point
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0).repeat(int(env_ids.numel()), 1)
        self.commands[env_ids] = cmd.to(device=self.device, dtype=torch.float32)
        self.stall_steps[env_ids] = 0

    def _sample_cmd(self, n: int) -> torch.Tensor:
        """
        Non-overlapping buckets:
          [0.00,0.25): forward/back (vy=0, omega=0)
          [0.25,0.50): left/right  (vx=0, omega=0)
          [0.50,0.70): pivot       (vx=0, vy=0)
          [0.70,1.00]: mixed
        """
        cfg = self.cfg
        d = self.device

        r = torch.rand(n, device=d)
        vx = torch.empty(n, device=d)
        vy = torch.empty(n, device=d)
        om = torch.empty(n, device=d)

        def urand(lo: float, hi: float, m: torch.Tensor) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(int(m.sum().item()), device=d)

        m0 = r < 0.25
        if m0.any():
            vx[m0] = urand(cfg.cmd_vx_min, cfg.cmd_vx_max, m0)
            vy[m0] = 0.0
            om[m0] = 0.0

        m1 = (r >= 0.25) & (r < 0.50)
        if m1.any():
            vx[m1] = 0.0
            vy[m1] = urand(cfg.cmd_vy_min, cfg.cmd_vy_max, m1)
            om[m1] = 0.0

        m2 = (r >= 0.50) & (r < 0.70)
        if m2.any():
            vx[m2] = 0.0
            vy[m2] = 0.0
            om[m2] = urand(cfg.cmd_omega_min, cfg.cmd_omega_max, m2)

        m3 = r >= 0.70
        if m3.any():
            vx[m3] = urand(cfg.cmd_vx_min, cfg.cmd_vx_max, m3)
            vy[m3] = urand(cfg.cmd_vy_min, cfg.cmd_vy_max, m3)
            om[m3] = urand(cfg.cmd_omega_min, cfg.cmd_omega_max, m3)

        cmd = torch.stack([vx, vy, om], dim=1)

        # avoid too many near-zero mixed commands
        lin = torch.sqrt(cmd[:, 0] ** 2 + cmd[:, 1] ** 2 + 1e-12)
        trivial = (lin < 0.10) & (cmd[:, 2].abs() < 0.15)
        if trivial.any():
            s = torch.where(torch.rand(int(trivial.sum().item()), device=d) > 0.5, 1.0, -1.0)
            cmd[trivial, 0] = s * 0.25
            cmd[trivial, 1] = 0.0
            cmd[trivial, 2] = 0.0

        return cmd

    def reset(self, env_ids: torch.Tensor):
        self.scene.reset(envs_idx=env_ids)
        n = int(env_ids.shape[0])

        noise = (torch.rand(n, self.num_actions, device=self.device) - 0.5) * 0.08
        q_init = self.q0.unsqueeze(0).repeat(n, 1) + noise
        q_init = self._clamp_joint_targets(q_init)

        self.robot.set_dofs_position(q_init, self.act_dofs, envs_idx=env_ids)
        self.robot.set_dofs_velocity(torch.zeros_like(q_init), self.act_dofs, envs_idx=env_ids)

        self.ep_len[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        self.stall_steps[env_ids] = 0
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

        return torch.cat([z, quat, vel_b, ang_b, q_rel, dq, self.prev_action, self.commands], dim=1)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # mid-episode command resample (optional)
        if int(self.cfg.cmd_resample_steps) > 0:
            interval = int(self.cfg.cmd_resample_steps)
            mask = (self.ep_len > 0) & ((self.ep_len % interval) == 0)
            if mask.any():
                ids = torch.nonzero(mask).squeeze(-1)
                self.commands[ids] = self._sample_cmd(int(ids.numel()))
                self.stall_steps[ids] = 0

        action = torch.clamp(action, -1.0, 1.0)
        prev_a = self.prev_action
        self.prev_action = action

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

        eul = quat_to_euler_wxyz(quat)
        roll = eul[:, 0]
        pitch = eul[:, 1]

        vel_b = world_to_body_vec(quat, vel_w)
        ang_b = world_to_body_vec(quat, ang_w)

        v_fwd = vel_b[:, 0]
        v_lat = vel_b[:, 1]
        yaw_rate = ang_b[:, 2]
        z = pos[:, 2]

        cmd_vx = self.commands[:, 0]
        cmd_vy = self.commands[:, 1]
        cmd_om = self.commands[:, 2]

        # ----- rewards -----
        # normalized tracking: prevents "do nothing" on tiny commands
        cmd_lin = torch.sqrt(cmd_vx**2 + cmd_vy**2 + 1e-12)
        lin_err2 = (v_fwd - cmd_vx) ** 2 + (v_lat - cmd_vy) ** 2
        lin_scale = cmd_lin**2 + 0.15**2
        r_lin = torch.exp(-2.5 * lin_err2 / lin_scale)

        yaw_err2 = (yaw_rate - cmd_om) ** 2
        yaw_scale = cmd_om**2 + 0.40**2
        r_yaw = torch.exp(-1.5 * yaw_err2 / yaw_scale)

        r_tracking = self.cfg.w_tracking * (r_lin + 0.5 * r_yaw)

        upright = torch.exp(-10.0 * (roll * roll + pitch * pitch))
        r_upright = self.cfg.w_upright * upright

        height = torch.exp(-80.0 * (z - self.cfg.z_target) ** 2)
        r_height = self.cfg.w_height * height

        dq = self.robot.get_dofs_velocity(self.act_dofs)
        p_energy = self.cfg.w_energy * torch.sum(dq * dq, dim=1)
        p_action = self.cfg.w_action * torch.sum(action * action, dim=1)
        p_smooth = self.cfg.w_smooth * torch.sum((action - prev_a) ** 2, dim=1)

        past_grace = (self.ep_len > self.cfg.stall_grace)

        actual_lin = torch.sqrt(v_fwd**2 + v_lat**2 + 1e-12)
        need_lin = cmd_lin > self.cfg.cmd_lin_thresh
        need_yaw = torch.abs(cmd_om) > self.cfg.cmd_yaw_thresh

        stalled_lin = need_lin & (actual_lin < self.cfg.lin_speed_thresh)
        stalled_yaw = need_yaw & (torch.abs(yaw_rate) < self.cfg.yaw_rate_thresh)
        is_stalled = stalled_lin | stalled_yaw

        p_stall = self.cfg.w_stall * is_stalled.float() * past_grace.float()

        reward = (r_tracking + r_upright + r_height) - (p_energy + p_action + p_smooth + p_stall)

        # ----- terminations -----
        tilted = (torch.abs(roll) > self.cfg.max_tilt) | (torch.abs(pitch) > self.cfg.max_tilt)
        fallen = z < self.cfg.min_z

        self.ep_len += 1
        time_out = self.ep_len >= self.cfg.max_ep_len

        self.stall_steps = torch.where(past_grace & is_stalled, self.stall_steps + 1, torch.zeros_like(self.stall_steps))
        stalled_out = torch.zeros_like(fallen)
        if int(self.cfg.stall_terminate) > 0:
            stalled_out = self.stall_steps >= int(self.cfg.stall_terminate)

        done = tilted | fallen | time_out | stalled_out
        reward = reward - self.cfg.fall_penalty * (tilted | fallen).float()

        if self.auto_reset:
            done_ids = torch.nonzero(done).squeeze(-1)
            if done_ids.numel() > 0:
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
    backend_name = os.getenv("GS_BACKEND", "vulkan").lower()
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

    # 1) Build cfg FIRST
    cfg = CFG.from_dict(ckpt.get("cfg", {}))

    # 2) Then allow env vars to override cfg (so VIDEO_CAM_* actually works when resuming)
    apply_env_overrides(cfg)

    # 3) Only override env count for recording (training still uses N_ENVS)
    cfg.n_envs = int(getattr(cfg, "video_envs", 1))

    gs.init(backend=pick_backend())
    env = MiniPupperBatched(cfg, with_camera=True, auto_reset=False)
    device = gs.device

    model = ActorCritic(env.obs_dim, env.num_actions).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("[video] settling physics...")
    for _ in range(40):
        env.robot.control_dofs_position(env.q0.unsqueeze(0).repeat(cfg.n_envs, 1), env.act_dofs)
        env.scene.step(update_visualizer=False, refresh_visualizer=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    env.cam.start_recording()

    env0 = torch.tensor([0], device=device, dtype=torch.int64)
    sequence = [
        ("Forward",  torch.tensor([ 0.40,  0.00,  0.00], device=device)),
        ("Backward", torch.tensor([-0.30,  0.00,  0.00], device=device)),
        ("Left",     torch.tensor([ 0.00,  0.25,  0.00], device=device)),
        ("Right",    torch.tensor([ 0.00, -0.25,  0.00], device=device)),
        ("Pivot",    torch.tensor([ 0.00,  0.00,  0.60], device=device)),
    ]
    steps_per_mode = int(max(1, cfg.video_cmd_switch))

    cam_state = {"pos": None, "look": None}

    try:
        for t in range(int(cfg.video_steps)):
            mode_idx = (t // steps_per_mode) % len(sequence)
            mode_name, cmd_vec = sequence[mode_idx]
            if (t % steps_per_mode) == 0:
                print(f"[video] {mode_name} for {steps_per_mode} steps (demo only)")
                env.set_command(env0, cmd_vec)

            obs = env.get_obs()
            a = model.act_deterministic(obs)
            obs, _, done = env.step(a)

            # Follow camera AFTER stepping, BEFORE rendering
            if getattr(cfg, "video_follow", False):
                pos0 = env.robot.get_pos()[0]
                quat0 = env.robot.get_quat()[0]
                _follow_cam_update(env.cam, pos0, quat0, cfg, cam_state)

            env.cam.render()

            if bool(done[0].item()):
                print(f"[video] env0 done at t={t}; resetting env0 and continuing demo...")
                env.reset(env0)
                env.set_command(env0, cmd_vec)
                cam_state["pos"] = None
                cam_state["look"] = None
                for _ in range(20):
                    env.robot.control_dofs_position(env.q0.unsqueeze(0).repeat(cfg.n_envs, 1), env.act_dofs)
                    env.scene.step(update_visualizer=False, refresh_visualizer=False)

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
    parser.add_argument("--resume", type=str, default=os.getenv("RESUME", ""))
    args = parser.parse_args()

    if args.record_only:
        rc = record_video_from_ckpt(args.ckpt, args.out)
        raise SystemExit(rc)

    cfg = None
    resume_path = args.resume.strip()

    start_update = 1
    ckpt_resume = None

    if resume_path:
        ckpt_resume = torch.load(resume_path, map_location="cpu")
        cfg = CFG.from_dict(ckpt_resume.get("cfg", {}))
        apply_env_overrides(cfg)
        # optional: allow changing N_ENVS when resuming via env var
        if "N_ENVS" in os.environ:
            cfg.n_envs = env_int("N_ENVS", cfg.n_envs)
        start_update = int(ckpt_resume.get("update", 0)) + 1
        print(f"↩️ resuming from {resume_path} (next upd={start_update})")
    else:
        cfg = CFG()

    os.makedirs(cfg.out_dir, exist_ok=True)

    gs.init(backend=pick_backend())
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # TRAINING uses cfg.n_envs (many parallel)
    env = MiniPupperBatched(cfg, with_camera=False, auto_reset=True)
    device = gs.device

    model = ActorCritic(env.obs_dim, env.num_actions).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    if ckpt_resume is not None:
        model.load_state_dict(ckpt_resume["model"])
        if "optim" in ckpt_resume:
            optim.load_state_dict(ckpt_resume["optim"])

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

    for update in range(start_update, cfg.total_updates + 1):
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

        # live metrics
        quat = env.robot.get_quat()
        vel_w = env.robot.get_vel()
        ang_w = env.robot.get_ang()
        vel_b = world_to_body_vec(quat, vel_w)
        ang_b = world_to_body_vec(quat, ang_w)

        v_fwd = vel_b[:, 0]
        v_lat = vel_b[:, 1]
        yaw_rate = ang_b[:, 2]

        err_fwd = (v_fwd - env.commands[:, 0]).abs().mean().item()
        err_lat = (v_lat - env.commands[:, 1]).abs().mean().item()
        err_yaw = (yaw_rate - env.commands[:, 2]).abs().mean().item()

        mean_z = env.robot.get_pos()[:, 2].mean().item()

        print(
            f"upd={update:05d}  env_fps={env_fps:10.0f}  "
            f"mean_rew={mean_rew:+.3f}  z={mean_z:+.3f} | "
            f"Err_Fwd={err_fwd:.3f}  Err_Lat={err_lat:.3f}  Err_Yaw={err_yaw:.3f}"
        )

        if (update % cfg.save_every) == 0:
            ckpt = {"update": update, "cfg": cfg.__dict__, "model": model.state_dict(), "optim": optim.state_dict()}
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{update:05d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"💾 saved {ckpt_path}")

        # Progress videos every VIDEO_EVERY PPO updates (this is what you meant)
        if cfg.record_video and (update % cfg.video_every) == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{update:05d}.pt")
            vid_path = os.path.join(cfg.out_dir, f"video_{update:05d}.mp4")

            if not os.path.exists(ckpt_path):
                ckpt = {"update": update, "cfg": cfg.__dict__, "model": model.state_dict(), "optim": optim.state_dict()}
                torch.save(ckpt, ckpt_path)

            spawn_record_video(ckpt_path, vid_path)


if __name__ == "__main__":
    main()
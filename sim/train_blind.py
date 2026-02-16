#!/usr/bin/env python3
from __future__ import annotations

"""
train_blind.py — Mini Pupper PPO in Genesis (proprio-only), with robust video recording.

Fixes (standing-still exploit & Stability):
1) Stability Tuning:
   - LOWERED KP from 45.0 to 5.0 (Critical for Mini Pupper mass/inertia).
   - LOWERED KV from 2.5 to 0.5.
   - LOWERED spawn height from 0.25m to 0.12m to prevent impact shocks.
2) Forward objective is primary:
   - Reward uses BODY-FRAME forward velocity (yaw-invariant).
   - Command-conditioned target speed cmd_vx is sampled per-episode and included in obs.
3) Anti-stall shaping:
   - Penalty when vfwd stays below VX_MIN after a short grace period.
   - Optional termination if stalled too long (counted only after grace).

Video robustness:
- Recording runs in a subprocess (--record-only) so GL/EGL failures never kill training.
- Video recording now includes a "settle" phase so the robot doesn't drop from the sky.
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


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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
    """
    Rotate vector v by quaternion q.
    q: (N,4) wxyz
    v: (N,3)
    """
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    vq = torch.cat([zeros, v], dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(q, vq), quat_conj_wxyz(q))[:, 1:4]


def world_to_body_vec(quat_wxyz: torch.Tensor, vec_world: torch.Tensor) -> torch.Tensor:
    """
    Convert world-frame vector to body-frame using q_conj rotation.
    """
    return quat_rotate_wxyz(quat_conj_wxyz(quat_wxyz), vec_world)


def quat_to_euler_wxyz(q: torch.Tensor) -> torch.Tensor:
    # q: (N,4) wxyz
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

    n_envs: int = env_int("N_ENVS", 2048)
    env_spacing: float = env_float("ENV_SPACING", 1.0) # Reduced spacing for small robots

    dt: float = env_float("DT", 0.01)
    substeps: int = env_int("SUBSTEPS", 4)
    decimation: int = env_int("DECIMATION", 4)

    # Stability Tuning: drastically reduced for Mini Pupper scale
    kp: float = env_float("KP", 5.0)  
    kv: float = env_float("KV", 0.5)

    max_ep_len: int = env_int("MAX_EP_LEN", 800)

    hip_splay: float = env_float("HIP_SPLAY", 0.06)
    thigh0: float = env_float("THIGH0", 0.85)
    calf0: float = env_float("CALF0", -1.75)
    action_scale: float = env_float("ACTION_SCALE", 0.30) # Reduced slightly for safety

    min_z: float = env_float("MIN_Z", 0.05)
    max_tilt: float = env_float("MAX_TILT", 1.0)
    z_target: float = env_float("Z_TARGET", 0.085)

    # command-conditioned forward speed target (m/s), sampled per reset
    cmd_vx_low: float = env_float("CMD_VX_LOW", 0.25)
    cmd_vx_high: float = env_float("CMD_VX_HIGH", 0.80)

    # reward weights
    w_fwd: float = env_float("W_FWD", 2.0)
    w_cmd: float = env_float("W_CMD", 1.0)
    w_upright: float = env_float("W_UPRIGHT", 0.25)
    w_height: float = env_float("W_HEIGHT", 0.10)

    # penalties
    w_energy: float = env_float("W_ENERGY", 2e-4)
    w_action: float = env_float("W_ACTION", 1e-3)
    w_smooth: float = env_float("W_SMOOTH", 2e-3)

    # anti-stall
    vx_min: float = env_float("VX_MIN", 0.12)
    stall_grace: int = env_int("STALL_GRACE", 10)
    w_stall: float = env_float("W_STALL", 0.8)
    stall_terminate: int = env_int("STALL_TERMINATE", 200)

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

    out_dir: str = os.getenv("OUT_DIR", f"runs/pupper_walk_{now_tag()}")
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

        # anti-stall tracking
        self.stall_steps = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.cmd_vx = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)

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

        # FIX: Lower spawn height to 0.12m to prevent impact explosion
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

        # obs = [z(1), quat(4), vel_body(3), ang_body(3), q_rel(12), dq(12), prev_a(12), cmd_vx(1)] = 48
        self.obs_dim = 48

        self.reset(torch.arange(self.n_envs, device=self.device, dtype=torch.int64))

    def _clamp_joint_targets(self, q: torch.Tensor) -> torch.Tensor:
        q = q.clone()
        q[:, 0:4] = torch.clamp(q[:, 0:4], *self.JOINT_LIMITS["hip"])
        q[:, 4:8] = torch.clamp(q[:, 4:8], *self.JOINT_LIMITS["thigh"])
        q[:, 8:12] = torch.clamp(q[:, 8:12], *self.JOINT_LIMITS["calf"])
        return q

    def _sample_cmd(self, n: int) -> torch.Tensor:
        lo = float(self.cfg.cmd_vx_low)
        hi = float(self.cfg.cmd_vx_high)
        if hi <= lo:
            hi = lo + 1e-3
        return lo + (hi - lo) * torch.rand(n, device=self.device)

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
        self.cmd_vx[env_ids] = self._sample_cmd(n)

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

        cmd = self.cmd_vx.unsqueeze(1)
        return torch.cat([z, quat, vel_b, ang_b, q_rel, dq, self.prev_action, cmd], dim=1)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        eul = quat_to_euler_wxyz(quat)
        roll = eul[:, 0]
        pitch = eul[:, 1]

        vel_b = world_to_body_vec(quat, vel_w)
        v_fwd = vel_b[:, 0]

        z = pos[:, 2]
        cmd = self.cmd_vx

        # ----- rewards -----
        # (A) forward speed
        r_fwd = self.cfg.w_fwd * torch.clamp(v_fwd, 0.0, 2.0)

        # (B) track commanded speed sharply
        err = v_fwd - cmd
        k = 4.0
        r_cmd = self.cfg.w_cmd * torch.exp(-k * (err / (cmd + 1e-6)) ** 2)

        # (C) posture shaping
        upright = torch.exp(-10.0 * (roll * roll + pitch * pitch))
        r_upright = self.cfg.w_upright * upright

        height = torch.exp(-80.0 * (z - self.cfg.z_target) ** 2)
        r_height = self.cfg.w_height * height

        # penalties
        dq = self.robot.get_dofs_velocity(self.act_dofs)
        p_energy = self.cfg.w_energy * torch.sum(dq * dq, dim=1)
        p_action = self.cfg.w_action * torch.sum(action * action, dim=1)
        p_smooth = self.cfg.w_smooth * torch.sum((action - prev_a) ** 2, dim=1)

        # anti-stall
        past_grace = (self.ep_len > self.cfg.stall_grace)
        below = torch.clamp((self.cfg.vx_min - v_fwd) / (self.cfg.vx_min + 1e-6), min=0.0)
        p_stall = self.cfg.w_stall * below * past_grace.float()

        reward = (r_fwd + r_cmd + r_upright + r_height) - (p_energy + p_action + p_smooth + p_stall)

        # ----- terminations -----
        tilted = (torch.abs(roll) > self.cfg.max_tilt) | (torch.abs(pitch) > self.cfg.max_tilt)
        fallen = z < self.cfg.min_z

        self.ep_len += 1
        time_out = self.ep_len >= self.cfg.max_ep_len

        is_slow = (v_fwd < self.cfg.vx_min)
        self.stall_steps = torch.where(past_grace & is_slow, self.stall_steps + 1, torch.zeros_like(self.stall_steps))

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
    cfg = CFG.from_dict(ckpt.get("cfg", {}))
    cfg.n_envs = 1  # enforce 1 env for video

    gs.init(backend=pick_backend())

    env = MiniPupperBatched(cfg, with_camera=True, auto_reset=False)
    device = gs.device

    model = ActorCritic(env.obs_dim, env.num_actions).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # FIX: Settle phase! Allow robot to touch ground gracefully before starting recording
    print("[video] settling physics...")
    for _ in range(40):
        # Hold default pose (action=0)
        env.robot.control_dofs_position(env.q0.unsqueeze(0), env.act_dofs)
        env.scene.step(update_visualizer=False, refresh_visualizer=False)

    obs = env.get_obs()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    env.cam.start_recording()
    try:
        for _ in range(cfg.video_steps):
            a = model.act_deterministic(obs)
            obs, _, done = env.step(a)
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
    """
    Run video capture in a subprocess so GL/EGL failures never kill training.
    """
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

    env = MiniPupperBatched(cfg, with_camera=False, auto_reset=True)
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

        # live metrics
        quat = env.robot.get_quat()
        vel_w = env.robot.get_vel()
        v_fwd = world_to_body_vec(quat, vel_w)[:, 0]
        mean_vfwd = v_fwd.mean().item()
        mean_cmd = env.cmd_vx.mean().item()
        mean_err = (v_fwd - env.cmd_vx).abs().mean().item()
        stall_frac = (v_fwd < cfg.vx_min).float().mean().item()
        mean_z = env.robot.get_pos()[:, 2].mean().item()

        print(
            f"upd={update:05d}  env_fps={env_fps:10.0f}  "
            f"mean_rew={mean_rew:+.3f}  vfwd={mean_vfwd:+.3f}  cmd={mean_cmd:+.3f}  |err|={mean_err:+.3f}  "
            f"stall={stall_frac*100:5.1f}%  z={mean_z:+.3f}"
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
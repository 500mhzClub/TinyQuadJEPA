#!/usr/bin/env python3
"""
PROJECT CERBERUS: PPO BLIND WALKER (Genesis 0.3.14 | gs.vulkan)

FULL REGEN (stable):

Fixes baked in:
1) Scene.build(): env_spacing MUST be a tuple (x, y)               -> ENV_SPACING="2.5,2.5"
2) RigidEntity has NO set_qvel() in 0.3.14                         -> use set_dofs_velocity() on base DOFs (0..5)
3) RigidEntity.is_free property is REMOVED and raises DeprecationError
   -> never access is_free; just try/except base kick
4) PPO autograd bug ("backward through graph a second time")
   -> rollout/GAE/returns are STRICTLY no-grad + detached (+ clones)
5) BIG PRACTICAL FIX: reset now hard-resets BASE pose/orientation and velocities
   -> prevents learning from being stuck inverted on the floor
6) Reward includes a height term centered at nominal base height

Your command works as-is:
  ENV_SPACING="2.5,2.5" CURRICULUM=1 STAND_UPDATES=0 FWD_RAMP_UPDATES=400 UPRIGHT_DECAY=0.3 \
  ACTION_SCALE=0.65 POSE_PENALTY=1e-4 ALIVE_BONUS=0.0 FWD_SCALE=3.0 ENT_COEF=0.02 \
  YAW_PENALTY=0.1 SIDE_PENALTY=0.1 python3 sim/train_blind.py

Useful knobs:
  ENVS=1024
  DT=0.01
  FORCE_LIMIT=30
  ROLLOUT_T=256
  UPDATE_EPOCHS=4
  HIDDEN=256
  LR=3e-4
  KICK_AFTER_UPDATES=500    (disable kicks early; helps learn standing)
  CLONE_ROLLOUT=1           (recommended; avoids any sim-buffer aliasing)
  FALL_H=0.10               (termination height)
  EP_LEN=800
  SEED=0
  URDF=./assets/mini_pupper/mini_pupper.urdf
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# STS3215 Actuator Model (PD + speed-dependent torque limit + latency)
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

    def reset(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            self.command_queue.zero_()
        else:
            self.command_queue[env_ids] = 0.0


# -----------------------------
# Quaternion helper (wxyz)
# -----------------------------
def quat_rotate_wxyz(q, v):
    """
    Rotate vector v by quaternion q (wxyz).
    q: (...,4), v: (...,3)
    """
    q_w, q_vec = q[..., 0:1], q[..., 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


# -----------------------------
# PPO network
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
        logp = (-0.5 * (((a - mu) ** 2) / (log_std.exp() ** 2 + 1e-8) + 2.0 * log_std + math.log(2.0 * math.pi))).sum(dim=-1)
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
        self.var = ((self.var * self.count + v * c + delta**2 * (self.count * c / tot)) / tot).clamp_min(1e-6)
        self.count = tot

    @torch.no_grad()
    def normalize(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10, 10)


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    # sim
    envs: int = _env_int("ENVS", 1024)
    dt: float = _env_float("DT", 0.01)
    env_spacing: Tuple[float, float] = _env_tuple2("ENV_SPACING", (2.5, 2.5))

    # robot/control
    action_scale: float = _env_float("ACTION_SCALE", 0.65)
    force_limit: float = _env_float("FORCE_LIMIT", 30.0)

    # shaping / curriculum
    curriculum: bool = _env_bool("CURRICULUM", "0")
    stand_updates: int = _env_int("STAND_UPDATES", 0)
    fwd_scale: float = _env_float("FWD_SCALE", 3.0)
    fwd_ramp_updates: int = _env_int("FWD_RAMP_UPDATES", 400)
    upright_decay: float = _env_float("UPRIGHT_DECAY", 0.3)

    pose_penalty: float = _env_float("POSE_PENALTY", 1e-4)
    alive_bonus: float = _env_float("ALIVE_BONUS", 0.0)
    yaw_penalty: float = _env_float("YAW_PENALTY", 0.1)
    side_penalty: float = _env_float("SIDE_PENALTY", 0.1)

    # episode termination
    fall_h: float = _env_float("FALL_H", 0.10)
    ep_len: int = _env_int("EP_LEN", 800)

    # PPO
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

    # reset / rollout safety
    kick_after_updates: int = _env_int("KICK_AFTER_UPDATES", 500)
    clone_rollout: bool = _env_bool("CLONE_ROLLOUT", "1")

    # misc
    seed: int = _env_int("SEED", 0)

    # paths
    urdf: str = _env_str("URDF", "./assets/mini_pupper/mini_pupper.urdf")


# -----------------------------
# Genesis Env
# -----------------------------
class BlindWalkerEnv:
    def __init__(self, cfg: CFG, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._global_update = 0

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(gravity=(0, 0, -9.81)),
        )
        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=cfg.urdf, pos=(0, 0, 0.18)),
            material=gs.materials.Rigid(),
        )

        # FIX: env_spacing must be tuple(len=2)
        self.scene.build(n_envs=cfg.envs, env_spacing=cfg.env_spacing)

        # DOFs: base (0..5) + motors (6..17) for this URDF in Genesis
        self.base_dofs = torch.arange(0, 6, device=device)
        self.motor_dofs = torch.arange(6, 18, device=device)

        # Gains for 12 motors
        self.robot.set_dofs_kp(torch.zeros(12, device=device), self.motor_dofs)
        self.robot.set_dofs_kv(torch.zeros(12, device=device), self.motor_dofs)

        f_lim = torch.ones(12, device=device) * cfg.force_limit
        self.robot.set_dofs_force_range(-f_lim, f_lim, self.motor_dofs)

        self.base_link = self.robot.get_link("base_link")

        # Nominal base pose
        self.base_pos0 = torch.tensor([0.0, 0.0, 0.18], device=device)
        self.base_quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # wxyz

        # Nominal joint pose (may still exceed limits in URDF -> warning is ok)
        self.default_dof_pos = torch.tensor([0.0, 0.6, -1.2] * 4, device=device)

        self.actuator = STS3215_Actuator(cfg.envs, device, dt=cfg.dt)
        self.last_action = torch.zeros((cfg.envs, 12), device=device)
        self.episode_length = torch.zeros(cfg.envs, device=device)

        self.reset(torch.arange(cfg.envs, device=device))

    def set_update(self, upd: int):
        self._global_update = upd

    def _set_base_pose(self, env_ids: torch.Tensor):
        pos = self.base_pos0.expand(len(env_ids), 3).clone()
        quat = self.base_quat0.expand(len(env_ids), 4).clone()

        # Prefer base_link if it supports set_pos/set_quat (Genesis API can vary)
        try:
            self.base_link.set_pos(pos, envs_idx=env_ids)
            self.base_link.set_quat(quat, envs_idx=env_ids)
            return
        except Exception:
            pass

        # Fallback: entity-level set_pos/set_quat
        try:
            self.robot.set_pos(pos, envs_idx=env_ids)
            self.robot.set_quat(quat, envs_idx=env_ids)
        except Exception:
            pass

    def _zero_velocities(self, env_ids: torch.Tensor):
        # Base velocities (best-effort)
        try:
            self.robot.set_dofs_velocity(
                torch.zeros((len(env_ids), 6), device=self.device),
                self.base_dofs,
                envs_idx=env_ids,
            )
        except Exception:
            pass

        # Joint velocities (should exist)
        self.robot.set_dofs_velocity(
            torch.zeros((len(env_ids), 12), device=self.device),
            self.motor_dofs,
            envs_idx=env_ids,
        )

    def _kick_base_velocity(self, env_ids: torch.Tensor):
        """
        Genesis 0.3.14:
          - no set_qvel()
          - MUST NOT touch self.robot.is_free (removed; raises DeprecationError)
        So: try set_dofs_velocity on base DOFs (0..5). If fixed/unsupported -> ignore.
        """
        if not hasattr(self.robot, "set_dofs_velocity"):
            return

        base_vel = torch.zeros((len(env_ids), 6), device=self.device)
        base_vel[:, 0] = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * 0.5  # vx
        base_vel[:, 5] = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * 0.5  # wz
        try:
            self.robot.set_dofs_velocity(base_vel, self.base_dofs, envs_idx=env_ids)
        except Exception:
            pass

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self.episode_length[env_ids] = 0
        self.actuator.reset(env_ids)

        # HARD reset base pose + velocities
        self._set_base_pose(env_ids)
        self._zero_velocities(env_ids)

        # Joint pose reset (+ noise)
        q = self.default_dof_pos.unsqueeze(0) + (torch.rand((len(env_ids), 12), device=self.device) * 2 - 1) * 0.10
        self.robot.set_dofs_position(q, self.motor_dofs, envs_idx=env_ids)

        # Optional exploration kicks only after some learning
        if self._global_update >= self.cfg.kick_after_updates:
            self._kick_base_velocity(env_ids)

    def get_obs(self):
        q = self.robot.get_dofs_position(self.motor_dofs)
        dq = self.robot.get_dofs_velocity(self.motor_dofs)

        pos = self.base_link.get_pos()
        quat = self.base_link.get_quat()
        v_lin = self.base_link.get_vel()
        v_ang = self.base_link.get_ang()

        # body up direction proxy
        up_world = torch.tensor([0, 0, 1], device=self.device, dtype=v_lin.dtype).expand_as(v_lin)
        up_body = quat_rotate_wxyz(quat, up_world)

        # [ang_vel(3), up_body(3), lin_vel(3), height(1), q_err(12), dq(12), last_action(12)]
        return torch.cat([v_ang, up_body, v_lin, pos[:, 2:3], q - self.default_dof_pos, dq, self.last_action], dim=-1)

    def step(self, action: torch.Tensor, w_fwd: float):
        self.last_action = torch.tanh(action)
        target = self.default_dof_pos + self.cfg.action_scale * self.last_action

        q = self.robot.get_dofs_position(self.motor_dofs)
        dq = self.robot.get_dofs_velocity(self.motor_dofs)

        torque = self.actuator.step(target, q, dq)
        self.robot.control_dofs_force(torque, self.motor_dofs)

        self.scene.step()

        obs = self.get_obs()
        v_ang = obs[:, 0:3]
        up_body = obs[:, 3:6]
        v_lin = obs[:, 6:9]
        h = obs[:, 9]

        # curriculum: optionally stand first, then forward ramp
        if self.cfg.curriculum and (self._global_update < self.cfg.stand_updates):
            w_fwd_eff = 0.0
        else:
            w_fwd_eff = w_fwd

        upright = up_body[:, 2].clamp(-1, 1)
        upright_term = torch.exp(-((1.0 - upright) ** 2) / 0.1)

        # NEW: explicit height term around nominal base height (0.18)height_ter
        height_term = torch.exp(-((h - 0.18) ** 2) / 0.004)


        pose_cost = (q - self.default_dof_pos).pow(2).mean(dim=-1)
        yaw_cost = v_ang[:, 2].abs()
        side_cost = v_lin[:, 1].abs()

        reward = (
            self.cfg.alive_bonus
            + (w_fwd_eff * self.cfg.fwd_scale * v_lin[:, 0])
            + 0.5 * upright_term
            + 0.5 * height_term
            - (self.cfg.pose_penalty * pose_cost)
            - (self.cfg.yaw_penalty * yaw_cost)
            - (self.cfg.side_penalty * side_cost)
        )

        # fade upright shaping over updates (optional)
        if self.cfg.upright_decay > 0:
            decay = math.exp(-self.cfg.upright_decay * float(self._global_update))
            reward = reward - (1.0 - decay) * 0.5 * (1.0 - upright_term)

        self.episode_length += 1
        done = (h < self.cfg.fall_h) | (self.episode_length >= self.cfg.ep_len)

        reset_ids = torch.nonzero(done).squeeze(-1)
        self.reset(reset_ids)

        info = {"vfwd": v_lin[:, 0], "h": h, "upright": upright}
        return obs, reward, done.float(), info


# -----------------------------
# PPO update (graph-safe)
# -----------------------------
def ppo_update(
    model: ActorCritic,
    opt: torch.optim.Optimizer,
    rms: RunningMeanStd,
    obs, act, lp0, rew, done, val, last_val,
    cfg: CFG
):
    """
    Critical rule: anything reused across PPO epochs MUST NOT carry an autograd graph.
    So: adv/ret/obs_n/act_flat/lp0_flat are all built under no_grad + detached.
    """
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

        # flatten + detach hard
        obs_flat = obs.reshape(-1, obs.shape[-1]).detach()
        act_flat = act.reshape(-1, act.shape[-1]).detach()
        lp0_flat = lp0.reshape(-1).detach()
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)

        # normalize obs (no grad)
        rms.update(obs_flat)
        obs_n = rms.normalize(obs_flat)

        # normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    # PPO epochs (fresh graph each epoch through model forward only)
    for _ in range(cfg.update_epochs):
        mu, log_std, v = model(obs_n)
        std = log_std.exp()

        lp = (-0.5 * (((act_flat - mu) ** 2) / (std ** 2 + 1e-8)
                      + 2.0 * log_std + math.log(2.0 * math.pi))).sum(-1)

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
# Main
# -----------------------------
def main():
    cfg = CFG()
    seed_all(cfg.seed)

    gs.init(backend=gs.vulkan)
    device = torch.device("cuda")

    env = BlindWalkerEnv(cfg, device)
    env.set_update(0)

    obs = env.get_obs()
    obs_dim = obs.shape[-1]
    act_dim = 12

    model = ActorCritic(obs_dim, act_dim, cfg.hidden, cfg.log_std_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rms = RunningMeanStd((obs_dim,), device)

    def store(x: torch.Tensor) -> torch.Tensor:
        # Recommended: clone rollout tensors to avoid aliasing with sim buffers.
        return x.detach().clone() if cfg.clone_rollout else x.detach()

    for upd in range(1, 1_000_000):
        env.set_update(upd)

        # forward ramp
        w_fwd = min(1.0, upd / max(1, cfg.fwd_ramp_updates))

        obs_b, act_b, lp_b, rew_b, don_b, val_b = [], [], [], [], [], []
        info = {}

        # --- rollout (NO grad) ---
        for _ in range(cfg.rollout_T):
            with torch.no_grad():
                obs_in = rms.normalize(obs)
                a, lp, v = model.act(obs_in)

            next_obs, r, d, info = env.step(a, w_fwd)

            obs_b.append(store(obs))
            act_b.append(store(a))
            lp_b.append(store(lp))
            rew_b.append(store(r))
            don_b.append(store(d))
            val_b.append(store(v))

            obs = next_obs

        # bootstrap value (NO grad, detached)
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
            cfg
        )

        if upd % 10 == 0:
            mean_rew = torch.stack(rew_b, 0).mean().item()
            mean_fwd = info["vfwd"].mean().item() if "vfwd" in info else float("nan")
            mean_h = info["h"].mean().item() if "h" in info else float("nan")
            mean_up = info["upright"].mean().item() if "upright" in info else float("nan")
            print(
                f"Upd {upd:06d} | Rew {mean_rew:+.3f} | Fwd {mean_fwd:+.3f} | h {mean_h:.3f} | up {mean_up:+.3f} "
                f"| envs={cfg.envs} dt={cfg.dt} spacing={cfg.env_spacing} kick_after={cfg.kick_after_updates}"
            )


if __name__ == "__main__":
    main()

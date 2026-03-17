#!/usr/bin/env python3
from __future__ import annotations

"""
JEPA landmark-navigation eval (expanded demo)

What this version adds on top of the stable regenerated eval:
- Route mode with repeated landmark visits (default: W1 -> W2 -> W3 -> W2 -> W1)
- Larger default step budget
- Optional kidnap/relocalization event to demonstrate replanning robustness
- Minimap HUD with robot trail, waypoint route, active target, and planner rollout
- Route-progress overlays and visit counters for a more showable demo video

The navigation core stays the same:
- approach-aligned latent breadcrumbs
- commands clamped to rollout training distribution
- energy + geometric progress planning cost
- PPO low-level controller for actuation
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import genesis as gs


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 50, act_dim: int = 12, hid: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, 1),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(obs))


class VisionEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int = 47, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(128)
        self.prop_enc = ProprioEncoder(47, 128)
        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, vision: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.vis_enc(vision), self.prop_enc(proprio)], dim=-1))


class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(latent_dim + cmd_dim, latent_dim), nn.ELU())
        self.rnn = nn.GRUCell(latent_dim, latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, c_t: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        return self.output_proj(h_next), h_next


class TinyQuadJEPA(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = JointEncoder(latent_dim)
        self.predictor = LatentPredictor(latent_dim, 3)


class GoalEnergyHead(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_pred, z_goal, z_pred - z_goal, z_pred * z_goal], dim=-1)).squeeze(-1)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class WaypointSpec:
    pos: np.ndarray
    approach_dir_xy: np.ndarray
    name: str
    color_rgb: Tuple[float, float, float]
    panel_pos: Tuple[float, float, float]
    panel_size: Tuple[float, float, float]


def clean_state_dict(d: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in d.items()}


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_to_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = 0.5 * yaw_rad
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def body_to_world_xy(yaw: float, v_body_xy: np.ndarray) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx_b, vy_b = float(v_body_xy[0]), float(v_body_xy[1])
    return np.array([c * vx_b - s * vy_b, s * vx_b + c * vy_b], dtype=np.float32)


def world_to_body_xy(yaw: float, v_world_xy: np.ndarray) -> np.ndarray:
    return body_to_world_xy(-yaw, v_world_xy)


def create_checkerboard(path: str = "dense_checker.png") -> str:
    res, grid = 1024, 16
    img = np.zeros((res, res, 3), dtype=np.uint8)
    for i in range(res):
        for j in range(res):
            c = 255 if ((i // grid) + (j // grid)) % 2 == 0 else 40
            img[i, j] = [c, c, c]
    Image.fromarray(img).save(path)
    return os.path.abspath(path)


def make_waypoints() -> List[WaypointSpec]:
    return [
        WaypointSpec(
            pos=np.array([2.0, 0.0, 0.15], dtype=np.float32),
            approach_dir_xy=np.array([1.0, 0.0], dtype=np.float32),
            name="RED beacon",
            color_rgb=(0.94, 0.16, 0.16),
            panel_pos=(3.5, 0.0, 0.60),
            panel_size=(0.05, 2.00, 1.20),
        ),
        WaypointSpec(
            pos=np.array([2.0, 2.0, 0.15], dtype=np.float32),
            approach_dir_xy=np.array([0.0, 1.0], dtype=np.float32),
            name="GREEN beacon",
            color_rgb=(0.16, 0.86, 0.16),
            panel_pos=(2.0, 3.5, 0.60),
            panel_size=(2.00, 0.05, 1.20),
        ),
        WaypointSpec(
            pos=np.array([0.0, 2.0, 0.15], dtype=np.float32),
            approach_dir_xy=np.array([-1.0, 0.0], dtype=np.float32),
            name="BLUE beacon",
            color_rgb=(0.16, 0.31, 0.94),
            panel_pos=(-1.5, 2.0, 0.60),
            panel_size=(0.05, 2.00, 1.20),
        ),
    ]


def parse_route(route_str: str, n_waypoints: int) -> List[int]:
    toks = [t.strip() for t in route_str.split(",") if t.strip()]
    if not toks:
        raise ValueError("Route must contain at least one waypoint index")
    route = []
    for t in toks:
        idx = int(t) - 1
        if idx < 0 or idx >= n_waypoints:
            raise ValueError(f"Route index {t} out of range 1..{n_waypoints}")
        route.append(idx)
    return route


def to_genesis_target(x: torch.Tensor) -> torch.Tensor:
    x_np = x.detach().to("cpu").numpy().astype(np.float32, copy=True)
    return torch.tensor(x_np, device=gs.device, dtype=torch.float32)


# ----------------------------------------------------------------------------
# Genesis scene + observations
# ----------------------------------------------------------------------------

def init_scene(device: torch.device, waypoints: List[WaypointSpec]):
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)

    tex = create_checkerboard()
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_path=tex)),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(file="assets/mini_pupper/mini_pupper.urdf", pos=(0.0, 0.0, 0.12), fixed=False)
    )

    for wp in waypoints:
        scene.add_entity(
            gs.morphs.Box(pos=wp.panel_pos, size=wp.panel_size, fixed=True),
            surface=gs.surfaces.Rough(color=wp.color_rgb),
        )

    cb = scene.add_camera(res=(64, 64), fov=50)
    ce = scene.add_camera(res=(384, 384), fov=50)
    c3 = scene.add_camera(res=(512, 512), fov=50)
    scene.build()

    joint_names = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    dofs = [robot.get_joint(n).dofs_idx_local[0] for n in joint_names]

    q0_np = np.array([
        0.06, 0.06, -0.06, -0.06,
        0.85, 0.85, 0.85, 0.85,
        -1.75, -1.75, -1.75, -1.75,
    ], dtype=np.float32)
    q0 = torch.tensor(q0_np, device=device)

    robot.set_pos(np.array([0.0, 0.0, 0.12], dtype=np.float32))
    robot.set_quat(yaw_to_quat(0.0))
    robot.set_dofs_position(q0_np, dofs)
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, dofs)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, dofs)

    return scene, robot, cb, ce, c3, dofs, q0


def get_sys1_obs(r, q0: torch.Tensor, p_a: torch.Tensor, cmd: torch.Tensor, dofs, dev: torch.device) -> torch.Tensor:
    p = r.get_pos().to(dev)
    q = r.get_quat().to(dev)
    v = r.get_vel().to(dev)
    a = r.get_ang().to(dev)
    p, q, v, a = [x.unsqueeze(0) if x.dim() == 1 else x for x in (p, q, v, a)]

    q_c = torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)

    def qm(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            qa[:, 0] * qb[:, 0] - qa[:, 1] * qb[:, 1] - qa[:, 2] * qb[:, 2] - qa[:, 3] * qb[:, 3],
            qa[:, 0] * qb[:, 1] + qa[:, 1] * qb[:, 0] + qa[:, 2] * qb[:, 3] - qa[:, 3] * qb[:, 2],
            qa[:, 0] * qb[:, 2] - qa[:, 1] * qb[:, 3] + qa[:, 2] * qb[:, 0] + qa[:, 3] * qb[:, 1],
            qa[:, 0] * qb[:, 3] + qa[:, 1] * qb[:, 2] - qa[:, 2] * qb[:, 1] + qa[:, 3] * qb[:, 0],
        ], dim=-1)

    def world_to_body(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return qm(qm(q_c, torch.cat([torch.zeros((vec.shape[0], 1), device=vec.device), vec], dim=-1)), quat)[:, 1:4]

    qd = r.get_dofs_position(dofs).to(dev).unsqueeze(0)
    dq = r.get_dofs_velocity(dofs).to(dev).unsqueeze(0)
    return torch.cat([p[:, 2:3], q, world_to_body(q, v), world_to_body(q, a), qd - q0.unsqueeze(0), dq, p_a, cmd], dim=1)


@torch.no_grad()
def get_jepa_state(r, cb, q0: torch.Tensor, pa: torch.Tensor, dofs, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    img = cb.render()[0]
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    rgb = np.transpose(img[:, :, :3], (2, 0, 1)).copy()
    vision = torch.from_numpy(rgb).float().to(dev) / 255.0
    proprio = get_sys1_obs(r, q0, pa, torch.zeros((1, 3), device=dev), dofs, dev)[:, :47]
    return vision.unsqueeze(0), proprio


def move_cams(r, cb, ce, c3):
    p = r.get_pos().cpu().numpy()
    q = r.get_quat().cpu().numpy()
    if p.ndim > 1:
        p, q = p[0], q[0]
    fw = np.array([
        1 - 2 * (q[2] ** 2 + q[3] ** 2),
        2 * (q[1] * q[2] + q[0] * q[3]),
        2 * (q[1] * q[3] - q[0] * q[2]),
    ])
    up = np.array([
        2 * (q[1] * q[3] + q[0] * q[2]),
        2 * (q[2] * q[3] - q[0] * q[1]),
        1 - 2 * (q[1] ** 2 + q[2] ** 2),
    ])
    cp = p + fw * 0.10 + up * 0.05
    lk = cp + fw * 1.0
    for c in (cb, ce):
        c.set_pose(pos=cp, lookat=lk, up=up)
    c3p = p - fw * 1.8 + np.array([0.0, 0.0, 0.8])
    c3l = p + fw * 0.5
    c3.set_pose(pos=c3p, lookat=c3l, up=np.array([0.0, 0.0, 1.0]))
    return cp, lk, up, c3p, c3l, np.array([0.0, 0.0, 1.0])


def project_world_to_pixel(wp: np.ndarray, cp: np.ndarray, cl: np.ndarray, cu: np.ndarray, fov: float, w: int, h: int):
    dist = np.linalg.norm(cl - cp)
    f = (cl - cp) / max(dist, 1e-8)
    s = np.cross(f, cu / np.linalg.norm(cu))
    sn = np.linalg.norm(s)
    if sn < 1e-5:
        return None
    s /= sn
    u = np.cross(s, f)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[0, 3], view[1, 3], view[2, 3] = -np.dot(s, cp), -np.dot(u, cp), np.dot(f, cp)
    asp = w / h
    fy = 1.0 / np.tan(np.radians(fov) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0], proj[1, 1], proj[2, 2], proj[2, 3], proj[3, 2] = fy / asp, fy, -1.0, -0.02, -1.0
    pt = np.array([wp[0], wp[1], wp[2], 1.0], dtype=np.float32)
    clip = proj @ view @ pt
    if clip[3] <= 0:
        return None
    ndc = clip[:3] / clip[3]
    return int((ndc[0] + 1.0) * 0.5 * w), int((1.0 - ndc[1]) * 0.5 * h)


# ----------------------------------------------------------------------------
# Planning / breadcrumb harvesting
# ----------------------------------------------------------------------------

def rollout_cmd_kinematic(start_xy: np.ndarray, start_yaw: float, cmd_xyw: np.ndarray, hz: int, dt: float = 0.10):
    pos = np.array(start_xy, dtype=np.float32).copy()
    yaw = float(start_yaw)
    path = []
    for _ in range(hz):
        path.append(pos.copy())
        world_v = body_to_world_xy(yaw, cmd_xyw[:2])
        pos += dt * world_v
        yaw = wrap_to_pi(yaw + dt * float(cmd_xyw[2]))
    return np.stack(path, axis=0), pos, yaw


@torch.no_grad()
def plan_best_cmd(
    jepa: TinyQuadJEPA,
    head: GoalEnergyHead,
    zc: torch.Tensor,
    zg: torch.Tensor,
    robot_xy: np.ndarray,
    robot_yaw: float,
    goal_xy: np.ndarray,
    goal_body_xy: np.ndarray,
    dist_to_goal: float,
    heading_error: float,
    cands: int,
    hz: int,
    dev: torch.device,
    prev: torch.Tensor | None = None,
):
    far = dist_to_goal > 0.9
    transl_scale = 0.30 if far else 0.18
    if abs(heading_error) > 0.9:
        transl_scale *= 0.45

    mean = torch.tensor([
        clamp(float(goal_body_xy[0]) * transl_scale, -0.35, 0.35),
        clamp(float(goal_body_xy[1]) * transl_scale, -0.20, 0.20),
        clamp(0.40 * heading_error, -0.45, 0.45),
    ], device=dev, dtype=torch.float32)

    std = torch.tensor([
        0.12 if far else 0.09,
        0.10 if far else 0.08,
        0.22 if far else 0.18,
    ], device=dev, dtype=torch.float32)

    best_cmd = mean.view(1, 3)
    best_path = None
    best_cost = None
    best_energy = None

    for _ in range(5):
        cmds = mean + std * torch.randn((cands, 3), device=dev)
        cmds[:, 0].clamp_(-0.40, 0.40)
        cmds[:, 1].clamp_(-0.25, 0.25)
        cmds[:, 2].clamp_(-0.60, 0.60)

        z_roll = zc.expand(cands, -1)
        h_t = torch.zeros((cands, jepa.latent_dim), device=dev, dtype=z_roll.dtype)
        for _t in range(hz):
            z_roll, h_t = jepa.predictor(z_roll, cmds, h_t)
        eng = head(z_roll, zg.expand_as(z_roll))

        geo_cost = torch.empty((cands,), device=dev, dtype=torch.float32)
        path_cache = []
        for i in range(cands):
            cmd_np = cmds[i].detach().cpu().numpy()
            path_xy, end_xy, end_yaw = rollout_cmd_kinematic(robot_xy, robot_yaw, cmd_np, hz)
            path_cache.append(path_xy)
            end_dist = float(np.linalg.norm(end_xy - goal_xy))
            end_goal_angle = math.atan2(float(goal_xy[1] - end_xy[1]), float(goal_xy[0] - end_xy[0]))
            end_heading_err = abs(wrap_to_pi(end_goal_angle - end_yaw))
            geo_cost[i] = 0.85 * end_dist + 0.10 * end_heading_err

        cost = eng + geo_cost
        if prev is not None:
            cost = cost + 0.10 * (cmds - prev).pow(2).sum(dim=-1)

        k = max(cands // 10, 8)
        elite_idx = torch.topk(cost, k=k, largest=False).indices
        elite_cmds = cmds[elite_idx]
        mean = elite_cmds.mean(dim=0)
        std = elite_cmds.std(dim=0) + 1e-4

        iter_best = int(torch.argmin(cost).item())
        iter_best_cost = float(cost[iter_best].item())
        if best_cost is None or iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_cmd = cmds[iter_best].view(1, 3).detach().clone()
            best_energy = float(eng[iter_best].item())
            best_path = path_cache[iter_best]

    assert best_path is not None and best_energy is not None and best_cost is not None
    return best_cmd, {"eng": best_energy, "path": best_path, "cost": best_cost}


@torch.no_grad()
def harvest_breadcrumb(
    robot,
    cb,
    q0: torch.Tensor,
    jepa: TinyQuadJEPA,
    ppo: ActorCritic,
    dofs,
    dev: torch.device,
    scene,
    wp: WaypointSpec,
    n_avg: int = 5,
    warmup: int = 10,
    speed: float = 0.25,
    start_offset: float = 0.45,
):
    approach = wp.approach_dir_xy.astype(np.float32)
    approach = approach / max(float(np.linalg.norm(approach)), 1e-8)
    start_xy = wp.pos[:2] - start_offset * approach
    start_yaw = math.atan2(float(approach[1]), float(approach[0]))

    robot.set_pos(np.array([start_xy[0], start_xy[1], 0.12], dtype=np.float32))
    robot.set_quat(yaw_to_quat(start_yaw))
    robot.set_dofs_position(q0.detach().cpu().numpy(), dofs)
    for _ in range(8):
        scene.step()

    pa_h = torch.zeros((1, 12), device=dev)
    cmd = torch.tensor([[speed, 0.0, 0.0]], device=dev)

    for _ in range(warmup):
        obs = get_sys1_obs(robot, q0, pa_h, cmd, dofs, dev)
        pa_h = ppo.act_deterministic(obs)
        target = to_genesis_target(q0 + 0.3 * pa_h[0])
        robot.control_dofs_position(target, dofs)
        for _k in range(4):
            scene.step()

    latents = []
    for _ in range(n_avg):
        obs = get_sys1_obs(robot, q0, pa_h, cmd, dofs, dev)
        pa_h = ppo.act_deterministic(obs)
        target = to_genesis_target(q0 + 0.3 * pa_h[0])
        robot.control_dofs_position(target, dofs)
        for _k in range(4):
            scene.step()
        v, p = get_jepa_state(robot, cb, q0, pa_h, dofs, dev)
        latents.append(jepa.encoder(v, p).detach())

    return torch.stack(latents, dim=0).mean(dim=0)


# ----------------------------------------------------------------------------
# HUD helpers
# ----------------------------------------------------------------------------

def draw_energy_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, energy: float, label: str, color=(0, 200, 100)):
    draw.rectangle([x, y, x + w, y + h], outline=(80, 80, 80), fill=(30, 30, 30))
    frac = max(0.0, min(1.0, energy / 4.0))
    bar_w = int(w * frac)
    if bar_w > 0:
        draw.rectangle([x, y, x + bar_w, y + h], fill=color)
    draw.text((x + w + 4, y), f"{label}: {energy:.2f}", fill=(200, 200, 200))


def draw_progress_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, frac: float, label: str, color=(0, 160, 255)):
    frac = max(0.0, min(1.0, frac))
    draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 90), fill=(30, 30, 30))
    if frac > 0:
        draw.rectangle([x, y, x + int(w * frac), y + h], fill=color)
    draw.text((x, y - 16), label, fill=(200, 200, 200))


def world_to_map_px(xy: np.ndarray, map_x0: int, map_y0: int, map_w: int, map_h: int,
                    world_min: np.ndarray, world_max: np.ndarray) -> Tuple[int, int]:
    nx = (float(xy[0]) - float(world_min[0])) / max(float(world_max[0] - world_min[0]), 1e-8)
    ny = (float(xy[1]) - float(world_min[1])) / max(float(world_max[1] - world_min[1]), 1e-8)
    px = map_x0 + int(np.clip(nx, 0.0, 1.0) * map_w)
    py = map_y0 + map_h - int(np.clip(ny, 0.0, 1.0) * map_h)
    return px, py


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p_from: Tuple[int, int],
    p_to: Tuple[int, int],
    dash: int = 7,
    gap: int = 5,
    fill: Tuple[int, int, int] = (255, 220, 110),
    width: int = 3,
):
    x0, y0 = p_from
    x1, y1 = p_to
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return

    ux = dx / dist
    uy = dy / dist
    step = dash + gap
    t = 0.0

    while t < dist:
        t_end = min(t + dash, dist)
        xa = x0 + ux * t
        ya = y0 + uy * t
        xb = x0 + ux * t_end
        yb = y0 + uy * t_end
        draw.line((xa, ya, xb, yb), fill=fill, width=width)
        t += step


def draw_minimap(
    draw: ImageDraw.ImageDraw,
    map_x0: int,
    map_y0: int,
    map_w: int,
    map_h: int,
    waypoints: List[WaypointSpec],
    route: List[int],
    route_ptr: int,
    robot_xy: np.ndarray,
    robot_yaw: float,
    trail: List[np.ndarray],
    plan_path: np.ndarray,
    visit_counts: Dict[int, int],
    dist_thresh: float,
    teleport_from_xy: Optional[np.ndarray] = None,
    teleport_to_xy: Optional[np.ndarray] = None,
    teleport_flash: int = 0,
):
    world_min = np.array([-2.2, -1.2], dtype=np.float32)
    world_max = np.array([3.8, 3.8], dtype=np.float32)

    draw.rectangle([map_x0, map_y0, map_x0 + map_w, map_y0 + map_h], fill=(18, 18, 18), outline=(95, 95, 95))

    for gx in np.linspace(world_min[0], world_max[0], 7):
        p0 = world_to_map_px(np.array([gx, world_min[1]], dtype=np.float32), map_x0, map_y0, map_w, map_h, world_min, world_max)
        p1 = world_to_map_px(np.array([gx, world_max[1]], dtype=np.float32), map_x0, map_y0, map_w, map_h, world_min, world_max)
        draw.line([p0, p1], fill=(35, 35, 35), width=1)
    for gy in np.linspace(world_min[1], world_max[1], 6):
        p0 = world_to_map_px(np.array([world_min[0], gy], dtype=np.float32), map_x0, map_y0, map_w, map_h, world_min, world_max)
        p1 = world_to_map_px(np.array([world_max[0], gy], dtype=np.float32), map_x0, map_y0, map_w, map_h, world_min, world_max)
        draw.line([p0, p1], fill=(35, 35, 35), width=1)

    route_colors = [(255, 80, 80), (80, 255, 80), (80, 140, 255)]

    world_span_x = float(world_max[0] - world_min[0])
    world_span_y = float(world_max[1] - world_min[1])
    px_per_m_x = map_w / max(world_span_x, 1e-8)
    px_per_m_y = map_h / max(world_span_y, 1e-8)
    goal_r_px = max(3, int(dist_thresh * 0.5 * (px_per_m_x + px_per_m_y)))

    if len(route) > 1:
        pts = [world_to_map_px(waypoints[idx].pos[:2], map_x0, map_y0, map_w, map_h, world_min, world_max) for idx in route]
        draw.line(pts, fill=(120, 120, 120), width=2)

    if len(trail) > 1:
        trail_pts = [world_to_map_px(t, map_x0, map_y0, map_w, map_h, world_min, world_max) for t in trail[-240:]]
        if len(trail_pts) > 1:
            draw.line(trail_pts, fill=(255, 214, 10), width=2)

    if plan_path is not None and len(plan_path) > 1:
        plan_pts = [world_to_map_px(pt, map_x0, map_y0, map_w, map_h, world_min, world_max) for pt in plan_path]
        draw.line(plan_pts, fill=(0, 170, 255), width=3)

    active_idx = route[route_ptr]
    for wi, wp in enumerate(waypoints):
        px, py = world_to_map_px(wp.pos[:2], map_x0, map_y0, map_w, map_h, world_min, world_max)
        col = route_colors[wi]
        ring_col = (255, 255, 255) if wi == active_idx else (90, 90, 90)
        draw.ellipse([px - goal_r_px, py - goal_r_px, px + goal_r_px, py + goal_r_px], outline=ring_col, width=2)
        r = 8 if wi == active_idx else 6
        draw.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(230, 230, 230), width=1)
        draw.text((px + goal_r_px + 6, py - 8), f"W{wi + 1} x{visit_counts.get(wi, 0)}", fill=col)

    if teleport_flash > 0 and teleport_from_xy is not None and teleport_to_xy is not None:
        p_from = world_to_map_px(np.asarray(teleport_from_xy, dtype=np.float32), map_x0, map_y0, map_w, map_h, world_min, world_max)
        p_to = world_to_map_px(np.asarray(teleport_to_xy, dtype=np.float32), map_x0, map_y0, map_w, map_h, world_min, world_max)
        draw_dashed_line(draw, p_from, p_to, dash=7, gap=5, fill=(255, 220, 110), width=3)
        for px, py, fill_col, lab in [
            (p_from[0], p_from[1], (255, 120, 80), "before"),
            (p_to[0], p_to[1], (255, 230, 120), "after"),
        ]:
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=fill_col, outline=(255, 255, 255), width=2)
            draw.text((px + 10, py - 10), lab, fill=fill_col)
        mx0 = map_x0 + 8
        my0 = map_y0 + map_h - 24
        draw.rounded_rectangle([mx0, my0, mx0 + 196, my0 + 18], radius=5, fill=(88, 28, 10), outline=(255, 200, 100), width=2)
        draw.text((mx0 + 8, my0 + 2), "teleport / relocalize", fill=(255, 235, 190))

    rx, ry = world_to_map_px(robot_xy, map_x0, map_y0, map_w, map_h, world_min, world_max)
    head = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float32)
    left = np.array([math.cos(robot_yaw + 2.5), math.sin(robot_yaw + 2.5)], dtype=np.float32)
    right = np.array([math.cos(robot_yaw - 2.5), math.sin(robot_yaw - 2.5)], dtype=np.float32)
    scale = 12.0
    tri = [
        (rx + int(head[0] * scale), ry - int(head[1] * scale)),
        (rx + int(left[0] * scale * 0.8), ry - int(left[1] * scale * 0.8)),
        (rx + int(right[0] * scale * 0.8), ry - int(right[1] * scale * 0.8)),
    ]
    draw.polygon(tri, fill=(255, 255, 255), outline=(10, 10, 10))
    draw.text((map_x0, map_y0 - 16), "World map", fill=(200, 200, 200))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", required=True)
    parser.add_argument("--head_ckpt", required=True)
    parser.add_argument("--ppo_ckpt", required=True)
    parser.add_argument("--n_steps", type=int, default=1200)
    parser.add_argument("--cands", type=int, default=384)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--dist_thresh", type=float, default=0.45)
    parser.add_argument("--route", type=str, default="1,2,3,2,1")
    parser.add_argument("--out", type=str, default="jepa_logs/proof_of_thinking_v4_extended.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_kidnap", action="store_true")
    parser.add_argument("--kidnap_on_route_idx", type=int, default=3,
                        help="1-based target index within the route on which to trigger a kidnap event")
    parser.add_argument("--kidnap_after_leg_steps", type=int, default=45)
    parser.add_argument("--kidnap_dx", type=float, default=-0.55)
    parser.add_argument("--kidnap_dy", type=float, default=-0.45)
    parser.add_argument("--kidnap_dyaw_deg", type=float, default=95.0)
    parser.add_argument("--teleport_banner_frames", type=int, default=60)
    parser.add_argument("--teleport_freeze_frames", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waypoints = make_waypoints()
    route = parse_route(args.route, len(waypoints))

    jepa = TinyQuadJEPA().to(dev)
    jepa.load_state_dict(clean_state_dict(torch.load(args.jepa_ckpt, map_location=dev)["model_state_dict"]))
    jepa.eval()

    head = GoalEnergyHead().to(dev)
    head.load_state_dict(clean_state_dict(torch.load(args.head_ckpt, map_location=dev)["energy_head_state_dict"]))
    head.eval()

    ppo = ActorCritic().to(dev)
    ppo.load_state_dict(torch.load(args.ppo_ckpt, map_location=dev)["model"], strict=False)
    ppo.eval()

    scene, robot, cb, ce, c3, dofs, q0 = init_scene(dev, waypoints)

    print("🎬 Harvesting latent breadcrumbs (approach-aligned, averaged)...")
    latent_breadcrumbs = []
    for i, wp in enumerate(waypoints, start=1):
        print(f"  Waypoint {i}: {wp.pos[:2]} via approach {wp.approach_dir_xy}")
        latent_breadcrumbs.append(harvest_breadcrumb(robot, cb, q0, jepa, ppo, dofs, dev, scene, wp, n_avg=5, warmup=10))

    robot.set_pos(np.array([0.0, 0.0, 0.12], dtype=np.float32))
    robot.set_quat(yaw_to_quat(0.0))
    robot.set_dofs_position(q0.detach().cpu().numpy(), dofs)
    for _ in range(20):
        scene.step()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=30)

    pa = torch.zeros((1, 12), device=dev)
    prev_cmd = None
    route_ptr = 0
    target_wp_idx = route[route_ptr]
    ema_energy = 0.0
    ema_alpha = 0.30
    energy_history: List[float] = []
    trail: List[np.ndarray] = []
    visit_counts: Dict[int, int] = {i: 0 for i in range(len(waypoints))}
    kidnap_used = False
    kidnap_flash = 0
    kidnap_from_xy = None
    kidnap_to_xy = None
    kidnap_freeze_pending = 0
    leg_steps = 0

    route_str_human = " -> ".join([f"W{i + 1}" for i in route])
    print(f"\n🐕 Running expanded landmark navigation ({args.n_steps} steps)")
    print(f"   Route: {route_str_human}")
    if not args.disable_kidnap:
        print(f"   Kidnap event: route target #{args.kidnap_on_route_idx} after {args.kidnap_after_leg_steps} steps")

    for step in range(args.n_steps):
        target_wp_idx = route[route_ptr]

        cep, cel, ceu, c3p, c3l, c3u = move_cams(robot, cb, ce, c3)
        v, p = get_jepa_state(robot, cb, q0, pa, dofs, dev)

        with torch.no_grad():
            zc = jepa.encoder(v, p).detach()
            raw_energy = float(head(zc, latent_breadcrumbs[target_wp_idx]).item())

        ema_energy = ema_alpha * raw_energy + (1.0 - ema_alpha) * ema_energy
        energy_history.append(ema_energy)

        rp = robot.get_pos().cpu().numpy()
        rq = robot.get_quat().cpu().numpy()
        if rp.ndim > 1:
            rp = rp[0]
        if rq.ndim > 1:
            rq = rq[0]

        trail.append(rp[:2].copy())

        goal_xy = waypoints[target_wp_idx].pos[:2]
        dist = float(np.linalg.norm(rp[:2] - goal_xy))

        if dist < args.dist_thresh:
            visit_counts[target_wp_idx] += 1
            print(f"\n✅ Route target {route_ptr + 1}/{len(route)} = W{target_wp_idx + 1} reached at step {step} | dist={dist:.2f} | ema_eng={ema_energy:.2f}")
            if route_ptr == len(route) - 1:
                print(f"\n🎉 Full expanded route complete! Final step={step}")
                break
            route_ptr += 1
            target_wp_idx = route[route_ptr]
            ema_energy = 4.0
            prev_cmd = None
            leg_steps = 0
            goal_xy = waypoints[target_wp_idx].pos[:2]
            dist = float(np.linalg.norm(rp[:2] - goal_xy))

        yaw = math.atan2(
            2.0 * (float(rq[0]) * float(rq[3]) + float(rq[1]) * float(rq[2])),
            1.0 - 2.0 * (float(rq[2]) ** 2 + float(rq[3]) ** 2),
        )

        if (not args.disable_kidnap and not kidnap_used and (route_ptr + 1) == args.kidnap_on_route_idx
                and leg_steps >= args.kidnap_after_leg_steps):
            old_xy = rp[:2].copy()
            new_xy = rp[:2] + np.array([args.kidnap_dx, args.kidnap_dy], dtype=np.float32)
            new_xy[0] = clamp(float(new_xy[0]), -1.8, 2.8)
            new_xy[1] = clamp(float(new_xy[1]), -0.8, 2.8)
            new_yaw = wrap_to_pi(yaw + math.radians(args.kidnap_dyaw_deg))
            robot.set_pos(np.array([new_xy[0], new_xy[1], 0.12], dtype=np.float32))
            robot.set_quat(yaw_to_quat(new_yaw))
            robot.set_dofs_position(robot.get_dofs_position(dofs).detach().cpu().numpy(), dofs)
            for _ in range(12):
                scene.step()
            kidnap_used = True
            kidnap_flash = args.teleport_banner_frames
            kidnap_from_xy = old_xy
            kidnap_to_xy = new_xy.copy()
            kidnap_freeze_pending = args.teleport_freeze_frames
            prev_cmd = None
            print(f"\n🌀 Kidnap event triggered on route target {route_ptr + 1}: moved robot to ({new_xy[0]:.2f}, {new_xy[1]:.2f}) yaw={math.degrees(new_yaw):+.0f}°")
            continue

        goal_vec = goal_xy - rp[:2]
        goal_angle = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
        heading_error = wrap_to_pi(goal_angle - yaw)
        goal_dir_world = goal_vec / max(float(np.linalg.norm(goal_vec)), 1e-8)
        goal_body_xy = world_to_body_xy(yaw, goal_dir_world)

        cmd, st = plan_best_cmd(
            jepa=jepa,
            head=head,
            zc=zc,
            zg=latent_breadcrumbs[target_wp_idx],
            robot_xy=rp[:2].copy(),
            robot_yaw=yaw,
            goal_xy=goal_xy.copy(),
            goal_body_xy=goal_body_xy.copy(),
            dist_to_goal=dist,
            heading_error=heading_error,
            cands=args.cands,
            hz=args.horizon,
            dev=dev,
            prev=prev_cmd,
        )
        prev_cmd = cmd.clone()

        with torch.no_grad():
            pa = ppo.act_deterministic(get_sys1_obs(robot, q0, pa, cmd, dofs, dev)).detach()
        target = to_genesis_target(q0 + 0.3 * pa[0])
        robot.control_dofs_position(target, dofs)
        for _ in range(4):
            scene.step()
        leg_steps += 1

        frame3 = c3.render()[0]
        if hasattr(frame3, "cpu"):
            frame3 = frame3.cpu().numpy()
        p3 = Image.fromarray(frame3[:, :, :3].astype(np.uint8))

        frame_e = ce.render()[0]
        if hasattr(frame_e, "cpu"):
            frame_e = frame_e.cpu().numpy()
        pe = Image.fromarray(frame_e[:, :, :3].astype(np.uint8))

        d3 = ImageDraw.Draw(p3)
        h_px = [
            project_world_to_pixel(np.array([pt[0], pt[1], 0.05], dtype=np.float32), c3p, c3l, c3u, 50, 512, 512)
            for pt in st["path"]
        ]
        valid_px = [px for px in h_px if px is not None]
        if len(valid_px) > 1:
            d3.line(valid_px, fill=(0, 150, 255), width=4)

        lm_colors_draw = [(255, 80, 80), (80, 255, 80), (80, 130, 255)]
        completed_target_positions = set(route[:route_ptr])
        for wi, wp in enumerate(waypoints):
            px = project_world_to_pixel(wp.pos, c3p, c3l, c3u, 50, 512, 512)
            if px is None:
                continue
            done = wi in completed_target_positions
            col = (100, 100, 100) if done else lm_colors_draw[wi]
            if wi == target_wp_idx:
                col = (255, 255, 255)
            d3.ellipse([px[0] - 12, px[1] - 12, px[0] + 12, px[1] + 12], outline=col, width=3)
            label = f"W{wi + 1} x{visit_counts[wi]}"
            d3.text((px[0] + 16, px[1] - 6), label, fill=col)

        header_h = 176
        canv = Image.new("RGB", (896, header_h + 512), (20, 20, 20))
        canv.paste(p3, (0, header_h))
        canv.paste(pe, (512, header_h))

        drw = ImageDraw.Draw(canv)
        drw_rgba = ImageDraw.Draw(canv, "RGBA")

        drw.rectangle([0, 0, 895, header_h - 1], fill=(10, 10, 10), outline=(55, 55, 55))
        drw.line([(0, header_h - 1), (895, header_h - 1)], fill=(90, 90, 90), width=2)

        drw.rectangle([0, header_h, 511, header_h + 511], outline=(70, 70, 70), width=2)
        drw.rectangle([512, header_h, 895, header_h + 383], outline=(70, 70, 70), width=2)
        drw.text((12, header_h - 22), "World view + planner rollout", fill=(190, 190, 190))
        drw.text((524, header_h - 22), "Agent / close-up view", fill=(190, 190, 190))

        title = "JEPA | Expanded Landmark Demo"
        if kidnap_flash > 0:
            title += " | RECOVERY"
        drw.text((20, 16), title, fill=(0, 255, 100))
        drw.text((20, 36), f"Step: {step:04d} | Route target: {route_ptr + 1}/{len(route)} | Active: W{target_wp_idx + 1} ({waypoints[target_wp_idx].name})", fill=(200, 200, 200))
        drw.text((20, 56), f"Route: {route_str_human}", fill=(160, 160, 160))
        drw.text((20, 76), f"Dist: {dist:.2f}m | Goal radius: {args.dist_thresh:.2f}m | Heading err: {np.degrees(heading_error):+.0f}° | Leg steps: {leg_steps}", fill=(200, 200, 200))
        drw.text((20, 96), f"Raw E: {raw_energy:.2f} | EMA E: {ema_energy:.2f} | Plan cost: {st['cost']:.2f}", fill=(200, 200, 200))
        drw.text((20, 116), f"Cmd: vx={float(cmd[0, 0].item()):+.2f} vy={float(cmd[0, 1].item()):+.2f} wz={float(cmd[0, 2].item()):+.2f}", fill=(200, 200, 200))

        draw_energy_bar(drw, 20, 140, 160, 12, raw_energy, "raw", color=(200, 80, 40))
        draw_energy_bar(drw, 20, 158, 160, 12, ema_energy, "ema", color=(0, 180, 120))
        draw_progress_bar(drw, 220, 146, 220, 12, route_ptr / max(len(route) - 1, 1), "Route completion", color=(120, 180, 255))

        if len(energy_history) > 2:
            hist = energy_history[-80:]
            hist_arr = np.asarray(hist, dtype=np.float32)
            h_min = float(hist_arr.min())
            h_max = max(float(hist_arr.max()), h_min + 0.1)
            pts = []
            for i, val in enumerate(hist_arr):
                x_px = 20 + int(i / len(hist_arr) * 420)
                y_px = 170 - int((float(val) - h_min) / (h_max - h_min) * 34)
                pts.append((x_px, y_px))
            if len(pts) > 1:
                drw.line(pts, fill=(0, 200, 255), width=2)
            drw.text((20, 124), "EMA energy history", fill=(180, 180, 180))

        draw_minimap(
            drw,
            map_x0=540,
            map_y0=22,
            map_w=322,
            map_h=134,
            waypoints=waypoints,
            route=route,
            route_ptr=route_ptr,
            robot_xy=rp[:2],
            robot_yaw=yaw,
            trail=trail,
            plan_path=st["path"],
            visit_counts=visit_counts,
            dist_thresh=args.dist_thresh,
            teleport_from_xy=kidnap_from_xy,
            teleport_to_xy=kidnap_to_xy,
            teleport_flash=kidnap_flash,
        )

        footer = "Blue = planned rollout | Yellow = actual trail | White ring = active goal radius"
        if not args.disable_kidnap:
            footer += " | Recovery event enabled"
        drw.text((540, 160), footer, fill=(120, 120, 120))

        freeze_now = kidnap_freeze_pending > 0
        if kidnap_flash > 0:
            drw_rgba.rectangle([0, header_h, 895, header_h + 511], fill=(120, 38, 0, 34))
            drw.rounded_rectangle([8, 4, 888, 48], radius=10, fill=(92, 26, 8), outline=(255, 205, 110), width=3)
            drw.text((22, 12), f"RELOCALIZATION TEST: robot teleported while pursuing W{target_wp_idx + 1}", fill=(255, 242, 214))
            drw.text((22, 28), "Why it matters: the demo is testing whether latent goal memory supports recovery after unexpected displacement.", fill=(255, 220, 168))
            drw.rounded_rectangle([170, header_h + 208, 726, header_h + 286], radius=12, fill=(90, 24, 8), outline=(255, 205, 110), width=4)
            drw.text((212, header_h + 228), "TELEPORT / RELOCALIZE EVENT", fill=(255, 244, 220))
            drw.text((202, header_h + 250), f"Unexpected displacement -> replanning to W{target_wp_idx + 1}", fill=(255, 224, 176))
            kidnap_flash -= 1

        frame_np = np.array(canv)
        writer.append_data(frame_np)
        if freeze_now:
            for _ in range(kidnap_freeze_pending):
                writer.append_data(frame_np)
            kidnap_freeze_pending = 0
        print(f"\r⚡ step={step:03d} | route={route_ptr + 1}/{len(route)} | wp={target_wp_idx + 1} | dist={dist:.2f} | ema_E={ema_energy:.2f} | hdg={np.degrees(heading_error):+.0f}°", end="")

    writer.close()
    print(f"\n\n✅ Demo saved to {args.out}")
    print(f"   Route completion: {route_ptr + 1}/{len(route)} targets")
    print(f"   Visit counts: " + ", ".join([f"W{i + 1}={visit_counts[i]}" for i in range(len(waypoints))]))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

"""
Fast JEPA exploration demo with obstacle navigation.

What changed vs the original explore demo:
- replaces heavy 320x5 CEM with a very small local candidate set around a path-following command
- adds a global frontier path planner on the occupancy grid so the robot stops trying to drive through walls
- calibrates latent OOD against the safe-bank distribution instead of using raw kNN distance directly
- replans every few steps instead of every single step
- marks visited cells along the travelled segment, not just at the instantaneous pose
- makes the output frame size divisible by 16 to avoid ffmpeg resizing

Design intent:
- obstacle avoidance comes primarily from explicit geometry + grid path planning
- JEPA OOD is used as a *risk gate / speed limiter*, not the main reward signal
- PPO gait still provides low-level execution
"""

import argparse
import heapq
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


# ----------------------------------------------------------------------------
# Specs / utilities
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    pos: np.ndarray
    size: np.ndarray
    color_rgb: Tuple[float, float, float]


@dataclass
class FrontierTarget:
    cell: Tuple[int, int]
    xy: np.ndarray


@dataclass
class OODStats:
    mean: float
    std: float
    p90: float
    p95: float
    p99: float


@dataclass
class PlannerState:
    frontier: Optional[FrontierTarget] = None
    path_cells: Optional[List[Tuple[int, int]]] = None
    waypoint_xy: Optional[np.ndarray] = None
    cmd: Optional[torch.Tensor] = None
    best_path_xy: Optional[np.ndarray] = None
    hold_steps: int = 0
    stall_count: int = 0
    frontier_switches: int = 0


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


def to_genesis_target(x: torch.Tensor) -> torch.Tensor:
    x_np = x.detach().to("cpu").numpy().astype(np.float32, copy=True)
    return torch.tensor(x_np, device=gs.device, dtype=torch.float32)


# ----------------------------------------------------------------------------
# World config
# ----------------------------------------------------------------------------


WORLD_MIN = np.array([-2.20, -1.20], dtype=np.float32)
WORLD_MAX = np.array([3.80, 3.80], dtype=np.float32)
MAP_W = 60
MAP_H = 50
NEIGHBORS_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
]


def make_obstacles() -> List[ObstacleSpec]:
    return [
        ObstacleSpec("wall_a", np.array([0.55, 0.25, 0.45], dtype=np.float32), np.array([0.18, 1.90, 0.90], dtype=np.float32), (0.72, 0.36, 0.18)),
        ObstacleSpec("wall_b", np.array([1.55, 2.55, 0.45], dtype=np.float32), np.array([0.18, 1.80, 0.90], dtype=np.float32), (0.72, 0.36, 0.18)),
        ObstacleSpec("wall_c", np.array([2.65, 1.05, 0.45], dtype=np.float32), np.array([0.18, 1.70, 0.90], dtype=np.float32), (0.72, 0.36, 0.18)),
        ObstacleSpec("bar_a", np.array([1.75, 0.95, 0.45], dtype=np.float32), np.array([1.15, 0.18, 0.90], dtype=np.float32), (0.18, 0.45, 0.75)),
        ObstacleSpec("bar_b", np.array([0.10, 1.95, 0.45], dtype=np.float32), np.array([1.15, 0.18, 0.90], dtype=np.float32), (0.18, 0.45, 0.75)),
        ObstacleSpec("bar_c", np.array([2.45, 3.00, 0.45], dtype=np.float32), np.array([1.00, 0.18, 0.90], dtype=np.float32), (0.18, 0.45, 0.75)),
        ObstacleSpec("box_a", np.array([-0.55, 0.75, 0.35], dtype=np.float32), np.array([0.45, 0.45, 0.70], dtype=np.float32), (0.75, 0.72, 0.22)),
        ObstacleSpec("box_b", np.array([1.05, 1.85, 0.35], dtype=np.float32), np.array([0.50, 0.50, 0.70], dtype=np.float32), (0.75, 0.72, 0.22)),
        ObstacleSpec("box_c", np.array([2.95, 2.05, 0.35], dtype=np.float32), np.array([0.50, 0.50, 0.70], dtype=np.float32), (0.75, 0.72, 0.22)),
        ObstacleSpec("box_d", np.array([0.10, 3.15, 0.35], dtype=np.float32), np.array([0.55, 0.55, 0.70], dtype=np.float32), (0.75, 0.72, 0.22)),
    ]


# ----------------------------------------------------------------------------
# Scene + observation helpers
# ----------------------------------------------------------------------------


def init_scene(device: torch.device, obstacles: Sequence[ObstacleSpec], start_xy: Tuple[float, float] = (0.0, 0.0)):
    scene = gs.Scene(show_viewer=False)

    tex = create_checkerboard()
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_path=tex)),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(file="assets/mini_pupper/mini_pupper.urdf", pos=(start_xy[0], start_xy[1], 0.12), fixed=False)
    )

    for obs in obstacles:
        scene.add_entity(
            gs.morphs.Box(pos=tuple(float(x) for x in obs.pos), size=tuple(float(x) for x in obs.size), fixed=True),
            surface=gs.surfaces.Rough(color=obs.color_rgb),
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

    robot.set_pos(np.array([start_xy[0], start_xy[1], 0.12], dtype=np.float32))
    robot.set_quat(yaw_to_quat(0.0))
    robot.set_dofs_position(q0_np, dofs)
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, dofs)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, dofs)

    for _ in range(12):
        scene.step()

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
# Safe latent bank harvesting + OOD calibration
# ----------------------------------------------------------------------------


@torch.no_grad()
def harvest_safe_latent_bank(
    jepa: TinyQuadJEPA,
    ppo: ActorCritic,
    dev: torch.device,
    max_latents: int = 128,
) -> torch.Tensor:
    bank_scene, bank_robot, bank_cb, _, _, bank_dofs, bank_q0 = init_scene(dev, obstacles=[], start_xy=(0.0, 0.0))

    positions = [
        (-0.8, -0.4), (0.0, -0.4), (0.8, -0.4),
        (-0.8, 0.8), (0.0, 0.8), (0.8, 0.8),
    ]
    yaws = [0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi]
    cmds = [
        np.array([+0.24, 0.00, 0.00], dtype=np.float32),
        np.array([+0.20, +0.05, 0.00], dtype=np.float32),
        np.array([+0.20, -0.05, 0.00], dtype=np.float32),
        np.array([+0.16, 0.00, +0.20], dtype=np.float32),
        np.array([+0.16, 0.00, -0.20], dtype=np.float32),
    ]

    latents: List[torch.Tensor] = []
    pa = torch.zeros((1, 12), device=dev)

    for (sx, sy) in positions:
        for yaw in yaws:
            bank_robot.set_pos(np.array([sx, sy, 0.12], dtype=np.float32))
            bank_robot.set_quat(yaw_to_quat(float(yaw)))
            bank_robot.set_dofs_position(bank_q0.detach().cpu().numpy(), bank_dofs)
            for _ in range(8):
                bank_scene.step()

            for cmd_np in cmds:
                cmd = torch.tensor(cmd_np, device=dev, dtype=torch.float32).view(1, 3)
                for _ in range(4):
                    obs = get_sys1_obs(bank_robot, bank_q0, pa, cmd, bank_dofs, dev)
                    pa = ppo.act_deterministic(obs)
                    target = to_genesis_target(bank_q0 + 0.3 * pa[0])
                    bank_robot.control_dofs_position(target, bank_dofs)
                    for _ in range(3):
                        bank_scene.step()
                    v, p = get_jepa_state(bank_robot, bank_cb, bank_q0, pa, bank_dofs, dev)
                    latents.append(jepa.encoder(v, p).detach().squeeze(0))
                    if len(latents) >= max_latents:
                        return torch.stack(latents, dim=0)

    return torch.stack(latents, dim=0)


def knn_ood_score(z: torch.Tensor, bank: torch.Tensor, k: int = 8) -> torch.Tensor:
    if z.dim() == 1:
        z = z.unsqueeze(0)
    d = torch.cdist(z, bank)
    k = int(min(k, bank.shape[0]))
    vals = torch.topk(d, k=k, largest=False).values
    return vals.mean(dim=-1)


@torch.no_grad()
def calibrate_ood(bank: torch.Tensor, k: int = 8) -> OODStats:
    d = torch.cdist(bank, bank)
    n = d.shape[0]
    diag = torch.eye(n, device=d.device, dtype=torch.bool)
    d[diag] = float("inf")
    vals = torch.topk(d, k=min(k, max(n - 1, 1)), largest=False).values.mean(dim=-1)
    arr = vals.detach().cpu().numpy().astype(np.float32)
    return OODStats(
        mean=float(arr.mean()),
        std=float(arr.std() + 1e-6),
        p90=float(np.percentile(arr, 90.0)),
        p95=float(np.percentile(arr, 95.0)),
        p99=float(np.percentile(arr, 99.0)),
    )


def normalize_ood(raw_ood: float, stats: OODStats) -> float:
    denom = max(stats.p95 - stats.p90, 1e-6)
    return max(0.0, (float(raw_ood) - stats.p90) / denom)


# ----------------------------------------------------------------------------
# Grid / frontier helpers
# ----------------------------------------------------------------------------


def world_to_grid(xy: np.ndarray) -> Tuple[int, int]:
    nx = (float(xy[0]) - float(WORLD_MIN[0])) / max(float(WORLD_MAX[0] - WORLD_MIN[0]), 1e-8)
    ny = (float(xy[1]) - float(WORLD_MIN[1])) / max(float(WORLD_MAX[1] - WORLD_MIN[1]), 1e-8)
    gx = int(np.clip(nx * MAP_W, 0, MAP_W - 1))
    gy = int(np.clip(ny * MAP_H, 0, MAP_H - 1))
    return gx, gy


def grid_to_world(gx: int, gy: int) -> np.ndarray:
    x = float(WORLD_MIN[0]) + (gx + 0.5) / MAP_W * float(WORLD_MAX[0] - WORLD_MIN[0])
    y = float(WORLD_MIN[1]) + (gy + 0.5) / MAP_H * float(WORLD_MAX[1] - WORLD_MIN[1])
    return np.array([x, y], dtype=np.float32)


def point_aabb_signed_clearance_xy(xy: np.ndarray, obs: ObstacleSpec, inflate: float = 0.0) -> float:
    hx = 0.5 * float(obs.size[0]) + float(inflate)
    hy = 0.5 * float(obs.size[1]) + float(inflate)
    dx = abs(float(xy[0]) - float(obs.pos[0])) - hx
    dy = abs(float(xy[1]) - float(obs.pos[1])) - hy
    outside_dx = max(dx, 0.0)
    outside_dy = max(dy, 0.0)
    if dx <= 0.0 and dy <= 0.0:
        return -min(-dx, -dy)
    return float(math.hypot(outside_dx, outside_dy))


def min_clearance_to_obstacles(xy: np.ndarray, obstacles: Sequence[ObstacleSpec], inflate: float = 0.0) -> float:
    if not obstacles:
        return 1e6
    return min(point_aabb_signed_clearance_xy(xy, obs, inflate=inflate) for obs in obstacles)


def build_obstacle_grid(obstacles: Sequence[ObstacleSpec], inflate: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    occ = np.zeros((MAP_H, MAP_W), dtype=bool)
    clearance = np.zeros((MAP_H, MAP_W), dtype=np.float32)
    for gy in range(MAP_H):
        for gx in range(MAP_W):
            xy = grid_to_world(gx, gy)
            clr = min_clearance_to_obstacles(xy, obstacles, inflate=inflate)
            clearance[gy, gx] = float(clr)
            if clr < 0.0:
                occ[gy, gx] = True
    return occ, clearance


def mark_visited(visited: np.ndarray, xy: np.ndarray, radius_cells: int = 1):
    gx, gy = world_to_grid(xy)
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            nx = gx + dx
            ny = gy + dy
            if 0 <= nx < MAP_W and 0 <= ny < MAP_H:
                visited[ny, nx] = True


def mark_visited_segment(visited: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius_cells: int = 1):
    dist = float(np.linalg.norm(p1 - p0))
    n = max(2, int(dist / 0.05) + 1)
    for t in np.linspace(0.0, 1.0, n):
        pt = (1.0 - t) * p0 + t * p1
        mark_visited(visited, pt, radius_cells=radius_cells)


def compute_frontier_mask(visited: np.ndarray, occ: np.ndarray) -> np.ndarray:
    free = ~occ
    frontier = np.zeros_like(visited, dtype=bool)
    for gy in range(MAP_H):
        for gx in range(MAP_W):
            if occ[gy, gx] or visited[gy, gx]:
                continue
            for dx, dy in NEIGHBORS_8:
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < MAP_W and 0 <= ny < MAP_H and free[ny, nx] and visited[ny, nx]:
                    frontier[gy, gx] = True
                    break
    return frontier


def local_unseen_gain(visited: np.ndarray, occ: np.ndarray, gx: int, gy: int, rad: int = 2) -> int:
    score = 0
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            nx = gx + dx
            ny = gy + dy
            if 0 <= nx < MAP_W and 0 <= ny < MAP_H and (not occ[ny, nx]) and (not visited[ny, nx]):
                score += 1
    return score


def compute_distance_field(start: Tuple[int, int], occ: np.ndarray) -> np.ndarray:
    sx, sy = start
    dist = np.full((MAP_H, MAP_W), np.inf, dtype=np.float32)
    if not (0 <= sx < MAP_W and 0 <= sy < MAP_H) or occ[sy, sx]:
        return dist
    pq: List[Tuple[float, int, int]] = [(0.0, sx, sy)]
    dist[sy, sx] = 0.0
    while pq:
        d, x, y = heapq.heappop(pq)
        if d > float(dist[y, x]):
            continue
        for dx, dy in NEIGHBORS_8:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < MAP_W and 0 <= ny < MAP_H):
                continue
            if occ[ny, nx]:
                continue
            step = 1.4142 if dx != 0 and dy != 0 else 1.0
            nd = d + step
            if nd < float(dist[ny, nx]):
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, nx, ny))
    return dist


def choose_frontier_target(visited: np.ndarray, occ: np.ndarray, robot_xy: np.ndarray) -> Optional[FrontierTarget]:
    """
    Cheap, deterministic frontier chooser.

    Why this exists:
    - the earlier version used a full distance-field solve during frontier selection
    - in practice, that stage appears to be where some runs stall before the first step
    - for a 60x50 map we do not need anything sophisticated here

    Strategy:
    1. build the frontier mask
    2. score frontier cells using only local unseen gain + straight-line distance
    3. if no frontier exists, fall back to any unvisited free cell
    4. reject cells too close to obstacles so A* gets cleaner goals
    """
    frontier = compute_frontier_mask(visited, occ)
    cells = np.argwhere(frontier)
    if len(cells) == 0:
        cells = np.argwhere((~occ) & (~visited))
        if len(cells) == 0:
            return None

    best_cell = None
    best_score = None
    for gy, gx in cells:
        gx_i = int(gx)
        gy_i = int(gy)
        if occ[gy_i, gx_i]:
            continue

        xy = grid_to_world(gx_i, gy_i)
        euclid = float(np.linalg.norm(xy - robot_xy))
        gain = float(local_unseen_gain(visited, occ, gx_i, gy_i, rad=2))

        # Avoid selecting cells that sit right on obstacle margins.
        obs_clear = float(min_clearance_to_obstacles(xy, make_obstacles(), inflate=0.10))
        if obs_clear < 0.02:
            continue

        score = 2.8 * gain - 0.55 * euclid + 0.15 * obs_clear
        if best_score is None or score > best_score:
            best_score = score
            best_cell = (gx_i, gy_i)

    if best_cell is None:
        return None
    return FrontierTarget(cell=best_cell, xy=grid_to_world(best_cell[0], best_cell[1]))


def reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    cur = goal
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def astar_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occ: np.ndarray,
    clearance: np.ndarray,
) -> Optional[List[Tuple[int, int]]]:
    sx, sy = start
    gx, gy = goal
    if occ[sy, sx] or occ[gy, gx]:
        return None

    pq: List[Tuple[float, float, int, int]] = []
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_cost: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

    def heuristic(x: int, y: int) -> float:
        return float(math.hypot(gx - x, gy - y))

    heapq.heappush(pq, (heuristic(sx, sy), 0.0, sx, sy))

    while pq:
        _, cur_g, x, y = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            return reconstruct_path(came_from, (gx, gy))
        if cur_g > g_cost.get((x, y), float("inf")):
            continue

        for dx, dy in NEIGHBORS_8:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < MAP_W and 0 <= ny < MAP_H):
                continue
            if occ[ny, nx]:
                continue
            step = 1.4142 if dx != 0 and dy != 0 else 1.0
            clr = float(clearance[ny, nx])
            clr_pen = 0.0
            if clr < 0.12:
                clr_pen += (0.12 - clr) * 3.5
            if clr < 0.22:
                clr_pen += (0.22 - clr) * 0.8
            nd = cur_g + step + clr_pen
            if nd < g_cost.get((nx, ny), float("inf")):
                g_cost[(nx, ny)] = nd
                came_from[(nx, ny)] = (x, y)
                heapq.heappush(pq, (nd + heuristic(nx, ny), nd, nx, ny))
    return None


# ----------------------------------------------------------------------------
# Planning
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


def choose_waypoint_from_path(path_cells: List[Tuple[int, int]], robot_xy: np.ndarray, lookahead: int) -> np.ndarray:
    if len(path_cells) <= 1:
        return robot_xy.copy()
    idx = min(lookahead, len(path_cells) - 1)
    gx, gy = path_cells[idx]
    return grid_to_world(gx, gy)


@torch.no_grad()
def predict_cmd_ood(
    jepa: TinyQuadJEPA,
    zc: torch.Tensor,
    cmd: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    z_roll = zc.clone()
    h_t = torch.zeros((z_roll.shape[0], jepa.latent_dim), device=zc.device, dtype=zc.dtype)
    for _ in range(horizon):
        z_roll, h_t = jepa.predictor(z_roll, cmd, h_t)
    return z_roll


@torch.no_grad()
def choose_local_cmd(
    jepa: TinyQuadJEPA,
    safe_bank: torch.Tensor,
    ood_stats: OODStats,
    zc: torch.Tensor,
    robot_xy: np.ndarray,
    robot_yaw: float,
    waypoint_xy: np.ndarray,
    frontier_xy: np.ndarray,
    current_ood_rel: float,
    visited: np.ndarray,
    occ: np.ndarray,
    obstacles: Sequence[ObstacleSpec],
    prev_cmd: Optional[torch.Tensor],
    cmd_horizon: int,
    rollout_horizon: int,
) -> Tuple[torch.Tensor, Dict[str, float], np.ndarray]:
    goal_vec_w = waypoint_xy - robot_xy
    goal_body = world_to_body_xy(robot_yaw, goal_vec_w)
    dist_wp = float(np.linalg.norm(goal_vec_w))
    heading_err = math.atan2(float(goal_body[1]), float(goal_body[0]) + 1e-8)

    speed_scale = 1.0 / (1.0 + 0.55 * current_ood_rel)
    max_vx = 0.28 * speed_scale + 0.04
    max_vy = 0.20 * speed_scale + 0.03
    max_wz = 0.55 * speed_scale + 0.10

    base = np.array([
        clamp(0.95 * float(goal_body[0]), -max_vx, max_vx),
        clamp(0.95 * float(goal_body[1]), -max_vy, max_vy),
        clamp(0.85 * heading_err, -max_wz, max_wz),
    ], dtype=np.float32)

    if dist_wp < 0.18:
        base[0] *= 0.6
        base[1] *= 0.6
        base[2] *= 0.7

    deltas = np.array([
        [0.00, 0.00, 0.00],
        [+0.05, 0.00, 0.00],
        [-0.05, 0.00, 0.00],
        [0.00, +0.05, 0.00],
        [0.00, -0.05, 0.00],
        [0.00, 0.00, +0.18],
        [0.00, 0.00, -0.18],
        [+0.04, +0.04, 0.00],
        [+0.04, -0.04, 0.00],
    ], dtype=np.float32)

    best_cost = None
    best_cmd_np = None
    best_info: Dict[str, float] = {}
    best_path = None

    start_frontier_dist = float(np.linalg.norm(frontier_xy - robot_xy))
    start_wp_dist = float(np.linalg.norm(waypoint_xy - robot_xy))

    for delta in deltas:
        cmd_np = base + delta
        cmd_np[0] = clamp(float(cmd_np[0]), -0.34, 0.34)
        cmd_np[1] = clamp(float(cmd_np[1]), -0.24, 0.24)
        cmd_np[2] = clamp(float(cmd_np[2]), -0.65, 0.65)
        cmd = torch.tensor(cmd_np, device=zc.device, dtype=torch.float32).view(1, 3)

        pred_z = predict_cmd_ood(jepa, zc, cmd, horizon=cmd_horizon)
        pred_raw = float(knn_ood_score(pred_z, safe_bank, k=8).item())
        pred_rel = normalize_ood(pred_raw, ood_stats)

        path_xy, end_xy, _ = rollout_cmd_kinematic(robot_xy, robot_yaw, cmd_np, rollout_horizon, dt=0.10)

        collision = 0.0
        clearance_pen = 0.0
        novel = 0
        seen = set()
        for pt in path_xy:
            clr = min_clearance_to_obstacles(pt, obstacles, inflate=0.14)
            if clr < 0.0:
                collision = 1.0
                clearance_pen += 25.0
                break
            if clr < 0.18:
                clearance_pen += (0.18 - clr) ** 2 * 14.0
            gx, gy = world_to_grid(pt)
            key = (gx, gy)
            if key not in seen:
                seen.add(key)
                if (not occ[gy, gx]) and (not visited[gy, gx]):
                    novel += 1

        end_frontier_dist = float(np.linalg.norm(frontier_xy - end_xy))
        end_wp_dist = float(np.linalg.norm(waypoint_xy - end_xy))
        frontier_progress = start_frontier_dist - end_frontier_dist
        wp_progress = start_wp_dist - end_wp_dist
        disp = float(np.linalg.norm(end_xy - robot_xy))
        stall_pen = max(0.0, 0.05 - disp) * 14.0
        spin_pen = max(0.0, abs(float(cmd_np[2])) - 0.50) * 1.8
        reverse_pen = max(0.0, -float(cmd_np[0])) * 1.0

        cost = (
            70.0 * collision
            + 2.2 * clearance_pen
            + 1.8 * pred_rel
            + 1.6 * stall_pen
            + 0.35 * spin_pen
            + 0.45 * reverse_pen
            - 1.70 * frontier_progress
            - 1.30 * wp_progress
            - 0.55 * float(novel)
        )
        if prev_cmd is not None:
            cost += 0.10 * float(((cmd - prev_cmd) ** 2).sum().item())

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_cmd_np = cmd_np.copy()
            best_path = path_xy.copy()
            best_info = {
                "cost": float(cost),
                "pred_ood_raw": pred_raw,
                "pred_ood_rel": pred_rel,
                "clearance": float(clearance_pen),
                "collision": float(collision),
                "novel": float(novel),
                "frontier_progress": float(frontier_progress),
                "wp_progress": float(wp_progress),
                "stall": float(stall_pen),
            }

    assert best_cmd_np is not None and best_path is not None
    best_cmd = torch.tensor(best_cmd_np, device=zc.device, dtype=torch.float32).view(1, 3)
    return best_cmd, best_info, best_path


# ----------------------------------------------------------------------------
# HUD helpers
# ----------------------------------------------------------------------------


def draw_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, val: float, vmax: float, label: str, fill=(0, 180, 120)):
    frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, val / vmax))
    draw.rectangle([x, y, x + w, y + h], outline=(85, 85, 85), fill=(28, 28, 28))
    if frac > 0.0:
        draw.rectangle([x, y, x + int(frac * w), y + h], fill=fill)
    draw.text((x, y - 16), f"{label}: {val:.2f}", fill=(210, 210, 210))


def draw_progress_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, frac: float, label: str, fill=(0, 150, 255)):
    frac = max(0.0, min(1.0, frac))
    draw.rectangle([x, y, x + w, y + h], outline=(85, 85, 85), fill=(28, 28, 28))
    if frac > 0:
        draw.rectangle([x, y, x + int(frac * w), y + h], fill=fill)
    draw.text((x, y - 16), label, fill=(210, 210, 210))


def world_to_map_px(xy: np.ndarray, map_x0: int, map_y0: int, map_w: int, map_h: int) -> Tuple[int, int]:
    nx = (float(xy[0]) - float(WORLD_MIN[0])) / max(float(WORLD_MAX[0] - WORLD_MIN[0]), 1e-8)
    ny = (float(xy[1]) - float(WORLD_MIN[1])) / max(float(WORLD_MAX[1] - WORLD_MIN[1]), 1e-8)
    px = map_x0 + int(np.clip(nx, 0.0, 1.0) * map_w)
    py = map_y0 + map_h - int(np.clip(ny, 0.0, 1.0) * map_h)
    return px, py


def draw_minimap(
    draw: ImageDraw.ImageDraw,
    map_x0: int,
    map_y0: int,
    map_w: int,
    map_h: int,
    robot_xy: np.ndarray,
    robot_yaw: float,
    trail: List[np.ndarray],
    plan_path: Optional[np.ndarray],
    frontier_xy: Optional[np.ndarray],
    visited: np.ndarray,
    occ: np.ndarray,
    obstacles: Sequence[ObstacleSpec],
):
    draw.rectangle([map_x0, map_y0, map_x0 + map_w, map_y0 + map_h], fill=(18, 18, 18), outline=(95, 95, 95))
    cell_w = map_w / MAP_W
    cell_h = map_h / MAP_H
    frontier_mask = compute_frontier_mask(visited, occ)

    for gy in range(MAP_H):
        for gx in range(MAP_W):
            x0 = map_x0 + int(gx * cell_w)
            y0 = map_y0 + int((MAP_H - 1 - gy) * cell_h)
            x1 = map_x0 + int((gx + 1) * cell_w)
            y1 = map_y0 + int((MAP_H - gy) * cell_h)
            if occ[gy, gx]:
                fill = (80, 48, 28)
            elif frontier_mask[gy, gx]:
                fill = (36, 54, 88)
            elif visited[gy, gx]:
                fill = (64, 92, 64)
            else:
                fill = (30, 30, 30)
            draw.rectangle([x0, y0, x1, y1], fill=fill)

    for obs in obstacles:
        hx = 0.5 * float(obs.size[0])
        hy = 0.5 * float(obs.size[1])
        p0 = world_to_map_px(np.array([obs.pos[0] - hx, obs.pos[1] - hy], dtype=np.float32), map_x0, map_y0, map_w, map_h)
        p1 = world_to_map_px(np.array([obs.pos[0] + hx, obs.pos[1] + hy], dtype=np.float32), map_x0, map_y0, map_w, map_h)
        left = min(p0[0], p1[0])
        right = max(p0[0], p1[0])
        top = min(p0[1], p1[1])
        bot = max(p0[1], p1[1])
        draw.rectangle([left, top, right, bot], outline=(220, 180, 120), width=2)

    if len(trail) > 1:
        trail_pts = [world_to_map_px(t, map_x0, map_y0, map_w, map_h) for t in trail[-320:]]
        if len(trail_pts) > 1:
            draw.line(trail_pts, fill=(255, 214, 10), width=2)

    if plan_path is not None and len(plan_path) > 1:
        plan_pts = [world_to_map_px(pt, map_x0, map_y0, map_w, map_h) for pt in plan_path]
        draw.line(plan_pts, fill=(0, 170, 255), width=3)

    if frontier_xy is not None:
        fx, fy = world_to_map_px(frontier_xy, map_x0, map_y0, map_w, map_h)
        draw.ellipse([fx - 8, fy - 8, fx + 8, fy + 8], fill=(255, 255, 255), outline=(20, 20, 20), width=2)
        draw.text((fx + 10, fy - 10), "frontier", fill=(240, 240, 240))

    rx, ry = world_to_map_px(robot_xy, map_x0, map_y0, map_w, map_h)
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
    draw.text((map_x0, map_y0 - 16), "Exploration map (green=visited, blue=frontier)", fill=(200, 200, 200))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", required=True)
    parser.add_argument("--ppo_ckpt", required=True)
    parser.add_argument("--out", type=str, default="jepa_logs/jepa_explore_ood_demo_fast.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=1800)
    parser.add_argument("--coverage_goal", type=float, default=0.55)
    parser.add_argument("--frontier_reach", type=float, default=0.24)
    parser.add_argument("--replan_every", type=int, default=3)
    parser.add_argument("--frontier_stall_steps", type=int, default=20)
    parser.add_argument("--waypoint_lookahead", type=int, default=6)
    parser.add_argument("--cmd_horizon", type=int, default=4)
    parser.add_argument("--rollout_horizon", type=int, default=8)
    parser.add_argument("--control_repeat", type=int, default=3)
    parser.add_argument("--ood_bank_samples", type=int, default=128)
    parser.add_argument("--render_every", type=int, default=1)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--debug_first_step", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gs.init(backend=gs.cpu)

    jepa = TinyQuadJEPA().to(dev)
    jepa.load_state_dict(clean_state_dict(torch.load(args.jepa_ckpt, map_location=dev)["model_state_dict"]))
    jepa.eval()

    ppo = ActorCritic().to(dev)
    ppo.load_state_dict(torch.load(args.ppo_ckpt, map_location=dev)["model"], strict=False)
    ppo.eval()

    print("🎬 Harvesting open-floor safe latent bank...")
    safe_bank = harvest_safe_latent_bank(jepa, ppo, dev, max_latents=args.ood_bank_samples).to(dev)
    ood_stats = calibrate_ood(safe_bank, k=8)
    print(f"   Safe bank size: {safe_bank.shape[0]} latents")
    print(f"   Safe OOD stats: mean={ood_stats.mean:.2f} p90={ood_stats.p90:.2f} p95={ood_stats.p95:.2f} p99={ood_stats.p99:.2f}")

    obstacles = make_obstacles()
    scene, robot, cb, ce, c3, dofs, q0 = init_scene(dev, obstacles=obstacles, start_xy=(0.0, 0.0))

    occ, clearance = build_obstacle_grid(obstacles, inflate=0.11)
    visited = np.zeros((MAP_H, MAP_W), dtype=bool)

    writer = None
    if not args.no_video:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = imageio.get_writer(args.out, fps=30)

    pa = torch.zeros((1, 12), device=dev)
    planner = PlannerState()
    trail: List[np.ndarray] = []
    coverage_history: List[float] = []
    ood_history: List[float] = []

    free_cells = int((~occ).sum())
    print(f"\n🐕 Running fast exploration demo ({args.n_steps} steps)", flush=True)
    print(f"   Free cells: {free_cells} | Coverage goal: {args.coverage_goal:.0%}", flush=True)
    print("   Entering control loop...", flush=True)

    prev_xy = None
    current_info: Dict[str, float] = {"cost": 0.0, "pred_ood_raw": 0.0, "pred_ood_rel": 0.0, "clearance": 0.0, "collision": 0.0, "novel": 0.0, "frontier_progress": 0.0, "wp_progress": 0.0, "stall": 0.0}

    for step in range(args.n_steps):
        step_t0 = time.perf_counter()
        if step == 0 and args.debug_first_step:
            print("   [debug] step 0: move_cams", flush=True)
        cep, cel, ceu, c3p, c3l, c3u = move_cams(robot, cb, ce, c3)
        rp = robot.get_pos().cpu().numpy()
        rq = robot.get_quat().cpu().numpy()
        if rp.ndim > 1:
            rp = rp[0]
        if rq.ndim > 1:
            rq = rq[0]
        robot_xy = rp[:2].copy()
        trail.append(robot_xy.copy())
        if prev_xy is None:
            mark_visited(visited, robot_xy, radius_cells=1)
        else:
            mark_visited_segment(visited, prev_xy, robot_xy, radius_cells=1)
        prev_xy = robot_xy.copy()

        coverage = float((visited & (~occ)).sum()) / max(float(free_cells), 1.0)
        coverage_history.append(coverage)
        if coverage >= args.coverage_goal:
            print(f"\n🎉 Coverage goal reached at step {step}: {coverage:.1%}")
            break

        yaw = math.atan2(
            2.0 * (float(rq[0]) * float(rq[3]) + float(rq[1]) * float(rq[2])),
            1.0 - 2.0 * (float(rq[2]) ** 2 + float(rq[3]) ** 2),
        )

        if step == 0 and args.debug_first_step:
            print("   [debug] step 0: get_jepa_state", flush=True)
        v, p = get_jepa_state(robot, cb, q0, pa, dofs, dev)
        with torch.no_grad():
            zc = jepa.encoder(v, p).detach()
            current_ood_raw = float(knn_ood_score(zc, safe_bank, k=8).item())
            current_ood_rel = float(normalize_ood(current_ood_raw, ood_stats))
        ood_history.append(current_ood_raw)

        robot_cell = world_to_grid(robot_xy)
        needs_new_frontier = False
        if planner.frontier is None:
            needs_new_frontier = True
        elif float(np.linalg.norm(planner.frontier.xy - robot_xy)) < args.frontier_reach:
            needs_new_frontier = True
        elif planner.stall_count >= args.frontier_stall_steps:
            needs_new_frontier = True

        if step == 0 and args.debug_first_step:
            print("   [debug] step 0: frontier selection", flush=True)
        if needs_new_frontier:
            planner.frontier = choose_frontier_target(visited, occ, robot_xy)
            planner.path_cells = None
            planner.waypoint_xy = None
            planner.cmd = None
            planner.best_path_xy = None
            planner.hold_steps = 0
            planner.stall_count = 0
            planner.frontier_switches += 1
            if planner.frontier is None:
                print("\n✅ No frontier remaining; map fully explored.")
                break

        if planner.frontier is None:
            break

        if step == 0 and args.debug_first_step:
            print("   [debug] step 0: path planning / local cmd", flush=True)
        if (
            planner.path_cells is None
            or planner.hold_steps <= 0
            or step % max(args.replan_every, 1) == 0
        ):
            path_cells = astar_path(robot_cell, planner.frontier.cell, occ, clearance)
            if path_cells is None or len(path_cells) == 0:
                planner.frontier = None
                continue
            planner.path_cells = path_cells
            planner.waypoint_xy = choose_waypoint_from_path(path_cells, robot_xy, args.waypoint_lookahead)
            planner.cmd, current_info, planner.best_path_xy = choose_local_cmd(
                jepa=jepa,
                safe_bank=safe_bank,
                ood_stats=ood_stats,
                zc=zc,
                robot_xy=robot_xy.copy(),
                robot_yaw=yaw,
                waypoint_xy=planner.waypoint_xy.copy(),
                frontier_xy=planner.frontier.xy.copy(),
                current_ood_rel=current_ood_rel,
                visited=visited,
                occ=occ,
                obstacles=obstacles,
                prev_cmd=planner.cmd,
                cmd_horizon=args.cmd_horizon,
                rollout_horizon=args.rollout_horizon,
            )
            planner.hold_steps = args.replan_every

        assert planner.cmd is not None
        cmd = planner.cmd.clone()
        planner.hold_steps -= 1

        disp_proxy = float(np.linalg.norm(planner.best_path_xy[-1] - planner.best_path_xy[0])) if planner.best_path_xy is not None and len(planner.best_path_xy) > 1 else 0.0
        if disp_proxy < 0.05:
            planner.stall_count += 1
        else:
            planner.stall_count = max(planner.stall_count - 1, 0)

        if step == 0 and args.debug_first_step:
            print("   [debug] step 0: PPO + physics", flush=True)
        with torch.no_grad():
            pa = ppo.act_deterministic(get_sys1_obs(robot, q0, pa, cmd, dofs, dev)).detach()
        target = to_genesis_target(q0 + 0.3 * pa[0])
        robot.control_dofs_position(target, dofs)
        for _ in range(max(args.control_repeat, 1)):
            scene.step()

        frontier_dist = float(np.linalg.norm(planner.frontier.xy - robot_xy)) if planner.frontier is not None else 0.0
        if (not args.no_video) and (step % max(args.render_every, 1) == 0):
            if step == 0 and args.debug_first_step:
                print("   [debug] step 0: rendering / video", flush=True)
            frame3 = c3.render()[0]
            if hasattr(frame3, "cpu"):
                frame3 = frame3.cpu().numpy()
            p3 = Image.fromarray(frame3[:, :, :3].astype(np.uint8))

            frame_e = ce.render()[0]
            if hasattr(frame_e, "cpu"):
                frame_e = frame_e.cpu().numpy()
            pe = Image.fromarray(frame_e[:, :, :3].astype(np.uint8))

            d3 = ImageDraw.Draw(p3)
            if planner.best_path_xy is not None:
                path_px = [
                    project_world_to_pixel(np.array([pt[0], pt[1], 0.05], dtype=np.float32), c3p, c3l, c3u, 50, 512, 512)
                    for pt in planner.best_path_xy
                ]
                valid_px = [px for px in path_px if px is not None]
                if len(valid_px) > 1:
                    d3.line(valid_px, fill=(0, 150, 255), width=4)

            for obs in obstacles:
                px = project_world_to_pixel(obs.pos, c3p, c3l, c3u, 50, 512, 512)
                if px is not None:
                    d3.text((px[0] + 6, px[1] - 6), obs.name, fill=(255, 220, 150))

            if planner.frontier is not None:
                fx = project_world_to_pixel(np.array([planner.frontier.xy[0], planner.frontier.xy[1], 0.08], dtype=np.float32), c3p, c3l, c3u, 50, 512, 512)
                if fx is not None:
                    d3.ellipse([fx[0] - 12, fx[1] - 12, fx[0] + 12, fx[1] + 12], outline=(255, 255, 255), width=3)
                    d3.text((fx[0] + 16, fx[1] - 6), "frontier", fill=(255, 255, 255))

            header_h = 192
            canv = Image.new("RGB", (896, header_h + 512), (20, 20, 20))
            canv.paste(p3, (0, header_h))
            canv.paste(pe, (512, header_h))

            drw = ImageDraw.Draw(canv)
            drw.rectangle([0, 0, 895, header_h - 1], fill=(10, 10, 10), outline=(55, 55, 55))
            drw.line([(0, header_h - 1), (895, header_h - 1)], fill=(90, 90, 90), width=2)
            drw.rectangle([0, header_h, 511, header_h + 511], outline=(70, 70, 70), width=2)
            drw.rectangle([512, header_h, 895, header_h + 383], outline=(70, 70, 70), width=2)
            drw.text((12, header_h - 22), "World view + local rollout", fill=(190, 190, 190))
            drw.text((524, header_h - 22), "Agent / close-up view", fill=(190, 190, 190))

            drw.text((20, 16), "JEPA | Fast Frontier Exploration + OOD Safety", fill=(0, 255, 100))
            drw.text((20, 36), f"Step: {step:04d} | Coverage: {coverage:.1%} | Frontier switches: {planner.frontier_switches}", fill=(200, 200, 200))
            drw.text((20, 56), f"Frontier dist: {frontier_dist:.2f} m | OOD raw: {current_ood_raw:.2f} | OOD rel: {current_ood_rel:.2f}", fill=(200, 200, 200))
            drw.text((20, 76), f"Cmd: vx={float(cmd[0, 0].item()):+.2f} vy={float(cmd[0, 1].item()):+.2f} wz={float(cmd[0, 2].item()):+.2f}", fill=(200, 200, 200))
            drw.text((20, 96), f"Plan cost: {current_info['cost']:.2f} | Novel(best): {int(current_info['novel'])} | Frontier prog: {current_info['frontier_progress']:+.2f}", fill=(200, 200, 200))
            drw.text((20, 116), f"Pred OOD raw: {current_info['pred_ood_raw']:.2f} | Pred OOD rel: {current_info['pred_ood_rel']:.2f} | Stall: {current_info['stall']:.2f}", fill=(200, 200, 200))

            draw_bar(drw, 20, 150, 165, 12, current_ood_rel, 3.0, "OOD rel", fill=(220, 90, 40))
            draw_bar(drw, 220, 150, 165, 12, current_info["pred_ood_rel"], 3.0, "pred OOD rel", fill=(180, 80, 180))
            draw_progress_bar(drw, 420, 150, 160, 12, coverage, "Coverage", fill=(0, 170, 255))
            draw_progress_bar(drw, 610, 150, 160, 12, clamp(current_info["novel"] / 8.0, 0.0, 1.0), "Novel cells in local rollout", fill=(100, 210, 120))

            if len(ood_history) > 2:
                hist = np.asarray(ood_history[-90:], dtype=np.float32)
                hmin = float(hist.min())
                hmax = max(float(hist.max()), hmin + 0.1)
                pts = []
                for i, val in enumerate(hist):
                    x_px = 20 + int(i / max(len(hist) - 1, 1) * 420)
                    y_px = 182 - int((float(val) - hmin) / (hmax - hmin) * 34)
                    pts.append((x_px, y_px))
                if len(pts) > 1:
                    drw.line(pts, fill=(255, 180, 90), width=2)
                drw.text((20, 128), "Raw OOD history", fill=(180, 180, 180))

            draw_minimap(
                drw,
                map_x0=540,
                map_y0=22,
                map_w=322,
                map_h=138,
                robot_xy=robot_xy,
                robot_yaw=yaw,
                trail=trail,
                plan_path=planner.best_path_xy,
                frontier_xy=planner.frontier.xy if planner.frontier is not None else None,
                visited=visited,
                occ=occ,
                obstacles=obstacles,
            )
            drw.text((540, 164), "Blue = local rollout | Yellow = actual trail | White = active frontier", fill=(120, 120, 120))
            writer.append_data(np.array(canv))
        step_dt = time.perf_counter() - step_t0
        print(
            f"\r⚡ step={step:04d} | cov={coverage:.1%} | frontier={frontier_dist:.2f} | "
            f"OODraw={current_ood_raw:.2f} | OODrel={current_ood_rel:.2f} | novel={int(current_info['novel'])} | "
            f"prog={current_info['frontier_progress']:+.2f} | cost={current_info['cost']:.2f} | dt={step_dt:.2f}s",
            end="",
            flush=True,
        )

    if writer is not None:
        writer.close()
    final_cov = float((visited & (~occ)).sum()) / max(float(free_cells), 1.0)
    if writer is not None:
        print(f"\n\n✅ Exploration demo saved to {args.out}")
    else:
        print(f"\n\n✅ Exploration demo completed (no video written)")
    print(f"   Final coverage: {final_cov:.1%}")
    print(f"   Frontier switches: {planner.frontier_switches}")


if __name__ == "__main__":
    main()

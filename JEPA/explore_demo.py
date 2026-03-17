#!/usr/bin/env python3
from __future__ import annotations

"""
JEPA exploration demo (sensor-frontier navigation)

Key properties:
- Does NOT preload obstacle boxes into the planner/map.
- Builds a local/world occupancy estimate only from what the robot camera sees.
- Uses JEPA latent rollout + safe-bank OOD as a soft exploration prior.
- Uses PPO as the low-level gait controller.
- Includes stuck detection + short recovery maneuvers.
- Robust camera parsing for Genesis render outputs that may contain None entries.
"""

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import genesis as gs


# -----------------------------------------------------------------------------
# World / map constants
# -----------------------------------------------------------------------------

WORLD_MIN = np.array([-2.2, -1.2], dtype=np.float32)
WORLD_MAX = np.array([3.8, 3.8], dtype=np.float32)

MAP_UNKNOWN = -1
MAP_FREE = 0
MAP_OCC = 1


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

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
        z = torch.cat([self.vis_enc(vision), self.prop_enc(proprio)], dim=-1)
        return self.fusion(z)


class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + cmd_dim, latent_dim),
            nn.ELU(),
        )
        self.rnn = nn.GRUCell(latent_dim, latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, c_t: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        z_next = self.output_proj(h_next)
        return z_next, h_next


class TinyQuadJEPA(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = JointEncoder(latent_dim)
        self.predictor = LatentPredictor(latent_dim, 3)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ObstacleSpec:
    pos: Tuple[float, float, float]
    size: Tuple[float, float, float]


@dataclass
class SensorMap:
    grid: np.ndarray
    free_visits: np.ndarray
    res: float

    @property
    def h(self) -> int:
        return int(self.grid.shape[0])

    @property
    def w(self) -> int:
        return int(self.grid.shape[1])


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def clean_state_dict(d: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in d.items()}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


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


def make_obstacles() -> List[ObstacleSpec]:
    return [
        ObstacleSpec((1.20, 0.20, 0.25), (0.25, 1.10, 0.50)),
        ObstacleSpec((2.20, 1.40, 0.25), (0.25, 1.00, 0.50)),
        ObstacleSpec((0.40, 2.20, 0.25), (1.10, 0.25, 0.50)),
        ObstacleSpec((2.90, 2.80, 0.25), (0.90, 0.25, 0.50)),
        ObstacleSpec((1.80, 3.10, 0.25), (0.25, 0.90, 0.50)),
        ObstacleSpec((0.20, 0.80, 0.25), (0.25, 0.80, 0.50)),
        ObstacleSpec((3.20, 0.40, 0.25), (0.25, 0.90, 0.50)),
        ObstacleSpec((3.20, 1.90, 0.25), (0.90, 0.25, 0.50)),
        ObstacleSpec((1.20, 1.60, 0.25), (0.70, 0.25, 0.50)),
        ObstacleSpec((2.40, 0.70, 0.25), (0.70, 0.25, 0.50)),
    ]


def to_genesis_target(x: torch.Tensor) -> torch.Tensor:
    x_np = x.detach().to("cpu").numpy().astype(np.float32, copy=True)
    return torch.tensor(x_np, device=gs.device, dtype=torch.float32)


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def nearest_bank_distance(z: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    # z: [B, D] or [D], bank: [N, D]
    if z.dim() == 1:
        z = z.unsqueeze(0)
    d = torch.cdist(z, bank)
    return d.min(dim=1).values


def pairwise_nn_stats(bank: torch.Tensor) -> Dict[str, float]:
    d = torch.cdist(bank, bank)
    eye = torch.eye(d.shape[0], device=d.device, dtype=torch.bool)
    d = d.masked_fill(eye, float("inf"))
    nn = d.min(dim=1).values.detach().cpu().numpy()
    return {
        "mean": float(np.mean(nn)),
        "p50": float(np.percentile(nn, 50)),
        "p90": float(np.percentile(nn, 90)),
        "p95": float(np.percentile(nn, 95)),
        "p99": float(np.percentile(nn, 99)),
    }


# -----------------------------------------------------------------------------
# Map helpers
# -----------------------------------------------------------------------------

def make_sensor_map(res: float) -> SensorMap:
    w = int(math.ceil((WORLD_MAX[0] - WORLD_MIN[0]) / res))
    h = int(math.ceil((WORLD_MAX[1] - WORLD_MIN[1]) / res))
    return SensorMap(
        grid=np.full((h, w), MAP_UNKNOWN, dtype=np.int8),
        free_visits=np.zeros((h, w), dtype=np.int32),
        res=float(res),
    )


def world_to_grid(sm: SensorMap, xy: np.ndarray) -> Optional[Tuple[int, int]]:
    gx = int((float(xy[0]) - float(WORLD_MIN[0])) / sm.res)
    gy = int((float(xy[1]) - float(WORLD_MIN[1])) / sm.res)
    if 0 <= gx < sm.w and 0 <= gy < sm.h:
        return gy, gx
    return None


def grid_to_world(sm: SensorMap, rc: Tuple[int, int]) -> np.ndarray:
    r, c = int(rc[0]), int(rc[1])
    x = float(WORLD_MIN[0]) + (c + 0.5) * sm.res
    y = float(WORLD_MIN[1]) + (r + 0.5) * sm.res
    return np.array([x, y], dtype=np.float32)


def mark_disc(sm: SensorMap, xy: np.ndarray, radius: float, value: int):
    g = world_to_grid(sm, xy)
    if g is None:
        return
    rr = max(1, int(radius / sm.res))
    r0, c0 = g
    for r in range(max(0, r0 - rr), min(sm.h, r0 + rr + 1)):
        for c in range(max(0, c0 - rr), min(sm.w, c0 + rr + 1)):
            p = grid_to_world(sm, (r, c))
            if float(np.linalg.norm(p - xy[:2])) <= radius:
                if value == MAP_FREE:
                    if sm.grid[r, c] != MAP_OCC:
                        sm.grid[r, c] = MAP_FREE
                        sm.free_visits[r, c] += 1
                elif value == MAP_OCC:
                    sm.grid[r, c] = MAP_OCC


def sample_cell(sm: SensorMap, xy: np.ndarray) -> int:
    g = world_to_grid(sm, xy)
    if g is None:
        return MAP_OCC
    return int(sm.grid[g[0], g[1]])


def local_unknown_gain(sm: SensorMap, xy: np.ndarray, radius: float = 0.50) -> float:
    g = world_to_grid(sm, xy)
    if g is None:
        return 0.0
    rr = max(1, int(radius / sm.res))
    r0, c0 = g
    unknown = 0
    for r in range(max(0, r0 - rr), min(sm.h, r0 + rr + 1)):
        for c in range(max(0, c0 - rr), min(sm.w, c0 + rr + 1)):
            p = grid_to_world(sm, (r, c))
            if float(np.linalg.norm(p - xy[:2])) <= radius and sm.grid[r, c] == MAP_UNKNOWN:
                unknown += 1
    return float(unknown) * 0.08


def local_clearance_penalty(sm: SensorMap, xy: np.ndarray, radius: float = 0.25) -> float:
    g = world_to_grid(sm, xy)
    if g is None:
        return 10.0
    rr = max(1, int(radius / sm.res))
    r0, c0 = g
    occ = 0
    for r in range(max(0, r0 - rr), min(sm.h, r0 + rr + 1)):
        for c in range(max(0, c0 - rr), min(sm.w, c0 + rr + 1)):
            p = grid_to_world(sm, (r, c))
            if float(np.linalg.norm(p - xy[:2])) <= radius and sm.grid[r, c] == MAP_OCC:
                occ += 1
    return 0.35 * float(occ)


def coverage_percent(sm: SensorMap) -> float:
    known = np.count_nonzero(sm.grid != MAP_UNKNOWN)
    return 100.0 * float(known) / float(sm.grid.size)


def free_cell_count(sm: SensorMap) -> int:
    return int(np.count_nonzero(sm.grid == MAP_FREE))


def occ_cell_count(sm: SensorMap) -> int:
    return int(np.count_nonzero(sm.grid == MAP_OCC))


# -----------------------------------------------------------------------------
# Genesis scene / obs / rendering
# -----------------------------------------------------------------------------

def resolve_sim_backend(sim_backend_arg: str):
    sim_backend_arg = sim_backend_arg.lower().strip()
    gpu_backend = getattr(gs, "gpu", getattr(gs, "cuda", None))

    if sim_backend_arg in ("cpu",):
        return gs.cpu, "CPU requested"
    if sim_backend_arg in ("gpu", "cuda"):
        return gpu_backend if gpu_backend is not None else gs.cpu, "GPU requested"
    if sim_backend_arg == "auto":
        if gpu_backend is not None:
            return gpu_backend, "AUTO requested (GPU preferred)"
        return gs.cpu, "AUTO requested (CPU fallback)"
    return gs.cpu, "CPU fallback"


def init_genesis_once(sim_backend_arg: str):
    backend, msg = resolve_sim_backend(sim_backend_arg)
    print(f"🌍 Initializing Genesis ({msg})...")
    gs.init(backend=backend)
    print("ℹ️ Genesis prints the actual backend it selected in its own log line above.")


def init_scene(with_obstacles: bool):
    scene = gs.Scene(show_viewer=False)

    tex = create_checkerboard()
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_path=tex)),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/mini_pupper/mini_pupper.urdf",
            pos=(0.0, 0.0, 0.12),
            fixed=False,
        )
    )

    if with_obstacles:
        for obs in make_obstacles():
            scene.add_entity(
                gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
                surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.60)),
            )

    cam_brain = scene.add_camera(res=(64, 64), fov=58)
    cam_eye = scene.add_camera(res=(384, 384), fov=58)
    cam_over = scene.add_camera(res=(512, 512), fov=55)

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

    robot.set_pos(np.array([0.0, 0.0, 0.12], dtype=np.float32))
    robot.set_quat(yaw_to_quat(0.0))
    robot.set_dofs_position(q0_np, dofs)
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, dofs)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, dofs)

    return scene, robot, cam_brain, cam_eye, cam_over, dofs, q0_np


def quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3],
        a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2],
        a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1],
        a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0],
    ], dim=-1)


def world_to_body_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_conj = quat_conj_wxyz(quat)
    vq = torch.cat([torch.zeros((vec.shape[0], 1), device=vec.device), vec], dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(q_conj, vq), quat)[:, 1:4]


def get_sys1_obs(robot, q0: torch.Tensor, prev_action: torch.Tensor, cmd: torch.Tensor, dofs, dev: torch.device) -> torch.Tensor:
    pos = robot.get_pos().to(dev)
    quat = robot.get_quat().to(dev)
    vel = robot.get_vel().to(dev)
    ang = robot.get_ang().to(dev)
    pos, quat, vel, ang = [x.unsqueeze(0) if x.dim() == 1 else x for x in (pos, quat, vel, ang)]

    q = robot.get_dofs_position(dofs).to(dev)
    dq = robot.get_dofs_velocity(dofs).to(dev)
    q = q.unsqueeze(0) if q.dim() == 1 else q
    dq = dq.unsqueeze(0) if dq.dim() == 1 else dq

    q0b = q0.unsqueeze(0)
    obs = torch.cat([
        pos[:, 2:3],
        quat,
        world_to_body_vec(quat, vel),
        world_to_body_vec(quat, ang),
        q - q0b,
        dq,
        prev_action,
        cmd,
    ], dim=1)
    return obs


def move_cams(robot, cam_brain, cam_eye, cam_over):
    p = robot.get_pos().cpu().numpy()
    q = robot.get_quat().cpu().numpy()
    if p.ndim > 1:
        p = p[0]
    if q.ndim > 1:
        q = q[0]

    w, x, y, z = q
    fw = np.array([
        1 - 2 * (y ** 2 + z ** 2),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ], dtype=np.float32)
    up = np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x ** 2 + y ** 2),
    ], dtype=np.float32)

    brain_pos = p + fw * 0.10 + up * 0.05
    brain_lk = brain_pos + fw * 1.00

    cam_brain.set_pose(pos=brain_pos, lookat=brain_lk, up=up)
    cam_eye.set_pose(pos=brain_pos, lookat=brain_lk, up=up)

    over_pos = p - fw * 1.7 + np.array([0.0, 0.0, 0.9], dtype=np.float32)
    over_lk = p + fw * 0.45
    cam_over.set_pose(pos=over_pos, lookat=over_lk, up=np.array([0.0, 0.0, 1.0], dtype=np.float32))

    yaw = math.atan2(float(fw[1]), float(fw[0]))
    return p, yaw, brain_pos, brain_lk


def render_rgb_depth(cam):
    attempts = [
        lambda: cam.render(rgb=True, depth=True),
        lambda: cam.render(depth=True),
        lambda: cam.render(),
    ]

    out = None
    last_err = None
    for fn in attempts:
        try:
            out = fn()
            break
        except TypeError as e:
            last_err = e
        except Exception as e:
            last_err = e

    if out is None:
        raise RuntimeError(f"Camera render failed; last error: {last_err}")

    rgb = None
    depth = None

    def maybe_take(arr):
        nonlocal rgb, depth
        if arr is None:
            return
        arr = np.asarray(arr)

        if arr.ndim == 3 and arr.shape[-1] >= 3 and rgb is None:
            rgb = arr[..., :3]
            return

        if arr.ndim == 2 and depth is None:
            depth = arr
            return

        if arr.ndim == 3 and arr.shape[-1] == 1 and depth is None:
            depth = arr[..., 0]
            return

    if isinstance(out, dict):
        maybe_take(out.get("rgb", None))
        maybe_take(out.get("color", None))
        maybe_take(out.get("image", None))
        maybe_take(out.get("depth", None))

    elif isinstance(out, (tuple, list)):
        for item in out:
            if item is None:
                continue
            arr = to_numpy(item)
            maybe_take(arr)

    else:
        maybe_take(to_numpy(out))

    if rgb is None:
        raise RuntimeError(
            f"Could not parse RGB image from Genesis camera render output. "
            f"Got type={type(out)}"
        )

    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            mx = float(np.nanmax(rgb)) if rgb.size else 1.0
            if mx <= 1.0:
                rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32)

    return rgb, depth


@torch.no_grad()
def get_jepa_state(robot, cam_brain, q0: torch.Tensor, prev_action: torch.Tensor, dofs, dev: torch.device):
    rgb, depth = render_rgb_depth(cam_brain)
    rgb_chw = np.transpose(rgb[:, :, :3], (2, 0, 1)).copy()
    vision = torch.from_numpy(rgb_chw).float().to(dev) / 255.0
    proprio = get_sys1_obs(
        robot,
        q0,
        prev_action,
        torch.zeros((1, 3), device=dev),
        dofs,
        dev,
    )[:, :47]
    return vision.unsqueeze(0), proprio, rgb, depth


# -----------------------------------------------------------------------------
# Sensor mapping
# -----------------------------------------------------------------------------

def normalize_depth(depth: Optional[np.ndarray], depth_max: float) -> Optional[np.ndarray]:
    if depth is None:
        return None

    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]

    if d.size == 0:
        return None

    # Heuristic: Genesis sometimes returns metric depth, sometimes normalized depth-like arrays.
    finite = np.isfinite(d)
    if not finite.any():
        return None

    d = np.where(finite, d, np.nan)
    mx = float(np.nanmax(d))
    mn = float(np.nanmin(d))

    if mx <= 1.05 and mn >= 0.0:
        d = np.clip(d, 0.0, 1.0) * depth_max
    else:
        d = np.clip(d, 0.0, depth_max)

    return np.nan_to_num(d, nan=depth_max, posinf=depth_max, neginf=0.0).astype(np.float32)


def update_sensor_map_from_depth(
    sm: SensorMap,
    robot_xy: np.ndarray,
    robot_yaw: float,
    depth_img: Optional[np.ndarray],
    fov_deg: float,
    depth_max: float,
    footprint_radius: float = 0.20,
):
    mark_disc(sm, robot_xy, footprint_radius, MAP_FREE)

    d = normalize_depth(depth_img, depth_max)
    if d is None:
        # Fallback: mark a small free wedge ahead so the map still grows.
        for a in np.linspace(-0.35, 0.35, 11):
            ray_yaw = robot_yaw + float(a)
            for t in np.linspace(0.08, 0.55, 8):
                p = robot_xy + np.array([math.cos(ray_yaw), math.sin(ray_yaw)], dtype=np.float32) * float(t)
                mark_disc(sm, p, 0.05, MAP_FREE)
        return

    h, w = d.shape
    cols = np.linspace(int(0.08 * w), int(0.92 * w), 41).astype(int)
    cols = np.unique(np.clip(cols, 0, w - 1))
    row0 = int(0.42 * h)
    row1 = int(0.88 * h)

    for c in cols:
        ray = d[row0:row1, c]
        if ray.size == 0:
            continue

        dist = float(np.nanmedian(ray))
        dist = clamp(dist, 0.05, depth_max)

        x_norm = (float(c) / max(float(w - 1), 1.0)) * 2.0 - 1.0
        ang = math.radians(0.5 * fov_deg) * x_norm
        ray_yaw = robot_yaw + ang

        # Free cells up to just before the hit.
        free_until = max(0.0, dist - 0.08)
        n_steps = max(2, int(free_until / max(sm.res * 0.7, 0.04)))
        for t in np.linspace(0.06, free_until, n_steps):
            p = robot_xy + np.array([math.cos(ray_yaw), math.sin(ray_yaw)], dtype=np.float32) * float(t)
            mark_disc(sm, p, 0.04, MAP_FREE)

        # If the ray ends before max range, mark occupied.
        if dist < depth_max * 0.96:
            p_occ = robot_xy + np.array([math.cos(ray_yaw), math.sin(ray_yaw)], dtype=np.float32) * float(dist)
            mark_disc(sm, p_occ, 0.08, MAP_OCC)


# -----------------------------------------------------------------------------
# Frontier selection
# -----------------------------------------------------------------------------

def select_frontier_target(
    sm: SensorMap,
    robot_xy: np.ndarray,
    blacklist: Optional[List[np.ndarray]] = None,
    blacklist_radius: float = 0.40,
) -> Tuple[np.ndarray, float]:
    robot_g = world_to_grid(sm, robot_xy)
    if robot_g is None:
        return robot_xy.copy(), 0.0

    bl = blacklist or []
    candidates: List[Tuple[float, np.ndarray]] = []

    for r in range(1, sm.h - 1):
        for c in range(1, sm.w - 1):
            if sm.grid[r, c] != MAP_FREE:
                continue

            unknown_n = 0
            occ_n = 0
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if rr == r and cc == c:
                        continue
                    v = int(sm.grid[rr, cc])
                    if v == MAP_UNKNOWN:
                        unknown_n += 1
                    elif v == MAP_OCC:
                        occ_n += 1

            if unknown_n <= 0:
                continue

            wp = grid_to_world(sm, (r, c))
            dist = float(np.linalg.norm(wp - robot_xy))
            if dist < 0.20:
                continue

            # Skip blacklisted regions.
            in_bl = False
            for bp in bl:
                if float(np.linalg.norm(wp - bp)) < blacklist_radius:
                    in_bl = True
                    break
            if in_bl:
                continue

            # Prefer nearby frontiers with lots of unknown — penalise distance.
            score = 0.45 * float(unknown_n) - 0.30 * dist - 0.35 * float(occ_n)
            candidates.append((score, wp))

    if not candidates:
        # If everything is blacklisted, clear blacklist and try any frontier.
        if bl:
            return select_frontier_target(sm, robot_xy, blacklist=None)
        return robot_xy.copy(), 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, wp = candidates[0]
    return wp, float(score)


# -----------------------------------------------------------------------------
# Latent bank harvesting
# -----------------------------------------------------------------------------

@torch.no_grad()
def scene_step_with_policy(
    scene,
    robot,
    q0: torch.Tensor,
    prev_action: torch.Tensor,
    cmd: torch.Tensor,
    dofs,
    ppo: ActorCritic,
    dev: torch.device,
    control_scale: float = 0.30,
    sim_substeps: int = 4,
) -> torch.Tensor:
    obs = get_sys1_obs(robot, q0, prev_action, cmd, dofs, dev)
    action = ppo.act_deterministic(obs).detach()
    target = to_genesis_target(q0 + control_scale * action[0])
    robot.control_dofs_position(target, dofs)
    for _ in range(sim_substeps):
        scene.step()
    return action


@torch.no_grad()
def harvest_safe_bank(
    scene,
    robot,
    cam_brain,
    q0: torch.Tensor,
    dofs,
    jepa: TinyQuadJEPA,
    ppo: ActorCritic,
    dev: torch.device,
    n_latents: int = 128,
):
    print("🎬 Harvesting open-floor safe latent bank...")
    prev_action = torch.zeros((1, 12), device=dev)

    cmds = []
    for k in range(n_latents * 3):
        phase = k % 24
        if phase < 8:
            cmd = [0.30, 0.00, 0.00]
        elif phase < 12:
            cmd = [0.22, 0.00, 0.30]
        elif phase < 16:
            cmd = [0.28, 0.00, -0.30]
        elif phase < 20:
            cmd = [0.18, 0.08, 0.10]
        else:
            cmd = [0.18, -0.08, -0.10]
        cmds.append(torch.tensor([cmd], device=dev, dtype=torch.float32))

    bank: List[torch.Tensor] = []
    for k, cmd in enumerate(cmds):
        p, yaw, _, _ = move_cams(robot, cam_brain, cam_brain, cam_brain)
        _ = (p, yaw)  # silence lint
        prev_action = scene_step_with_policy(scene, robot, q0, prev_action, cmd, dofs, ppo, dev)
        if k % 3 == 0:
            vis, prop, _, _ = get_jepa_state(robot, cam_brain, q0, prev_action, dofs, dev)
            z = jepa.encoder(vis, prop).detach().squeeze(0)
            bank.append(z)
            if len(bank) >= n_latents:
                break

    bank_t = torch.stack(bank, dim=0)
    stats = pairwise_nn_stats(bank_t)

    print(f"   Safe bank size: {bank_t.shape[0]} latents")
    print(
        "   Safe OOD stats: "
        f"mean={stats['mean']:.2f} p90={stats['p90']:.2f} "
        f"p95={stats['p95']:.2f} p99={stats['p99']:.2f}"
    )
    return bank_t, stats


# -----------------------------------------------------------------------------
# Kinematic rollout / local planner
# -----------------------------------------------------------------------------

def rollout_cmd_kinematic(
    start_xy: np.ndarray,
    start_yaw: float,
    cmd_xyw: np.ndarray,
    hz: int,
    dt: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    pos = np.array(start_xy, dtype=np.float32).copy()
    yaw = float(start_yaw)
    path = []
    for _ in range(hz):
        path.append(pos.copy())
        world_v = body_to_world_xy(yaw, cmd_xyw[:2])
        pos += dt * world_v
        yaw = wrap_to_pi(yaw + dt * float(cmd_xyw[2]))
    return np.stack(path, axis=0), pos.copy(), float(yaw)


def path_collision_penalty(sm: SensorMap, path_xy: np.ndarray) -> float:
    pen = 0.0
    for pt in path_xy:
        cell = sample_cell(sm, pt)
        if cell == MAP_OCC:
            pen += 1.8
        # Unknown is not punished: the map is sensor-built only.
        pen += local_clearance_penalty(sm, pt, radius=0.18) * 0.25
    return float(pen)


@torch.no_grad()
def plan_local_cmd(
    jepa: TinyQuadJEPA,
    zc: torch.Tensor,
    safe_bank: torch.Tensor,
    safe_stats: Dict[str, float],
    sm: SensorMap,
    robot_xy: np.ndarray,
    robot_yaw: float,
    frontier_xy: np.ndarray,
    prev_cmd: Optional[torch.Tensor],
    cands: int,
    hz: int,
    dev: torch.device,
):
    goal_vec = frontier_xy - robot_xy
    goal_dist = float(np.linalg.norm(goal_vec))
    if goal_dist < 1e-6:
        return torch.zeros((1, 3), device=dev), {
            "path": np.zeros((hz, 2), dtype=np.float32),
            "frontier": 0.0,
            "OODraw": 0.0,
            "OODrel": 0.0,
            "novel": 0,
            "prog": 0.0,
            "cost": 0.0,
        }

    goal_dir_world = goal_vec / max(goal_dist, 1e-8)
    goal_body_xy = world_to_body_xy(robot_yaw, goal_dir_world)
    goal_angle = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
    heading_error = wrap_to_pi(goal_angle - robot_yaw)

    far = goal_dist > 0.8
    transl_scale = 0.28 if far else 0.18
    if abs(heading_error) > 0.9:
        transl_scale *= 0.50

    mean = torch.tensor([
        clamp(float(goal_body_xy[0]) * transl_scale, -0.40, 0.40),
        clamp(float(goal_body_xy[1]) * transl_scale, -0.20, 0.20),
        clamp(0.40 * heading_error, -0.60, 0.60),
    ], device=dev, dtype=torch.float32)

    std = torch.tensor([
        0.12 if far else 0.09,
        0.10 if far else 0.07,
        0.24 if far else 0.18,
    ], device=dev, dtype=torch.float32)

    best = None

    for _ in range(4):
        cmds = mean + std * torch.randn((cands, 3), device=dev)
        cmds[:, 0].clamp_(-0.40, 0.40)
        cmds[:, 1].clamp_(-0.25, 0.25)
        cmds[:, 2].clamp_(-0.60, 0.60)

        z_roll = zc.expand(cands, -1)
        h_t = torch.zeros((cands, jepa.latent_dim), device=dev, dtype=z_roll.dtype)
        for _t in range(hz):
            z_roll, h_t = jepa.predictor(z_roll, cmds, h_t)

        ood_raw = nearest_bank_distance(z_roll, safe_bank)
        ood_rel = ood_raw - float(safe_stats["p90"])

        costs = []
        paths = []
        frontier_vals = []
        novel_flags = []
        progress_vals = []

        for i in range(cands):
            cmd_np = cmds[i].detach().cpu().numpy()
            path_xy, end_xy, end_yaw = rollout_cmd_kinematic(robot_xy, robot_yaw, cmd_np, hz)
            _ = end_yaw
            paths.append(path_xy)

            end_dist = float(np.linalg.norm(end_xy - frontier_xy))
            progress = goal_dist - end_dist
            progress_vals.append(progress)

            frontier_gain = local_unknown_gain(sm, end_xy, radius=0.55)
            frontier_vals.append(frontier_gain)

            coll_pen = path_collision_penalty(sm, path_xy)
            smooth_pen = 0.0 if prev_cmd is None else 0.12 * float(torch.sum((cmds[i] - prev_cmd[0]) ** 2).item())
            reverse_pen = 0.08 if float(cmds[i, 0].item()) < -0.02 else 0.0

            end_g = world_to_grid(sm, end_xy)
            novel = 0
            if end_g is not None and sm.free_visits[end_g[0], end_g[1]] == 0:
                novel = 1
            novel_flags.append(novel)

            # Lower is better.
            # OOD weight kept very low: the safe bank is from open floor,
            # so anything near obstacles will look OOD — don't let it
            # dominate over actually reaching the frontier.
            ood_cost = min(0.50, max(0.0, float(ood_rel[i].item())))
            cost = (
                coll_pen
                + 0.05 * ood_cost
                + smooth_pen
                + reverse_pen
                - 2.50 * progress
                - 0.70 * frontier_gain
                - 0.25 * float(novel)
            )
            costs.append(cost)

        costs_np = np.asarray(costs, dtype=np.float32)
        elite_k = max(8, cands // 10)
        elite_idx = np.argsort(costs_np)[:elite_k]
        elite_cmds = cmds[torch.as_tensor(elite_idx, device=dev, dtype=torch.long)]
        mean = elite_cmds.mean(dim=0)
        std = elite_cmds.std(dim=0) + 1e-4

        i_best = int(np.argmin(costs_np))
        cur = {
            "cmd": cmds[i_best].detach().clone().view(1, 3),
            "path": paths[i_best],
            "frontier": float(frontier_vals[i_best]),
            "OODraw": float(ood_raw[i_best].item()),
            "OODrel": float(ood_rel[i_best].item()),
            "novel": int(novel_flags[i_best]),
            "prog": float(progress_vals[i_best]),
            "cost": float(costs_np[i_best]),
        }
        if best is None or cur["cost"] < best["cost"]:
            best = cur

    assert best is not None
    return best["cmd"], best


# -----------------------------------------------------------------------------
# Recovery guard
# -----------------------------------------------------------------------------

def choose_recovery_cmd(depth_img: Optional[np.ndarray], depth_max: float, dev: torch.device) -> torch.Tensor:
    d = normalize_depth(depth_img, depth_max)
    if d is None:
        return torch.tensor([[-0.20, 0.10, 0.55]], device=dev, dtype=torch.float32)

    h, w = d.shape
    band = d[int(0.45 * h):int(0.90 * h), :]
    if band.size == 0:
        return torch.tensor([[-0.20, 0.10, 0.55]], device=dev, dtype=torch.float32)

    left = float(np.nanmedian(band[:, :max(1, w // 3)]))
    right = float(np.nanmedian(band[:, 2 * w // 3:]))
    mid = float(np.nanmedian(band[:, w // 3:2 * w // 3]))

    # Turn hard toward the side with more clearance while backing up.
    if left > right + 0.10:
        return torch.tensor([[-0.18, 0.12, 0.55]], device=dev, dtype=torch.float32)
    if right > left + 0.10:
        return torch.tensor([[-0.18, -0.12, -0.55]], device=dev, dtype=torch.float32)
    if mid < 0.60 * depth_max:
        return torch.tensor([[-0.22, 0.00, 0.60]], device=dev, dtype=torch.float32)

    return torch.tensor([[-0.18, 0.08, 0.45]], device=dev, dtype=torch.float32)


# -----------------------------------------------------------------------------
# HUD / video
# -----------------------------------------------------------------------------

def world_to_map_px(xy: np.ndarray, map_x0: int, map_y0: int, map_w: int, map_h: int) -> Tuple[int, int]:
    nx = (float(xy[0]) - float(WORLD_MIN[0])) / max(float(WORLD_MAX[0] - WORLD_MIN[0]), 1e-8)
    ny = (float(xy[1]) - float(WORLD_MIN[1])) / max(float(WORLD_MAX[1] - WORLD_MIN[1]), 1e-8)
    px = map_x0 + int(np.clip(nx, 0.0, 1.0) * map_w)
    py = map_y0 + map_h - int(np.clip(ny, 0.0, 1.0) * map_h)
    return px, py


def draw_minimap(
    draw: ImageDraw.ImageDraw,
    sm: SensorMap,
    map_x0: int,
    map_y0: int,
    map_w: int,
    map_h: int,
    robot_xy: np.ndarray,
    robot_yaw: float,
    target_xy: np.ndarray,
    trail: List[np.ndarray],
    plan_path: np.ndarray,
):
    draw.rectangle([map_x0, map_y0, map_x0 + map_w, map_y0 + map_h], fill=(18, 18, 18), outline=(95, 95, 95))

    for r in range(sm.h):
        for c in range(sm.w):
            x0 = map_x0 + int(c / sm.w * map_w)
            y0 = map_y0 + int(r / sm.h * map_h)
            x1 = map_x0 + int((c + 1) / sm.w * map_w)
            y1 = map_y0 + int((r + 1) / sm.h * map_h)

            v = int(sm.grid[r, c])
            if v == MAP_FREE:
                fill = (48, 48, 48)
            elif v == MAP_OCC:
                fill = (180, 70, 70)
            else:
                fill = (25, 25, 25)
            draw.rectangle([x0, y0, x1, y1], fill=fill)

    if len(trail) > 1:
        pts = [world_to_map_px(t, map_x0, map_y0, map_w, map_h) for t in trail[-300:]]
        draw.line(pts, fill=(255, 220, 80), width=2)

    if plan_path is not None and len(plan_path) > 1:
        pts = [world_to_map_px(t, map_x0, map_y0, map_w, map_h) for t in plan_path]
        draw.line(pts, fill=(0, 170, 255), width=3)

    tx, ty = world_to_map_px(target_xy, map_x0, map_y0, map_w, map_h)
    draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(255, 255, 255), outline=(10, 10, 10), width=2)

    rx, ry = world_to_map_px(robot_xy, map_x0, map_y0, map_w, map_h)
    head = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float32)
    left = np.array([math.cos(robot_yaw + 2.5), math.sin(robot_yaw + 2.5)], dtype=np.float32)
    right = np.array([math.cos(robot_yaw - 2.5), math.sin(robot_yaw - 2.5)], dtype=np.float32)
    scale = 11.0
    tri = [
        (rx + int(head[0] * scale), ry - int(head[1] * scale)),
        (rx + int(left[0] * scale * 0.8), ry - int(left[1] * scale * 0.8)),
        (rx + int(right[0] * scale * 0.8), ry - int(right[1] * scale * 0.8)),
    ]
    draw.polygon(tri, fill=(255, 255, 255), outline=(10, 10, 10))


def compose_video_frame(
    over_rgb: np.ndarray,
    eye_rgb: np.ndarray,
    sm: SensorMap,
    robot_xy: np.ndarray,
    robot_yaw: float,
    target_xy: np.ndarray,
    trail: List[np.ndarray],
    plan_path: np.ndarray,
    status_text: List[str],
) -> np.ndarray:
    pov = Image.fromarray(over_rgb[:, :, :3].astype(np.uint8))
    eye = Image.fromarray(eye_rgb[:, :, :3].astype(np.uint8))
    eye = eye.resize((384, 384))

    canvas = Image.new("RGB", (896, 560), (20, 20, 20))
    canvas.paste(pov, (0, 48))
    canvas.paste(eye, (512, 96))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 895, 47], fill=(12, 12, 12), outline=(70, 70, 70))
    draw.text((14, 14), "JEPA exploration demo | sensor-frontier navigation", fill=(0, 255, 110))

    draw.text((14, 50), "World / follow view", fill=(190, 190, 190))
    draw.text((526, 78), "Robot-eye view", fill=(190, 190, 190))

    # Status text below the world-view panel.
    for i, line in enumerate(status_text):
        draw.text((14, 510 + 14 * i), line, fill=(200, 200, 200))

    draw_minimap(
        draw=draw,
        sm=sm,
        map_x0=512,
        map_y0=490 - 120,
        map_w=340,
        map_h=120,
        robot_xy=robot_xy,
        robot_yaw=robot_yaw,
        target_xy=target_xy,
        trail=trail,
        plan_path=plan_path,
    )
    draw.text((512, 344), "Sensor map (dark=unknown, grey=free, red=occupied)", fill=(190, 190, 190))

    return np.asarray(canvas)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", required=True)
    parser.add_argument("--ppo_ckpt", required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sim_backend", type=str, default="auto")
    parser.add_argument("--n_steps", type=int, default=1800)
    parser.add_argument("--cands", type=int, default=192)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--map_res", type=float, default=0.12)
    parser.add_argument("--depth_max", type=float, default=1.80)
    parser.add_argument("--coverage_goal", type=float, default=55.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--out", type=str, default="jepa_logs/explore_demo.mp4")
    parser.add_argument("--debug_first_step", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested for torch, but torch.cuda.is_available() is False. Falling back to CPU.")
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    print(f"🚀 Loading brains into Genesis on {dev.type}...")
    init_genesis_once(args.sim_backend)

    jepa = TinyQuadJEPA().to(dev)
    jepa.load_state_dict(clean_state_dict(torch.load(args.jepa_ckpt, map_location=dev)["model_state_dict"]))
    jepa.eval()

    ppo = ActorCritic().to(dev)
    ppo.load_state_dict(torch.load(args.ppo_ckpt, map_location=dev)["model"], strict=False)
    ppo.eval()

    print("✅ Both checkpoints loaded successfully!")

    # Safe-bank scene: no obstacles.
    safe_scene, safe_robot, safe_cam_brain, _, _, safe_dofs, safe_q0_np = init_scene(with_obstacles=False)
    safe_q0 = torch.tensor(safe_q0_np, device=dev, dtype=torch.float32)
    for _ in range(12):
        safe_scene.step()
    move_cams(safe_robot, safe_cam_brain, safe_cam_brain, safe_cam_brain)
    safe_bank, safe_stats = harvest_safe_bank(
        safe_scene, safe_robot, safe_cam_brain, safe_q0, safe_dofs, jepa, ppo, dev, n_latents=128
    )

    # Exploration scene: actual obstacles.
    print("🌍 Booting exploration scene...")
    scene, robot, cam_brain, cam_eye, cam_over, dofs, q0_np = init_scene(with_obstacles=True)
    q0 = torch.tensor(q0_np, device=dev, dtype=torch.float32)

    for _ in range(20):
        scene.step()

    prev_action = torch.zeros((1, 12), device=dev)
    prev_cmd = None

    sm = make_sensor_map(args.map_res)
    trail: List[np.ndarray] = []
    recent_pos: Deque[np.ndarray] = deque(maxlen=18)
    recent_cov: Deque[float] = deque(maxlen=40)

    writer = None
    if not args.no_video:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = imageio.get_writer(args.out, fps=30)

    frontier_xy = np.array([0.5, 0.0], dtype=np.float32)
    frontier_switches = 0
    frontier_blacklist: List[np.ndarray] = []

    # How long (steps) we've been targeting the current frontier without
    # meaningful coverage gain — used to detect unreachable frontiers.
    frontier_age = 0
    cov_at_frontier_start = 0.0
    FRONTIER_PATIENCE = 60  # steps before we blacklist a stale frontier

    guard_mode = "none"
    guard_steps = 0
    guard_cmd = torch.zeros((1, 3), device=dev)
    stuck_count = 0  # consecutive stuck episodes

    # Wander mode: random walk when persistently stuck.
    wander_steps = 0
    wander_cmd = torch.zeros((1, 3), device=dev)

    # Lightweight metric accumulators for the summary JSON.
    metric_cov_history: List[float] = []
    metric_guard_counts: Dict[str, int] = {"recover": 0, "wander": 0}
    metric_step_times: List[float] = []

    print(f"\n🐕 Running fast exploration demo ({args.n_steps} steps)")
    print(f"   Free cells: {sm.grid.size} | Coverage goal: {args.coverage_goal:.0f}%")
    print("   Entering control loop...")

    for step in range(args.n_steps):
        t0 = time.time()

        if args.debug_first_step and step == 0:
            print("   [debug] step 0: move_cams")
        robot_pos_3d, robot_yaw, _, _ = move_cams(robot, cam_brain, cam_eye, cam_over)
        robot_xy = robot_pos_3d[:2].astype(np.float32)

        if args.debug_first_step and step == 0:
            print("   [debug] step 0: get_jepa_state")
        vis, prop, _, depth = get_jepa_state(robot, cam_brain, q0, prev_action, dofs, dev)
        zc = jepa.encoder(vis, prop).detach()

        update_sensor_map_from_depth(
            sm=sm,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            depth_img=depth,
            fov_deg=58.0,
            depth_max=args.depth_max,
            footprint_radius=0.20,
        )

        cov = coverage_percent(sm)
        trail.append(robot_xy.copy())
        recent_pos.append(robot_xy.copy())
        recent_cov.append(cov)

        # --- Frontier staleness tracking ---
        frontier_age += 1
        cov_gain = cov - cov_at_frontier_start

        # If we've been targeting the same frontier for too long with
        # negligible coverage gain, blacklist it and force a switch.
        force_new_frontier = False
        if frontier_age >= FRONTIER_PATIENCE and cov_gain < 0.5:
            frontier_blacklist.append(frontier_xy.copy())
            force_new_frontier = True
            frontier_age = 0
            cov_at_frontier_start = cov

        if args.debug_first_step and step == 0:
            print("   [debug] step 0: frontier selection")

        frontier_xy_new, frontier_score = select_frontier_target(
            sm, robot_xy, blacklist=frontier_blacklist
        )
        dist_to_new = float(np.linalg.norm(frontier_xy_new - frontier_xy))
        if dist_to_new > 0.18 or force_new_frontier:
            frontier_switches += 1
            frontier_xy = frontier_xy_new
            frontier_age = 0
            cov_at_frontier_start = cov

        if args.debug_first_step and step == 0:
            print("   [debug] step 0: path planning / local cmd")

        cmd, plan_stats = plan_local_cmd(
            jepa=jepa,
            zc=zc,
            safe_bank=safe_bank,
            safe_stats=safe_stats,
            sm=sm,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            frontier_xy=frontier_xy,
            prev_cmd=prev_cmd,
            cands=args.cands,
            hz=args.horizon,
            dev=dev,
        )

        # --- Stuck detection ---
        disp_recent = 0.0
        if len(recent_pos) >= recent_pos.maxlen:
            disp_recent = float(np.linalg.norm(recent_pos[-1] - recent_pos[0]))

        # Coverage stall: if coverage hasn't moved in the recent window,
        # we're stuck even if the robot is drifting slightly.
        cov_stall = False
        if len(recent_cov) >= recent_cov.maxlen:
            cov_stall = (max(recent_cov) - min(recent_cov)) < 0.3

        is_stuck = (
            guard_steps <= 0
            and wander_steps <= 0
            and len(recent_pos) >= recent_pos.maxlen
            and (disp_recent < 0.12 or cov_stall)
        )

        if is_stuck:
            stuck_count += 1
            # Blacklist the current frontier — it's clearly unreachable.
            frontier_blacklist.append(frontier_xy.copy())

            if stuck_count >= 3:
                # Persistent stuck: enter wander mode — pick a random
                # direction and commit to it for a while.
                wander_angle = robot_yaw + np.random.uniform(-math.pi, math.pi)
                wander_cmd = torch.tensor([[
                    0.30 * math.cos(wander_angle - robot_yaw),
                    0.30 * math.sin(wander_angle - robot_yaw),
                    float(np.clip(wrap_to_pi(wander_angle - robot_yaw) * 0.5, -0.6, 0.6)),
                ]], device=dev, dtype=torch.float32)
                wander_steps = 30
                guard_mode = "wander"
                stuck_count = 0
            else:
                # Normal recovery: back up + turn hard.
                guard_mode = "stuck_recover"
                guard_steps = 14
                guard_cmd = choose_recovery_cmd(depth, args.depth_max, dev)

        # Apply guard / wander overrides.
        if wander_steps > 0:
            cmd = wander_cmd
            guard_mode = "wander"
            wander_steps -= 1
            if wander_steps <= 0:
                guard_mode = "none"
                # Force frontier re-selection after wander.
                frontier_age = FRONTIER_PATIENCE + 1
                cov_at_frontier_start = cov - 10.0
        elif guard_steps > 0:
            cmd = guard_cmd
            guard_mode = "recover_hold"
            guard_steps -= 1
            if guard_steps <= 0:
                guard_mode = "none"
                stuck_count = max(0, stuck_count - 1)
                # Force frontier re-selection after recovery.
                frontier_blacklist.append(frontier_xy.copy())
        else:
            guard_mode = "none"

        prev_cmd = cmd.detach().clone()

        if args.debug_first_step and step == 0:
            print("   [debug] step 0: PPO + physics")
        prev_action = scene_step_with_policy(
            scene=scene,
            robot=robot,
            q0=q0,
            prev_action=prev_action,
            cmd=cmd,
            dofs=dofs,
            ppo=ppo,
            dev=dev,
            control_scale=0.30,
            sim_substeps=4,
        )

        ood_raw = float(plan_stats["OODraw"])
        ood_rel = float(plan_stats["OODrel"])
        novel = int(plan_stats["novel"])
        prog = float(plan_stats["prog"])
        cost = float(plan_stats["cost"])

        dt = time.time() - t0

        # Accumulate metrics.
        metric_step_times.append(dt)
        metric_cov_history.append(cov)
        if guard_mode == "stuck_recover":
            metric_guard_counts["recover"] += 1
        elif guard_mode == "wander":
            metric_guard_counts["wander"] += 1

        print(
            f"\r⚡ step={step+1:04d} | cov={cov:4.1f}% | frontier={frontier_score:4.2f} | "
            f"prog={prog:+.2f} | cost={cost:.2f} | dt={dt:.2f}s | "
            f"guard={guard_mode} | bl={len(frontier_blacklist)}",
            end="",
            flush=True,
        )

        if writer is not None:
            over_rgb, _ = render_rgb_depth(cam_over)
            eye_rgb, _ = render_rgb_depth(cam_eye)
            status = [
                f"coverage={cov:.1f}%  free={free_cell_count(sm)}  occ={occ_cell_count(sm)}  blacklisted={len(frontier_blacklist)}",
                f"prog={prog:+.2f}  cost={cost:.2f}  guard={guard_mode}",
                f"cmd=[{float(cmd[0,0]):+.2f}, {float(cmd[0,1]):+.2f}, {float(cmd[0,2]):+.2f}]",
            ]
            frame = compose_video_frame(
                over_rgb=over_rgb,
                eye_rgb=eye_rgb,
                sm=sm,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                target_xy=frontier_xy,
                trail=trail,
                plan_path=plan_stats["path"],
                status_text=status,
            )
            writer.append_data(frame)

        if cov >= args.coverage_goal:
            break

    if writer is not None:
        writer.close()

    # --- Write summary JSON ---------------------------------------------------
    final_cov = coverage_percent(sm)
    steps_taken = min(step + 1, args.n_steps)
    mean_dt = float(np.mean(metric_step_times)) if metric_step_times else 0.0

    # Find when coverage milestones were first reached.
    milestones = {}
    for pct in [10, 20, 30, 40, 50]:
        for i, c in enumerate(metric_cov_history):
            if c >= pct:
                milestones[f"step_at_{pct}pct"] = i
                break

    summary = {
        "jepa_ckpt": args.jepa_ckpt,
        "ppo_ckpt": args.ppo_ckpt,
        "steps_budget": args.n_steps,
        "steps_taken": steps_taken,
        "coverage_goal": args.coverage_goal,
        "final_coverage_pct": round(final_cov, 2),
        "goal_reached": final_cov >= args.coverage_goal,
        "free_cells": free_cell_count(sm),
        "occupied_cells": occ_cell_count(sm),
        "frontier_switches": frontier_switches,
        "frontier_blacklisted": len(frontier_blacklist),
        "recovery_episodes": metric_guard_counts["recover"],
        "wander_episodes": metric_guard_counts["wander"],
        "mean_step_time_s": round(mean_dt, 3),
        "fps": round(1.0 / mean_dt, 1) if mean_dt > 0 else 0.0,
        "coverage_milestones": milestones,
        "video_path": args.out if not args.no_video else None,
    }

    summary_path = os.path.join(
        os.path.dirname(args.out) or ".", "explore_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n")
    if args.no_video:
        print("✅ Exploration demo completed (no video written)")
    else:
        print(f"✅ Exploration demo saved to {args.out}")
    print(f"   Final coverage: {final_cov:.1f}%")
    print(f"   Frontier switches: {frontier_switches}")
    print(f"   Summary: {summary_path}")


if __name__ == "__main__":
    main()
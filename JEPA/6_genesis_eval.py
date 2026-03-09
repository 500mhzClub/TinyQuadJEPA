#!/usr/bin/env python3
"""
Open-floor JEPA navigation demo (drop-in style replacement)

Purpose:
- remove obstacle-avoidance confounds
- show that the current JEPA-style model can drive the robot toward
  reachable latent goals in open space
- produce a cleaner demo with forward / strafe / turn primitives

Usage:
    python JEPA/6_genesis_eval.py \
        --jepa_ckpt jepa_checkpoints/jepa_epoch_8_step_3000.pt \
        --ppo_ckpt runs/pupper_omni_20260225_150134/ckpt_20000.pt \
        --device cpu
"""
import os
import json
import time
import math
import argparse
from typing import List, Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

import genesis as gs


# -----------------------------------------
# Quaternion helpers
# -----------------------------------------
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


def quat_to_yaw_wxyz_np(q: np.ndarray) -> float:
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# -----------------------------------------
# Projection helper for HUD goal marker
# -----------------------------------------
def project_3d_to_2d(pt_3d, cam_pos, look_at, fov_deg=50, res=(768, 768)):
    forward = look_at - cam_pos
    dist = np.linalg.norm(forward)
    if dist < 1e-5:
        return None
    forward = forward / dist

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    right = np.cross(forward, world_up)
    rnorm = np.linalg.norm(right)
    if rnorm < 1e-6:
        return None
    right = right / rnorm

    up = np.cross(right, forward)
    up = up / max(np.linalg.norm(up), 1e-6)

    v = pt_3d - cam_pos
    z_cam = np.dot(v, forward)
    if z_cam <= 0.01:
        return None

    x_cam = np.dot(v, right)
    y_cam = np.dot(v, up)

    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)

    x_ndc = (x_cam * f) / z_cam
    y_ndc = (y_cam * f) / z_cam

    u = (x_ndc + 1.0) * 0.5 * res[0]
    v = (1.0 - y_ndc) * 0.5 * res[1]
    return int(u), int(v)


# -----------------------------------------
# Architectures
# -----------------------------------------
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

    def act_deterministic(self, obs: torch.Tensor):
        return torch.tanh(self.actor(obs))


class VisionEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ELU(),
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
            nn.Linear(256, feature_dim), nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(feature_dim=128)
        self.prop_enc = ProprioEncoder(input_dim=47, feature_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, vision: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        v_feat = self.vis_enc(vision)
        p_feat = self.prop_enc(proprio)
        return self.fusion(torch.cat([v_feat, p_feat], dim=-1))


class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(latent_dim + cmd_dim, latent_dim), nn.ELU())
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, c_t: torch.Tensor, h_t: torch.Tensor):
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        return self.output_proj(h_next), h_next


class EBM_TinyQuadJEPA(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = JointEncoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim, cmd_dim=3)


# -----------------------------------------
# Scene setup (open floor)
# -----------------------------------------
def init_genesis_scene(device):
    print("🌍 Booting Genesis Simulator (Open-Floor Demo Mode)...")
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)

    tex_path = os.path.abspath("checkerboard.png")
    if not os.path.exists(tex_path):
        checker = np.indices((32, 32)).sum(axis=0) % 2
        checker = np.repeat(np.repeat(checker, 32, axis=0), 32, axis=1)
        checker = (checker * 255).astype(np.uint8)
        Image.fromarray(checker).save(tex_path)

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=tex_path)
        ),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/mini_pupper/mini_pupper.urdf",
            pos=(0.0, 0.0, 0.12),
            fixed=False,
            merge_fixed_links=False,
            requires_jac_and_IK=False,
        )
    )

    cam_brain = scene.add_camera(res=(64, 64), pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50)
    cam_ego_vis = scene.add_camera(res=(768, 768), pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50)
    cam_3rd = scene.add_camera(res=(768, 768), pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50)

    scene.build()

    actuated_joints = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    dofs_idx = [robot.get_joint(name).dofs_idx_local[0] for name in actuated_joints]

    q0 = np.array([
        0.06, 0.06, -0.06, -0.06,
        0.85, 0.85, 0.85, 0.85,
        -1.75, -1.75, -1.75, -1.75,
    ], dtype=np.float32)
    robot.set_dofs_position(q0, dofs_idx)
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, dofs_idx)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, dofs_idx)

    return scene, robot, cam_brain, cam_ego_vis, cam_3rd, dofs_idx, torch.tensor(q0, device=device)


# -----------------------------------------
# State builders
# -----------------------------------------
def get_system1_obs(robot, q0, prev_action, cmd, act_dofs, device):
    pos = robot.get_pos().to(device)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    quat = robot.get_quat().to(device)
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    vel_w = robot.get_vel().to(device)
    if vel_w.dim() == 1:
        vel_w = vel_w.unsqueeze(0)
    ang_w = robot.get_ang().to(device)
    if ang_w.dim() == 1:
        ang_w = ang_w.unsqueeze(0)

    vel_b = world_to_body_vec(quat, vel_w)
    ang_b = world_to_body_vec(quat, ang_w)

    q = robot.get_dofs_position(act_dofs).to(device)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    dq = robot.get_dofs_velocity(act_dofs).to(device)
    if dq.dim() == 1:
        dq = dq.unsqueeze(0)

    z = pos[:, 2:3]
    q_rel = q - q0.unsqueeze(0)
    return torch.cat([z, quat, vel_b, ang_b, q_rel, dq, prev_action, cmd], dim=1)


def get_jepa_state(robot, cam_brain, q0, prev_action, act_dofs, device):
    render_out = cam_brain.render()
    img = render_out[0] if isinstance(render_out, tuple) else render_out
    if hasattr(img, 'cpu'):
        img = img.cpu().numpy()
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img_chw = np.transpose(img.astype(np.uint8), (2, 0, 1))
    vis_tensor = torch.from_numpy(img_chw).float().to(device) / 255.0

    dummy_cmd = torch.zeros((1, 3), device=device)
    sys1_obs = get_system1_obs(robot, q0, prev_action, dummy_cmd, act_dofs, device)
    return vis_tensor.unsqueeze(0), sys1_obs[:, :47].clone()


# -----------------------------------------
# Cameras and HUD
# -----------------------------------------
def move_cameras(robot, cam_brain, cam_ego_vis, cam_3rd=None, goal_pos=None):
    r_pos = robot.get_pos().cpu().numpy()
    r_quat = robot.get_quat().cpu().numpy()
    if r_pos.ndim > 1:
        r_pos, r_quat = r_pos[0], r_quat[0]

    w, x, y, z = r_quat
    fx = 1 - 2 * (y**2 + z**2)
    fy = 2 * (x * y + w * z)
    fz = 2 * (x * z - w * y)
    forward = np.array([fx, fy, fz], dtype=np.float32)

    ux = 2 * (x * z + w * y)
    uy = 2 * (y * z - w * x)
    uz = 1 - 2 * (x**2 + y**2)
    up = np.array([ux, uy, uz], dtype=np.float32)

    cam_pos = r_pos + (forward * 0.10) + (up * 0.05)
    look_target = cam_pos + (forward * 1.0)

    for cam in [cam_brain, cam_ego_vis]:
        try:
            cam.set_pose(pos=cam_pos, lookat=look_target, up=up)
        except TypeError:
            cam.set_pose(pos=cam_pos, lookat=look_target)

    cam_3rd_pos, look_at_pt = None, None
    if cam_3rd is not None:
        cam_3rd_pos = r_pos + np.array([-1.3, 0.0, 0.9], dtype=np.float32)
        if goal_pos is not None:
            look_at_pt = 0.55 * r_pos + 0.45 * goal_pos
        else:
            look_at_pt = r_pos + forward * 0.5
        try:
            cam_3rd.set_pose(pos=cam_3rd_pos, lookat=look_at_pt)
        except TypeError:
            cam_3rd.set_pose(pos=cam_3rd_pos, lookat=look_at_pt)

    return cam_3rd_pos, look_at_pt


def render_camera_rgb(cam) -> np.ndarray:
    out = cam.render()
    img = out[0] if isinstance(out, tuple) else out
    if hasattr(img, 'cpu'):
        img = img.cpu().numpy()
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def draw_goal_marker(img_3rd_pil: Image.Image, cam_3rd_pos, look_at_pt, goal_pos_np):
    if cam_3rd_pos is None or look_at_pt is None:
        return
    uv = project_3d_to_2d(goal_pos_np, cam_3rd_pos, look_at_pt, fov_deg=50, res=(768, 768))
    if uv is None:
        return
    u, v = uv
    r = 15
    draw = ImageDraw.Draw(img_3rd_pil)
    draw.ellipse((u-r, v-r, u+r, v+r), outline="red", width=4)
    draw.line((u, v-r-20, u, v+r+20), fill="red", width=4)
    draw.line((u-r-20, v, u+r+20, v), fill="red", width=4)


def compose_frame(img_ego, img_3rd, hud_lines: List[str]) -> np.ndarray:
    combined = Image.new('RGB', (1536, 768))
    combined.paste(Image.fromarray(img_ego), (0, 0))
    combined.paste(Image.fromarray(img_3rd), (768, 0))
    draw = ImageDraw.Draw(combined)
    draw.rectangle([(8, 8), (520, 190)], fill=(0, 0, 0))
    draw.text((16, 14), "Left: ego view used by JEPA", fill=(255, 255, 255))
    draw.text((784, 14), "Right: external open-floor tracking view", fill=(255, 255, 255))

    y = 40
    for line in hud_lines:
        draw.text((16, y), line, fill=(255, 255, 255))
        y += 18
    return np.array(combined)


# -----------------------------------------
# Open-floor primitive demonstration path
# -----------------------------------------
def primitive_script():
    return [
        ("forward",  24, [0.28,  0.00,  0.00]),
        ("left",     16, [0.00,  0.16,  0.00]),
        ("right",    16, [0.00, -0.16,  0.00]),
        ("turn_l",   18, [0.00,  0.00,  0.55]),
        ("forward2", 18, [0.24,  0.00,  0.00]),
        ("turn_r",   18, [0.00,  0.00, -0.55]),
        ("diag",     18, [0.18,  0.10,  0.10]),
    ]


def capture_demo_waypoints(scene, robot, cam_brain, cam_ego_vis, act_dofs, q0, ppo, jepa, device,
                           action_scale: float, waypoint_stride: int, min_waypoint_spacing: float):
    print("\n🎬 Driving primitive open-floor demo to generate reachable latent waypoints...")

    waypoints_z: List[np.ndarray] = []
    waypoints_pos: List[np.ndarray] = []
    waypoints_yaw: List[float] = []
    prev_action_demo = torch.zeros((1, 12), device=device)
    demo_step_count = 0
    last_keep_xy = None
    total_demo_steps = sum(d for _, d, _ in primitive_script())

    for phase_name, duration, cmd_vals in primitive_script():
        demo_cmd = torch.tensor([cmd_vals], device=device, dtype=torch.float32)
        for _ in range(duration):
            sys1_obs = get_system1_obs(robot, q0, prev_action_demo, demo_cmd, act_dofs, device)
            with torch.no_grad():
                action = ppo.act_deterministic(sys1_obs)
            prev_action_demo = action.clone()

            q_tgt = q0.unsqueeze(0) + action_scale * action
            robot.control_dofs_position(q_tgt[0].detach().to(gs.device), act_dofs)

            for _ in range(4):
                scene.step()
            move_cameras(robot, cam_brain, cam_ego_vis)

            demo_step_count += 1
            print(f"\rDriving primitive path... {demo_step_count}/{total_demo_steps} [{phase_name}]", end="")

            if demo_step_count % waypoint_stride == 0:
                pos_w = robot.get_pos().cpu().numpy()
                quat_w = robot.get_quat().cpu().numpy()
                if pos_w.ndim > 1:
                    pos_w = pos_w[0]
                    quat_w = quat_w[0]
                keep = False
                if last_keep_xy is None:
                    keep = True
                else:
                    keep = np.linalg.norm(pos_w[:2] - last_keep_xy) >= min_waypoint_spacing
                if keep:
                    vis_w, prop_w = get_jepa_state(robot, cam_brain, q0, prev_action_demo, act_dofs, device)
                    with torch.no_grad():
                        z_w = jepa.encoder(vis_w, prop_w).detach().cpu().numpy()[0]
                    pos_keep = pos_w.copy()
                    pos_keep[2] = 0.10
                    waypoints_z.append(z_w)
                    waypoints_pos.append(pos_keep)
                    waypoints_yaw.append(quat_to_yaw_wxyz_np(quat_w))
                    last_keep_xy = pos_keep[:2].copy()

    print(f"\n✅ Captured {len(waypoints_z)} spaced open-floor waypoints.")
    return np.stack(waypoints_z), np.stack(waypoints_pos), np.asarray(waypoints_yaw, dtype=np.float32)


def save_waypoints(path: str, z: np.ndarray, pos: np.ndarray, yaw: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, waypoints_z=z, waypoints_pos=pos, waypoints_yaw=yaw)
    print(f"💾 Saved waypoint file to {path}")


def load_waypoints(path: str):
    data = np.load(path)
    if "waypoints_yaw" in data:
        return data["waypoints_z"], data["waypoints_pos"], data["waypoints_yaw"]
    # backward-compatible fallback
    n = len(data["waypoints_pos"])
    return data["waypoints_z"], data["waypoints_pos"], np.zeros((n,), dtype=np.float32)


# -----------------------------------------
# Planner
# -----------------------------------------
def build_candidate_bank(prev_best_cmd: torch.Tensor, n: int, device: torch.device):
    templates = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.20, 0.00, 0.00],
        [0.30, 0.00, 0.00],
        [0.00, 0.16, 0.00],
        [0.00, -0.16, 0.00],
        [0.10, 0.08, 0.00],
        [0.10, -0.08, 0.00],
        [0.00, 0.00, 0.35],
        [0.00, 0.00, -0.35],
        [0.00, 0.00, 0.55],
        [0.00, 0.00, -0.55],
        [0.18, 0.10, 0.10],
        [0.18, -0.10, -0.10],
    ], device=device, dtype=torch.float32)

    if prev_best_cmd is not None:
        prev = prev_best_cmd.detach().clone().view(1, 3)
        template_extra = torch.cat([
            prev,
            prev + torch.tensor([[0.06, 0.00, 0.00]], device=device),
            prev + torch.tensor([[-0.06, 0.00, 0.00]], device=device),
            prev + torch.tensor([[0.00, 0.05, 0.00]], device=device),
            prev + torch.tensor([[0.00, -0.05, 0.00]], device=device),
            prev + torch.tensor([[0.00, 0.00, 0.12]], device=device),
            prev + torch.tensor([[0.00, 0.00, -0.12]], device=device),
        ], dim=0)
        templates = torch.cat([templates, template_extra], dim=0)

    remaining = max(0, n - templates.shape[0])
    rand = (torch.rand((remaining, 3), device=device) * 2.0) - 1.0
    rand[:, 0] *= 0.40
    rand[:, 1] *= 0.22
    rand[:, 2] *= 0.60

    if prev_best_cmd is not None and remaining > 0:
        half = remaining // 2
        noise = torch.randn((half, 3), device=device) * torch.tensor([0.08, 0.05, 0.15], device=device)
        local = prev_best_cmd.view(1, 3) + noise
        local[:, 0] = local[:, 0].clamp(-0.40, 0.40)
        local[:, 1] = local[:, 1].clamp(-0.22, 0.22)
        local[:, 2] = local[:, 2].clamp(-0.60, 0.60)
        rand[:half] = local

    cmds = torch.cat([templates, rand], dim=0)
    cmds[:, 0] = cmds[:, 0].clamp(-0.40, 0.40)
    cmds[:, 1] = cmds[:, 1].clamp(-0.22, 0.22)
    cmds[:, 2] = cmds[:, 2].clamp(-0.60, 0.60)
    return cmds[:n]


def plan_best_cmd(jepa, z_current, z_goal, h_exec, prev_best_cmd, candidates: int, horizon: int, device):
    candidate_cmds = build_candidate_bank(prev_best_cmd, candidates, device)
    z_batch = z_current.expand(candidate_cmds.shape[0], -1)
    h_batch = h_exec.expand(candidate_cmds.shape[0], -1).clone()

    z_pred = z_batch
    cmd_seq = candidate_cmds.unsqueeze(1).expand(-1, horizon, -1)

    with torch.no_grad():
        for t in range(horizon):
            z_pred, h_batch = jepa.predictor(z_pred, cmd_seq[:, t], h_batch)

        latent_cost = 1.0 - F.cosine_similarity(z_pred, z_goal.expand_as(z_pred), dim=-1)
        reg_turn = 0.05 * candidate_cmds[:, 2].abs()
        reg_side = 0.03 * candidate_cmds[:, 1].abs()
        reg_stop = 0.02 * F.relu(0.04 - candidate_cmds[:, 0])
        smooth = torch.zeros_like(latent_cost)
        if prev_best_cmd is not None:
            smooth = 0.05 * (candidate_cmds - prev_best_cmd.view(1, 3)).pow(2).sum(dim=-1)
        total_cost = latent_cost + reg_turn + reg_side + reg_stop + smooth

        best_idx = torch.argmin(total_cost)
        best_cmd = candidate_cmds[best_idx].view(1, 3)
        stats = {
            "best_total_cost": float(total_cost[best_idx].item()),
            "best_latent_cost": float(latent_cost[best_idx].item()),
            "candidate_cost_min": float(total_cost.min().item()),
            "candidate_cost_mean": float(total_cost.mean().item()),
            "candidate_cost_max": float(total_cost.max().item()),
        }
    return best_cmd, stats


# -----------------------------------------
# Main
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", type=str, required=True)
    parser.add_argument("--ppo_ckpt", type=str, required=True)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--plan_freq", type=int, default=5)
    parser.add_argument("--out", type=str, default="eval_output_open_nav.mp4")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--waypoints_file", type=str, default="")
    parser.add_argument("--save_waypoints", type=str, default="jepa_logs/open_nav_waypoints.npz")
    parser.add_argument("--waypoint_stride", type=int, default=5)
    parser.add_argument("--min_waypoint_spacing", type=float, default=0.10)
    parser.add_argument("--reach_tol", type=float, default=0.11)
    parser.add_argument("--yaw_tol", type=float, default=0.45)
    parser.add_argument("--summary_json", type=str, default="jepa_logs/eval_open_nav_summary.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Loading brains into Genesis on {device}...")

    jepa = EBM_TinyQuadJEPA().to(device)
    jepa_ckpt = torch.load(args.jepa_ckpt, map_location=device, weights_only=True)
    jepa.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in jepa_ckpt['model_state_dict'].items()})
    jepa.eval()

    ppo = ActorCritic(obs_dim=50, act_dim=12).to(device)
    ppo.load_state_dict(torch.load(args.ppo_ckpt, map_location=device)["model"])
    ppo.eval()
    print("✅ Both checkpoints loaded successfully!")

    scene, robot, cam_brain, cam_ego_vis, cam_3rd, act_dofs, q0 = init_genesis_scene(device)
    q0_np = q0.cpu().numpy()
    action_scale = 0.30

    for _ in range(10):
        scene.step()

    if args.waypoints_file and os.path.exists(args.waypoints_file):
        print(f"📦 Loading waypoint file from {args.waypoints_file}")
        waypoints_z_np, waypoints_pos_np, waypoints_yaw_np = load_waypoints(args.waypoints_file)
    else:
        waypoints_z_np, waypoints_pos_np, waypoints_yaw_np = capture_demo_waypoints(
            scene, robot, cam_brain, cam_ego_vis, act_dofs, q0, ppo, jepa, device,
            action_scale=action_scale,
            waypoint_stride=args.waypoint_stride,
            min_waypoint_spacing=args.min_waypoint_spacing,
        )
        if args.save_waypoints:
            save_waypoints(args.save_waypoints, waypoints_z_np, waypoints_pos_np, waypoints_yaw_np)

    waypoints_z = torch.from_numpy(waypoints_z_np).float().to(device)

    print("⏪ Resetting to origin to begin open-floor latent tracking test...")
    start_pos = np.array([0.0, 0.0, 0.12], dtype=np.float32)
    try:
        robot.set_pos(start_pos)
        robot.set_dofs_position(q0_np, act_dofs)
    except Exception:
        pass
    for _ in range(20):
        scene.step()

    video_writer = imageio.get_writer(args.out, fps=30)
    prev_action = torch.zeros((1, 12), device=device)
    prev_best_cmd = torch.zeros((1, 3), device=device)
    h_exec = torch.zeros((1, 256), device=device)
    last_plan_stats = {
        "best_total_cost": float("nan"),
        "best_latent_cost": float("nan"),
        "candidate_cost_min": float("nan"),
        "candidate_cost_mean": float("nan"),
        "candidate_cost_max": float("nan"),
    }

    target_idx = 0
    z_goal = waypoints_z[target_idx:target_idx + 1]
    goal_pos_np = waypoints_pos_np[target_idx].copy()
    goal_yaw = float(waypoints_yaw_np[target_idx])

    steps_taken = 0
    replans = 0
    reached = 0
    completed = False
    latent_cost_history = []
    dist_history = []
    yaw_err_history = []

    ckpt_epoch = int(jepa_ckpt.get("epoch", -1)) + 1 if isinstance(jepa_ckpt.get("epoch", None), int) else jepa_ckpt.get("epoch", "Unknown")
    ckpt_step = int(jepa_ckpt.get("batch_idx", -1)) + 1 if isinstance(jepa_ckpt.get("batch_idx", None), int) else jepa_ckpt.get("batch_idx", "Unknown")

    print(f"\n🐕 Running open-floor JEPA navigation demo for up to {args.steps} steps...\n")

    try:
        for step_count in range(args.steps):
            loop_start = time.perf_counter()
            steps_taken = step_count + 1

            cam_3rd_pos, look_at_pt = move_cameras(robot, cam_brain, cam_ego_vis, cam_3rd, goal_pos_np)
            vis_t, prop_t = get_jepa_state(robot, cam_brain, q0, prev_action, act_dofs, device)

            with torch.no_grad():
                z_current = jepa.encoder(vis_t, prop_t)

            r_pos_current = robot.get_pos().cpu().numpy()
            r_quat_current = robot.get_quat().cpu().numpy()
            if r_pos_current.ndim > 1:
                r_pos_current = r_pos_current[0]
                r_quat_current = r_quat_current[0]
            yaw_now = quat_to_yaw_wxyz_np(r_quat_current)
            yaw_err = abs(angle_wrap(yaw_now - goal_yaw))
            dist_to_goal = float(np.linalg.norm(r_pos_current[:2] - goal_pos_np[:2]))

            if dist_to_goal < args.reach_tol and yaw_err < args.yaw_tol:
                if target_idx < len(waypoints_pos_np) - 1:
                    target_idx += 1
                    reached = target_idx
                    z_goal = waypoints_z[target_idx:target_idx + 1]
                    goal_pos_np = waypoints_pos_np[target_idx].copy()
                    goal_yaw = float(waypoints_yaw_np[target_idx])
                    print(f"\n✅ Reached waypoint {target_idx}/{len(waypoints_pos_np)}. Advancing...")
                else:
                    completed = True
                    reached = len(waypoints_pos_np)
                    print("\n\n🎉 Final waypoint reached. Demo complete.")
                    break

            if step_count % args.plan_freq == 0:
                best_cmd, plan_stats = plan_best_cmd(
                    jepa=jepa,
                    z_current=z_current,
                    z_goal=z_goal,
                    h_exec=h_exec,
                    prev_best_cmd=prev_best_cmd,
                    candidates=args.candidates,
                    horizon=args.horizon,
                    device=device,
                )
                prev_best_cmd = best_cmd.clone()
                last_plan_stats = plan_stats
                replans += 1
            else:
                best_cmd = prev_best_cmd
                plan_stats = last_plan_stats

            with torch.no_grad():
                _, h_exec = jepa.predictor(z_current, best_cmd, h_exec)
                sys1_obs = get_system1_obs(robot, q0, prev_action, best_cmd, act_dofs, device)
                action = ppo.act_deterministic(sys1_obs)
                prev_action = action.clone()

            q_tgt = q0.unsqueeze(0) + action_scale * action
            robot.control_dofs_position(q_tgt[0].detach().to(gs.device), act_dofs)
            for _ in range(4):
                scene.step()

            img_ego = render_camera_rgb(cam_ego_vis)
            img_3rd = render_camera_rgb(cam_3rd)
            img_3rd_pil = Image.fromarray(img_3rd)
            draw_goal_marker(img_3rd_pil, cam_3rd_pos, look_at_pt, goal_pos_np)

            latent_cost_history.append(plan_stats["best_latent_cost"])
            dist_history.append(dist_to_goal)
            yaw_err_history.append(yaw_err)

            hz = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
            hud_lines = [
                f"Checkpoint: epoch={ckpt_epoch} step={ckpt_step}",
                f"Waypoint: {target_idx + 1}/{len(waypoints_pos_np)}   reached={reached}",
                f"Distance: {dist_to_goal:.3f} m   yaw err: {yaw_err:.2f} rad",
                f"Cmd: vx={best_cmd[0,0]:+.2f} vy={best_cmd[0,1]:+.2f} wz={best_cmd[0,2]:+.2f}",
                f"Latent cost: {plan_stats['best_latent_cost']:.3f}   total: {plan_stats['best_total_cost']:.3f}",
                f"Planner mean/min cost: {plan_stats['candidate_cost_mean']:.3f} / {plan_stats['candidate_cost_min']:.3f}",
                f"Replans: {replans}   sim loop: {hz:5.1f} Hz",
            ]
            frame = compose_frame(img_ego, np.array(img_3rd_pil), hud_lines)
            video_writer.append_data(frame)

            print(
                f"\r⚡ step={step_count+1:04d}/{args.steps} | wpt={target_idx+1:02d}/{len(waypoints_pos_np)} "
                f"| dist={dist_to_goal:.3f} | yaw={yaw_err:.2f} "
                f"| cmd=[{best_cmd[0,0]:+.2f}, {best_cmd[0,1]:+.2f}, {best_cmd[0,2]:+.2f}] "
                f"| latent={plan_stats['best_latent_cost']:.3f} | hz={hz:4.1f}",
                end="",
            )

    except KeyboardInterrupt:
        print("\n\n🛑 Simulation interrupted.")
    finally:
        video_writer.close()

    final_pos = robot.get_pos().cpu().numpy()
    final_quat = robot.get_quat().cpu().numpy()
    if final_pos.ndim > 1:
        final_pos = final_pos[0]
        final_quat = final_quat[0]
    final_goal_dist = float(np.linalg.norm(final_pos[:2] - goal_pos_np[:2]))
    final_yaw_err = float(abs(angle_wrap(quat_to_yaw_wxyz_np(final_quat) - goal_yaw)))

    summary = {
        "jepa_ckpt": args.jepa_ckpt,
        "ppo_ckpt": args.ppo_ckpt,
        "epoch": ckpt_epoch,
        "step": ckpt_step,
        "steps_budget": args.steps,
        "steps_taken": steps_taken,
        "completed": completed,
        "waypoints_total": int(len(waypoints_pos_np)),
        "waypoints_reached": int(reached),
        "waypoint_fraction": float(reached / max(len(waypoints_pos_np), 1)),
        "replans": int(replans),
        "final_goal_dist": final_goal_dist,
        "final_yaw_err": final_yaw_err,
        "mean_latent_cost": float(np.nanmean(latent_cost_history)) if latent_cost_history else float("nan"),
        "mean_goal_dist": float(np.nanmean(dist_history)) if dist_history else float("nan"),
        "mean_yaw_err": float(np.nanmean(yaw_err_history)) if yaw_err_history else float("nan"),
        "video_path": args.out,
    }

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Summary written to {args.summary_json}")

    print(f"\n✅ Video saved as {args.out}")
    print("📊 Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

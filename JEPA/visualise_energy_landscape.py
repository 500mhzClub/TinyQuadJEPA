#!/usr/bin/env python3
"""
JEPA latent energy visualiser (drop-in replacement)

What this script does:
- builds a reachable goal state by physically driving the PPO controller
- resets to the start state
- scans a (vx, wz) command grid through the latent predictor
- renders a polished figure with a heatmap and 3D surface
- marks the lowest-cost command and writes the raw grid to disk

Usage:
    python JEPA/visualise_energy_landscape.py \
        --ckpt jepa_checkpoints/jepa_epoch_8_step_3000.pt \
        --ppo_ckpt runs/pupper_omni_20260225_150134/ckpt_20000.pt \
        --device cpu
"""
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image

import genesis as gs


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
            nn.Linear(128 + 128, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, vision: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        v_feat = self.vis_enc(vision)
        p_feat = self.prop_enc(proprio)
        return self.fusion(torch.cat([v_feat, p_feat], dim=-1))


class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + cmd_dim, latent_dim), nn.ELU()
        )
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
# Simulator helpers
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


def init_genesis_scene(device):
    print("🌍 Booting Genesis Simulator (Headless CPU Mode)...")
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

    scene.add_entity(gs.morphs.Box(pos=(0.5, 0.4, 0.075), size=(0.15, 0.15, 0.15), fixed=True))
    scene.add_entity(gs.morphs.Box(pos=(0.9, -0.3, 0.05), size=(0.1, 0.2, 0.1), fixed=True))
    scene.add_entity(gs.morphs.Box(pos=(1.3, 0.2, 0.1), size=(0.2, 0.1, 0.2), fixed=True))
    scene.add_entity(gs.morphs.Box(pos=(0.2, -0.5, 0.06), size=(0.12, 0.12, 0.12), fixed=True))

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
    return scene, robot, cam_brain, dofs_idx, torch.tensor(q0, device=device)


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
    img = None
    if isinstance(render_out, tuple) and len(render_out) > 0:
        img = render_out[0]
    elif isinstance(render_out, dict):
        img = render_out.get('rgb', render_out.get('color'))
    elif hasattr(render_out, 'shape'):
        img = render_out
    if img is None:
        raise RuntimeError("Camera render failed")

    if hasattr(img, 'cpu'):
        img = img.cpu().numpy()
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    if img.shape[-1] == 3:
        img = np.transpose(img, (2, 0, 1))

    vis_tensor = torch.from_numpy(img.copy()).float().to(device) / 255.0
    dummy_cmd = torch.zeros((1, 3), device=device)
    sys1_obs = get_system1_obs(robot, q0, prev_action, dummy_cmd, act_dofs, device)
    prop_tensor = sys1_obs[:, :47].clone()
    return vis_tensor.unsqueeze(0), prop_tensor


def move_camera(robot, cam_brain):
    r_pos = robot.get_pos().cpu().numpy()
    r_quat = robot.get_quat().cpu().numpy()
    if r_pos.ndim > 1:
        r_pos = r_pos[0]
        r_quat = r_quat[0]

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
    try:
        cam_brain.set_pose(pos=cam_pos, lookat=look_target, up=up)
    except TypeError:
        cam_brain.set_pose(pos=cam_pos, lookat=look_target)


def rollout_to_goal(scene, robot, cam_brain, q0, act_dofs, ppo, jepa, action_scale, horizon, device, cmd_vals):
    print(f"🎬 Driving PPO for {horizon} predictor steps to capture a reachable target...")
    prev_action = torch.zeros((1, 12), device=device)
    cmd = torch.tensor([cmd_vals], device=device, dtype=torch.float32)
    for _ in range(horizon):
        sys1_obs = get_system1_obs(robot, q0, prev_action, cmd, act_dofs, device)
        with torch.no_grad():
            action = ppo.act_deterministic(sys1_obs)
        prev_action = action.clone()
        q_tgt = q0.unsqueeze(0) + action_scale * action
        robot.control_dofs_position(q_tgt[0].detach().to(gs.device), act_dofs)
        for _ in range(4):
            scene.step()
    move_camera(robot, cam_brain)
    vis_goal, prop_goal = get_jepa_state(robot, cam_brain, q0, prev_action, act_dofs, device)
    with torch.no_grad():
        z_goal = jepa.encoder(vis_goal, prop_goal).detach()
    goal_pos = robot.get_pos().cpu().numpy()
    if goal_pos.ndim > 1:
        goal_pos = goal_pos[0]
    return z_goal, goal_pos


# -----------------------------------------
# Main
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to JEPA checkpoint")
    parser.add_argument("--ppo_ckpt", type=str, required=True, help="Path to PPO checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--res", type=int, default=35, help="Grid resolution")
    parser.add_argument("--goal_vx", type=float, default=0.40)
    parser.add_argument("--goal_vy", type=float, default=0.00)
    parser.add_argument("--goal_wz", type=float, default=0.00)
    parser.add_argument("--out", type=str, default="jepa_logs/energy_landscape.png")
    parser.add_argument("--grid_npz", type=str, default="jepa_logs/energy_landscape_grid.npz")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Loading brains into Genesis on {device}...")

    jepa = EBM_TinyQuadJEPA().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    jepa.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()})
    jepa.eval()

    ppo = ActorCritic(obs_dim=50, act_dim=12).to(device)
    ppo.load_state_dict(torch.load(args.ppo_ckpt, map_location=device)["model"], strict=False)
    ppo.eval()

    scene, robot, cam_brain, act_dofs, q0 = init_genesis_scene(device)
    q0_np = q0.cpu().numpy()
    action_scale = 0.30

    for _ in range(10):
        scene.step()

    z_goal, goal_pos = rollout_to_goal(
        scene, robot, cam_brain, q0, act_dofs, ppo, jepa,
        action_scale=action_scale,
        horizon=args.horizon,
        device=device,
        cmd_vals=[args.goal_vx, args.goal_vy, args.goal_wz],
    )

    print("⏪ Resetting to origin to map energy landscape...")
    try:
        robot.set_pos(np.array([0.0, 0.0, 0.12], dtype=np.float32))
        robot.set_dofs_position(q0_np, act_dofs)
    except Exception:
        pass
    for _ in range(20):
        scene.step()
    move_camera(robot, cam_brain)

    prev_action_start = torch.zeros((1, 12), device=device)
    vis_start, prop_start = get_jepa_state(robot, cam_brain, q0, prev_action_start, act_dofs, device)
    with torch.no_grad():
        z_start = jepa.encoder(vis_start, prop_start).detach()

    print(f"🧠 Scanning latent command grid at {args.res}x{args.res}...")
    vx_vals = np.linspace(-0.40, 0.40, args.res)
    om_vals = np.linspace(-0.60, 0.60, args.res)
    VX, OM = np.meshgrid(vx_vals, om_vals)
    COSTS = np.zeros_like(VX, dtype=np.float32)

    with torch.no_grad():
        for i in range(args.res):
            for j in range(args.res):
                cmd = torch.tensor([[VX[i, j], 0.0, OM[i, j]]], device=device, dtype=torch.float32)
                z_pred = z_start.clone()
                h_t = torch.zeros(1, 256, device=device)
                for _ in range(args.horizon):
                    z_pred, h_t = jepa.predictor(z_pred, cmd, h_t)
                COSTS[i, j] = float((1.0 - F.cosine_similarity(z_pred, z_goal, dim=-1)).item())

    best_flat = int(np.argmin(COSTS))
    bi, bj = np.unravel_index(best_flat, COSTS.shape)
    best_vx = float(VX[bi, bj])
    best_wz = float(OM[bi, bj])
    best_cost = float(COSTS[bi, bj])

    os.makedirs(os.path.dirname(args.grid_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.grid_npz,
        VX=VX,
        OM=OM,
        COSTS=COSTS,
        best_vx=best_vx,
        best_wz=best_wz,
        best_cost=best_cost,
        goal_pos=goal_pos,
    )

    epoch = ckpt.get('epoch', 'Unknown')
    batch_idx = ckpt.get('batch_idx', 'Unknown')

    print("🎨 Rendering figure...")
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(
        COSTS,
        origin='lower',
        extent=[vx_vals.min(), vx_vals.max(), om_vals.min(), om_vals.max()],
        aspect='auto',
        interpolation='bicubic',
    )
    ax1.scatter([best_vx], [best_wz], marker='x', s=100)
    ax1.set_title("Latent cost heatmap")
    ax1.set_xlabel("Forward velocity vx")
    ax1.set_ylabel("Yaw rate wz")
    cbar1 = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Predicted terminal latent cost")
    ax1.text(
        0.02, 0.02,
        f"best: vx={best_vx:+.2f}, wz={best_wz:+.2f}\ncost={best_cost:.3f}",
        transform=ax1.transAxes,
        va='bottom', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(VX, OM, COSTS, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.95)
    ax2.scatter([best_vx], [best_wz], [best_cost], s=50)
    ax2.set_title("Latent cost surface")
    ax2.set_xlabel("vx")
    ax2.set_ylabel("wz")
    ax2.set_zlabel("cost")
    fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=18, pad=0.08)

    fig.suptitle(
        f"JEPA latent command landscape | epoch={epoch} step={batch_idx} | goal≈({goal_pos[0]:.2f}, {goal_pos[1]:.2f})",
        fontsize=14,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches='tight')
    print(f"✅ Energy landscape saved to {args.out}")
    print(f"💾 Raw grid saved to {args.grid_npz}")
    print(f"⭐ Best command on grid: vx={best_vx:+.3f}, wz={best_wz:+.3f}, cost={best_cost:.4f}")


if __name__ == "__main__":
    main()

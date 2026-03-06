#!/usr/bin/env python3
"""
System 1 + System 2 EBM JEPA - Genesis Simulator Evaluation
Runs the MPC loop using JEPA for navigation and PPO for low-level motor control.

Usage:
    python JEPA/6_genesis_eval.py \
        --jepa_ckpt jepa_checkpoints/jepa_epoch_2.pt \
        --ppo_ckpt runs/pupper_omni_20260225_150134/ckpt_20000.pt \
        --device cpu
"""
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import imageio
import genesis as gs

# -----------------------------------------
# Quaternion Helpers (System 1)
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

def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# -----------------------------------------
# 1. System 1: PPO Low-Level Motor Control (Full Class)
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 50, act_dim: int = 12, hid: int = 256):
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

# -----------------------------------------
# 2. System 2: EBM JEPA Architecture
# -----------------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int = 47, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class JointEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(feature_dim=128)
        self.prop_enc = ProprioEncoder(input_dim=47, feature_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ELU(),
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
            nn.Linear(latent_dim + cmd_dim, latent_dim),
            nn.ELU()
        )
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, z_t: torch.Tensor, c_t: torch.Tensor, h_t: torch.Tensor):
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        return self.output_proj(h_next), h_next

class EBM_TinyQuadJEPA(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = JointEncoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim, cmd_dim=3)

# -----------------------------------------
# 3. Genesis Simulator & State Building
# -----------------------------------------
def init_genesis_scene(device):
    print("🌍 Booting Genesis Simulator (Headless CPU Mode)...")
    gs.init(backend=gs.cpu) 
    scene = gs.Scene(show_viewer=False) 
    plane = scene.add_entity(gs.morphs.Plane()) 
    
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/mini_pupper/mini_pupper.urdf",
            pos=(0.0, 0.0, 0.12),
            fixed=False,
            merge_fixed_links=False,
            requires_jac_and_IK=False,
        )
    )
    
    cam_brain = scene.add_camera(
        res=(64, 64), pos=(0.8, -0.8, 0.45), lookat=(0.0, 0.0, 0.12), fov=50
    )
    cam_render = scene.add_camera(
        res=(512, 512), pos=(0.8, -0.8, 0.45), lookat=(0.0, 0.0, 0.12), fov=50
    )
    
    scene.build()

    actuated_joints = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    dofs_idx = [robot.get_joint(name).dofs_idx_local[0] for name in actuated_joints]
    
    # Init position
    q0 = np.array([0.06, 0.06, -0.06, -0.06, 0.85, 0.85, 0.85, 0.85, -1.75, -1.75, -1.75, -1.75], dtype=np.float32)
    robot.set_dofs_position(q0, dofs_idx)

    # PID values exactly like train_blind.py
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, dofs_idx)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, dofs_idx)

    return scene, robot, cam_brain, cam_render, dofs_idx, torch.tensor(q0, device=device)

def get_jepa_state(robot, cam_brain, device):
    cam_brain.render()
    try:
        img = cam_brain.get_image() if hasattr(cam_brain, 'get_image') else cam_brain.rgb
        if isinstance(img, np.ndarray) and img.shape[-1] in [3, 4]:
            img = img[:, :, :3] 
            img = np.transpose(img, (2, 0, 1)) 
            vis_tensor = torch.from_numpy(img).float().to(device) / 255.0
        else:
            vis_tensor = torch.zeros((3, 64, 64), device=device)
    except Exception:
        vis_tensor = torch.zeros((3, 64, 64), device=device)

    try:
        raw_prop = robot.get_dofs_position().cpu().numpy()
        if raw_prop.ndim == 2: raw_prop = raw_prop[0] 
        prop_array = np.zeros(47, dtype=np.float32)
        prop_array[:min(47, len(raw_prop))] = raw_prop[:min(47, len(raw_prop))]
        prop_tensor = torch.from_numpy(prop_array).float().to(device)
    except Exception:
        prop_tensor = torch.zeros(47, device=device)
        
    return vis_tensor.unsqueeze(0), prop_tensor.unsqueeze(0)

def get_system1_obs(robot, q0, prev_action, cmd, act_dofs, device):
    """Builds the exact 50D tensor that PPO expects, and forces EVERYTHING to the right device."""
    pos = robot.get_pos().to(device)
    if pos.dim() == 1: pos = pos.unsqueeze(0)
        
    quat = robot.get_quat().to(device)
    if quat.dim() == 1: quat = quat.unsqueeze(0)
        
    vel_w = robot.get_vel().to(device)
    if vel_w.dim() == 1: vel_w = vel_w.unsqueeze(0)
        
    ang_w = robot.get_ang().to(device)
    if ang_w.dim() == 1: ang_w = ang_w.unsqueeze(0)

    vel_b = world_to_body_vec(quat, vel_w)
    ang_b = world_to_body_vec(quat, ang_w)

    q = robot.get_dofs_position(act_dofs).to(device)
    if q.dim() == 1: q = q.unsqueeze(0)
        
    dq = robot.get_dofs_velocity(act_dofs).to(device)
    if dq.dim() == 1: dq = dq.unsqueeze(0)

    z = pos[:, 2:3]
    q_rel = q - q0.unsqueeze(0)

    # obs = [z(1), quat(4), vel_b(3), ang_b(3), q_rel(12), dq(12), prev_action(12), cmd(3)] = 50
    obs = torch.cat([z, quat, vel_b, ang_b, q_rel, dq, prev_action, cmd], dim=1)
    return obs

# -----------------------------------------
# 4. Main Loop
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", type=str, required=True, help="Path to JEPA (System 2)")
    parser.add_argument("--ppo_ckpt", type=str, required=True, help="Path to PPO (System 1)")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", type=str, default="eval_output.mp4")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Loading brains into Genesis on {device}...")

    # Load System 2 (JEPA)
    jepa = EBM_TinyQuadJEPA().to(device)
    jepa_ckpt = torch.load(args.jepa_ckpt, map_location=device, weights_only=True)
    jepa.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in jepa_ckpt['model_state_dict'].items()})
    jepa.eval()
    
    # Load System 1 (PPO)
    ppo = ActorCritic(obs_dim=50, act_dim=12).to(device)
    ppo_ckpt = torch.load(args.ppo_ckpt, map_location=device)
    ppo.load_state_dict(ppo_ckpt['model'])
    ppo.eval()
    print(f"✅ Both checkpoints loaded successfully!")

    scene, robot, cam_brain, cam_render, act_dofs, q0 = init_genesis_scene(device)
    cam_render.start_recording() 

    prev_action = torch.zeros((1, 12), device=device)
    action_scale = 0.30 

    print(f"\n🐕 Running integrated MPC (System 2 -> System 1). Recording {args.steps} steps...\n")
    
    try:
        for step_count in range(args.steps):
            loop_start = time.perf_counter()

            # --- SYSTEM 2: JEPA THINKS ---
            with torch.no_grad():
                vis_t, prop_t = get_jepa_state(robot, cam_brain, device)
                z_current = jepa.encoder(vis_t, prop_t)
                cam_render.render() 

                z_batch = z_current.expand(args.candidates, -1)
                h_t = torch.zeros(args.candidates, 256, device=device)
                
                # Constrain JEPA commands (vx, vy, omega) for Mini Pupper
                candidate_cmds = (torch.rand((args.candidates, args.horizon, 3), device=device) * 2.0) - 1.0
                candidate_cmds[:, :, 0] *= 0.60 # max vx
                candidate_cmds[:, :, 1] *= 0.30 # max vy
                candidate_cmds[:, :, 2] *= 0.80 # max omega
                
                total_energy = torch.zeros(args.candidates, device=device)
                z_pred = z_batch
                for t in range(args.horizon):
                    z_pred, h_t = jepa.predictor(z_pred, candidate_cmds[:, t], h_t)
                    total_energy += torch.norm(z_pred, dim=-1)

                best_idx = torch.argmin(total_energy)
                best_cmd = candidate_cmds[best_idx][0].unsqueeze(0) # [1, 3]

            # --- SYSTEM 1: PPO EXECUTES ---
            with torch.no_grad():
                sys1_obs = get_system1_obs(robot, q0, prev_action, best_cmd, act_dofs, device)
                action = ppo.act_deterministic(sys1_obs)
                prev_action = action.clone()

                # Convert PPO actions to joint angles
                q_tgt = q0.unsqueeze(0) + action_scale * action
                
                # Force it into Genesis device format to be safe
                q_tgt_gs = q_tgt[0].detach().to(gs.device)
                robot.control_dofs_position(q_tgt_gs, act_dofs)

            # --- PHYSICS ENGINE ---
            # decimation = 4 from your config
            for _ in range(4):
                scene.step()

            hz = 1.0 / (time.perf_counter() - loop_start)
            print(f"\r⚡ Sim Step: {step_count+1}/{args.steps} | Freq: {hz:5.1f} Hz | Cmd: [{best_cmd[0,0]:+.2f}, {best_cmd[0,1]:+.2f}, {best_cmd[0,2]:+.2f}]", end="")

    except KeyboardInterrupt:
        print("\n\n🛑 Simulation interrupted.")
    finally:
        try:
            cam_render.stop_recording(save_to_filename=args.out) 
        except TypeError:
            cam_render.stop_recording(save_path=args.out) 
        print(f"\n\n✅ Video saved as {args.out}")

if __name__ == "__main__":
    main()

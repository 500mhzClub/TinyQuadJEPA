#!/usr/bin/env python3
"""
System 2 JEPA MPC Planner.

This script demonstrates how to load the frozen JEPA backbone and the trained
GoalEnergyHead to perform latent-space Model Predictive Control (MPC).
It generates thousands of candidate action sequences, rolls them forward entirely 
in the GPU's memory (without touching the physics engine), and picks the 
commands that minimize the energy distance to a target goal.

Usage:
    python JEPA/mpc_inference.py \
        --jepa_ckpt jepa_checkpoints/jepa_epoch_20.pt \
        --head_ckpt energy_head_checkpoints/energy_head_best.pt
"""
from __future__ import annotations

import argparse
import time
import torch
import torch.nn as nn

# -----------------------------
# Architecture Definitions
# -----------------------------
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
            nn.LayerNorm(feature_dim)
        )
    def forward(self, x): return self.net(x)

class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int = 47, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, feature_dim), nn.LayerNorm(feature_dim)
        )
    def forward(self, x): return self.net(x)

class JointEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(feature_dim=128)
        self.prop_enc = ProprioEncoder(input_dim=47, feature_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
        )
    def forward(self, vision, proprio):
        return self.fusion(torch.cat([self.vis_enc(vision), self.prop_enc(proprio)], dim=-1))

class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(latent_dim + cmd_dim, latent_dim), nn.ELU())
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, z_t, c_t, h_t):
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        return self.output_proj(h_next), h_next

class TinyQuadJEPA(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = JointEncoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim, cmd_dim=3)

class GoalEnergyHead(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, z_pred, z_goal):
        x = torch.cat([z_pred, z_goal, z_pred - z_goal, z_pred * z_goal], dim=-1)
        return self.net(x).squeeze(-1)

# -----------------------------
# MPC Logic
# -----------------------------
@torch.no_grad()
def plan_mpc(
    backbone: TinyQuadJEPA,
    energy_head: GoalEnergyHead,
    vis_current: torch.Tensor,
    prop_current: torch.Tensor,
    vis_goal: torch.Tensor,
    prop_goal: torch.Tensor,
    num_samples: int = 10000,
    horizon: int = 15,
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Executes a zero-shot latent planning loop.
    """
    # 1. Encode starting and goal states into the latent space [1, 256]
    z_start = backbone.encoder(vis_current.unsqueeze(0), prop_current.unsqueeze(0))
    z_goal = backbone.encoder(vis_goal.unsqueeze(0), prop_goal.unsqueeze(0))
    
    # Expand to simulate `num_samples` parallel futures [N, 256]
    z_start = z_start.expand(num_samples, -1)
    z_goal = z_goal.expand(num_samples, -1)
    
    # 2. Generate massive batch of random command sequences (vx, vy, yaw)
    # Scaled roughly to the limits used in physics rollout
    cmds = torch.rand((num_samples, horizon, 3), device=device) * 2 - 1 
    cmds[:, :, 0] *= 0.40  # vx limit
    cmds[:, :, 1] *= 0.25  # vy limit
    cmds[:, :, 2] *= 0.60  # yaw limit
    
    # 3. Rollout all futures simultaneously in the latent space
    z_roll = z_start
    h_t = torch.zeros((num_samples, backbone.latent_dim), device=device)
    
    for t in range(horizon):
        z_roll, h_t = backbone.predictor(z_roll, cmds[:, t], h_t)
        
    # 4. Evaluate how close each imagined future got to the goal latent
    energies = energy_head(z_roll, z_goal)
    
    # 5. Extract the sequence that minimized the energy
    best_idx = torch.argmin(energies)
    best_action_sequence = cmds[best_idx]
    
    # In standard MPC, you only execute the immediate next step, then replan
    best_first_action = best_action_sequence[0]
    
    return best_first_action, best_action_sequence, energies[best_idx].item()

def clean_state_dict(state_dict):
    """Removes compiler/DDP prefixes if present."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", type=str, required=True)
    parser.add_argument("--head_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=15000, help="Number of trajectories to hallucinate")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"⚙️ Loading models onto {device}...")
    backbone = TinyQuadJEPA().to(device)
    backbone.load_state_dict(clean_state_dict(torch.load(args.jepa_ckpt, map_location=device)["model_state_dict"]))
    backbone.eval()

    energy_head = GoalEnergyHead().to(device)
    energy_head.load_state_dict(clean_state_dict(torch.load(args.head_ckpt, map_location=device)["energy_head_state_dict"]))
    energy_head.eval()

    # Create dummy tensors representing a live feed from the robot
    print("🤖 Simulating a live camera and proprioceptive feed...")
    vis_current = torch.rand((3, 64, 64), device=device)
    prop_current = torch.rand((47,), device=device)
    
    # Create dummy tensors representing the desired target state
    vis_goal = torch.rand((3, 64, 64), device=device)
    prop_goal = torch.rand((47,), device=device)

    print(f"🧠 Hallucinating {args.samples:,} parallel futures of length 15...")
    t0 = time.time()
    
    best_action, full_seq, final_energy = plan_mpc(
        backbone, energy_head, vis_current, prop_current, vis_goal, prop_goal,
        num_samples=args.samples, horizon=15, device=device
    )
    
    t_end = time.time()
    
    print("\n✅ Planning Complete!")
    print(f"⏱️  Time taken: {(t_end - t0) * 1000:.2f} ms")
    print(f"📉 Lowest Energy Found: {final_energy:.4f}")
    print(f"🚀 Optimal Next Command (vx, vy, yaw):")
    print(f"   {best_action.cpu().numpy()}")

if __name__ == "__main__":
    main()
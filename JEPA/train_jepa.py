#!/usr/bin/env python3
"""
System 2 EBM JEPA Training Loop (VICReg Formulation)
Optimized for high-VRAM / high-core-count workstations.

Usage:
    python JEPA/train_jepa.py
"""
import os
import glob
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Enable hardware autotuning for the CNN operations
torch.backends.cudnn.benchmark = True

# -----------------------------------------
# 1. Sequence Dataset Loader
# -----------------------------------------
class JEPADataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 16):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.h5_files = sorted(glob.glob(os.path.join(data_dir, "*_rgb.h5")))
        
        if not self.h5_files:
            raise FileNotFoundError(f"❌ No HDF5 datasets found in {data_dir}")
            
        self.samples = []
        print("🔍 Indexing S2W Dataset for valid temporal sequences...")
        
        for file_idx, fpath in enumerate(self.h5_files):
            try:
                with h5py.File(fpath, 'r') as h5f:
                    N, T = h5f['vision'].shape[:2]
                    max_start_t = T - self.seq_len
                    for env_idx in range(N):
                        # We slide a window of size `seq_len` across the trajectory
                        for start_t in range(max_start_t + 1):
                            self.samples.append((file_idx, env_idx, start_t))
            except Exception as e:
                print(f"⚠️ Warning: Could not index {fpath} ({e})")
                
        print(f"✅ Found {len(self.samples):,} valid {seq_len}-step sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, env_idx, start_t = self.samples[idx]
        fpath = self.h5_files[file_idx]
        end_t = start_t + self.seq_len
        
        with h5py.File(fpath, 'r') as h5f:
            # (Time, Channels, Height, Width) -> Normalized to [0.0, 1.0]
            vision = torch.from_numpy(h5f['vision'][env_idx, start_t:end_t]).float() / 255.0
            proprio = torch.from_numpy(h5f['proprio'][env_idx, start_t:end_t]).float()
            cmds = torch.from_numpy(h5f['cmds'][env_idx, start_t:end_t]).float()
            dones = torch.from_numpy(h5f['dones'][env_idx, start_t:end_t]).bool()

        return vision, proprio, cmds, dones

# -----------------------------------------
# 2. EBM JEPA Architecture
# -----------------------------------------
class VisionEncoder(nn.Module):
    """Compresses the 64x64 optical flow camera stream."""
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
    """Compresses joint angles and IMU data."""
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
    """Fuses modalities into the abstract state variable z_t."""
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
    """Predicts future states using a recurrent memory of momentum (GRU)."""
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
    """The overarching Energy-Based Model using VICReg geometry."""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = JointEncoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim, cmd_dim=3)

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def vicreg_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor):
        # 1. INVARIANCE: Push predictions and reality together
        sim_loss = F.mse_loss(z_pred, z_target)
        
        # 2. VARIANCE: Force batch embeddings to spread out (prevent point collapse)
        std_target = torch.sqrt(z_target.var(dim=0) + 1e-04)
        std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - std_target)) + torch.mean(F.relu(1 - std_pred))

        # 3. COVARIANCE: Decorrelate dimensions (prevent informational collapse)
        z_target_centered = z_target - z_target.mean(dim=0)
        z_pred_centered = z_pred - z_pred.mean(dim=0)
        batch_size = z_target.shape[0]
        
        cov_target = (z_target_centered.T @ z_target_centered) / (batch_size - 1)
        cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
        
        cov_loss = self.off_diagonal(cov_target).pow_(2).sum() / self.latent_dim + \
                   self.off_diagonal(cov_pred).pow_(2).sum() / self.latent_dim

        # Standard VICReg hyperparameters
        sim_weight, var_weight, cov_weight = 25.0, 25.0, 1.0
        total_loss = (sim_weight * sim_loss) + (var_weight * var_loss) + (cov_weight * cov_loss)
        
        return total_loss, sim_loss, var_loss, cov_loss

    def forward_step(self, vis_t, prop_t, cmd_t, vis_next, prop_next, h_t):
        z_t = self.encoder(vis_t, prop_t)
        z_pred, h_next = self.predictor(z_t, cmd_t, h_t)
        z_target = self.encoder(vis_next, prop_next)
        
        loss, sim, var, cov = self.vicreg_loss(z_pred, z_target)
        return loss, h_next, (sim.item(), var.item(), cov.item())

# -----------------------------------------
# 3. Backpropagation Through Time (BPTT)
# -----------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing EBM JEPA Training on {device}...")

    # Data Loading Strategy
    dataset = JEPADataset(data_dir="jepa_final_dataset", seq_len=16)
    dataloader = DataLoader(
        dataset, 
        batch_size=256,       # High batch size for geometric variance calculations
        shuffle=True, 
        num_workers=8,        # Feed the GPU fast enough to prevent idling
        pin_memory=True,      # Lock memory pages for faster CPU->GPU transfer
        prefetch_factor=2, 
        drop_last=True
    )

    model = EBM_TinyQuadJEPA().to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    os.makedirs("jepa_checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        total_epoch_loss = 0
        
        for batch_idx, (vis, prop, cmds, dones) in enumerate(pbar):
            # Move data sequence to GPU
            vis = vis.to(device, non_blocking=True)
            prop = prop.to(device, non_blocking=True)
            cmds = cmds.to(device, non_blocking=True)
            
            batch_size, seq_len = vis.shape[0], vis.shape[1]

            # Initialize working memory for the GRU
            h_t = torch.zeros(batch_size, 256, device=device)
            seq_loss = 0
            sim_avg, var_avg, cov_avg = 0, 0, 0

            # Unroll the sequence
            for t in range(seq_len - 1):
                loss, h_t, metrics = model.forward_step(
                    vis[:, t], prop[:, t], cmds[:, t],
                    vis[:, t+1], prop[:, t+1], h_t
                )
                
                # Prevent backprop across episode resets (if robot fell over)
                reset_mask = dones[:, t+1].float().to(device)
                loss = loss * (1.0 - reset_mask).mean()
                
                seq_loss += loss
                sim_avg += metrics[0]; var_avg += metrics[1]; cov_avg += metrics[2]

            # Average loss over the temporal steps
            seq_loss = seq_loss / (seq_len - 1)
            
            optimizer.zero_grad(set_to_none=True)
            seq_loss.backward()
            
            # Clip exploding RNN gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += seq_loss.item()
            
            # Update telemetry
            if batch_idx % 5 == 0:
                pbar.set_postfix({
                    "Energy": f"{seq_loss.item():.2f}",
                    "Sim": f"{sim_avg/(seq_len-1):.3f}",
                    "Var": f"{var_avg/(seq_len-1):.3f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.1e}"
                })

        # End of Epoch
        scheduler.step()
        avg_loss = total_epoch_loss / len(dataloader)
        print(f"🏁 Epoch {epoch+1} Complete | Avg Energy: {avg_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"jepa_checkpoints/jepa_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
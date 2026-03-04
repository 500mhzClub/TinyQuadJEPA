#!/usr/bin/env python3
"""
System 2 EBM JEPA Training Loop (VICReg Formulation)
Optimized with AMP, PyTorch Compiler, Worker-Side Batching, and Late GPU Casting.

Usage:
    python JEPA/train_jepa.py
"""
import os
import glob
import random
import h5py
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from collections import defaultdict

torch.backends.cudnn.benchmark = True

# -----------------------------------------
# 1. RAM-Buffered Streaming Dataset
# -----------------------------------------
class StreamingJEPADataset(IterableDataset):
    def __init__(self, data_dir: str, seq_len: int = 16, envs_per_chunk: int = 64, global_batch_size: int = 2048):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.envs_per_chunk = envs_per_chunk
        self.global_batch_size = global_batch_size
        self.h5_files = sorted(glob.glob(os.path.join(data_dir, "*_rgb.h5")))
        
        if not self.h5_files:
            raise FileNotFoundError(f"❌ No HDF5 datasets found in {data_dir}")
            
        self.global_envs = []
        print("🔍 Mapping global environments...")
        for fpath in self.h5_files:
            try:
                with h5py.File(fpath, 'r') as h5f:
                    N = h5f['vision'].shape[0]
                    for env_idx in range(N):
                        self.global_envs.append((fpath, env_idx))
            except Exception as e:
                print(f"⚠️ Warning: Could not index {fpath} ({e})")
                
        print(f"✅ Indexed {len(self.global_envs):,} environments.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            my_envs = self.global_envs
            seed = 0
        else:
            # Distribute environments evenly across active workers
            my_envs = self.global_envs[worker_info.id :: worker_info.num_workers]
            seed = worker_info.id

        rng = random.Random(seed)
        rng.shuffle(my_envs)
        
        # The worker builds the entire batch itself to bypass PyTorch collation overhead
        batch_v, batch_p, batch_c, batch_d = [], [], [], []

        for i in range(0, len(my_envs), self.envs_per_chunk):
            chunk_envs = my_envs[i : i + self.envs_per_chunk]
            
            vision_list, prop_list, cmd_list, done_list = [], [], [], []
            
            file_groups = defaultdict(list)
            for fpath, env_idx in chunk_envs:
                file_groups[fpath].append(env_idx)

            for fpath, indices in file_groups.items():
                indices.sort() 
                with h5py.File(fpath, 'r') as h5f:
                    v = h5f['vision'][indices] 
                    p = h5f['proprio'][indices]
                    c = h5f['cmds'][indices]
                    d = h5f['dones'][indices]
                    
                    for local_idx in range(len(indices)):
                        # 🔥 Loaded as strict uint8/float32 to minimize RAM footprint
                        vision_list.append(torch.from_numpy(v[local_idx])) 
                        prop_list.append(torch.from_numpy(p[local_idx]).float())
                        cmd_list.append(torch.from_numpy(c[local_idx]).float())
                        done_list.append(torch.from_numpy(d[local_idx]).bool())

            sequences = []
            num_envs_loaded = len(vision_list)
            T = vision_list[0].shape[0]
            max_start_t = T - self.seq_len

            for e_idx in range(num_envs_loaded):
                for start_t in range(max_start_t + 1):
                    sequences.append((e_idx, start_t))

            rng.shuffle(sequences)

            # 🔥 Worker-Side Batching
            for e_idx, start_t in sequences:
                end_t = start_t + self.seq_len
                
                batch_v.append(vision_list[e_idx][start_t:end_t])
                batch_p.append(prop_list[e_idx][start_t:end_t])
                batch_c.append(cmd_list[e_idx][start_t:end_t])
                batch_d.append(done_list[e_idx][start_t:end_t])
                
                if len(batch_v) == self.global_batch_size:
                    # Yield one massive, contiguous block of uint8 memory to the main thread
                    yield (
                        torch.stack(batch_v),
                        torch.stack(batch_p),
                        torch.stack(batch_c),
                        torch.stack(batch_d)
                    )
                    batch_v, batch_p, batch_c, batch_d = [], [], [], []
            
            # 🔥 Force garbage collection to ensure RAM is freed before loading next chunk
            del vision_list, prop_list, cmd_list, done_list
            gc.collect()

# -----------------------------------------
# 2. EBM JEPA Architecture
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

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def vicreg_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor):
        sim_loss = F.mse_loss(z_pred, z_target)
        
        std_target = torch.sqrt(z_target.var(dim=0) + 1e-04)
        std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - std_target)) + torch.mean(F.relu(1 - std_pred))

        z_target_centered = z_target - z_target.mean(dim=0)
        z_pred_centered = z_pred - z_pred.mean(dim=0)
        batch_size = z_target.shape[0]
        
        cov_target = (z_target_centered.T @ z_target_centered) / (batch_size - 1)
        cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
        
        cov_loss = self.off_diagonal(cov_target).pow_(2).sum() / self.latent_dim + \
                   self.off_diagonal(cov_pred).pow_(2).sum() / self.latent_dim

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

    global_batch_size = 2048
    dataset = StreamingJEPADataset(data_dir="jepa_final_dataset", seq_len=16, envs_per_chunk=64, global_batch_size=global_batch_size)
    
    total_batches = 4925 
    
    dataloader = DataLoader(
        dataset, 
        batch_size=None,      # 🔥 Set to None: The Dataset yields complete batches natively
        num_workers=12,       
        pin_memory=True,      
        prefetch_factor=2,
    )

    model = EBM_TinyQuadJEPA().to(device)
    
    if hasattr(torch, 'compile'):
        print("⚙️ Compiling model into optimized GPU assembly (this will hang for ~1 minute on startup)...")
        model = torch.compile(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = GradScaler('cuda')
    
    os.makedirs("jepa_checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, total=total_batches, desc=f"Epoch {epoch+1}/{epochs}")
        
        total_epoch_loss = 0
        
        for batch_idx, (vis, prop, cmds, dones) in enumerate(pbar):
            # 🔥 Late GPU Casting: Convert uint8 to float32 only once it hits the VRAM
            vis = vis.to(device, non_blocking=True).float() / 255.0
            prop = prop.to(device, non_blocking=True)
            cmds = cmds.to(device, non_blocking=True)
            
            h_t = torch.zeros(global_batch_size, 256, device=device)
            seq_loss = 0
            sim_avg, var_avg, cov_avg = 0, 0, 0

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.bfloat16):
                for t in range(15):
                    loss, h_t, metrics = model.forward_step(
                        vis[:, t], prop[:, t], cmds[:, t],
                        vis[:, t+1], prop[:, t+1], h_t
                    )
                    
                    reset_mask = dones[:, t+1].float().to(device)
                    loss = loss * (1.0 - reset_mask).mean()
                    
                    seq_loss += loss
                    sim_avg += metrics[0]; var_avg += metrics[1]; cov_avg += metrics[2]

                seq_loss = seq_loss / 15
            
            scaler.scale(seq_loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_epoch_loss += seq_loss.item()
            
            if batch_idx % 5 == 0:
                pbar.set_postfix({
                    "Energy": f"{seq_loss.item():.2f}",
                    "Sim": f"{sim_avg/15:.3f}",
                    "Var": f"{var_avg/15:.3f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.1e}"
                })

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
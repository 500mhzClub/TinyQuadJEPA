#!/usr/bin/env python3
"""
Train a post-hoc scalar energy head on top of an existing TinyQuadJEPA checkpoint.

The backbone encoder + predictor are loaded from an existing JEPA checkpoint and
frozen by default. The script rolls the predictor forward for H steps under the
recorded command sequence, encodes the true future latent at step H, and trains
an energy network E(z_pred_H, z_goal_H) to give lower energy to matched pairs
than to shuffled in-batch negatives.

Example:
    python JEPA/train_energy_head.py \
        --jepa_ckpt jepa_checkpoints/jepa_epoch_20.pt \
        --data_dir jepa_final_dataset \
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


class StreamingSequenceDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        envs_per_chunk: int = 64,
        global_batch_size: int = 512,
        split: str = "train",
        val_frac: float = 0.10,
        split_seed: int = 42,
        require_no_done: bool = True,
    ) -> None:
        super().__init__()
        assert split in {"train", "val"}
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.envs_per_chunk = envs_per_chunk
        self.global_batch_size = global_batch_size
        self.split = split
        self.val_frac = val_frac
        self.split_seed = split_seed
        self.require_no_done = require_no_done

        self.h5_files = sorted(glob.glob(os.path.join(data_dir, "*_rgb.h5")))
        if not self.h5_files:
            raise FileNotFoundError(f"No HDF5 datasets found in {data_dir}")

        print(f"🔍 Indexing environments from {data_dir}...")
        all_envs: List[Tuple[str, int]] = []
        for fpath in self.h5_files:
            try:
                with h5py.File(fpath, "r") as h5f:
                    n_envs = h5f["vision"].shape[0]
                    for env_idx in range(n_envs):
                        all_envs.append((fpath, env_idx))
            except Exception as exc:
                print(f"⚠️  Skipping {fpath}: {exc}")

        if not all_envs:
            raise RuntimeError("No environments were indexed from the dataset")

        rng = random.Random(split_seed)
        rng.shuffle(all_envs)
        n_val = max(1, int(len(all_envs) * val_frac))
        self.val_envs = all_envs[:n_val]
        self.train_envs = all_envs[n_val:]
        self.selected_envs = self.train_envs if split == "train" else self.val_envs

        print(
            f"✅ Indexed {len(all_envs):,} envs | "
            f"train={len(self.train_envs):,} | val={len(self.val_envs):,}"
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_envs = list(self.selected_envs)
            seed = self.split_seed + (1000 if self.split == "val" else 0)
        else:
            my_envs = list(self.selected_envs[worker_info.id :: worker_info.num_workers])
            seed = self.split_seed + worker_info.id + (1000 if self.split == "val" else 0)

        rng = random.Random(seed)
        rng.shuffle(my_envs)

        batch_v: List[torch.Tensor] = []
        batch_p: List[torch.Tensor] = []
        batch_c: List[torch.Tensor] = []
        batch_d: List[torch.Tensor] = []

        for i in range(0, len(my_envs), self.envs_per_chunk):
            chunk_envs = my_envs[i : i + self.envs_per_chunk]
            file_groups: Dict[str, List[int]] = defaultdict(list)
            for fpath, env_idx in chunk_envs:
                file_groups[fpath].append(env_idx)

            vision_list: List[torch.Tensor] = []
            prop_list: List[torch.Tensor] = []
            cmd_list: List[torch.Tensor] = []
            done_list: List[torch.Tensor] = []

            for fpath, indices in file_groups.items():
                indices.sort()
                with h5py.File(fpath, "r") as h5f:
                    v = h5f["vision"][indices]
                    p = h5f["proprio"][indices]
                    c = h5f["cmds"][indices]
                    d = h5f["dones"][indices]
                    for local_idx in range(len(indices)):
                        vision_list.append(torch.from_numpy(v[local_idx]))
                        prop_list.append(torch.from_numpy(p[local_idx]).float())
                        cmd_list.append(torch.from_numpy(c[local_idx]).float())
                        done_list.append(torch.from_numpy(d[local_idx]).bool())

            if not vision_list:
                continue

            sequences: List[Tuple[int, int]] = []
            num_envs_loaded = len(vision_list)
            t_total = int(vision_list[0].shape[0])
            max_start_t = t_total - self.seq_len

            for e_idx in range(num_envs_loaded):
                for start_t in range(max_start_t + 1):
                    end_t = start_t + self.seq_len
                    if self.require_no_done and bool(done_list[e_idx][start_t + 1 : end_t].any().item()):
                        continue
                    sequences.append((e_idx, start_t))

            rng.shuffle(sequences)

            for e_idx, start_t in sequences:
                end_t = start_t + self.seq_len
                batch_v.append(vision_list[e_idx][start_t:end_t])
                batch_p.append(prop_list[e_idx][start_t:end_t])
                batch_c.append(cmd_list[e_idx][start_t:end_t])
                batch_d.append(done_list[e_idx][start_t:end_t])

                if len(batch_v) == self.global_batch_size:
                    yield (
                        torch.stack(batch_v),
                        torch.stack(batch_p),
                        torch.stack(batch_c),
                        torch.stack(batch_d),
                    )
                    batch_v, batch_p, batch_c, batch_d = [], [], [], []

            del vision_list, prop_list, cmd_list, done_list
            gc.collect()


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
            nn.LayerNorm(feature_dim),
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
            nn.LayerNorm(feature_dim),
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
            nn.ELU(),
        )
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, c_t: torch.Tensor, h_t: torch.Tensor):
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
        in_dim = latent_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_pred, z_goal, z_pred - z_goal, z_pred * z_goal], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass
class StepStats:
    loss: float
    pos_energy: float
    neg_energy: float
    gap: float
    ranking_acc: float


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_backbone_checkpoint(path: str, device: torch.device) -> Tuple[TinyQuadJEPA, Dict]:
    ckpt = torch.load(path, map_location=device)
    model = TinyQuadJEPA().to(device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    return model, ckpt


def rollout_terminal_latent(
    model: TinyQuadJEPA,
    vis: torch.Tensor,
    prop: torch.Tensor,
    cmds: torch.Tensor,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_roll = model.encoder(vis[:, 0], prop[:, 0])
    h_t = torch.zeros((z_roll.shape[0], model.latent_dim), device=z_roll.device, dtype=z_roll.dtype)
    for t in range(horizon):
        z_roll, h_t = model.predictor(z_roll, cmds[:, t], h_t)
    z_goal = model.encoder(vis[:, horizon], prop[:, horizon])
    return z_roll, z_goal


def sample_negative_goals(z_goal: torch.Tensor, num_negatives: int) -> torch.Tensor:
    """Pure PyTorch vectorized negative sampling to avoid CPU-GPU syncs."""
    bsz = z_goal.shape[0]
    if bsz <= 1:
        return z_goal.unsqueeze(1).expand(-1, num_negatives, -1)
    
    # Generate random offsets in [1, bsz-1] to strictly avoid positive pairs (index i == i)
    offsets = torch.randint(1, bsz, (bsz, num_negatives), device=z_goal.device)
    base_idx = torch.arange(bsz, device=z_goal.device).unsqueeze(1)
    
    # Add offset and wrap around batch size
    neg_idx = (base_idx + offsets) % bsz
    return z_goal[neg_idx]  # [B, K, D]


def energy_ranking_loss(
    head: GoalEnergyHead,
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
    z_neg: torch.Tensor,
    margin: float,
    reg_weight: float,
) -> Tuple[torch.Tensor, StepStats]:
    bsz, k_neg, dim = z_neg.shape
    pos_energy = head(z_pred, z_goal)  # [B]
    z_pred_rep = z_pred[:, None, :].expand(-1, k_neg, -1).reshape(bsz * k_neg, dim)
    z_neg_flat = z_neg.reshape(bsz * k_neg, dim)
    neg_energy = head(z_pred_rep, z_neg_flat).view(bsz, k_neg)

    rank_loss = F.softplus((pos_energy[:, None] - neg_energy) + margin).mean()
    reg_loss = reg_weight * (pos_energy.square().mean() + neg_energy.square().mean())
    loss = rank_loss + reg_loss

    stats = StepStats(
        loss=float(loss.detach().item()),
        pos_energy=float(pos_energy.detach().mean().item()),
        neg_energy=float(neg_energy.detach().mean().item()),
        gap=float((neg_energy.detach().mean() - pos_energy.detach().mean()).item()),
        ranking_acc=float((pos_energy[:, None] < neg_energy).float().mean().detach().item()),
    )
    return loss, stats


@torch.no_grad()
def run_validation(
    backbone: TinyQuadJEPA,
    head: GoalEnergyHead,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
    num_negatives: int,
    margin: float,
    reg_weight: float,
    amp_enabled: bool,
    max_batches: int,
) -> StepStats:
    head.eval()
    
    # Prevent eval mode leak if freeze_backbone was set to False
    backbone_was_training = backbone.training
    backbone.eval()
    
    losses: List[float] = []
    pos_vals: List[float] = []
    neg_vals: List[float] = []
    gaps: List[float] = []
    accs: List[float] = []

    it = iter(loader)
    for _ in range(max_batches):
        try:
            vis, prop, cmds, _ = next(it)
        except StopIteration:
            break

        vis = vis.to(device, non_blocking=True).float().div_(255.0)
        prop = prop.to(device, non_blocking=True)
        cmds = cmds.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            z_pred, z_goal = rollout_terminal_latent(backbone, vis, prop, cmds, horizon)
            z_neg = sample_negative_goals(z_goal, num_negatives)
            _, stats = energy_ranking_loss(head, z_pred, z_goal, z_neg, margin, reg_weight)

        losses.append(stats.loss)
        pos_vals.append(stats.pos_energy)
        neg_vals.append(stats.neg_energy)
        gaps.append(stats.gap)
        accs.append(stats.ranking_acc)
        
    if backbone_was_training:
        backbone.train()

    if not losses:
        return StepStats(math.nan, math.nan, math.nan, math.nan, math.nan)

    return StepStats(
        loss=float(sum(losses) / len(losses)),
        pos_energy=float(sum(pos_vals) / len(pos_vals)),
        neg_energy=float(sum(neg_vals) / len(neg_vals)),
        gap=float(sum(gaps) / len(gaps)),
        ranking_acc=float(sum(accs) / len(accs)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a post-hoc scalar energy head on an existing TinyQuadJEPA checkpoint.")
    parser.add_argument("--jepa_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="jepa_final_dataset")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--envs_per_chunk", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--reg_weight", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--max_batches_per_epoch", type=int, default=0, help="0 = consume the full iterable epoch")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="energy_head_checkpoints")
    parser.add_argument("--logdir", type=str, default="energy_head_logs")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    seq_len = args.horizon + 1

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    print(f"🚀 Loading JEPA backbone from {args.jepa_ckpt} on {device}...")
    backbone, raw_ckpt = load_backbone_checkpoint(args.jepa_ckpt, device)
    if args.freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
    else:
        backbone.train()

    head = GoalEnergyHead(args.latent_dim, args.hidden_dim, args.dropout).to(device)

    trainable_params = list(head.parameters())
    if not args.freeze_backbone:
        trainable_params.extend([p for p in backbone.parameters() if p.requires_grad])

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(device.type, enabled=amp_enabled)

    train_dataset = StreamingSequenceDataset(
        data_dir=args.data_dir,
        seq_len=seq_len,
        envs_per_chunk=args.envs_per_chunk,
        global_batch_size=args.batch_size,
        split="train",
        val_frac=args.val_frac,
        split_seed=args.seed,
        require_no_done=True,
    )
    val_dataset = StreamingSequenceDataset(
        data_dir=args.data_dir,
        seq_len=seq_len,
        envs_per_chunk=args.envs_per_chunk,
        global_batch_size=args.batch_size,
        split="val",
        val_frac=args.val_frac,
        split_seed=args.seed,
        require_no_done=True,
    )

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(**train_loader_kwargs)

    val_workers = max(1, args.num_workers // 2)
    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=None,
        num_workers=val_workers,
        pin_memory=(device.type == "cuda"),
    )
    if val_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2
    val_loader = DataLoader(**val_loader_kwargs)

    csv_path = os.path.join(args.logdir, "energy_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "batch", "global_step", "split", "loss",
                "pos_energy", "neg_energy", "gap", "ranking_acc", "lr"
            ])

    best_val = float("inf")
    global_step = 0
    backbone_epoch = int(raw_ckpt.get("epoch", -1)) + 1 if isinstance(raw_ckpt, dict) else None
    backbone_batch = raw_ckpt.get("batch_idx") if isinstance(raw_ckpt, dict) else None

    for epoch in range(1, args.epochs + 1):
        head.train()
        backbone.eval() if args.freeze_backbone else backbone.train()

        train_losses: List[float] = []
        train_gaps: List[float] = []
        train_accs: List[float] = []

        pbar = tqdm(train_loader, desc=f"EnergyHead Epoch {epoch}/{args.epochs}")
        for batch_idx, (vis, prop, cmds, _) in enumerate(pbar, start=1):
            if args.max_batches_per_epoch > 0 and batch_idx > args.max_batches_per_epoch:
                break

            vis = vis.to(device, non_blocking=True).float().div_(255.0)
            prop = prop.to(device, non_blocking=True)
            cmds = cmds.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if args.freeze_backbone:
                with torch.no_grad():
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                        z_pred, z_goal = rollout_terminal_latent(backbone, vis, prop, cmds, args.horizon)
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    z_neg = sample_negative_goals(z_goal, args.num_negatives)
                    loss, stats = energy_ranking_loss(head, z_pred, z_goal, z_neg, args.margin, args.reg_weight)
            else:
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    z_pred, z_goal = rollout_terminal_latent(backbone, vis, prop, cmds, args.horizon)
                    z_neg = sample_negative_goals(z_goal.detach(), args.num_negatives)
                    loss, stats = energy_ranking_loss(head, z_pred, z_goal, z_neg, args.margin, args.reg_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(stats.loss)
            train_gaps.append(stats.gap)
            train_accs.append(stats.ranking_acc)
            global_step += 1
            lr = optimizer.param_groups[0]["lr"]

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, batch_idx, global_step, "train", stats.loss,
                    stats.pos_energy, stats.neg_energy, stats.gap,
                    stats.ranking_acc, lr
                ])

            if batch_idx % args.log_every == 0:
                n = min(args.log_every, len(train_losses))
                pbar.set_postfix({
                    "loss": f"{sum(train_losses[-n:]) / n:.4f}",
                    "gap": f"{sum(train_gaps[-n:]) / n:.4f}",
                    "acc": f"{sum(train_accs[-n:]) / n:.3f}",
                    "lr": f"{lr:.2e}",
                })

            if args.save_every > 0 and global_step % args.save_every == 0:
                step_ckpt = {
                    "energy_head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "train_loss": stats.loss,
                    "jepa_ckpt_path": args.jepa_ckpt,
                    "backbone_epoch": backbone_epoch,
                    "backbone_batch_idx": backbone_batch,
                    "config": vars(args),
                }
                torch.save(step_ckpt, os.path.join(args.outdir, f"energy_head_step_{global_step}.pt"))

        scheduler.step()

        val_stats = run_validation(
            backbone=backbone,
            head=head,
            loader=val_loader,
            device=device,
            horizon=args.horizon,
            num_negatives=args.num_negatives,
            margin=args.margin,
            reg_weight=args.reg_weight,
            amp_enabled=amp_enabled,
            max_batches=args.val_batches,
        )

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 0, global_step, "val", val_stats.loss,
                val_stats.pos_energy, val_stats.neg_energy, val_stats.gap,
                val_stats.ranking_acc, optimizer.param_groups[0]["lr"]
            ])

        mean_train_loss = sum(train_losses) / max(1, len(train_losses))
        print(
            f"\n📊 Epoch {epoch} complete | "
            f"train_loss={mean_train_loss:.4f} | "
            f"val_loss={val_stats.loss:.4f} | "
            f"val_gap={val_stats.gap:.4f} | "
            f"val_acc={val_stats.ranking_acc:.3f}"
        )

        ckpt = {
            "energy_head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_stats.loss,
            "val_gap": val_stats.gap,
            "val_ranking_acc": val_stats.ranking_acc,
            "jepa_ckpt_path": args.jepa_ckpt,
            "backbone_epoch": backbone_epoch,
            "backbone_batch_idx": backbone_batch,
            "config": vars(args),
        }
        torch.save(ckpt, os.path.join(args.outdir, f"energy_head_epoch_{epoch}.pt"))
        torch.save(ckpt, os.path.join(args.outdir, "energy_head_last.pt"))

        if math.isfinite(val_stats.loss) and val_stats.loss < best_val:
            best_val = val_stats.loss
            torch.save(ckpt, os.path.join(args.outdir, "energy_head_best.pt"))
            print(f"💾 New best checkpoint saved (val_loss={best_val:.4f})")

    print("\n✅ Energy head training complete.")
    print(f"   Best checkpoint : {os.path.join(args.outdir, 'energy_head_best.pt')}")
    print(f"   Last checkpoint : {os.path.join(args.outdir, 'energy_head_last.pt')}")
    print(f"   CSV log         : {csv_path}")


if __name__ == "__main__":
    main()
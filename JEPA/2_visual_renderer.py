#!/usr/bin/env python3
"""
System 2 JEPA Multiprocessed Visual S2W Renderer.
Spawns parallel Vulkan workers to completely saturate GPU VRAM and CPU cores.

Usage:
    python JEPA/2_visual_renderer.py --workers 8
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import h5py
from tqdm import tqdm
import math

# -----------------------------
# Math Helpers
# -----------------------------
def world_to_body_vec(quat_wxyz: torch.Tensor, vec_world: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((vec_world.shape[0], 1), device=vec_world.device, dtype=vec_world.dtype)
    vq = torch.cat([zeros, vec_world], dim=-1)
    q_conj = torch.stack([quat_wxyz[:, 0], -quat_wxyz[:, 1], -quat_wxyz[:, 2], -quat_wxyz[:, 3]], dim=-1)
    
    def quat_mul(a, b):
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        w = aw * bw - ax * bx - ay * by - az * bz
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack([w, x, y, z], dim=-1)
        
    return quat_mul(quat_mul(q_conj, vq), quat_wxyz)[:, 1:4]

def get_backend_str() -> str:
    return os.getenv("GS_BACKEND", "vulkan").lower()

# -----------------------------
# Independent Vulkan Worker
# -----------------------------
def render_worker(args):
    """Runs a completely isolated Genesis instance to render a subset of environments."""
    worker_id, chunk_file, start_env, end_env, tmp_file, backend_str = args
    
    import genesis as gs
    
    # 💥 CRITICAL FIX: Map the string back to the actual Genesis object INSIDE the worker
    if backend_str == "vulkan":
        backend_obj = gs.vulkan
    elif backend_str in ("amdgpu", "amd", "hip") and hasattr(gs, "amdgpu"):
        backend_obj = gs.amdgpu
    else:
        backend_obj = gs.gpu
        
    # Initialize isolated Vulkan context
    gs.init(backend=backend_obj, logging_level="warning")
    
    scene = gs.Scene(show_viewer=False)
    plane = scene.add_entity(gs.morphs.Plane())
    
    urdf = "assets/mini_pupper/mini_pupper.urdf"
    robot = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=False, merge_fixed_links=False))
    
    cam = scene.add_camera(res=(64, 64), fov=90, GUI=False)
    scene.build(n_envs=1)
    
    JOINTS_ACTUATED = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]
    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)

    cam_link_idx = [l.idx for l in robot.links if l.name == "camera_link"][0]
    
    # Load trajectory math
    data = np.load(chunk_file)
    T = data["base_pos"].shape[1]
    N_subset = end_env - start_env
    
    with h5py.File(tmp_file, 'w') as f:
        # Create an expanding HDF5 dataset so we use almost 0 System RAM
        h5_rgb = f.create_dataset('vision', (N_subset, T, 3, 64, 64), dtype='uint8', compression="gzip")
        
        for local_idx, env_idx in enumerate(range(start_env, end_env)):
            base_pos_seq  = torch.tensor(data["base_pos"][env_idx], device=gs.device)
            base_quat_seq = torch.tensor(data["base_quat"][env_idx], device=gs.device)
            joint_pos_seq = torch.tensor(data["joint_pos"][env_idx], device=gs.device)
            
            env_video = np.zeros((T, 3, 64, 64), dtype=np.uint8)
            
            for step in range(T):
                robot.set_pos(base_pos_seq[step].unsqueeze(0))
                robot.set_quat(base_quat_seq[step].unsqueeze(0))
                robot.set_dofs_position(joint_pos_seq[step].unsqueeze(0), act_dofs)
                
                scene.step(update_visualizer=False) 
                
                cam_pos = robot.get_links_pos()[0, cam_link_idx]
                cam_quat = robot.get_links_quat()[0, cam_link_idx]
                
                fwd_vec = world_to_body_vec(cam_quat.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=gs.device))[0]
                cam.set_pose(pos=cam_pos.cpu().numpy(), lookat=(cam_pos + fwd_vec).cpu().numpy())

                render_out = cam.render(rgb=True)
                rgb = render_out[0]
                if hasattr(rgb, "cpu"): 
                    rgb = rgb.cpu().numpy()
                rgb = rgb.astype(np.float32) / 255.0
                
                # POST-RENDER VDR (Sim-to-World augmentation)
                brightness = np.random.uniform(-0.4, 0.4)
                contrast = np.random.uniform(0.5, 1.5)
                noise = np.random.normal(0, 0.05, rgb.shape)
                
                rgb_vdr = (rgb * contrast) + brightness + noise
                rgb_vdr = np.clip(rgb_vdr * 255.0, 0, 255).astype(np.uint8)
                
                env_video[step] = np.transpose(rgb_vdr, (2, 0, 1))
                
            # Stream video to disk immediately
            h5_rgb[local_idx] = env_video
            
    return tmp_file

# -----------------------------
# Main Orchestrator
# -----------------------------
def main():
    # REQUIRED for spawning independent Vulkan processes
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing chunks")
    parser.add_argument("--workers", type=int, default=8, help="Parallel Genesis instances.")
    args = parser.parse_args()

    raw_files = sorted(glob.glob("jepa_raw_data/chunk_*.npz"))
    if not raw_files:
        print("❌ No raw data found. Run 1_physics_rollout.py first!")
        return

    os.makedirs("jepa_final_dataset", exist_ok=True)
    backend_str = get_backend_str()

    for file_path in raw_files:
        chunk_name = os.path.basename(file_path).split('.')[0]
        out_path = f"jepa_final_dataset/{chunk_name}_rgb.h5"
        
        if os.path.exists(out_path) and not args.force:
            try:
                with h5py.File(out_path, 'r') as h5f:
                    if 'vision' not in h5f: raise ValueError()
                print(f"⏭️ Skipping {chunk_name}, completely exists at {out_path}")
                continue
            except Exception:
                print(f"⚠️ {out_path} corrupted. Overwriting...")
            
        print(f"\n🎨 Rendering {chunk_name} -> {out_path}")
        
        data = np.load(file_path)
        N, T = data["base_pos"].shape[:2]
        
        envs_per_worker = math.ceil(N / args.workers)
        tasks = []
        tmp_files = []
        
        for i in range(args.workers):
            start = i * envs_per_worker
            end = min(start + envs_per_worker, N)
            if start >= end: 
                break
            
            tmp = f"jepa_final_dataset/{chunk_name}_tmp_{i}.h5"
            tmp_files.append(tmp)
            tasks.append((i, file_path, start, end, tmp, backend_str))
            
        print(f"🔥 Spawning {len(tasks)} isolated Vulkan engines...")
        
        with mp.Pool(len(tasks)) as pool:
            list(tqdm(pool.imap_unordered(render_worker, tasks), total=len(tasks), desc="Vulkan Workers Completed"))
            
        print(f"🧵 Stitching temporary chunks into final HDF5...")
        with h5py.File(out_path, 'w') as h5f:
            h5_rgb = h5f.create_dataset("vision", (N, T, 3, 64, 64), dtype='uint8', chunks=(1, T, 3, 64, 64), compression="gzip")
            
            # Use data arrays directly from memory to avoid re-reading
            h5f.create_dataset("proprio", data=data["proprio"], compression="gzip")
            h5f.create_dataset("cmds", data=data["cmds"], compression="gzip")
            h5f.create_dataset("dones", data=data["dones"], compression="gzip")
            
            for i, (tmp_file, task) in enumerate(zip(tmp_files, tasks)):
                start, end = task[2], task[3]
                try:
                    with h5py.File(tmp_file, 'r') as tmp_in:
                        h5_rgb[start:end] = tmp_in['vision'][:]
                    os.remove(tmp_file) 
                except Exception as e:
                    print(f"❌ Failed to stitch {tmp_file}: {e}")
                
        print(f"✅ Chunk {chunk_name} completely saved and compressed!")

if __name__ == "__main__":
    main()
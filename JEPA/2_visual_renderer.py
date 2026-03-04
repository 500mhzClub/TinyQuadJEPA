#!/usr/bin/env python3
"""
System 2 JEPA Multiprocessed Visual S2W Renderer.
Fault-tolerant version with optical flow checkerboard and fixed camera tracking.

Usage:
    python JEPA/2_visual_renderer.py --workers 4
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
from PIL import Image

# -----------------------------
# Math Helpers
# -----------------------------
def body_to_world_vec(quat_wxyz: torch.Tensor, vec_body: torch.Tensor) -> torch.Tensor:
    """Correctly rotates a body-frame vector into the world frame."""
    zeros = torch.zeros((vec_body.shape[0], 1), device=vec_body.device, dtype=vec_body.dtype)
    vq = torch.cat([zeros, vec_body], dim=-1)
    
    # Conjugate quaternion
    q_conj = torch.stack([quat_wxyz[:, 0], -quat_wxyz[:, 1], -quat_wxyz[:, 2], -quat_wxyz[:, 3]], dim=-1)
    
    def quat_mul(a, b):
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        w = aw * bw - ax * bx - ay * by - az * bz
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack([w, x, y, z], dim=-1)
        
    # q * v * q_conj
    return quat_mul(quat_mul(quat_wxyz, vq), q_conj)[:, 1:4]

def get_backend_str() -> str:
    return os.getenv("GS_BACKEND", "vulkan").lower()

# -----------------------------
# Independent Vulkan Worker
# -----------------------------
def render_worker(args):
    worker_id, chunk_file, start_env, end_env, tmp_file, backend_str, tex_path = args
    N_subset = end_env - start_env
    
    if os.path.exists(tmp_file):
        try:
            with h5py.File(tmp_file, 'r') as f:
                if 'vision' in f and f['vision'].shape[0] == N_subset:
                    return tmp_file  
        except Exception:
            pass 

    import genesis as gs
    
    if backend_str == "vulkan":
        backend_obj = gs.vulkan
    elif backend_str in ("amdgpu", "amd", "hip") and hasattr(gs, "amdgpu"):
        backend_obj = gs.amdgpu
    else:
        backend_obj = gs.gpu
        
    gs.init(backend=backend_obj, logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    
    # Optical Flow Checkerboard
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=tex_path)
        )
    )
    
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

    data = np.load(chunk_file)
    T = data["base_pos"].shape[1]
    
    # Fixed camera offset: 10cm forward, 5cm up from the absolute center of the robot
    local_cam_offset = torch.tensor([[0.10, 0.0, 0.05]], device=gs.device)
    local_fwd = torch.tensor([[1.0, 0.0, 0.0]], device=gs.device)
    local_up = torch.tensor([[0.0, 0.0, 1.0]], device=gs.device)
    
    with h5py.File(tmp_file, 'w') as f:
        h5_rgb = f.create_dataset('vision', (N_subset, T, 3, 64, 64), dtype='uint8', compression="gzip")
        worker_pbar = tqdm(range(start_env, end_env), desc=f"Worker {worker_id}", position=worker_id, leave=False)
        
        for local_idx, env_idx in enumerate(worker_pbar):
            base_pos_seq  = torch.tensor(data["base_pos"][env_idx], device=gs.device)
            base_quat_seq = torch.tensor(data["base_quat"][env_idx], device=gs.device)
            joint_pos_seq = torch.tensor(data["joint_pos"][env_idx], device=gs.device)
            
            env_video = np.zeros((T, 3, 64, 64), dtype=np.uint8)
            
            for step in range(T):
                base_pos = base_pos_seq[step].unsqueeze(0)
                base_quat = base_quat_seq[step].unsqueeze(0)
                
                robot.set_pos(base_pos)
                robot.set_quat(base_quat)
                robot.set_dofs_position(joint_pos_seq[step].unsqueeze(0), act_dofs)
                
                scene.step(update_visualizer=False) 
                
                # Mathematically staple the camera to the base_pos trajectory
                world_offset = body_to_world_vec(base_quat, local_cam_offset)[0]
                cam_pos = base_pos[0] + world_offset
                
                fwd_vec = body_to_world_vec(base_quat, local_fwd)[0]
                up_vec = body_to_world_vec(base_quat, local_up)[0]
                
                # Explicit up vector prevents upside-down gimbal lock
                cam.set_pose(
                    pos=cam_pos.cpu().numpy(), 
                    lookat=(cam_pos + fwd_vec).cpu().numpy(),
                    up=up_vec.cpu().numpy()
                )

                render_out = cam.render(rgb=True)
                rgb = render_out[0]
                if hasattr(rgb, "cpu"): 
                    rgb = rgb.cpu().numpy()
                rgb = rgb.astype(np.float32) / 255.0
                
                brightness = np.random.uniform(-0.4, 0.4)
                contrast = np.random.uniform(0.5, 1.5)
                noise = np.random.normal(0, 0.05, rgb.shape)
                
                rgb_vdr = (rgb * contrast) + brightness + noise
                rgb_vdr = np.clip(rgb_vdr * 255.0, 0, 255).astype(np.uint8)
                
                env_video[step] = np.transpose(rgb_vdr, (2, 0, 1))
                
            h5_rgb[local_idx] = env_video
            
    return tmp_file

# -----------------------------
# Main Orchestrator
# -----------------------------
def main():
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing chunks")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Genesis instances.")
    args = parser.parse_args()

    raw_files = sorted(glob.glob("jepa_raw_data/chunk_*.npz"))
    if not raw_files:
        print("❌ No raw data found. Run 1_physics_rollout.py first!")
        return

    os.makedirs("jepa_final_dataset", exist_ok=True)
    backend_str = get_backend_str()
    
    tex_path = os.path.abspath("checkerboard.png")
    if not os.path.exists(tex_path):
        print("🎨 Generating optical flow checkerboard texture...")
        checker = np.indices((32, 32)).sum(axis=0) % 2
        checker = np.repeat(np.repeat(checker, 32, axis=0), 32, axis=1) # 1024x1024
        checker = (checker * 255).astype(np.uint8)
        Image.fromarray(checker).save(tex_path)

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
            tasks.append((i, file_path, start, end, tmp, backend_str, tex_path))
            
        print(f"🔥 Spawning {len(tasks)} isolated Vulkan engines...")
        
        with mp.Pool(len(tasks)) as pool:
            list(tqdm(pool.imap_unordered(render_worker, tasks), total=len(tasks), desc="Total Workers", position=args.workers+1))
            
        print(f"\n🧵 Stitching temporary chunks into final HDF5...")
        with h5py.File(out_path, 'w') as h5f:
            h5_rgb = h5f.create_dataset("vision", (N, T, 3, 64, 64), dtype='uint8', chunks=(1, T, 3, 64, 64), compression="gzip")
            
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
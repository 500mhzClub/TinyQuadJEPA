#!/usr/bin/env python3
"""
System 2 JEPA Dataset Visualizer.
Randomly samples trajectories from the HDF5 dataset and exports them as GIFs
to verify rendering, lighting randomization, and tensor alignment.

Usage:
    python JEPA/3_verify_dataset.py
"""
import os
import glob
import h5py
import numpy as np
import imageio
import random

def main():
    os.makedirs("verification_videos", exist_ok=True)
    h5_files = sorted(glob.glob("jepa_final_dataset/*_rgb.h5"))
    
    if not h5_files:
        print("❌ No HDF5 files found in jepa_final_dataset/")
        return

    num_spot_checks = 5
    print(f"🔍 Generating {num_spot_checks} random S2W spot-check videos...\n")

    for i in range(num_spot_checks):
        # Pick a random chunk and open it
        target_file = random.choice(h5_files)
        chunk_name = os.path.basename(target_file).split('.')[0]
        
        with h5py.File(target_file, 'r') as h5f:
            N, T = h5f['vision'].shape[:2]
            
            # Pick a random robot environment
            env_idx = random.randint(0, N - 1)
            
            # Extract the raw PyTorch-formatted data
            # Vision shape is (T, 3, 64, 64)
            vision_data = h5f['vision'][env_idx] 
            cmds_data = h5f['cmds'][env_idx]
            
            # Convert from PyTorch format (Time, Channels, Height, Width) 
            # to standard Video format (Time, Height, Width, Channels)
            video_frames = np.transpose(vision_data, (0, 2, 3, 1))
            
            # Grab the first 150 frames (5 seconds at 30fps) for a quick preview
            clip_frames = video_frames[:150]
            
            # Get the average command [vx, vy, omega] over this clip
            avg_cmd = np.mean(cmds_data[:150], axis=0) 
            
            out_path = f"verification_videos/spotcheck_{i+1}_{chunk_name}_env{env_idx}.gif"
            
            # Save the sequence as an animated GIF
            imageio.mimsave(out_path, clip_frames, fps=30)
            
            print(f"✅ Saved {out_path}")
            print(f"   ↳ Avg Command: Forward(vx)={avg_cmd[0]:.2f}, Sideways(vy)={avg_cmd[1]:.2f}, Turn(w)={avg_cmd[2]:.2f}\n")

if __name__ == "__main__":
    main()
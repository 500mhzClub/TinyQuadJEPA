import os
import struct
from pathlib import Path
import numpy as np # Genesis env has numpy

def fix_stl_files():
    src_dir = Path("assets/mini_pupper/meshes")
    dst_dir = Path("assets/mini_pupper/meshes_fixed")
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # The Magic Offset we found from the "Sky Drop" test
    OFFSET_Z = 1.822 

    print(f"🛠️  Fixing meshes in {src_dir}...")
    print(f"    Applying Z-Offset: +{OFFSET_Z}m")

    for stl_file in src_dir.glob("*.stl"):
        print(f"    Processing {stl_file.name}...", end="")
        
        try:
            with open(stl_file, "rb") as f:
                header = f.read(80)
                # Check for ASCII (starts with 'solid')
                if header.strip().startswith(b'solid'):
                    print(" [ASCII detected] -> converting...")
                    fix_ascii_stl(stl_file, dst_dir / stl_file.name, OFFSET_Z)
                else:
                    print(" [Binary detected] -> fixing...")
                    fix_binary_stl(stl_file, dst_dir / stl_file.name, OFFSET_Z)
        except Exception as e:
            print(f" ❌ Error: {e}")

    print("✅ All meshes fixed!")

def fix_binary_stl(src, dst, dz):
    with open(src, "rb") as f:
        header = f.read(80)
        count_bytes = f.read(4)
        num_triangles = struct.unpack("<I", count_bytes)[0]
        
        with open(dst, "wb") as out:
            out.write(header)
            out.write(count_bytes)
            
            # Record format: Normal(3f) + V1(3f) + V2(3f) + V3(3f) + Attr(2b) = 50 bytes
            for _ in range(num_triangles):
                data = f.read(50)
                if len(data) < 50: break
                
                # Unpack 12 floats (Normal + 3 Vertices) + 1 uint16
                floats = list(struct.unpack("<12f", data[:48]))
                attr = data[48:]
                
                # Shift Z for Vertex 1, 2, 3 (Indices 5, 8, 11)
                floats[5] += dz
                floats[8] += dz
                floats[11] += dz
                
                out.write(struct.pack("<12f", *floats))
                out.write(attr)

def fix_ascii_stl(src, dst, dz):
    with open(src, "r") as f:
        lines = f.readlines()
        
    with open(dst, "w") as out:
        for line in lines:
            if "vertex" in line:
                parts = line.split()
                # parts: ['vertex', 'x', 'y', 'z']
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                z += dz
                out.write(f"      vertex {x} {y} {z}\n")
            else:
                out.write(line)

if __name__ == "__main__":
    fix_stl_files()
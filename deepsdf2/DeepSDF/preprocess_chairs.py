import os
import glob
import trimesh
import numpy as np
from mesh_to_sdf import sample_sdf_near_surface

def generate_xyz_sdf(filename):
    mesh = trimesh.load(filename, force='mesh')
    xyz, sdf = sample_sdf_near_surface(mesh, number_of_points=15000)
    return xyz, sdf

def write_sdf_to_npz(xyz, sdfs, filename):
    pos, neg = [], []
    for i in range(len(xyz)):
        v, s = xyz[i], sdfs[i]
        (pos if s > 0 else neg).extend([*v, s])
    np.savez(filename, pos=np.array(pos).reshape(-1, 4), neg=np.array(neg).reshape(-1, 4))

def process_mesh(obj_path, target_path):
    xyz, sdf = generate_xyz_sdf(obj_path)
    write_sdf_to_npz(xyz, sdf, target_path)

# === Setup ===
source_root = "dataset/03001627/03001627"
output_root = "./processed_data/chair_train"
os.makedirs(output_root, exist_ok=True)

mesh_paths = sorted(glob.glob(f"{source_root}/*/models/model_normalized.obj"))
print(f"Found {len(mesh_paths)} .obj files")

for idx, mesh_path in enumerate(mesh_paths[:2000]):
    out_path = os.path.join(output_root, f"{idx}.npz")
    
    if os.path.exists(out_path):
        print(f"[{idx+1}/2000] Skipped {mesh_path} (already processed)")
        continue

    process_mesh(mesh_path, out_path)
    print(f"[{idx+1}/2000] Processed {mesh_path}")

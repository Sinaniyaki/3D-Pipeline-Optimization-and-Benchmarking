import os
import time
import torch
import numpy as np
import open3d as o3d
from chamferdist import ChamferDistance
from model.dataset import ShapeNet_Dataset
from model.decoder import Decoder
import model.reconstruct as reconstruct


def load_pointcloud_from_mesh(path, n_points=10000):
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_triangles():
        raise ValueError("Mesh has no triangles")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return torch.tensor(np.asarray(pcd.points)).unsqueeze(0).float().cuda()


def voxelize_mesh(path, voxel_size=0.05):
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_triangles():
        raise ValueError("Mesh has no triangles")
    return o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)


def compute_iou(pred_voxel, gt_voxel):
    pred_voxels = set([tuple(voxel.grid_index) for voxel in pred_voxel.get_voxels()])
    gt_voxels = set([tuple(voxel.grid_index) for voxel in gt_voxel.get_voxels()])
    intersection = len(pred_voxels & gt_voxels)
    union = len(pred_voxels | gt_voxels)
    return intersection / union if union != 0 else 0.0


def compute_chamfer(gt_mesh_path, pred_mesh_path):
    gt = load_pointcloud_from_mesh(gt_mesh_path)
    pred = load_pointcloud_from_mesh(pred_mesh_path)
    cd = ChamferDistance()
    dist = cd(gt, pred)
    return dist.item()


def reconstruct_with_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ShapeNet_Dataset("./processed_data/chair_train/")

    decoder = Decoder().to(device)
    checkpoint = torch.load("./checkpoints/chairs/model_epoch_822.pt", map_location=device)
    decoder.load_state_dict(checkpoint["model"])
    decoder.eval()

    os.makedirs("./reconstruction/chairs", exist_ok=True)

    chamfers, ious, inference_times, memory_usages = [], [], [], []
    successful_cd, successful_iou = 0, 0

    for idx in range(len(dataset)):
        mesh_id, sample = dataset[idx]
        filename = f"./reconstruction/chairs/example_{idx}"
        gt_mesh_path = f"dataset/03001627/03001627/{mesh_id}/models/model_normalized.obj"
        out_mesh_path = filename + ".ply"

        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        reconstruct.reconstruct(sample, decoder, filename,
                                lat_iteration=800, lat_lr=5e-4,
                                N=256, max_batch=32**3)

        elapsed = time.time() - start_time
        inference_times.append(elapsed)
        mem_used = torch.cuda.max_memory_allocated() / 1024**2
        memory_usages.append(mem_used)

        cd_val = None
        try:
            cd_val = compute_chamfer(gt_mesh_path, out_mesh_path)
            successful_cd += 1
        except Exception as e:
            print(f"[{idx}] Chamfer Error: {e}")
        chamfers.append(cd_val)

        iou_val = None
        try:
            pred_vox = voxelize_mesh(out_mesh_path, voxel_size=0.05)
            gt_vox = voxelize_mesh(gt_mesh_path, voxel_size=0.05)
            iou_val = compute_iou(pred_vox, gt_vox)
            successful_iou += 1
        except Exception as e:
            print(f"[{idx}] IoU Error: {e}")
        ious.append(iou_val)

        cd_str = f"{cd_val:.6f}" if cd_val is not None else "N/A"
        iou_str = f"{iou_val:.4f}" if iou_val is not None else "N/A"
        print(f"[{idx}] CD: {cd_str} | IoU: {iou_str} | Time: {elapsed:.2f}s | Mem: {mem_used:.2f}MB")

    valid_chamfers = [cd for cd in chamfers if cd is not None]
    valid_ious = [iou for iou in ious if iou is not None]

    print("\nFinal Benchmark Summary:")
    print(f"Successful Chamfer: {successful_cd}/{len(dataset)}")
    print(f"Successful IoU:     {successful_iou}/{len(dataset)}")
    if valid_chamfers:
        print(f"Avg Chamfer Distance: {np.mean(valid_chamfers):.6f}")
    if valid_ious:
        print(f"Avg IoU:              {np.mean(valid_ious):.4f}")
    print(f"Avg Inference Time:   {np.mean(inference_times):.2f} sec")
    print(f"Avg Peak GPU Mem:     {np.mean(memory_usages):.2f} MB")


if __name__ == "__main__":
    reconstruct_with_metrics()

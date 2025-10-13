import torch
from torch.utils.data import Dataset

import numpy as np
import random
from pathlib import Path

class ShapeNet_Dataset(Dataset):

    def __init__(self, dataset_path):
        self.dataset = []
        self.mesh_ids = []  # NEW: store mesh ID (npz filename without extension)

        for file_path in sorted(Path(dataset_path).glob("*.npz")):
            npz = np.load(file_path)
            pos_tensor = torch.from_numpy(npz["pos"])
            neg_tensor = torch.from_numpy(npz["neg"])

            half = 15000 // 2

            # Sample pos
            if pos_tensor.shape[0] <= half:
                idxs = torch.randint(0, pos_tensor.shape[0], (half,))
                sample_pos = pos_tensor[idxs]
            else:
                start = random.randint(0, pos_tensor.shape[0] - half)
                sample_pos = pos_tensor[start:start + half]

            # Sample neg
            if neg_tensor.shape[0] <= half:
                idxs = torch.randint(0, neg_tensor.shape[0], (half,))
                sample_neg = neg_tensor[idxs]
            else:
                start = random.randint(0, neg_tensor.shape[0] - half)
                sample_neg = neg_tensor[start:start + half]

            sample = torch.cat([sample_pos, sample_neg], dim=0)
            self.dataset.append(sample)

            mesh_id = file_path.stem  # Removes ".npz"
            self.mesh_ids.append(mesh_id)

    def __getitem__(self, index):
        return self.mesh_ids[index], self.dataset[index]  # Return mesh_id

    def __len__(self):
        return len(self.dataset)

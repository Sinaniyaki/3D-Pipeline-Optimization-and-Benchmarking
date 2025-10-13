import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import trange
from torch.utils.data import DataLoader

from model.dataset import ShapeNet_Dataset
from model.decoder import Decoder


def train_decoder(
    epochs=6000,
    batch_size=10,
    latent_size=256,
    lat_vecs_std=0.01,
    decoder_lr=0.0005,
    lat_vecs_lr=0.001,
    train_data_path="./processed_data/train",
    checkpoint_save_path="./checkpoints/"
):
    # ------------ Reproducibility ------------
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # ------------ Load Dataset ------------
    dataset = ShapeNet_Dataset(train_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    num_shapes = len(dataset)
    print(f"📦 Loaded {num_shapes} training shapes.")

    # ------------ Initialize Model and Latents ------------
    model = Decoder(latent_size=latent_size).to(device)
    latent_vectors = torch.nn.Embedding(num_shapes, latent_size, max_norm=1.0).to(device)
    torch.nn.init.normal_(latent_vectors.weight, mean=0.0, std=lat_vecs_std)

    # ------------ Optimizer ------------
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": decoder_lr},
        {"params": latent_vectors.parameters(), "lr": lat_vecs_lr}
    ])

    # ------------ Loss Function ------------
    loss_fn = torch.nn.L1Loss(reduction="sum")
    clamp_min, clamp_max = -0.1, 0.1
    loss_log = []

    # ------------ Training Loop ------------
    print("🚀 Starting training...")
    for epoch in trange(epochs, desc="Training", unit="epoch"):
        model.train()
        epoch_losses = []

        for indices, samples in dataloader:
            samples = samples.reshape(-1, 4).to(device)  # (batch_size * 15000, 4)
            sdf_gt = samples[:, 3].unsqueeze(1).clamp(clamp_min, clamp_max)
            xyz = samples[:, :3]

            indices = indices.to(device).unsqueeze(-1).repeat(1, 15000).view(-1)
            latents = latent_vectors(indices)

            inputs = torch.cat([latents, xyz], dim=1)  # (N, 256+3)

            optimizer.zero_grad()
            sdf_pred = model(inputs).clamp(clamp_min, clamp_max)
            loss = loss_fn(sdf_pred, sdf_gt) / samples.shape[0]
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        loss_log.append(avg_loss)
        if epoch % 100 == 0:
            print(f"📉 Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}")

        # ------------ Save Checkpoint ------------
        if epoch % 500 == 0 or epoch == epochs - 1:
            os.makedirs(checkpoint_save_path, exist_ok=True)
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'latent_vectors': latent_vectors.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_log': loss_log,
            }
            ckpt_path = os.path.join(checkpoint_save_path, f"model_epoch_{epoch+1}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

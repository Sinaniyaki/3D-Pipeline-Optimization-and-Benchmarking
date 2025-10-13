import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import trange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.dataset import ShapeNet_Dataset
from model.decoder import Decoder

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_decoder(
    epochs=822,
    batch_size=10,
    latent_size=256,
    lat_vecs_std=0.01,
    decoder_lr=0.0005,
    lat_vecs_lr=0.001,
    train_data_path="./processed_data/chair_train",
    checkpoint_save_path="./checkpoints/chairs",
    tensorboard_log_dir="./runs/chair"
):
    # ------------ Setup ------------
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(checkpoint_save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # ------------ Load Dataset ------------
    dataset = ShapeNet_Dataset(train_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print(f"Loaded {len(dataset)} training shapes.")

    # ------------ Model & Latents ------------
    model = Decoder(latent_size=latent_size).to(device)
    latent_vectors = torch.nn.Embedding(len(dataset), latent_size, max_norm=1.0).to(device)
    torch.nn.init.normal_(latent_vectors.weight, mean=0.0, std=lat_vecs_std)

    # ------------ Optimizer ------------
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": decoder_lr},
        {"params": latent_vectors.parameters(), "lr": lat_vecs_lr}
    ])

    # ------------ Info Logs ------------
    print(f"Model has {count_params(model):,} parameters")
    torch.save(model.state_dict(), "temp_model.pt")
    print(f"Model file size: {os.path.getsize('temp_model.pt') / 1e6:.2f} MB")
    os.remove("temp_model.pt")

    # ------------ Training Loop ------------
    loss_fn = torch.nn.L1Loss(reduction="sum")
    clamp_min, clamp_max = -0.1, 0.1
    loss_log = []

    print("Starting training...")
    total_start_time = time.time()

    for epoch in trange(epochs, desc="Training", unit="epoch"):
        model.train()
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats()

        epoch_losses = []
        for indices, samples in dataloader:
            samples = samples.reshape(-1, 4).to(device)
            sdf_gt = samples[:, 3].unsqueeze(1).clamp(clamp_min, clamp_max)
            xyz = samples[:, :3]

            indices = indices.to(device).unsqueeze(-1).repeat(1, 15000).view(-1)
            latents = latent_vectors(indices)

            inputs = torch.cat([latents, xyz], dim=1)

            optimizer.zero_grad()
            sdf_pred = model(inputs).clamp(clamp_min, clamp_max)
            loss = loss_fn(sdf_pred, sdf_gt) / samples.shape[0]
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # ------------ Logs ------------
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        loss_log.append(avg_loss)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)
        writer.add_scalar("Memory/peak_MB", peak_mem, epoch)

        print(f"📉 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Time: {epoch_time:.2f}s - Mem: {peak_mem:.2f}MB")

        # ------------ Save Checkpoint More Frequently ------------
        if epoch % 100 == 0 or epoch == epochs - 1:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'latent_vectors': latent_vectors.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_log': loss_log,
            }
            ckpt_path = os.path.join(checkpoint_save_path, f"model_epoch_{epoch+1}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    # ------------ Done ------------
    total_time = time.time() - total_start_time
    print(f"Training complete in {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    writer.close()

import model.train_chair as train

if __name__ == "__main__":
    train.train_decoder(
        train_data_path="./processed_data/chair_train/",
        checkpoint_save_path="./checkpoints/chairs/",
        tensorboard_log_dir="./runs/chair/",
        epochs=822,
        batch_size=10,
        latent_size=256,
        lat_vecs_std=0.01,
        decoder_lr=0.0005,
        lat_vecs_lr=0.001
    )

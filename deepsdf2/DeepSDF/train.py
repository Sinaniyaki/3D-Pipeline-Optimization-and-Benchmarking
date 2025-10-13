import model.train as train

if __name__ == "__main__":
    train.train_decoder(
        epochs=6000,
        batch_size=10,
        latent_size=256,
        lat_vecs_std=0.01,
        decoder_lr=0.0005,
        lat_vecs_lr=0.001,
        train_data_path="./processed_data/train/",
        checkpoint_save_path="./checkpoints/"
    )

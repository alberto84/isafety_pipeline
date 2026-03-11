import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# CONSTANTS
# -------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 96
NUM_EPOCHS = 512
LEARNING_RATE = 1e-5
NUM_EPOCHS_BERT = 10
MAX_TOKENS = 512
LOG_DIR = "log"

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"2_data_reduction_{timestamp}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filepath, mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# -------------------------------
# LinearUNet
# -------------------------------
class LinearUNet_768_512(nn.Module):
    def __init__(self, input_dim=768, latent_dim=512):
        super(LinearUNet_768_512, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(input_dim, 512)

        # Bottleneck
        self.bottleneck = nn.Linear(512, latent_dim)

        # Decoder
        self.dec1 = nn.Linear(latent_dim + 512, input_dim)

        self.activation = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, x):
        # Encoder
        e1 = self.activation(self.bn1(self.enc1(x)))

        # Bottleneck
        z = self.bottleneck(e1)

        # Decoder with skip connections (concat)
        out = self.dec1(torch.cat([z, e1], dim=1))

        return out, z  # out = reconstruction, z = latent


class LinearUNet_768_512_256(nn.Module):
    def __init__(self, input_dim=768, latent_dim=256):
        super(LinearUNet_768_512_256, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(input_dim, 512)
        self.enc2 = nn.Linear(512, 256)

        # Bottleneck
        self.bottleneck = nn.Linear(256, latent_dim)

        # Decoder
        self.dec2 = nn.Linear(latent_dim + 256, 256)
        self.dec1 = nn.Linear(256 + 512, input_dim)

        self.activation = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        # Encoder
        e1 = self.activation(self.bn1(self.enc1(x)))
        e2 = self.activation(self.bn2(self.enc2(e1)))

        # Bottleneck
        z = self.bottleneck(e2)

        # Decoder with skip connections (concat)
        d2 = self.activation(self.dec2(torch.cat([z, e2], dim=1)))
        out = self.dec1(torch.cat([d2, e1], dim=1))

        return out, z  # out = reconstruction, z = latent


class LinearUNet_768_512_256_128(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128):
        super(LinearUNet_768_512_256_128, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(input_dim, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, 128)

        # Bottleneck
        self.bottleneck = nn.Linear(128, latent_dim)

        # Decoder
        self.dec3 = nn.Linear(latent_dim + 128, 128)
        self.dec2 = nn.Linear(128 + 256, 256)
        self.dec1 = nn.Linear(256 + 512, input_dim)

        self.activation = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Encoder
        e1 = self.activation(self.bn1(self.enc1(x)))
        e2 = self.activation(self.bn2(self.enc2(e1)))
        e3 = self.activation(self.bn3(self.enc3(e2)))

        # Bottleneck
        z = self.bottleneck(e3)

        # Decoder with skip connections (concat)
        d3 = self.activation(self.dec3(torch.cat([z, e3], dim=1)))
        d2 = self.activation(self.dec2(torch.cat([d3, e2], dim=1)))
        out = self.dec1(torch.cat([d2, e1], dim=1))

        return out, z  # out = reconstruction, z = latent


# -------------------------------
# AUTOENCODER TRAINING
# -------------------------------
def train_autoencoder(embeddings,
                      batch_size=32,
                      latent_dim=128,
                      epochs=NUM_EPOCHS,
                      learning_rate=LEARNING_RATE,
                      save_model=False):

    logger.info(f"Training LinearUNet autoencoder with embeddings len={len(embeddings)}, e latent_dim={latent_dim}... ")

    embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(embeddings_tensor), batch_size=batch_size, shuffle=True)

    if latent_dim == 128:
        model = LinearUNet_768_512_256_128()
        logger.info("Using 128-dim latent space.")
    elif latent_dim == 256:
        model = LinearUNet_768_512_256()
        logger.info("Using 256-dim latent space.")
    elif latent_dim == 512:
        model = LinearUNet_768_512()
        logger.info("Using 512-dim latent space.")
    else:
        raise ValueError("Invalid latent_dim value. Must be 128, 256, or 512.")

    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_loss = float('inf')
    best_epoch = 0
    start_time = datetime.now()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            inputs = batch[0].to(DEVICE)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1:03d}/{epochs} - Loss: {avg_loss:.6f}")

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Autoencoder training finished in {duration:.2f}s - Best Epoch {best_epoch}, Best Loss {best_loss:.6f}")

    if save_model:
        os.makedirs("output", exist_ok=True)
        torch.save(model.state_dict(), "output/autoencoder-aria-osha.pt")
        logger.info("Autoencoder model saved: output/autoencoder-aria-osha.pt")

    return model


# -------------------------------
# EMBEDDING REDUCTION
# -------------------------------
def get_reduced_embeddings(model, embeddings):
    logger.info("Reducing embeddings using trained autoencoder...")
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
        _, encoded = model(inputs)
    reduced = encoded.cpu().numpy()
    logger.info(f"Reduced embeddings shape: {reduced.shape}")
    return reduced


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main(latent_dim):
    logger.info("===== STARTING PIPELINE - STEP 2 DATA REDUCTION =====")

    logging.info("\nStage 1: Loading datasets...")
    df_osha = pd.read_parquet('output/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS_embeddings.parquet')

    raw_embeddings = df_osha['embeddings'].tolist()
    autoencoder_model = train_autoencoder(raw_embeddings,
                                          latent_dim=latent_dim,
                                          epochs=NUM_EPOCHS,
                                          save_model=False)

    logging.info(f"\nStage 4: Reducing embeddings using trained autoencoder...")
    reduced = get_reduced_embeddings(autoencoder_model, raw_embeddings)
    df_osha['embeddings_reduced'] = list(reduced)

    logging.info(f"\nStage 5: Saving reduced embeddings to parquet file...")
    output_file = f"output/NEW_OSHA_DATABASE_FILTERED_SIC_NAICS_embeddings_reduced.parquet"
    df_osha.to_parquet(output_file, engine='pyarrow')
    logging.info(f"Parquet saved: {output_file}")

    logger.info("===== STEP 2 COMPLETE =====")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline di classificazione con modelli BERT")
    parser.add_argument('--latent_dim',
                        type=int,
                        default=128,
                        help=f'Specifica la dimensione dello spazio latente, default: {128}')
    args = parser.parse_args()
    main(args.latent_dim)

import os
import glob
import copy

import numpy as np
import rasterio
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ==============================
# USER SETTINGS
# ==============================

# Folder with the GEE-exported composite tiles
TILES_DIR = r"C:\Users\hvt632\Presto_embedded_model"

# Output folder for embedding rasters
OUT_DIR = os.path.join(TILES_DIR, "embeddings_64d")

# Embedding model parameters
EMBED_DIM   = 64       # size of embedding vector (can change to 128, etc.)
HIDDEN_DIM  = 256      # hidden layer size in autoencoder

# Training parameters
N_TRAIN_PIXELS = 200_000   # total number of pixels to sample for training
BATCH_SIZE_TRAIN = 1024
N_EPOCHS = 20

# Inference parameters
BATCH_SIZE_INFER = 4096

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed (optional, for reproducibility)
np.random.seed(42)
torch.manual_seed(42)

# ==============================
# 1. LIST TILES
# ==============================

os.makedirs(OUT_DIR, exist_ok=True)

tile_paths = sorted(glob.glob(os.path.join(TILES_DIR, "*.tif")))
if not tile_paths:
    raise RuntimeError(f"No .tif files found in {TILES_DIR}")

print(f"Found {len(tile_paths)} tiles.")

# ==============================
# 2. COLLECT TRAINING PIXELS FROM TILES
# ==============================

def collect_training_samples(tile_paths, n_train_pixels):
    """
    Iterates over tiles and collects up to n_train_pixels pixel vectors for training.
    Treats all-zero pixels as invalid.
    """
    samples = []
    total = 0

    print(f"Collecting up to {n_train_pixels} training pixels...")

    for tile_path in tile_paths:
        print(f"  Reading {tile_path}")
        with rasterio.open(tile_path) as src:
            arr = src.read()  # shape: (bands, H, W)
            arr = arr.astype(np.float32)

        bands, H, W = arr.shape
        data = arr.reshape(bands, -1).T  # (N_pixels, bands)

        # Mask out pixels that are all zeros (likely outside ROI / nodata)
        mask_valid = ~np.all(data == 0, axis=1)
        data = data[mask_valid]

        if data.shape[0] == 0:
            continue

        if total + data.shape[0] <= n_train_pixels:
            samples.append(data)
            total += data.shape[0]
        else:
            need = n_train_pixels - total
            idx = np.random.choice(data.shape[0], size=need, replace=False)
            samples.append(data[idx])
            total += need
            break

        if total >= n_train_pixels:
            break

    if not samples:
        raise RuntimeError("No valid pixels collected for training.")

    X_train = np.vstack(samples)
    print(f"Collected {X_train.shape[0]} pixels for training, "
          f"feature dimension = {X_train.shape[1]}")
    return X_train

X_train = collect_training_samples(tile_paths, N_TRAIN_PIXELS)

# ==============================
# 3. DEFINE DATASET & MODEL
# ==============================

class PixelDataset(Dataset):
    def __init__(self, X):
        # X: numpy array (N, D)
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


input_dim = X_train.shape[1]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

model = Autoencoder(input_dim, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ==============================
# 4. TRAIN AUTOENCODER
# ==============================

print("\nTraining autoencoder...")
train_ds = PixelDataset(X_train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

for epoch in range(N_EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in train_dl:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon, z = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)

    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch+1}/{N_EPOCHS} - Loss: {avg_loss:.6f}")

# Optionally save model
model_path = os.path.join(TILES_DIR, f"ae_model_{EMBED_DIM}d.pth")
torch.save({
    "state_dict": model.state_dict(),
    "input_dim": input_dim,
    "embed_dim": EMBED_DIM,
    "hidden_dim": HIDDEN_DIM
}, model_path)
print(f"Saved model to {model_path}")

# ==============================
# 5. APPLY MODEL TO EACH TILE
# ==============================

def compute_embeddings_for_tile(tile_path, model, out_dir):
    """
    Loads one composite tile, encodes all valid pixels into embeddings,
    and writes an embedding raster with EMBED_DIM bands.
    """
    base = os.path.basename(tile_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}_embed_{EMBED_DIM}d.tif")

    # Skip if already done
    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} already exists")
        return

    print(f"  Processing tile: {tile_path}")
    with rasterio.open(tile_path) as src:
        arr = src.read().astype(np.float32)  # (bands, H, W)
        profile = src.profile

    bands, H, W = arr.shape
    data = arr.reshape(bands, -1).T  # (N_pixels, bands)

    # mask valid pixels (non-all-zero)
    mask_valid = ~np.all(data == 0, axis=1)
    valid_data = data[mask_valid]

    print(f"    Valid pixels: {valid_data.shape[0]} / {data.shape[0]}")

    # Compute embeddings for valid pixels
    model.eval()
    emb_valid = np.zeros((valid_data.shape[0], EMBED_DIM), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, valid_data.shape[0], BATCH_SIZE_INFER):
            batch_np = valid_data[i:i+BATCH_SIZE_INFER]
            batch = torch.from_numpy(batch_np).to(DEVICE)
            z = model.encoder(batch)
            emb_valid[i:i+BATCH_SIZE_INFER] = z.cpu().numpy()

    # Create full embedding array with zeros for invalid
    emb_full = np.zeros((data.shape[0], EMBED_DIM), dtype=np.float32)
    emb_full[mask_valid] = emb_valid

    # Reshape to raster format (bands, H, W)
    emb_raster = emb_full.T.reshape(EMBED_DIM, H, W)

    # Set output profile
    out_profile = copy.deepcopy(profile)
    out_profile.update({
        "count": EMBED_DIM,
        "dtype": "float32",
        "nodata": 0.0  # all-zero vector = "invalid" pixels
    })

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(emb_raster)

    print(f"    Saved embedding tile: {out_path}")


print("\nApplying encoder to all tiles...")
for tile_path in tqdm(tile_paths):
    compute_embeddings_for_tile(tile_path, model, OUT_DIR)

print("\nDONE. All embedding tiles saved in:")
print(OUT_DIR)

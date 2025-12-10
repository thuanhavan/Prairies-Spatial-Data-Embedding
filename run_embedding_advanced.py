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
OUT_DIR = os.path.join(TILES_DIR, "embeddings_128d")

# Embedding model parameters
EMBED_DIM   = 128      # embedding size (up from 64)
HIDDEN_DIM  = 512      # wider hidden layers

# Training parameters
N_TRAIN_PIXELS   = 500_000   # more training pixels (if you can)
BATCH_SIZE_TRAIN = 1024
N_EPOCHS         = 50

# Inference parameters
BATCH_SIZE_INFER = 4096

# Denoising autoencoder noise level
NOISE_STD = 0.05   # 0.0 = no noise, ~0.05 is a nice start

# Optimization
LR          = 5e-4
WEIGHT_DECAY = 1e-5
SCHED_PATIENCE = 5   # epochs without improvement before LR is reduced
EARLY_STOP_PATIENCE = 10  # epochs without improvement before stopping

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
# 2. CLEANING / SAMPLING
# ==============================

def clean_pixel_matrix(data):
    """
    data: (N_pixels, n_bands) float32
    - remove non-finite rows (NaN/inf)
    - remove rows that are all zero
    """
    data = data.astype(np.float32)
    finite_mask = np.all(np.isfinite(data), axis=1)
    nonzero_mask = ~np.all(data == 0, axis=1)
    mask = finite_mask & nonzero_mask
    return data[mask], mask


def collect_training_samples(tile_paths, n_train_pixels):
    """
    Iterates over tiles and collects up to n_train_pixels pixel vectors for training.
    """
    samples = []
    total = 0

    print(f"\nCollecting up to {n_train_pixels} training pixels...")

    for tile_path in tile_paths:
        print(f"  Reading {tile_path}")
        with rasterio.open(tile_path) as src:
            arr = src.read()  # (bands, H, W)

        bands, H, W = arr.shape
        data = arr.reshape(bands, -1).T  # (N_pixels, bands)

        data_clean, mask = clean_pixel_matrix(data)
        if data_clean.shape[0] == 0:
            continue

        if total + data_clean.shape[0] <= n_train_pixels:
            samples.append(data_clean)
            total += data_clean.shape[0]
        else:
            need = n_train_pixels - total
            idx = np.random.choice(data_clean.shape[0], size=need, replace=False)
            samples.append(data_clean[idx])
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
# 3. FEATURE NORMALIZATION
# ==============================

feat_mean = np.nanmean(X_train, axis=0)
feat_std  = np.nanstd(X_train, axis=0)
feat_std[feat_std == 0] = 1.0

X_train_norm = (X_train - feat_mean) / feat_std

print("\nFeature stats:")
print("  mean (first 5):", feat_mean[:5])
print("  std  (first 5):", feat_std[:5])

# ==============================
# 4. DATASET & MODEL
# ==============================

class PixelDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


input_dim = X_train_norm.shape[1]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super().__init__()
        # Deeper encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        # Deeper decoder (mirror)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


model = Autoencoder(input_dim, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

# LR scheduler: reduce LR when loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=SCHED_PATIENCE, verbose=True
)

# ==============================
# 5. TRAINING LOOP (with denoising + early stopping)
# ==============================

print("\nTraining autoencoder...")
train_ds = PixelDataset(X_train_norm)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

best_loss = float('inf')
best_state = model.state_dict()  # initial weights
patience_counter = 0

for epoch in range(N_EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_dl:
        batch = batch.to(DEVICE)

        # Denoising: add small noise during training
        if NOISE_STD > 0:
            noisy = batch + NOISE_STD * torch.randn_like(batch)
        else:
            noisy = batch

        optimizer.zero_grad()
        recon, z = model(noisy)
        loss = loss_fn(recon, batch)  # reconstruct original, not noisy
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)

    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch+1}/{N_EPOCHS} - Loss: {avg_loss:.6f}")

    # Step LR scheduler
    scheduler.step(avg_loss)

    # Early stopping check
    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        patience_counter = 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

# Load best weights before saving and inference
model.load_state_dict(best_state)

# Save model + normalization stats
model_path = os.path.join(TILES_DIR, f"ae_model_{EMBED_DIM}d_advanced.pth")
torch.save({
    "state_dict": model.state_dict(),
    "input_dim": input_dim,
    "embed_dim": EMBED_DIM,
    "hidden_dim": HIDDEN_DIM,
    "feat_mean": feat_mean,
    "feat_std": feat_std
}, model_path)
print(f"Saved model to {model_path}")

# ==============================
# 6. APPLY MODEL TO EACH TILE
# ==============================

def compute_embeddings_for_tile(tile_path, model, feat_mean, feat_std, out_dir):
    """
    Loads one composite tile, encodes all valid pixels into embeddings,
    and writes an embedding raster with EMBED_DIM bands.
    Uses the same normalization (feat_mean/std) as training.
    """
    base = os.path.basename(tile_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}_embed_{EMBED_DIM}d.tif")

    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} already exists")
        return

    print(f"  Processing tile: {tile_path}")
    with rasterio.open(tile_path) as src:
        arr = src.read().astype(np.float32)  # (bands, H, W)
        profile = src.profile

    bands, H, W = arr.shape
    data = arr.reshape(bands, -1).T  # (N_pixels, bands)

    # Clean same as training
    data_clean, mask_valid = clean_pixel_matrix(data)
    n_valid = data_clean.shape[0]
    print(f"    Valid pixels: {n_valid} / {data.shape[0]}")

    if n_valid == 0:
        print("    No valid pixels, skipping.")
        return

    # Normalize
    data_norm = (data_clean - feat_mean) / feat_std

    # Compute embeddings
    model.eval()
    emb_valid = np.zeros((n_valid, EMBED_DIM), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_valid, BATCH_SIZE_INFER):
            batch_np = data_norm[i:i+BATCH_SIZE_INFER]
            batch = torch.from_numpy(batch_np).to(DEVICE)
            z = model.encoder(batch)
            emb_valid[i:i+BATCH_SIZE_INFER] = z.cpu().numpy()

    # Build full embedding grid: 0 for invalid pixels
    emb_full = np.zeros((data.shape[0], EMBED_DIM), dtype=np.float32)
    emb_full[mask_valid] = emb_valid

    # Reshape to (bands, H, W)
    emb_raster = emb_full.T.reshape(EMBED_DIM, H, W)

    # Output profile
    out_profile = copy.deepcopy(profile)
    out_profile.update({
        "count": EMBED_DIM,
        "dtype": "float32",
        "nodata": 0.0  # all-zero vector = invalid
    })

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(emb_raster)

    print(f"    Saved embedding tile: {out_path}")


print("\nApplying encoder to all tiles...")
for tile_path in tqdm(tile_paths):
    compute_embeddings_for_tile(tile_path, model, feat_mean, feat_std, OUT_DIR)

print("\nDONE. All embedding tiles saved in:")
print(OUT_DIR)

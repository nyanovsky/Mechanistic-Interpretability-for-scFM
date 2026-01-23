"""Train Top-K Sparse Autoencoder on AIDO.Cell layer 12 activations.

Streams data from HDF5 file to handle ~51M samples efficiently.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
BASE_DIR = "/biodata/nyanovsky/datasets/pbmc3k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
CELLS_PER_BATCH = 8  # Number of cells per batch (each cell = ~19k gene embeddings)
# Effective batch size = CELLS_PER_BATCH * n_genes (~150k samples per batch)
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
NUM_WORKERS = 4

print(f"Device: {DEVICE}")


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder.

    Uses exact top-k sparsity instead of L1 regularization.
    Only the K largest latent activations are kept; rest are zeroed.
    """

    def __init__(self, input_dim=640, expansion=4, k=32):
        super().__init__()
        latent_dim = input_dim * expansion
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.k = k
        self.latent_dim = latent_dim

    def encode(self, x):
        """Encode input to sparse latent representation."""
        # Pure Top-K: no ReLU, sparsity comes from top-k selection
        latents = self.encoder(x)

        # Top-K: keep only k largest activations (by absolute value, keep sign)
        topk_vals, topk_idx = latents.topk(self.k, dim=-1)
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(-1, topk_idx, topk_vals)

        return sparse_latents

    def decode(self, sparse_latents):
        """Decode sparse latents back to input space."""
        return self.decoder(sparse_latents)

    def forward(self, x):
        """Forward pass: encode to sparse latents, decode back."""
        sparse_latents = self.encode(x)
        recon = self.decode(sparse_latents)
        return recon, sparse_latents


class H5CellDataset(Dataset):
    """Dataset that loads full cells from HDF5, then flattens to gene embeddings.

    Much faster than per-gene access: O(n_cells) reads instead of O(n_cells * n_genes).
    Each __getitem__ returns all gene embeddings for one cell [n_genes, hidden_dim].

    Args:
        h5_path: Path to HDF5 file
        cell_indices: Optional list of cell indices to include (for train/val split)
    """

    def __init__(self, h5_path, cell_indices=None):
        self.h5_path = h5_path

        # Get dimensions without keeping file open
        with h5py.File(h5_path, 'r') as f:
            self.total_cells, self.n_genes, self.hidden_dim = f['activations'].shape

        # Use subset of cells if specified (for train/val split)
        if cell_indices is not None:
            self.cell_indices = np.array(cell_indices)
            self.n_cells = len(cell_indices)
        else:
            self.cell_indices = np.arange(self.total_cells)
            self.n_cells = self.total_cells

        # Keep file handle open for faster access (will be reopened per worker)
        self._h5_file = None

    def _get_h5_file(self):
        """Get or create HDF5 file handle (worker-safe)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        actual_cell_idx = self.cell_indices[idx]
        f = self._get_h5_file()
        # Return full cell: [n_genes, hidden_dim]
        return torch.from_numpy(f['activations'][actual_cell_idx, :, :].astype(np.float32))


def worker_init_fn(worker_id):
    """Initialize HDF5 file handle per worker."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._h5_file = None  # Force reopen in this worker


def collate_flatten(batch):
    """Collate function that flattens cells to gene embeddings.

    Input: list of [n_genes, hidden_dim] tensors (one per cell)
    Output: [batch_cells * n_genes, hidden_dim] flattened tensor
    """
    # Stack cells: [batch_cells, n_genes, hidden_dim]
    stacked = torch.stack(batch, dim=0)
    # Flatten to gene embeddings: [batch_cells * n_genes, hidden_dim]
    return stacked.view(-1, stacked.shape[-1])


def main():
    parser = argparse.ArgumentParser(description="Train SAE on AIDO.Cell activations")
    parser.add_argument("--layer", type=int, default=12, help="Layer to train on")
    parser.add_argument("--expansion", type=int, default=8, help="Expansion factor")
    parser.add_argument("--k", type=int, default=32, help="Top-K sparsity")
    args = parser.parse_args()

    target_layer = args.layer
    expansion = args.expansion
    k = args.k
    
    activations_file = os.path.join(BASE_DIR, f"layer_{target_layer}", f"layer{target_layer}_activations.h5")
    
    # Save SAE in a subfolder specific to hyperparameters
    input_dim = 640
    latent_dim = input_dim * expansion
    output_dir = os.path.join(BASE_DIR, f"layer_{target_layer}", f"sae_k_{k}_{latent_dim}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plot directory
    plot_dir = os.path.join(script_dir, f'../plots/sae/layer_{target_layer}')
    os.makedirs(plot_dir, exist_ok=True)

    # Check if activations file exists
    if not os.path.exists(activations_file):
        print(f"ERROR: Activations file not found at {activations_file}")
        print("Run extract_activations.py first!")
        return

    print("="*60)
    print(f"Training Top-K SAE on Layer {target_layer}")
    print("="*60)
    print(f"Input dim: {input_dim}")
    print(f"Expansion: {expansion}x -> {latent_dim} latent features")
    print(f"Top-K: {k} (~{k/latent_dim*100:.2f}% sparsity)")
    print(f"Cells per batch: {CELLS_PER_BATCH}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("="*60)

    # Create train/val split at cell level
    print("\nLoading dataset...")
    with h5py.File(activations_file, 'r') as f:
        n_cells = f['activations'].shape[0]

    train_cells, val_cells = train_test_split(
        list(range(n_cells)),
        test_size=0.1,
        random_state=42
    )
    print(f"Train cells: {len(train_cells)}, Val cells: {len(val_cells)}")

    train_dataset = H5CellDataset(activations_file, cell_indices=train_cells)
    val_dataset = H5CellDataset(activations_file, cell_indices=val_cells)

    n_genes = train_dataset.n_genes
    effective_batch = CELLS_PER_BATCH * n_genes
    print(f"Train cells: {len(train_dataset)}, Val cells: {len(val_dataset)}")
    print(f"Genes per cell: {n_genes:,}")
    print(f"Effective batch size: {CELLS_PER_BATCH} cells × {n_genes:,} genes = {effective_batch:,} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CELLS_PER_BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_flatten,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CELLS_PER_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_flatten,
        pin_memory=True
    )

    # Initialize SAE
    sae = TopKSAE(
        input_dim=input_dim,
        expansion=expansion,
        k=k
    ).to(DEVICE)

    print(f"\nSAE parameters: {sum(p.numel() for p in sae.parameters()):,}")

    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        # Training
        sae.train()
        epoch_train_loss = 0
        n_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{NUM_EPOCHS}")

        for batch in pbar:
            batch = batch.to(DEVICE)

            # Forward pass
            recon, _ = sae(batch)

            # Reconstruction loss (MSE)
            loss = F.mse_loss(recon, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()

        avg_train_loss = epoch_train_loss / n_train_batches
        train_losses.append(avg_train_loss)

        # Validation
        sae.eval()
        epoch_val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                recon, _ = sae(batch)
                loss = F.mse_loss(recon, batch)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / n_val_batches
        val_losses.append(avg_val_loss)

        # Compute sparsity stats on a sample batch
        with torch.no_grad():
            sample_batch = next(iter(train_loader)).to(DEVICE)
            _, sample_latents = sae(sample_batch)
            l0_sparsity = (sample_latents != 0).float().sum(dim=-1).mean().item()

        print(f"Epoch {epoch+1}: Train = {avg_train_loss:.6f}, Val = {avg_val_loss:.6f}, L0 = {l0_sparsity:.1f}")

    print("\nTraining complete!")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Top-K SAE Training Layer {target_layer} (K={k}, expansion={expansion}x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'sae_training_curve.png'), dpi=150)
    plt.close()

    # Save model
    model_path = os.path.join(output_dir, 'topk_sae.pt')
    torch.save({
        'model_state_dict': sae.state_dict(),
        'input_dim': input_dim,
        'expansion': expansion,
        'k': k,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, model_path)
    print(f"Model saved to {model_path}")

    # Save decoder weights separately for easy steering
    decoder_path = os.path.join(output_dir, 'sae_decoder.pt')
    torch.save(sae.decoder.state_dict(), decoder_path)
    print(f"Decoder weights saved to {decoder_path}")


if __name__ == "__main__":
    main()

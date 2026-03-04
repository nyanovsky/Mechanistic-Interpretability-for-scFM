"""Train Top-K SAE on PBMC3K with hyperparameter grid search.

Streams data from HDF5. Loops over a grid of (expansion, k, lr),
with early stopping and optional decoder weight normalization.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import TopKSAE

BASE_DIR = "/biodata/nyanovsky/datasets/pbmc3k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 640
BATCH_BY_EXPANSION = {4: 16, 8: 8, 16: 4}
NUM_WORKERS = 4
MAX_EPOCHS = 20
PATIENCE = 3

GRID = [
    # (expansion, k, lr)
    (4,   32, 3e-4),
    (8,   32, 3e-4),
    (8,   64, 3e-4),
    (16,  64, 3e-4),
    (16, 128, 3e-4),
]

print(f"Device: {DEVICE}")


# ── Decoder normalization helper ──────────────────────────────────────────────

@torch.no_grad()
def normalize_decoder(sae):
    sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)


# ── Data ──────────────────────────────────────────────────────────────────────

class H5CellDataset(Dataset):
    def __init__(self, h5_path, cell_indices=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.total_cells, self.n_genes, self.hidden_dim = f['activations'].shape
        self.cell_indices = np.array(cell_indices) if cell_indices is not None else np.arange(self.total_cells)
        self.n_cells = len(self.cell_indices)
        self._h5_file = None

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self): return self.n_cells

    def __getitem__(self, idx):
        return torch.from_numpy(self._get_h5()['activations'][self.cell_indices[idx]].astype(np.float32))


def worker_init_fn(worker_id):
    torch.utils.data.get_worker_info().dataset._h5_file = None


def collate_flatten(batch):
    stacked = torch.stack(batch, dim=0)
    return stacked.view(-1, stacked.shape[-1])


def make_loaders(layer, batch_size):
    path = os.path.join(BASE_DIR, f"layer_{layer}", f"layer{layer}_activations.h5")
    with h5py.File(path, 'r') as f:
        n_cells = f['activations'].shape[0]
    train_cells, val_cells = train_test_split(list(range(n_cells)), test_size=0.1, random_state=42)
    kwargs = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
                  worker_init_fn=worker_init_fn, collate_fn=collate_flatten, pin_memory=True)
    train_loader = DataLoader(H5CellDataset(path, train_cells), shuffle=True,  **kwargs)
    val_loader   = DataLoader(H5CellDataset(path, val_cells),   shuffle=False, **kwargs)
    return train_loader, val_loader


# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(sae, loader, optimizer=None, weight_norm=False):
    training = optimizer is not None
    sae.train() if training else sae.eval()
    total_loss, total_sq, n_batches = 0.0, 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(DEVICE)
            recon, _ = sae(batch)
            loss = F.mse_loss(recon, batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if weight_norm:
                    normalize_decoder(sae)
            total_loss += loss.item()
            total_sq   += batch.pow(2).mean().item()
            n_batches  += 1
    avg_mse = total_loss / n_batches
    avg_sq  = total_sq  / n_batches
    rel_err = avg_mse / avg_sq if avg_sq > 0 else float('nan')
    return avg_mse, rel_err


def train_config(sae, train_loader, val_loader, lr, label, weight_norm=False):
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    train_losses, val_losses = [], []
    best_val_mse = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        train_mse, _ = run_epoch(sae, train_loader, optimizer, weight_norm=weight_norm)
        val_mse, val_rel = run_epoch(sae, val_loader)
        scheduler.step()

        train_losses.append(train_mse)
        val_losses.append(val_mse)
        print(f"  [{label}] epoch {epoch+1:2d}: train={train_mse:.5f}  val={val_mse:.5f}  rel={val_rel*100:.1f}%")

        if val_mse < best_val_mse - 1e-6:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in sae.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    sae.load_state_dict(best_state)
    return train_losses, val_losses


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--b-pre", action="store_true",
                        help="Use learned pre-encoder centering bias")
    parser.add_argument("--weight-norm", action="store_true",
                        help="Normalize decoder weights to unit norm after each step")
    args = parser.parse_args()

    config_tag = []
    if args.b_pre:
        config_tag.append("bpre")
    if args.weight_norm:
        config_tag.append("wnorm")
    config_suffix = "_" + "_".join(config_tag) if config_tag else ""
    print(f"Config: b_pre={args.b_pre}, weight_norm={args.weight_norm}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, f"../../plots/sae/layer_{args.layer}{config_suffix}")
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    results = []
    best_rel_err = float('inf')
    best_sae_state, best_config = None, None

    for expansion, k, lr in GRID:
        latent_dim = INPUT_DIM * expansion
        label = f"exp={expansion} k={k}"
        print(f"\n{'='*60}\n{label}  lr={lr}\n{'='*60}")
        batch_size = BATCH_BY_EXPANSION[expansion]
        print(f"Loading data (batch_size={batch_size})...")
        train_loader, val_loader = make_loaders(args.layer, batch_size)

        sae = TopKSAE(INPUT_DIM, expansion, k, use_b_pre=args.b_pre).to(DEVICE)
        train_losses, val_losses = train_config(sae, train_loader, val_loader, lr, label,
                                                 weight_norm=args.weight_norm)

        _, rel_err = run_epoch(sae, val_loader)
        print(f"  → Relative reconstruction error: {rel_err*100:.1f}%")
        results.append((label, rel_err, val_losses))

        # Save config model
        out_dir = os.path.join(BASE_DIR, f"layer_{args.layer}", f"sae_k_{k}_{latent_dim}{config_suffix}")
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            'model_state_dict': sae.state_dict(),
            'input_dim': INPUT_DIM, 'expansion': expansion, 'k': k,
            'use_b_pre': args.b_pre, 'weight_norm': args.weight_norm,
            'train_losses': train_losses, 'val_losses': val_losses,
            'relative_recon_error': rel_err,
        }, os.path.join(out_dir, 'topk_sae.pt'))
        torch.save(sae.decoder.state_dict(), os.path.join(out_dir, 'sae_decoder.pt'))

        if rel_err < best_rel_err:
            best_rel_err = rel_err
            best_sae_state = {k_: v.clone() for k_, v in sae.state_dict().items()}
            best_config = (expansion, k, lr)

        ax.plot(val_losses, label=f"{label} ({rel_err*100:.1f}%)")

    # Save best SAE
    best_exp, best_k, _ = best_config
    best_dir = os.path.join(BASE_DIR, f"layer_{args.layer}", f"sae_best{config_suffix}")
    os.makedirs(best_dir, exist_ok=True)
    torch.save({'model_state_dict': best_sae_state, 'input_dim': INPUT_DIM,
                'expansion': best_exp, 'k': best_k,
                'use_b_pre': args.b_pre, 'weight_norm': args.weight_norm,
                'relative_recon_error': best_rel_err},
               os.path.join(best_dir, 'topk_sae.pt'))

    # Training curves
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val MSE')
    ax.set_title(f'SAE Grid Search — Layer {args.layer} (legend: final rel. error)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'sae_grid_search.png'), dpi=150)
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("Grid Search Summary (sorted by rel. error)")
    print(f"{'='*60}")
    for label, rel_err, _ in sorted(results, key=lambda x: x[1]):
        print(f"  {label:25s}  {rel_err*100:.1f}%")
    print(f"\nBest: exp={best_exp} k={best_k} → {best_rel_err*100:.1f}%  saved to {best_dir}")


if __name__ == "__main__":
    main()

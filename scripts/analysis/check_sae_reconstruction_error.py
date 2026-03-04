"""
Analyze SAE reconstruction error using pre-computed activations.

Usage:
    python check_sae_reconstruction_error.py
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import utility functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import TopKSAE, load_sae
from utils.data_utils import get_expressed_genes_mask
from aido_cell.utils import align_adata


def main():
    # Paths
    LAYER = 12
    BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{LAYER}"
    SAE_DIR = f"{BASE_DIR}/sae_k_32_5120"
    ACTIVATIONS_FILE = f"{BASE_DIR}/layer{LAYER}_activations.h5"
    RAW_DATA_FILE = "../../data/pbmc/pbmc3k_raw.h5ad"
    OUTPUT_DIR = "../../plots/sae/layer_12/reconstruction_error"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8  # Process 8 cells at a time

    # Training MSE baselines (from training curves)
    TRAINING_MSE = {9: 0.83, 12: 1.35, 15: 2.65}

    print("="*80)
    print("SAE RECONSTRUCTION ERROR ANALYSIS")
    print("="*80)

    # Load SAE
    print("\n1. Loading SAE...")
    sae = load_sae(SAE_DIR, device)

    # Get expressed genes mask
    print("\n2. Loading expressed genes mask...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, RAW_DATA_FILE)
    expressed_mask = get_expressed_genes_mask(raw_path)
    n_expressed = expressed_mask.sum()
    n_total = len(expressed_mask)
    print(f"   Expressed genes: {n_expressed}/{n_total} ({100*n_expressed/n_total:.1f}%)")

    # Load activations and compute reconstruction error
    print(f"\n3. Loading activations from {ACTIVATIONS_FILE}...")
    print(f"   Processing in batches of {batch_size} cells...")

    error_norms = []
    activation_norms = []
    relative_errors = []

    with h5py.File(ACTIVATIONS_FILE, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        print(f"   Shape: ({n_cells} cells, {n_genes} genes, {hidden_dim} hidden_dim)")

        with torch.no_grad():
            for cell_idx in tqdm(range(0, n_cells, batch_size), desc="Processing cells"):
                # Load batch of cells: (batch_size, n_genes, hidden_dim)
                end_idx = min(cell_idx + batch_size, n_cells)
                batch_activations = f['activations'][cell_idx:end_idx, :, :].astype(np.float32)

                # Filter to expressed genes only: (batch_size, n_expressed, hidden_dim)
                batch_activations = batch_activations[:, expressed_mask, :]

                # Flatten to (batch_size * n_expressed, hidden_dim)
                batch_size_actual = batch_activations.shape[0]
                flat_activations = batch_activations.reshape(-1, hidden_dim)
                flat_tensor = torch.from_numpy(flat_activations).to(device)

                # Forward pass through SAE
                recon, features = sae(flat_tensor)

                # Compute error: (batch_size * n_genes, hidden_dim)
                error = flat_tensor - recon

                # Norms per activation vector
                error_norm = torch.norm(error, dim=1)
                activation_norm = torch.norm(flat_tensor, dim=1)
                relative_error = error_norm / (activation_norm + 1e-8)

                error_norms.append(error_norm.cpu().numpy())
                activation_norms.append(activation_norm.cpu().numpy())
                relative_errors.append(relative_error.cpu().numpy())

    # Concatenate results
    error_norms = np.concatenate(error_norms)
    activation_norms = np.concatenate(activation_norms)
    relative_errors = np.concatenate(relative_errors)

    # Summary statistics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nRECONSTRUCTION QUALITY:")
    print(f"  Mean relative error:   {relative_errors.mean():.4f}")
    print(f"  Median relative error: {np.median(relative_errors):.4f}")
    print(f"  Min relative error:    {relative_errors.min():.4f}")
    print(f"  Max relative error:    {relative_errors.max():.4f}")

    print(f"\nABSOLUTE MAGNITUDES:")
    print(f"  Mean activation norm:  {activation_norms.mean():.4f}")
    print(f"  Mean error norm:       {error_norms.mean():.4f}")
    print(f"  Error/Activation:      {error_norms.mean() / activation_norms.mean():.4f}")

    # Interpretation
    mean_rel_error = relative_errors.mean()
    print(f"\nINTERPRETATION:")
    if mean_rel_error < 0.1:
        print("  ✓ EXCELLENT - Reconstruction is very good. Steering should work well.")
    elif mean_rel_error < 0.3:
        print("  ✓ GOOD - Reconstruction is acceptable. Steering should be effective.")
    elif mean_rel_error < 0.5:
        print("  ⚠ MODERATE - Reconstruction is marginal. Steering may have reduced effectiveness.")
    else:
        print("  ✗ POOR - Error term dominates. Steering effects may be washed out.")

    # Plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Relative error distribution
    axes[0].hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(relative_errors.mean(), color='red', linestyle='--',
                   label=f'Mean: {relative_errors.mean():.3f}')
    axes[0].set_xlabel('Relative Error (||error|| / ||activation||)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('SAE Reconstruction Error')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Absolute magnitudes
    axes[1].hist(activation_norms, bins=50, alpha=0.5, label='Activation', edgecolor='black')
    axes[1].hist(error_norms, bins=50, alpha=0.5, label='Error', edgecolor='black')
    axes[1].set_xlabel('L2 Norm')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Activation vs Error Magnitude')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Scatter: activation norm vs error norm (sample 10k points for visibility)
    sample_idx = np.random.choice(len(activation_norms), min(10000, len(activation_norms)), replace=False)
    axes[2].scatter(activation_norms[sample_idx], error_norms[sample_idx], alpha=0.3, s=5)
    axes[2].set_xlabel('Activation Norm')
    axes[2].set_ylabel('Error Norm')
    axes[2].set_title('Error vs Activation Magnitude')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, 'reconstruction_error.png')
    plt.savefig(plot_file, dpi=150)
    print(f"\n✓ Plot saved to: {plot_file}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

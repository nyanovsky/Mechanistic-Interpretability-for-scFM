"""Compute ICA baseline feature matrices for comparison with SAE.

Fits FastICA on the same residual stream activations the SAE operates on,
producing drop-in replacement matrices for the downstream interpretation pipeline.

Outputs (same interface as compute_feature_matrices.py):
- feature_gene_matrix.npy: [n_components, n_genes] mean |projection| per gene per IC
- cell_feature_matrix.npy: [n_cells, n_components] mean |projection| per cell (expressed genes)
- feature_participation_ratios.npy: [n_components] PR values
"""

import os
import sys
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import FastICA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import get_expressed_genes, compute_participation_ratio


def subsample_activations(h5_path, n_samples, seed=42):
    """Load a random subsample of activation vectors for ICA fitting.

    Args:
        h5_path: Path to HDF5 file with shape [n_cells, n_genes, hidden_dim]
        n_samples: Number of vectors to sample
        seed: Random seed

    Returns:
        samples: [n_samples, hidden_dim] array
    """
    rng = np.random.RandomState(seed)

    with h5py.File(h5_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        total_vectors = n_cells * n_genes

        n_samples = min(n_samples, total_vectors)
        print(f"Sampling {n_samples:,} vectors from {total_vectors:,} total ({n_cells} cells x {n_genes} genes)")

        # Sample random (cell, gene) pairs
        flat_indices = rng.choice(total_vectors, size=n_samples, replace=False)
        cell_indices = flat_indices // n_genes
        gene_indices = flat_indices % n_genes

        # Sort by cell index for sequential HDF5 reads
        sort_order = np.argsort(cell_indices)
        cell_indices = cell_indices[sort_order]
        gene_indices = gene_indices[sort_order]

        samples = np.empty((n_samples, hidden_dim), dtype=np.float32)

        # Group by cell for efficient reads
        unique_cells, cell_starts = np.unique(cell_indices, return_index=True)
        cell_ends = np.append(cell_starts[1:], n_samples)

        for i, (cell_idx, start, end) in enumerate(
            tqdm(zip(unique_cells, cell_starts, cell_ends),
                 total=len(unique_cells), desc="Loading samples")
        ):
            cell_data = f['activations'][cell_idx, :, :]  # [n_genes, hidden_dim]
            genes_for_cell = gene_indices[start:end]
            samples[start:end] = cell_data[genes_for_cell]

    return samples


def compute_ica_feature_gene_matrix(h5_path, unmixing_matrix, mean, batch_size=8):
    """Compute feature-gene matrix by projecting activations through ICA unmixing matrix.

    Args:
        h5_path: Path to HDF5 file
        unmixing_matrix: [n_components, hidden_dim] ICA unmixing matrix
        mean: [hidden_dim] mean vector to subtract before projection
        batch_size: Cells per batch

    Returns:
        [n_components, n_genes] mean |projection| across cells
    """
    n_components = unmixing_matrix.shape[0]

    with h5py.File(h5_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        feature_gene_sum = np.zeros((n_components, n_genes), dtype=np.float64)

        n_batches = (n_cells + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Computing feature-gene matrix"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_cells)

            # [batch, n_genes, hidden_dim]
            batch_data = f['activations'][batch_start:batch_end, :, :].astype(np.float32)

            # Center then project (matching ICA transform)
            projected = np.abs((batch_data - mean) @ unmixing_matrix.T)

            # Accumulate: sum over batch -> [n_genes, n_components] -> transpose to [n_components, n_genes]
            feature_gene_sum += projected.sum(axis=0).T

        feature_gene_mean = (feature_gene_sum / n_cells).astype(np.float32)

    return feature_gene_mean


def compute_ica_cell_feature_matrix(h5_path, unmixing_matrix, mean, expressed_mask, batch_size=8):
    """Compute cell-feature matrix: mean |projection| per cell across expressed genes.

    Args:
        h5_path: Path to HDF5 file
        unmixing_matrix: [n_components, hidden_dim] ICA unmixing matrix
        mean: [hidden_dim] mean vector to subtract before projection
        expressed_mask: Boolean mask for expressed genes
        batch_size: Cells per batch

    Returns:
        [n_cells, n_components] matrix
    """
    n_components = unmixing_matrix.shape[0]

    with h5py.File(h5_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        cell_feature_matrix = np.zeros((n_cells, n_components), dtype=np.float32)

        n_batches = (n_cells + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Computing cell-feature matrix"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_cells)

            batch_data = f['activations'][batch_start:batch_end, :, :].astype(np.float32)

            # Center then project (matching ICA transform)
            projected = np.abs((batch_data - mean) @ unmixing_matrix.T)  # [batch, n_genes, n_components]

            # Mean over expressed genes
            for i in range(batch_end - batch_start):
                cell_feature_matrix[batch_start + i] = projected[i, expressed_mask].mean(axis=0)

    return cell_feature_matrix


def main():
    parser = argparse.ArgumentParser(description='Compute ICA baseline feature matrices')
    parser.add_argument('--layer', type=int, default=12,
                        help='AIDO.Cell layer (default: 12)')
    parser.add_argument('--n_samples', type=int, default=500000,
                        help='Number of vectors to subsample for ICA fitting (default: 500000)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Cells per batch for matrix computation (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--max_iter', type=int, default=500,
                        help='Max iterations for FastICA (default: 500)')
    args = parser.parse_args()

    BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
    ACTIVATIONS_FILE = f"{BASE_DIR}/layer{args.layer}_activations.h5"
    RAW_DATA_FILE = "data/pbmc/pbmc3k_raw.h5ad"
    OUTPUT_DIR = f"{BASE_DIR}/ica_baseline"

    print("=" * 70)
    print("ICA Baseline Feature Matrix Computation")
    print("=" * 70)
    print(f"Layer: {args.layer}")
    print(f"Samples for fitting: {args.n_samples:,}")
    print(f"Seed: {args.seed}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    if not os.path.exists(ACTIVATIONS_FILE):
        print(f"ERROR: Activations file not found at {ACTIVATIONS_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get expressed genes mask
    print("\nLoading gene metadata and filtering to expressed genes...")
    _, expressed_names, expressed_mask = get_expressed_genes(
        RAW_DATA_FILE, min_mean_expr=0.01, min_pct_cells=0.5
    )
    print(f"Expressed genes: {expressed_mask.sum()} / {len(expressed_mask)}")

    # Step 1: Subsample activations for ICA fitting
    print("\n" + "=" * 70)
    print("Step 1: Subsample activations")
    print("=" * 70)
    samples = subsample_activations(ACTIVATIONS_FILE, args.n_samples, seed=args.seed)
    print(f"Loaded samples: {samples.shape}")

    # Clean NaN/Inf (zero-masked genes in AIDO.Cell can produce these)
    nan_mask = np.isnan(samples).any(axis=1) | np.isinf(samples).any(axis=1)
    if nan_mask.any():
        print(f"Removing {nan_mask.sum():,} samples with NaN/Inf ({100*nan_mask.mean():.1f}%)")
        samples = samples[~nan_mask]
        print(f"Remaining samples: {samples.shape[0]:,}")

    # Use float64 for numerical stability during ICA fitting
    samples = samples.astype(np.float64)

    # Remove near-zero-variance dimensions (cause whitening to blow up)
    dim_std = samples.std(axis=0)
    low_var_mask = dim_std < 1e-8
    if low_var_mask.any():
        print(f"Removing {low_var_mask.sum()} near-zero-variance dimensions")
        samples = samples[:, ~low_var_mask]
        print(f"Remaining dimensions: {samples.shape[1]}")

    # Step 2: Fit FastICA
    print("\n" + "=" * 70)
    print("Step 2: Fit FastICA")
    print("=" * 70)
    n_components = samples.shape[1]
    print(f"Fitting FastICA with {n_components} components on {samples.shape[0]:,} samples...")

    ica = FastICA(
        n_components=n_components,
        random_state=args.seed,
        max_iter=args.max_iter,
        whiten='unit-variance',
        tol=1e-3,
    )
    ica.fit(samples)
    del samples  # free memory

    # Reconstruct full unmixing matrix and mean [n_components, 640] with zeros for dropped dims
    if low_var_mask.any():
        full_unmixing = np.zeros((n_components, len(low_var_mask)), dtype=np.float32)
        full_unmixing[:, ~low_var_mask] = ica.components_.astype(np.float32)
        unmixing_matrix = full_unmixing
        full_mean = np.zeros(len(low_var_mask), dtype=np.float32)
        full_mean[~low_var_mask] = ica.mean_.astype(np.float32)
        ica_mean = full_mean
    else:
        unmixing_matrix = ica.components_.astype(np.float32)
        ica_mean = ica.mean_.astype(np.float32)
    print(f"ICA unmixing matrix: {unmixing_matrix.shape}")
    print(f"ICA mean norm: {np.linalg.norm(ica_mean):.4f}")

    # Save the ICA model components and mean
    np.save(os.path.join(OUTPUT_DIR, 'ica_unmixing_matrix.npy'), unmixing_matrix)
    np.save(os.path.join(OUTPUT_DIR, 'ica_mean.npy'), ica_mean)

    # Step 3: Compute feature-gene matrix
    print("\n" + "=" * 70)
    print("Step 3: Compute feature-gene matrix")
    print("=" * 70)
    feature_gene_matrix = compute_ica_feature_gene_matrix(
        ACTIVATIONS_FILE, unmixing_matrix, ica_mean, batch_size=args.batch_size
    )
    print(f"Feature-gene matrix: {feature_gene_matrix.shape}")
    np.save(os.path.join(OUTPUT_DIR, 'feature_gene_matrix.npy'), feature_gene_matrix)

    # Step 4: Compute participation ratios
    print("\n" + "=" * 70)
    print("Step 4: Compute participation ratios")
    print("=" * 70)
    pr_values = compute_participation_ratio(feature_gene_matrix, axis=1)
    print(f"PR statistics:")
    print(f"  Min: {pr_values.min():.2f}, Max: {pr_values.max():.2f}")
    print(f"  Mean: {pr_values.mean():.2f}, Median: {np.median(pr_values):.2f}")
    np.save(os.path.join(OUTPUT_DIR, 'feature_participation_ratios.npy'), pr_values)

    # Step 5: Compute cell-feature matrix
    print("\n" + "=" * 70)
    print("Step 5: Compute cell-feature matrix")
    print("=" * 70)
    cell_feature_matrix = compute_ica_cell_feature_matrix(
        ACTIVATIONS_FILE, unmixing_matrix, ica_mean, expressed_mask, batch_size=args.batch_size
    )
    print(f"Cell-feature matrix: {cell_feature_matrix.shape}")
    np.save(os.path.join(OUTPUT_DIR, 'cell_feature_matrix.npy'), cell_feature_matrix)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"All matrices saved to {OUTPUT_DIR}")
    print(f"  feature_gene_matrix.npy: {feature_gene_matrix.shape}")
    print(f"  feature_participation_ratios.npy: {pr_values.shape}")
    print(f"  cell_feature_matrix.npy: {cell_feature_matrix.shape}")
    print(f"  ica_unmixing_matrix.npy: {unmixing_matrix.shape}")


if __name__ == "__main__":
    main()

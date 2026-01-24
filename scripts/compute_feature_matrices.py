"""Compute feature-gene and cell-feature activation matrices with different pooling strategies.

This script consolidates:
1. Matrix computation logic from interpret_sae_parallel.py
2. Participation Ratio (PR) calculation from analyze_feature_scale.py

Supports two pooling strategies:
- mean: Original mean pooling across cells (baseline)
- custom: Double-adaptive pooling using Cell PR and 95th percentile
"""

import os
import sys
import argparse
import torch
import h5py
import numpy as np
import anndata as ad
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ModelGenerator/huggingface/aido.cell'))
from aido_cell.utils import align_adata


class TopKSAE(torch.nn.Module):
    """Top-K Sparse Autoencoder (must match train_sae.py)."""

    def __init__(self, input_dim=640, expansion=4, k=32):
        super().__init__()
        latent_dim = input_dim * expansion
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, input_dim)
        self.k = k
        self.latent_dim = latent_dim

    def encode(self, x):
        latents = self.encoder(x)
        topk_vals, topk_idx = latents.topk(self.k, dim=-1)
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(-1, topk_idx, topk_vals)
        return sparse_latents

    def forward(self, x):
        sparse_latents = self.encode(x)
        recon = self.decoder(sparse_latents)
        return recon, sparse_latents


def get_expressed_genes_mask(raw_data_path, min_mean_expr=0.01, min_pct_cells=0.5):
    """Return boolean mask for genes with sufficient expression.

    Filters out genes with zero/low expression to avoid spurious correlations.

    Args:
        raw_data_path: Path to raw h5ad file
        min_mean_expr: Minimum mean expression across cells (default: 0.01)
        min_pct_cells: Minimum % of cells with nonzero expression (default: 0.5%)

    Returns:
        expressed_mask: Boolean array of shape (n_genes,)
        expressed_names: List of expressed gene names
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, raw_data_path)

    adata_raw = ad.read_h5ad(raw_path)
    adata_aligned, attention_mask = align_adata(adata_raw)

    X = adata_aligned.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X_filtered = X[:, attention_mask.astype(bool)]

    gene_names = adata_aligned.var_names[attention_mask.astype(bool)]

    # Compute expression stats per gene
    mean_expr = X_filtered.mean(axis=0)
    pct_nonzero = (X_filtered > 0).sum(axis=0) / X_filtered.shape[0] * 100

    # Filter: mean expression > threshold OR expressed in > N% of cells
    expressed_mask = (mean_expr > min_mean_expr) | (pct_nonzero > min_pct_cells)

    expressed_names = [gene_names[i] for i in np.where(expressed_mask)[0]]

    return expressed_mask, expressed_names


def compute_participation_ratio(matrix, axis=1):
    """Compute Participation Ratio (PR) for each row.

    PR measures the effective number of dimensions that contribute to a distribution.
    Higher PR = more dimensions participate (broader feature).
    Lower PR = fewer dimensions participate (sparser feature).

    Args:
        matrix: [n_features, n_dims] array
        axis: Axis along which to compute PR (default: 1, across columns)

    Returns:
        pr: [n_features] array of participation ratios
    """
    # Square the activations to get "energy"
    energy = matrix ** 2

    # Normalize to get probability distribution P_i for each feature
    energy_sum = energy.sum(axis=axis, keepdims=True)

    # Avoid division by zero for dead features
    energy_sum[energy_sum == 0] = 1.0

    probs = energy / energy_sum

    # PR = 1 / Sum(P_i^2) (Inverse Participation Ratio)
    ipr = (probs ** 2).sum(axis=axis)

    # Handle dead features (ipr will be 0 if energy was 0)
    pr = np.zeros_like(ipr)
    mask = ipr > 0
    pr[mask] = 1.0 / ipr[mask]

    return pr


def compute_feature_gene_activations_mean(sae, h5_path, device, batch_size=8):
    """Compute feature-gene matrix using mean pooling (original approach).

    Args:
        sae: The SAE model
        h5_path: Path to HDF5 file
        device: 'cuda' or 'cpu'
        batch_size: Number of cells to process in parallel (default: 8)

    Returns: [n_features, n_genes] matrix of mean activations across cells
    """
    print(f"Computing feature-gene matrix with MEAN pooling (batch_size={batch_size})...")

    with h5py.File(h5_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        n_features = sae.latent_dim

        # Accumulate activations per gene
        feature_gene_sum = np.zeros((n_features, n_genes), dtype=np.float32)

        # Process cells in batches
        n_batches = (n_cells + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_cells)
            current_batch_size = batch_end - batch_start

            # Load batch: [batch_size, n_genes, hidden_dim]
            batch_activations = torch.from_numpy(
                f['activations'][batch_start:batch_end, :, :].astype(np.float32)
            ).to(device)

            # Reshape to [batch_size * n_genes, hidden_dim] for encoding
            batch_activations_flat = batch_activations.reshape(-1, hidden_dim)

            # Encode through SAE: [batch_size * n_genes, n_features]
            with torch.no_grad():
                sparse_latents_flat = sae.encode(batch_activations_flat)

            # Reshape back: [batch_size, n_genes, n_features]
            sparse_latents = sparse_latents_flat.reshape(current_batch_size, n_genes, n_features)

            # Transpose to [batch_size, n_features, n_genes] and sum over batch
            # Then accumulate to [n_features, n_genes]
            feature_gene_sum += sparse_latents.permute(0, 2, 1).sum(dim=0).cpu().numpy()

        # Average across cells
        feature_gene_mean = feature_gene_sum / n_cells

    return feature_gene_mean


def compute_feature_gene_activations_custom(sae, h5_path, device, batch_size=8,
                                           cell_pr_scale=0.6, min_cells=10,
                                           feature_batch_size=100):
    """Compute feature-gene matrix using custom double-adaptive pooling.

    Two-pass strategy for memory efficiency:
    Pass 1: Compute cell activities and Cell PR, determine selected cells per feature
    Pass 2: Process features in batches, collecting activations and computing percentiles

    Args:
        sae: The SAE model
        h5_path: Path to HDF5 file with activations [n_cells, n_genes, hidden_dim]
        device: 'cuda' or 'cpu'
        batch_size: Number of cells to process in parallel (default: 8)
        cell_pr_scale: Scale factor for selecting cells (default: 0.6 = 60% of Cell PR)
        min_cells: Minimum cells to use per feature (default: 10)
        feature_batch_size: Number of features to process together (default: 100)

    Returns:
        feature_gene_matrix: [n_features, n_genes] array
        cell_pr_values: [n_features] array of Cell PR values
    """
    print(f"Computing feature-gene matrix with CUSTOM double-adaptive pooling (batch_size={batch_size})...")
    print(f"  Cell PR scale: {cell_pr_scale}")
    print(f"  Min cells per feature: {min_cells}")
    print(f"  Feature batch size: {feature_batch_size}")

    with h5py.File(h5_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        n_features = sae.latent_dim

        # ==================== PASS 1: Compute Cell Activities ====================
        print(f"\nPASS 1: Computing cell activities and Cell PR...")
        # cell_activities: [n_features, n_cells]
        cell_activities = np.zeros((n_features, n_cells), dtype=np.float32)

        n_cell_batches = (n_cells + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_cell_batches), desc="Computing cell activities"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_cells)
            current_batch_size = batch_end - batch_start

            # Load batch: [batch_size, n_genes, hidden_dim]
            batch_activations = torch.from_numpy(
                f['activations'][batch_start:batch_end, :, :].astype(np.float32)
            ).to(device)

            # Reshape to [batch_size * n_genes, hidden_dim] for encoding
            batch_activations_flat = batch_activations.reshape(-1, hidden_dim)

            # Encode through SAE: [batch_size * n_genes, n_features]
            with torch.no_grad():
                sparse_latents_flat = sae.encode(batch_activations_flat)

            # Reshape back: [batch_size, n_genes, n_features]
            sparse_latents = sparse_latents_flat.reshape(current_batch_size, n_genes, n_features)

            # Sum across genes for each cell: [batch_size, n_features]
            cell_activities_batch = sparse_latents.sum(dim=1).cpu().numpy()

            # Store in cell_activities: [n_features, n_cells]
            cell_activities[:, batch_start:batch_end] = cell_activities_batch.T

        # Compute Cell PR for each feature
        print("Computing Cell PR for each feature...")
        cell_pr_values = compute_participation_ratio(cell_activities, axis=1)

        print(f"  Cell PR statistics:")
        print(f"    Min: {cell_pr_values.min():.2f}")
        print(f"    Max: {cell_pr_values.max():.2f}")
        print(f"    Mean: {cell_pr_values.mean():.2f}")
        print(f"    Median: {np.median(cell_pr_values):.2f}")

        # Determine selected cells for each feature
        print("Determining selected cells for each feature...")
        selected_cells_per_feature = {}
        for feat_idx in range(n_features):
            n_cells_to_use = int(cell_pr_values[feat_idx] * cell_pr_scale)
            n_cells_to_use = max(min_cells, min(n_cells_to_use, n_cells))

            # Select top cells by activity
            cell_activity_feat = cell_activities[feat_idx]
            top_cell_indices = np.argsort(cell_activity_feat)[::-1][:n_cells_to_use]
            selected_cells_per_feature[feat_idx] = set(top_cell_indices)

        # ==================== PASS 2: Collect Activations and Compute Percentiles ====================
        print(f"\nPASS 2: Computing 95th percentiles (processing {n_features} features in batches of {feature_batch_size})...")

        feature_gene_matrix = np.zeros((n_features, n_genes), dtype=np.float32)
        n_feature_batches = (n_features + feature_batch_size - 1) // feature_batch_size

        for feat_batch_idx in range(n_feature_batches):
            feat_batch_start = feat_batch_idx * feature_batch_size
            feat_batch_end = min(feat_batch_start + feature_batch_size, n_features)
            feature_batch = range(feat_batch_start, feat_batch_end)

            print(f"\n  Processing features {feat_batch_start}-{feat_batch_end-1} ({len(feature_batch)} features)...")

            # Pre-allocate arrays for this feature batch (avoids list over-allocation)
            feature_gene_arrays = {}
            feature_cell_counters = {}  # Track how many cells we've filled in

            for feat_idx in feature_batch:
                n_selected = len(selected_cells_per_feature[feat_idx])
                # Pre-allocate: [n_selected_cells, n_genes]
                feature_gene_arrays[feat_idx] = np.zeros((n_selected, n_genes), dtype=np.float32)
                feature_cell_counters[feat_idx] = 0

            # Create mapping from cell_idx to position in pre-allocated array
            cell_to_position = {}
            for feat_idx in feature_batch:
                cell_to_position[feat_idx] = {
                    cell_idx: pos
                    for pos, cell_idx in enumerate(sorted(selected_cells_per_feature[feat_idx]))
                }

            # Make one pass through all cells, collecting activations for this feature batch
            for cell_batch_idx in tqdm(range(n_cell_batches), desc=f"  Collecting activations"):
                cell_batch_start = cell_batch_idx * batch_size
                cell_batch_end = min(cell_batch_start + batch_size, n_cells)
                current_batch_size = cell_batch_end - cell_batch_start

                # Load batch: [batch_size, n_genes, hidden_dim]
                batch_activations = torch.from_numpy(
                    f['activations'][cell_batch_start:cell_batch_end, :, :].astype(np.float32)
                ).to(device)

                # Reshape to [batch_size * n_genes, hidden_dim] for encoding
                batch_activations_flat = batch_activations.reshape(-1, hidden_dim)

                # Encode through SAE: [batch_size * n_genes, n_features]
                with torch.no_grad():
                    sparse_latents_flat = sae.encode(batch_activations_flat)

                # Reshape back: [batch_size, n_genes, n_features]
                sparse_latents = sparse_latents_flat.reshape(current_batch_size, n_genes, n_features)

                # Extract only features in current batch to save memory
                # [batch_size, n_genes, feature_batch_size]
                sparse_latents_subset = sparse_latents[:, :, feat_batch_start:feat_batch_end].cpu().numpy()

                # For each cell in batch, check if any feature in current batch needs it
                for i in range(current_batch_size):
                    cell_idx = cell_batch_start + i

                    for local_feat_idx, feat_idx in enumerate(feature_batch):
                        if cell_idx in selected_cells_per_feature[feat_idx]:
                            # Get position in pre-allocated array
                            pos = cell_to_position[feat_idx][cell_idx]
                            # Fill in the pre-allocated array (no copy needed!)
                            feature_gene_arrays[feat_idx][pos] = sparse_latents_subset[i, :, local_feat_idx]

            # Compute percentiles for this feature batch
            print(f"  Computing percentiles for features {feat_batch_start}-{feat_batch_end-1}...")
            for feat_idx in feature_batch:
                # Compute 95th percentile across cells (axis=0)
                # feature_gene_arrays[feat_idx] is already [n_selected_cells, n_genes]
                feature_gene_matrix[feat_idx] = np.percentile(feature_gene_arrays[feat_idx], 95, axis=0)

    return feature_gene_matrix, cell_pr_values


def compute_cell_feature_activations(sae, h5_path, expressed_mask, device, batch_size=8):
    """Compute mean SAE feature activation per cell across expressed genes.

    Args:
        sae: The SAE model
        h5_path: Path to HDF5 file with activations
        expressed_mask: Boolean mask for expressed genes
        device: 'cuda' or 'cpu'
        batch_size: Number of cells to process in parallel (default: 8)

    Returns: [n_cells, n_features] matrix of mean activations
    """
    print(f"Computing cell-feature activation matrix (batch_size={batch_size})...")

    with h5py.File(h5_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        n_features = sae.latent_dim

        cell_feature_matrix = np.zeros((n_cells, n_features), dtype=np.float32)

        # Process cells in batches
        n_batches = (n_cells + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_cells)
            current_batch_size = batch_end - batch_start

            # Load batch: [batch_size, n_genes, hidden_dim]
            batch_activations = torch.from_numpy(
                f['activations'][batch_start:batch_end, :, :].astype(np.float32)
            ).to(device)

            # Reshape to [batch_size * n_genes, hidden_dim] for encoding
            batch_activations_flat = batch_activations.reshape(-1, hidden_dim)

            # Encode through SAE: [batch_size * n_genes, n_features]
            with torch.no_grad():
                sparse_latents_flat = sae.encode(batch_activations_flat)

            # Reshape back: [batch_size, n_genes, n_features]
            sparse_latents = sparse_latents_flat.reshape(current_batch_size, n_genes, n_features)

            # Mean activation across EXPRESSED genes only
            # sparse_latents: [batch_size, n_genes, n_features]
            # Apply mask and mean over gene dimension (axis=1)
            for i in range(current_batch_size):
                cell_feature_matrix[batch_start + i] = sparse_latents[i, expressed_mask].mean(dim=0).cpu().numpy()

    return cell_feature_matrix


def main():
    parser = argparse.ArgumentParser(description='Compute feature-gene and cell-feature matrices')
    parser.add_argument('--layer', type=int, default=12,
                       help='AIDO.Cell layer (default: 12)')
    parser.add_argument('--expansion', type=int, default=8,
                       help='SAE expansion factor (default: 8)')
    parser.add_argument('--k', type=int, default=32,
                       help='Top-K sparsity (default: 32)')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'custom'],
                       help='Pooling strategy: mean (baseline) or custom (double-adaptive)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Number of cells to process in parallel (default: 8)')
    parser.add_argument('--cell_pr_scale', type=float, default=1,
                       help='Cell PR scale factor for custom pooling (default: 0.6)')
    parser.add_argument('--min_cells', type=int, default=10,
                       help='Minimum cells per feature for custom pooling (default: 10)')
    parser.add_argument('--feature_batch_size', type=int, default=100,
                       help='Number of features to process together in custom pooling (default: 100)')
    parser.add_argument('--online', action='store_true',
                       help='Use online SAE directory')
    args = parser.parse_args()

    # Build paths
    INPUT_DIM = 640
    LATENT_DIM = INPUT_DIM * args.expansion
    BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
    SAE_SUFFIX = "_online" if args.online else ""
    SAE_DIR = f"{BASE_DIR}/sae_k_{args.k}_{LATENT_DIM}{SAE_SUFFIX}"

    ACTIVATIONS_FILE = f"{BASE_DIR}/layer{args.layer}_activations.h5"
    SAE_MODEL_FILE = f"{SAE_DIR}/topk_sae.pt"

    # Output directory with pooling strategy suffix
    OUTPUT_DIR = f"{SAE_DIR}/interpretations_{args.pooling}_pooling"

    RAW_DATA_FILE = "../data/pbmc/pbmc3k_raw.h5ad"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("Feature Matrix Computation")
    print("="*70)
    print(f"Layer: {args.layer}")
    print(f"SAE: {args.expansion}x expansion, K={args.k}")
    print(f"Pooling strategy: {args.pooling}")
    print(f"Batch size: {args.batch_size}")
    if args.pooling == 'custom':
        print(f"  Cell PR scale: {args.cell_pr_scale}")
        print(f"  Min cells: {args.min_cells}")
        print(f"  Feature batch size: {args.feature_batch_size}")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check files exist
    if not os.path.exists(ACTIVATIONS_FILE):
        print(f"ERROR: Activations file not found at {ACTIVATIONS_FILE}")
        return
    if not os.path.exists(SAE_MODEL_FILE):
        print(f"ERROR: SAE model not found at {SAE_MODEL_FILE}")
        return

    # Load SAE model
    print("\nLoading SAE model...")
    checkpoint = torch.load(SAE_MODEL_FILE, map_location=DEVICE)

    # Handle both checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Offline format
        input_dim = checkpoint['input_dim']
        expansion = checkpoint['expansion']
        k = checkpoint['k']
        state_dict = checkpoint['model_state_dict']
    else:
        # Online format - infer parameters from weight shapes
        input_dim = checkpoint['decoder.weight'].shape[0]
        latent_dim = checkpoint['decoder.weight'].shape[1]
        expansion = latent_dim // input_dim
        k = args.k
        state_dict = checkpoint

    sae = TopKSAE(input_dim=input_dim, expansion=expansion, k=k).to(DEVICE)
    sae.load_state_dict(state_dict)
    sae.eval()
    print(f"Loaded SAE: {input_dim} -> {sae.latent_dim} (K={k})")

    # Get expressed genes mask
    print("\nLoading gene metadata and filtering to expressed genes...")
    expressed_mask, expressed_names = get_expressed_genes_mask(
        RAW_DATA_FILE, min_mean_expr=0.01, min_pct_cells=0.5
    )
    print(f"Expressed genes: {expressed_mask.sum()} / {len(expressed_mask)}")

    # ==================== Feature-Gene Matrix ====================
    print("\n" + "="*70)
    print("Computing Feature-Gene Matrix")
    print("="*70)

    if args.pooling == 'mean':
        feature_gene_matrix = compute_feature_gene_activations_mean(
            sae, ACTIVATIONS_FILE, DEVICE, batch_size=args.batch_size
        )

        # Compute Gene PR for mean pooling (for consistency with old workflow)
        print("\nComputing Gene PR (Participation Ratio)...")
        gene_pr_values = compute_participation_ratio(feature_gene_matrix, axis=1)

        print(f"\nGene PR statistics:")
        print(f"  Min: {gene_pr_values.min():.2f}")
        print(f"  Max: {gene_pr_values.max():.2f}")
        print(f"  Mean: {gene_pr_values.mean():.2f}")
        print(f"  Median: {np.median(gene_pr_values):.2f}")

        # Save Gene PR
        np.save(os.path.join(OUTPUT_DIR, 'feature_participation_ratios.npy'), gene_pr_values)
        print(f"Saved Gene PR to {OUTPUT_DIR}/feature_participation_ratios.npy")

    elif args.pooling == 'custom':
        feature_gene_matrix, cell_pr_values = compute_feature_gene_activations_custom(
            sae, ACTIVATIONS_FILE, DEVICE, batch_size=args.batch_size,
            cell_pr_scale=args.cell_pr_scale,
            min_cells=args.min_cells,
            feature_batch_size=args.feature_batch_size
        )

        # Save Cell PR
        np.save(os.path.join(OUTPUT_DIR, 'cell_participation_ratios.npy'), cell_pr_values)
        print(f"\nSaved Cell PR to {OUTPUT_DIR}/cell_participation_ratios.npy")

        # Also compute Gene PR on the resulting matrix for comparison
        print("\nComputing Gene PR on custom-pooled matrix...")
        gene_pr_values = compute_participation_ratio(feature_gene_matrix, axis=1)
        print(f"Gene PR statistics:")
        print(f"  Min: {gene_pr_values.min():.2f}")
        print(f"  Max: {gene_pr_values.max():.2f}")
        print(f"  Mean: {gene_pr_values.mean():.2f}")
        print(f"  Median: {np.median(gene_pr_values):.2f}")

        # Save Gene PR
        np.save(os.path.join(OUTPUT_DIR, 'feature_participation_ratios.npy'), gene_pr_values)
        print(f"Saved Gene PR to {OUTPUT_DIR}/feature_participation_ratios.npy")

    # Save feature-gene matrix
    print(f"\nSaving feature-gene matrix: {feature_gene_matrix.shape}")
    np.save(os.path.join(OUTPUT_DIR, 'feature_gene_matrix.npy'), feature_gene_matrix)

    # ==================== Cell-Feature Matrix ====================
    print("\n" + "="*70)
    print("Computing Cell-Feature Matrix")
    print("="*70)

    cell_feature_matrix = compute_cell_feature_activations(
        sae, ACTIVATIONS_FILE, expressed_mask, DEVICE, batch_size=args.batch_size
    )

    print(f"Saving cell-feature matrix: {cell_feature_matrix.shape}")
    np.save(os.path.join(OUTPUT_DIR, 'cell_feature_matrix.npy'), cell_feature_matrix)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"All matrices saved to {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - feature_gene_matrix.npy: {feature_gene_matrix.shape}")
    print(f"  - cell_feature_matrix.npy: {cell_feature_matrix.shape}")
    print(f"  - feature_participation_ratios.npy (Gene PR): {gene_pr_values.shape}")
    if args.pooling == 'custom':
        print(f"  - cell_participation_ratios.npy (Cell PR): {cell_pr_values.shape}")


if __name__ == "__main__":
    main()

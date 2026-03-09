"""Interpret SAE features via GO enrichment and cell type correlation.

For each SAE feature:
1. Find genes with highest mean activation across cells
2. Run GO enrichment on top genes
3. Correlate feature activation with cell types
"""

import os
import sys
import argparse
import torch
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from collections import defaultdict


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_sae
from utils.data_utils import load_gene_names, get_expressed_genes
from utils.go_utils import run_go_enrichment

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Interpret SAE features via GO enrichment')
    parser.add_argument('--layer', type=int, default=12,
                        help='AIDO.Cell layer (12 or 15)')
    parser.add_argument('--expansion', type=int, default=8,
                        help='SAE expansion factor (4 or 8)')
    parser.add_argument('--k', type=int, default=32,
                        help='Top-K sparsity')
    parser.add_argument('--n_features', type=int, default=50,
                        help='Number of top features to analyze for GO enrichment')
    return parser.parse_args()

args = parse_args()

# Build paths from arguments
INPUT_DIM = 640
LATENT_DIM = INPUT_DIM * args.expansion
BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
SAE_DIR = f"{BASE_DIR}/sae_k_{args.k}_{LATENT_DIM}"

ACTIVATIONS_FILE = f"{BASE_DIR}/layer{args.layer}_activations.h5"
SAE_MODEL_FILE = f"{SAE_DIR}/topk_sae.pt"
OUTPUT_DIR = f"{SAE_DIR}/interpretations_filter_zero_expressed"
PLOT_DIR = f"../../plots/sae/layer_{args.layer}"

# Static paths (don't depend on layer/SAE config)
PROCESSED_DATA_FILE = "../../data/pbmc/pbmc3k_processed.h5ad"
RAW_DATA_FILE = "../../data/pbmc/pbmc3k_raw.h5ad"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Analysis parameters
TOP_GENES_PER_FEATURE = 30  # Number of top genes for GO enrichment (legacy, only used if USE_ADAPTIVE_SELECTION=False)
TOP_FEATURES_TO_ANALYZE = args.n_features  # Analyze top N most active features

# Adaptive gene selection parameters
USE_ADAPTIVE_SELECTION = True  # Set to False for original fixed-30 behavior
MIN_GENES_PER_FEATURE = 10  # Minimum genes for statistical power
MAX_GENES_PER_FEATURE = 100  # Maximum to avoid dilution
PR_SCALE_FACTOR = 0.6  # Select 60% of effective genes (PR * 0.6)
USE_ACTIVATION_THRESHOLD = False  # Optional: filter genes <10% of max activation

print(f"Configuration: Layer {args.layer}, {args.expansion}x expansion (K={args.k})")
print(f"SAE: {SAE_MODEL_FILE}")
print(f"Output: {OUTPUT_DIR}")



def compute_feature_gene_activations(sae, h5_path, batch_size=1000):
    """Compute mean SAE feature activation per gene across all cells.

    Returns: [n_features, n_genes] matrix of mean activations
    """
    print("Computing feature-gene activation matrix...")

    with h5py.File(h5_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        n_features = sae.latent_dim

        # Accumulate activations per gene
        feature_gene_sum = np.zeros((n_features, n_genes), dtype=np.float32)

        # Process cell by cell to manage memory
        for cell_idx in tqdm(range(n_cells), desc="Processing cells"):
            # Load all genes for this cell [n_genes, hidden_dim]
            cell_activations = torch.from_numpy(
                f['activations'][cell_idx, :, :].astype(np.float32)
            ).to(DEVICE)

            # Encode through SAE [n_genes, n_features]
            with torch.no_grad():
                sparse_latents = sae.encode(cell_activations)

            # Accumulate [n_features, n_genes]
            feature_gene_sum += sparse_latents.T.cpu().numpy()

        # Average across cells
        feature_gene_mean = feature_gene_sum / n_cells

    return feature_gene_mean


def compute_cell_feature_activations(sae, h5_path, expressed_mask=None):
    """Compute mean SAE feature activation per cell across expressed genes.

    Args:
        sae: The SAE model
        h5_path: Path to HDF5 file with activations
        expressed_mask: Boolean mask for expressed genes. If None, uses all genes.

    Returns: [n_cells, n_features] matrix of mean activations
    """
    print("Computing cell-feature activation matrix...")

    with h5py.File(h5_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape
        n_features = sae.latent_dim

        cell_feature_matrix = np.zeros((n_cells, n_features), dtype=np.float32)

        for cell_idx in tqdm(range(n_cells), desc="Processing cells"):
            cell_activations = torch.from_numpy(
                f['activations'][cell_idx, :, :].astype(np.float32)
            ).to(DEVICE)

            with torch.no_grad():
                sparse_latents = sae.encode(cell_activations)

            # Mean activation across EXPRESSED genes only
            if expressed_mask is not None:
                cell_feature_matrix[cell_idx] = sparse_latents[expressed_mask].mean(dim=0).cpu().numpy()
            else:
                cell_feature_matrix[cell_idx] = sparse_latents.mean(dim=0).cpu().numpy()

    return cell_feature_matrix


def select_genes_adaptive(feature_activations, pr, use_threshold=False,
                         min_genes=10, max_genes=100, scale_factor=0.6):
    """
    Select genes for GO enrichment from EXPRESSED genes only.

    This function operates on an already-filtered set of expressed genes to avoid
    contamination from AIDO.Cell's zero-expression issues (e.g., olfactory genes).

    Args:
        feature_activations: Activations for EXPRESSED genes only (already filtered)
        pr: Participation ratio for this feature
        use_threshold: If True, filter by 10% of max activation (optional refinement)
        min_genes: Minimum genes for statistical power
        max_genes: Maximum genes to avoid dilution
        scale_factor: Proportion of PR to select (default 0.6 = 60%)

    Returns:
        Indices into the expressed_genes array
    """
    # 1. Compute target count from PR
    target_count = int(pr * scale_factor)  # e.g., PR=50 → 30 genes
    target_count = max(min_genes, min(target_count, max_genes))  # Bounds: [min, max]

    # 2. Select top N genes by activation
    sorted_indices = np.argsort(feature_activations)[::-1]
    top_indices = sorted_indices[:target_count]

    # 3. Optional: Filter by minimum activation (10% of max)
    if use_threshold:
        max_activation = feature_activations[top_indices[0]]
        threshold = max_activation * 0.1
        valid_mask = feature_activations[top_indices] >= threshold
        selected = top_indices[valid_mask]

        # Ensure minimum for statistical power
        if len(selected) < min_genes:
            selected = top_indices[:min_genes]
    else:
        selected = top_indices

    return selected




def analyze_cell_type_correlations(cell_feature_matrix, cell_types):
    """Compute correlation between SAE features and cell types."""
    print("Computing cell type correlations...")

    unique_types = cell_types.unique()
    n_features = cell_feature_matrix.shape[1]

    # Create binary indicators for each cell type
    correlations = {}
    for cell_type in unique_types:
        indicator = (cell_types == cell_type).astype(float)
        correlations[cell_type] = []

        for feat_idx in range(n_features):
            corr, _ = spearmanr(cell_feature_matrix[:, feat_idx], indicator)
            correlations[cell_type].append(corr if not np.isnan(corr) else 0)

    return pd.DataFrame(correlations)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check files exist
    if not os.path.exists(ACTIVATIONS_FILE):
        print(f"ERROR: Activations file not found at {ACTIVATIONS_FILE}")
        return
    if not os.path.exists(SAE_MODEL_FILE):
        print(f"ERROR: SAE model not found at {SAE_MODEL_FILE}")
        return

    print("="*60)
    print("SAE Feature Interpretation")
    print("="*60)

    # Load SAE model
    print("\nLoading SAE model...")
    sae = load_sae(os.path.dirname(SAE_MODEL_FILE), device=DEVICE)
    print(f"Loaded SAE: {sae.input_dim} -> {sae.latent_dim} (K={sae.k})")

    # Load gene names
    print("\nLoading gene names...")
    gene_names = load_gene_names(RAW_DATA_FILE)
    print(f"Loaded {len(gene_names)} gene names")

    # Load cell type labels
    print("\nLoading cell type labels...")
    processed_path = os.path.join(script_dir, PROCESSED_DATA_FILE)
    adata_processed = ad.read_h5ad(processed_path)

    # Get cell names from HDF5
    with h5py.File(ACTIVATIONS_FILE, 'r') as f:
        cell_names = [name.decode() for name in f['cell_names'][:]]

    # Align cell types with activation order
    cell_types = adata_processed.obs.loc[cell_names, 'louvain']
    print(f"Cell types: {cell_types.value_counts().to_dict()}")

    # ==================== Feature-Gene Analysis ====================
    print("\n" + "="*60)
    print("Feature-Gene Analysis")
    print("="*60)

    # load feature_gene_matrix if it exists, compute otherwise:
    try:
        feature_gene_matrix = np.load(os.path.join(OUTPUT_DIR, 'feature_gene_matrix.npy'))
        print(f"Loaded feature-gene matrix from {OUTPUT_DIR}")
    except:
        print("Computing feature-gene matrix...")

        feature_gene_matrix = compute_feature_gene_activations(sae, ACTIVATIONS_FILE)
        print(f"Feature-gene matrix shape: {feature_gene_matrix.shape}")

    # Find most active features (highest total activation)
    feature_activity = feature_gene_matrix.sum(axis=1)
    top_features = np.argsort(feature_activity)[::-1][:TOP_FEATURES_TO_ANALYZE]

    print(f"\nAnalyzing top {TOP_FEATURES_TO_ANALYZE} most active features...")

    # Get expressed genes only (filter out zero-expressed genes like olfactory receptors)
    print("\nFiltering to expressed genes only...")
    _, expressed_names, expressed_mask = get_expressed_genes(
        RAW_DATA_FILE, min_mean_expr=0.01, min_pct_cells=0.5
    )
    print(f"Filtered from {len(gene_names)} to {len(expressed_names)} expressed genes")

    # Load PR values if using adaptive selection
    if USE_ADAPTIVE_SELECTION:
        pr_path = os.path.join(OUTPUT_DIR, 'feature_participation_ratios.npy')
        if not os.path.exists(pr_path):
            print(f"ERROR: PR values not found at {pr_path}")
            print("Please run analyze_feature_scale.py first to compute PR values.")
            return
        pr_values = np.load(pr_path)
        print(f"Loaded PR values. Using adaptive selection (factor={PR_SCALE_FACTOR}, bounds=[{MIN_GENES_PER_FEATURE},{MAX_GENES_PER_FEATURE}])")

    # GO enrichment for top features
    go_results = {}
    skipped = 0
    for feat_idx in tqdm(top_features, desc="GO enrichment"):
        # Check if GO enrichment already exists for this feature
        enrichr_dir = os.path.join(OUTPUT_DIR, f'feature_{feat_idx}_enrichr')
        if os.path.exists(enrichr_dir) and len(os.listdir(enrichr_dir)) > 0:
            skipped += 1
            continue

        # Filter feature-gene activations to expressed genes only
        gene_activations = feature_gene_matrix[feat_idx][expressed_mask]

        # Select genes (adaptive or fixed)
        if USE_ADAPTIVE_SELECTION:
            pr = pr_values[feat_idx]
            top_local_indices = select_genes_adaptive(
                gene_activations, pr,
                use_threshold=USE_ACTIVATION_THRESHOLD,
                min_genes=MIN_GENES_PER_FEATURE,
                max_genes=MAX_GENES_PER_FEATURE,
                scale_factor=PR_SCALE_FACTOR
            )
        else:
            top_local_indices = np.argsort(gene_activations)[::-1][:TOP_GENES_PER_FEATURE]

        top_genes = [expressed_names[i] for i in top_local_indices]

        # Run GO enrichment
        result = run_go_enrichment(top_genes, OUTPUT_DIR, identifier=f"feature_{feat_idx}")
        if result is not None and len(result) > 0:
            go_results[feat_idx] = result

    if skipped > 0:
        print(f"Skipped {skipped} features with existing GO enrichment results")

    # Save feature-gene matrix if it didnt exist
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'feature_gene_matrix.npy')):
        print(f"Saving feature-gene matrix to {OUTPUT_DIR}")
    
        # Save feature-gene matrix
        np.save(os.path.join(OUTPUT_DIR, 'feature_gene_matrix.npy'), feature_gene_matrix)

    # ==================== Cell-Type Analysis ====================
    print("\n" + "="*60)
    print("Cell-Type Analysis")
    print("="*60)

    # load cell_feature_matrix if it exists, compute otherwise:
    try:
        cell_feature_matrix = np.load(os.path.join(OUTPUT_DIR, 'cell_feature_matrix.npy'))
        correlations_df = pd.read_csv(
            os.path.join(OUTPUT_DIR, 'feature_celltype_correlations.csv'),
            index_col=0
        )
        print(f"Loaded cell-feature matrix and correlations from {OUTPUT_DIR}")
    except:
        print("Computing cell-type correlations...")

        cell_feature_matrix = compute_cell_feature_activations(sae, ACTIVATIONS_FILE, expressed_mask)
        print(f"Cell-feature matrix shape: {cell_feature_matrix.shape}")

        # Compute correlations
        correlations_df = analyze_cell_type_correlations(cell_feature_matrix, cell_types)

    # Find features most correlated with each cell type
    print("\nTop features per cell type:")
    for cell_type in correlations_df.columns:
        top_feat = correlations_df[cell_type].abs().idxmax()
        corr_val = correlations_df.loc[top_feat, cell_type]
        print(f"  {cell_type}: Feature {top_feat} (r={corr_val:.3f})")

    # Save cell-feature matrix and correlations_df if it didnt exist:
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'cell_feature_matrix.npy')):
        print(f"Saving cell-feature matrix to {OUTPUT_DIR}")
        np.save(os.path.join(OUTPUT_DIR, 'cell_feature_matrix.npy'), cell_feature_matrix)
        correlations_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_celltype_correlations.csv'))

    # ==================== Visualization ====================
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    # Heatmap of top features vs cell types
    plt.figure(figsize=(12, 10))
    top_corr = correlations_df.iloc[top_features[:30]]
    sns.heatmap(top_corr, cmap='RdBu_r', center=0, annot=False)
    plt.xlabel('Cell Type')
    plt.ylabel('SAE Feature')
    plt.title('Top 30 SAE Features vs Cell Types')
    plt.tight_layout()
    plot_dir = os.path.join(script_dir, PLOT_DIR)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'sae_feature_celltype_heatmap.png'), dpi=150)
    plt.close()

    # Distribution of feature activations
    plt.figure(figsize=(10, 5))
    plt.hist(feature_activity, bins=50, alpha=0.7)
    plt.xlabel('Total Feature Activation')
    plt.ylabel('Count')
    plt.title('Distribution of SAE Feature Activity')
    plt.savefig(os.path.join(plot_dir, 'sae_feature_activity_dist.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()

"""Interpret SAE features via GO enrichment and cell type correlation (PARALLELIZED VERSION).

For each SAE feature:
1. Find genes with highest mean activation across cells (from pre-computed matrices)
2. Run GO enrichment on top genes (PARALLELIZED)
3. Correlate feature activation with cell types

This version parallelizes GO enrichment API calls for significantly faster execution.

IMPORTANT: This script requires pre-computed matrices. Run compute_feature_matrices.py first.

Usage:
    # Small dataset (e.g., PBMC3K): load gene info from raw h5ad
    python scripts/interpret_sae_parallel.py \\
        --matrix-path /path/to/feature_gene_matrix.npy \\
        --raw-data /path/to/data.h5ad \\
        --plot-dir /path/to/plots

    # Large dataset (e.g., bone marrow): use pre-computed gene names and mask
    python scripts/interpret_sae_parallel.py \\
        --matrix-path /path/to/feature_gene_matrix.npy \\
        --gene-names /path/to/gene_names.txt \\
        --expressed-mask /path/to/expressed_genes_mask.npy \\
        --celltype-file /path/to/cell_types.txt \\
        --plot-dir /path/to/plots
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import get_expressed_genes
from utils.go_utils import run_go_enrichment

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Interpret SAE features via GO enrichment (parallel)')

    # Required paths
    parser.add_argument('--matrix-path', type=str, required=True,
                        help='Path to feature_gene_matrix.npy')
    parser.add_argument('--raw-data', type=str, default=None,
                        help='Path to raw h5ad file (used for gene alignment via AIDO.Cell). '
                             'Not needed if --gene-names and --expressed-mask are provided.')
    parser.add_argument('--plot-dir', type=str, required=True,
                        help='Directory to save plots')

    # Pre-computed gene metadata (alternative to --raw-data)
    parser.add_argument('--gene-names', type=str, default=None,
                        help='Path to pre-computed gene names file (one gene per line). '
                             'When provided, skips loading raw data for gene names.')
    parser.add_argument('--expressed-mask', type=str, default=None,
                        help='Path to pre-computed expressed genes mask (.npy boolean array). '
                             'When provided with --gene-names, skips loading raw data entirely.')

    # Optional paths
    parser.add_argument('--processed-data', type=str, default=None,
                        help='Path to processed h5ad file with cell type labels (only needed if raw data lacks cell_type column)')
    parser.add_argument('--celltype-file', type=str, default=None,
                        help='Path to pre-computed cell types file (one cell type per line). '
                             'Alternative to loading cell types from h5ad files.')

    # Analysis parameters
    parser.add_argument('--n_features', type=int, default=50,
                        help='Number of top features to analyze for GO enrichment')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers for GO enrichment (default: 8)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between API calls in seconds (default: 0.1)')
    parser.add_argument('--celltype-col', type=str, default='cell_type',
                        help='Column name in .obs for cell type labels (default: cell_type)')
    return parser.parse_args()

args = parse_args()

# Paths from CLI arguments
MATRIX_PATH = args.matrix_path
MATRIX_DIR = os.path.dirname(MATRIX_PATH)
OUTPUT_DIR = MATRIX_DIR  # Save GO enrichment results to same directory
PLOT_DIR = args.plot_dir
RAW_DATA_FILE = args.raw_data
PROCESSED_DATA_FILE = args.processed_data  # Optional, only needed if raw data lacks cell_type
CELLTYPE_FILE = args.celltype_file
CELLTYPE_COL = args.celltype_col

# Analysis parameters
TOP_GENES_PER_FEATURE = 30  # Number of top genes for GO enrichment (legacy, only used if USE_ADAPTIVE_SELECTION=False)
TOP_FEATURES_TO_ANALYZE = args.n_features  # Analyze top N most active features

# Adaptive gene selection parameters
USE_ADAPTIVE_SELECTION = True  # Set to False for original fixed-30 behavior
MIN_GENES_PER_FEATURE = 10  # Minimum genes for statistical power
MAX_GENES_PER_FEATURE = 100  # Maximum to avoid dilution
PR_SCALE_FACTOR = 1 # 0.6  Select 60% of effective genes (PR * 0.6)
USE_ACTIVATION_THRESHOLD = False  # Optional: filter genes <10% of max activation

GENE_NAMES_FILE = args.gene_names
EXPRESSED_MASK_FILE = args.expressed_mask

# Validate: either raw-data or (gene-names) must be provided
if RAW_DATA_FILE is None and GENE_NAMES_FILE is None:
    print("ERROR: Either --raw-data or --gene-names must be provided.")
    sys.exit(1)

print(f"Configuration:")
print(f"  Matrix path: {MATRIX_PATH}")
if RAW_DATA_FILE:
    print(f"  Raw data: {RAW_DATA_FILE}")
if GENE_NAMES_FILE:
    print(f"  Gene names file: {GENE_NAMES_FILE}")
if EXPRESSED_MASK_FILE:
    print(f"  Expressed mask file: {EXPRESSED_MASK_FILE}")
if PROCESSED_DATA_FILE:
    print(f"  Processed data: {PROCESSED_DATA_FILE}")
print(f"  Cell type column: {CELLTYPE_COL}")
print(f"  Plot directory: {PLOT_DIR}")
print(f"  Parallelization: {args.workers} workers, {args.delay}s delay")



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


def process_feature_go_enrichment(task_data):
    """Process GO enrichment for a single feature (for parallel execution)."""
    feat_idx, gene_activations, expressed_names, pr_values, background = task_data

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

    # Run GO enrichment with expressed genes as background
    result = run_go_enrichment(top_genes, OUTPUT_DIR, background=background, identifier=f"feature_{feat_idx}", verbose=False)

    return feat_idx, result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("SAE Feature Interpretation (PARALLELIZED)")
    print("="*60)

    # ==================== Load Pre-computed Matrices ====================
    print("\n" + "="*60)
    print("Loading Pre-computed Matrices")
    print("="*60)

    # Load feature-gene matrix (REQUIRED)
    if not os.path.exists(MATRIX_PATH):
        print(f"\nERROR: Feature-gene matrix not found at {MATRIX_PATH}")
        return

    print(f"Loading feature-gene matrix from {MATRIX_PATH}...")
    feature_gene_matrix = np.load(MATRIX_PATH)
    print(f"  Shape: {feature_gene_matrix.shape}")

    # Load PR values (REQUIRED for adaptive selection)
    pr_path = os.path.join(MATRIX_DIR, 'feature_participation_ratios.npy')
    if USE_ADAPTIVE_SELECTION:
        if not os.path.exists(pr_path):
            print(f"\nERROR: PR values not found at {pr_path}")
            print("\nPlease run compute_feature_matrices.py first (it computes PR values automatically)")
            return
        pr_values = np.load(pr_path)
        print(f"  PR values loaded. Adaptive selection: factor={PR_SCALE_FACTOR}, bounds=[{MIN_GENES_PER_FEATURE},{MAX_GENES_PER_FEATURE}]")
    else:
        pr_values = None

    # Load gene names and expressed genes
    print("\nLoading gene names and expressed genes...")
    _, expressed_names, expressed_mask = get_expressed_genes(
        raw_data_path=RAW_DATA_FILE,
        gene_names_file=GENE_NAMES_FILE,
        expressed_mask_file=EXPRESSED_MASK_FILE,
    )
    print(f"  Loaded {len(expressed_names)} expressed gene names")

    # Detect if feature_gene_matrix is already filtered to expressed genes
    matrix_already_filtered = (feature_gene_matrix.shape[1] == len(expressed_names))
    if matrix_already_filtered:
        print(f"  Feature-gene matrix already filtered to expressed genes")
    elif expressed_mask is not None:
        print(f"  Will filter feature-gene matrix from {feature_gene_matrix.shape[1]} to {len(expressed_names)} genes")
    else:
        print(f"\nERROR: Feature-gene matrix has {feature_gene_matrix.shape[1]} genes but no expressed mask to filter.")
        return

    # Load cell type labels
    cell_types = None
    print("\nLoading cell type labels...")

    if CELLTYPE_FILE is not None:
        with open(CELLTYPE_FILE) as f:
            cell_types = pd.Series(f.read().strip().split('\n'))
        print(f"  Loaded {len(cell_types)} cell types from {CELLTYPE_FILE}")
    elif RAW_DATA_FILE is not None:
        adata_raw = ad.read_h5ad(RAW_DATA_FILE)

        if CELLTYPE_COL in adata_raw.obs.columns:
            print(f"  Found '{CELLTYPE_COL}' column in raw data")
            cell_types = adata_raw.obs[CELLTYPE_COL]

    if cell_types is None and PROCESSED_DATA_FILE is not None:
        adata_processed = ad.read_h5ad(PROCESSED_DATA_FILE)
        if CELLTYPE_COL in adata_processed.obs.columns:
            cell_types = adata_processed.obs[CELLTYPE_COL]
        else:
            print(f"  WARNING: '{CELLTYPE_COL}' column not found.")

    if cell_types is not None:
        print(f"  Cell types: {cell_types.value_counts().to_dict()}")
    else:
        print("  No cell type labels available. Skipping cell-type analysis.")

    # ==================== Feature-Gene Analysis ====================
    print("\n" + "="*60)
    print("Feature-Gene Analysis")
    print("="*60)

    # Find most active features (highest total activation)
    feature_activity = feature_gene_matrix.sum(axis=1)
    top_features = np.argsort(feature_activity)[::-1][:TOP_FEATURES_TO_ANALYZE]

    print(f"Analyzing top {TOP_FEATURES_TO_ANALYZE} most active features...")

    # ==================== PARALLELIZED GO ENRICHMENT ====================
    print(f"\nRunning GO enrichment with {args.workers} parallel workers...")

    # Check which features need processing
    features_to_process = []
    skipped = 0
    for feat_idx in top_features:
        enrichr_dir = os.path.join(OUTPUT_DIR, f'feature_{feat_idx}_enrichr')
        if os.path.exists(enrichr_dir) and len(os.listdir(enrichr_dir)) > 0:
            skipped += 1
            continue

        if matrix_already_filtered:
            gene_activations = feature_gene_matrix[feat_idx]
        else:
            gene_activations = feature_gene_matrix[feat_idx][expressed_mask]
        features_to_process.append((feat_idx, gene_activations, expressed_names, pr_values, expressed_names))

    if skipped > 0:
        print(f"Skipping {skipped} features with existing GO enrichment results")

    print(f"Processing {len(features_to_process)} features...")

    # Run GO enrichment in parallel
    go_results = {}
    if len(features_to_process) > 0:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_feature_go_enrichment, task): task for task in features_to_process}

            with tqdm(total=len(features_to_process), desc="GO enrichment") as pbar:
                for future in as_completed(futures):
                    try:
                        feat_idx, result = future.result()
                        if result is not None and len(result) > 0:
                            go_results[feat_idx] = result
                    except Exception as e:
                        task = futures[future]
                        feat_idx = task[0]
                        print(f"\nError processing feature {feat_idx}: {e}")
                    finally:
                        pbar.update(1)

    # ==================== Cell-Type Analysis ====================
    correlations_df = None
    if cell_types is not None:
        print("\n" + "="*60)
        print("Cell-Type Analysis")
        print("="*60)

        cell_feature_matrix_path = os.path.join(MATRIX_DIR, 'cell_feature_matrix.npy')
        if not os.path.exists(cell_feature_matrix_path):
            print(f"\nWARNING: Cell-feature matrix not found at {cell_feature_matrix_path}")
            print("Skipping cell-type analysis.")
        else:
            cell_feature_matrix = np.load(cell_feature_matrix_path)
            print(f"Loaded cell-feature matrix: {cell_feature_matrix.shape}")
            

            # Load or compute correlations
            correlations_path = os.path.join(MATRIX_DIR, 'feature_celltype_correlations.csv')
            if os.path.exists(correlations_path):
                correlations_df = pd.read_csv(correlations_path, index_col=0)
                print(f"Loaded pre-computed correlations from {correlations_path}")
            else:
                print("Computing cell-type correlations...")
                correlations_df = analyze_cell_type_correlations(cell_feature_matrix, cell_types)
                correlations_df.to_csv(correlations_path)
                print(f"Saved correlations to {correlations_path}")

            # Find features most correlated with each cell type
            print("\nTop features per cell type:")
            for cell_type in correlations_df.columns:
                top_feat = correlations_df[cell_type].abs().idxmax()
                corr_val = correlations_df.loc[top_feat, cell_type]
                print(f"  {cell_type}: Feature {top_feat} (r={corr_val:.3f})")
    else:
        print("\nSkipping cell-type analysis (no cell type labels).")

    # ==================== Visualization ====================
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    plot_dir = os.path.abspath(PLOT_DIR)
    os.makedirs(plot_dir, exist_ok=True)

    if correlations_df is not None:
        # Heatmap of top features vs cell types
        plt.figure(figsize=(12, 10))
        top_corr = correlations_df.iloc[top_features[:30]]
        sns.heatmap(top_corr, cmap='RdBu_r', center=0, annot=False)
        plt.xlabel('Cell Type')
        plt.ylabel('SAE Feature')
        plt.title('Top 30 SAE Features vs Cell Types')
        plt.tight_layout()
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

    print(f"\nNew GO enrichment results saved to {OUTPUT_DIR}")
    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()

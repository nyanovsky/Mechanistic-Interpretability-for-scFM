"""Interpret SAE features via GO enrichment and cell type correlation (PARALLELIZED VERSION).

For each SAE feature:
1. Find genes with highest mean activation across cells (from pre-computed matrices)
2. Run GO enrichment on top genes (PARALLELIZED)
3. Correlate feature activation with cell types

This version parallelizes GO enrichment API calls for significantly faster execution.

IMPORTANT: This script requires pre-computed matrices. Run compute_feature_matrices.py first.

Usage:
    # If raw data has cell_type column:
    python scripts/interpret_sae_parallel.py \\
        --matrix-dir /path/to/interpretations \\
        --raw-data /path/to/data.h5ad \\
        --plot-dir /path/to/plots

    # If cell types are in a separate processed file (e.g., PBMC3K):
    python scripts/interpret_sae_parallel.py \\
        --matrix-dir /path/to/interpretations \\
        --raw-data /path/to/raw.h5ad \\
        --processed-data /path/to/processed.h5ad \\
        --celltype-col louvain \\
        --plot-dir /path/to/plots

Arguments:
    --raw-data: Used for gene alignment via AIDO.Cell's align_adata() and cell type labels if available
    --processed-data: Optional, only needed if raw data lacks the cell type column
    --celltype-col: Column name in .obs for cell type labels (default: cell_type)
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

# Try to import gseapy for GO enrichment
try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    print("WARNING: gseapy not installed. GO enrichment will be skipped.")
    print("Install with: pip install gseapy")
    HAS_GSEAPY = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ModelGenerator/huggingface/aido.cell'))
from aido_cell.utils import align_adata

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Interpret SAE features via GO enrichment (parallel)')

    # Required paths
    parser.add_argument('--matrix-dir', type=str, required=True,
                        help='Directory containing pre-computed matrices (feature_gene_matrix.npy, etc.)')
    parser.add_argument('--raw-data', type=str, required=True,
                        help='Path to raw h5ad file (used for gene alignment via AIDO.Cell)')
    parser.add_argument('--plot-dir', type=str, required=True,
                        help='Directory to save plots')

    # Optional paths
    parser.add_argument('--processed-data', type=str, default=None,
                        help='Path to processed h5ad file with cell type labels (only needed if raw data lacks cell_type column)')

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
MATRIX_DIR = args.matrix_dir
OUTPUT_DIR = MATRIX_DIR  # Save GO enrichment results to same directory
PLOT_DIR = args.plot_dir
RAW_DATA_FILE = args.raw_data
PROCESSED_DATA_FILE = args.processed_data  # Optional, only needed if raw data lacks cell_type
CELLTYPE_COL = args.celltype_col

# Analysis parameters
TOP_GENES_PER_FEATURE = 30  # Number of top genes for GO enrichment (legacy, only used if USE_ADAPTIVE_SELECTION=False)
TOP_FEATURES_TO_ANALYZE = args.n_features  # Analyze top N most active features

# Adaptive gene selection parameters
USE_ADAPTIVE_SELECTION = True  # Set to False for original fixed-30 behavior
MIN_GENES_PER_FEATURE = 10  # Minimum genes for statistical power
MAX_GENES_PER_FEATURE = 100  # Maximum to avoid dilution
PR_SCALE_FACTOR = 0.6  # Select 60% of effective genes (PR * 0.6)
USE_ACTIVATION_THRESHOLD = False  # Optional: filter genes <10% of max activation

print(f"Configuration:")
print(f"  Matrix directory: {MATRIX_DIR}")
print(f"  Raw data: {RAW_DATA_FILE}")
if PROCESSED_DATA_FILE:
    print(f"  Processed data: {PROCESSED_DATA_FILE}")
print(f"  Cell type column: {CELLTYPE_COL}")
print(f"  Plot directory: {PLOT_DIR}")
print(f"  Parallelization: {args.workers} workers, {args.delay}s delay")


def load_gene_names(raw_data_path):
    """Load gene names from original data aligned with AIDO.Cell attention mask.

    Args:
        raw_data_path: Path to raw h5ad file (can be absolute or relative to script)
    """
    # Handle both absolute and relative paths
    if not os.path.isabs(raw_data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_path = os.path.join(script_dir, raw_data_path)

    adata_raw = ad.read_h5ad(raw_data_path)
    # Make gene names unique to avoid reindex errors
    adata_raw.var_names_make_unique()
    adata_aligned, attention_mask = align_adata(adata_raw)

    # Get gene names for valid genes
    gene_names = adata_aligned.var_names[attention_mask.astype(bool)]
    return gene_names.tolist()


def get_expressed_genes(raw_data_path, min_mean_expr=0.01, min_pct_cells=0.5):
    """Return indices and names of genes with sufficient expression.

    Filters out genes with zero/low expression to avoid spurious GO enrichments
    from genes like olfactory receptors that have embeddings but no expression.

    Args:
        raw_data_path: Path to raw h5ad file (can be absolute or relative to script)
        min_mean_expr: Minimum mean expression across cells (default: 0.01)
        min_pct_cells: Minimum % of cells with nonzero expression (default: 0.5%)

    Returns:
        expressed_indices: Indices of expressed genes in the full gene list
        expressed_names: Names of expressed genes
        expressed_mask: Boolean mask for expressed genes
    """
    # Handle both absolute and relative paths
    if not os.path.isabs(raw_data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_path = os.path.join(script_dir, raw_data_path)

    adata_raw = ad.read_h5ad(raw_data_path)
    # Make gene names unique to avoid reindex errors
    adata_raw.var_names_make_unique()
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

    expressed_indices = np.where(expressed_mask)[0]
    expressed_names = [gene_names[i] for i in expressed_indices]

    return expressed_indices, expressed_names, expressed_mask


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


def run_go_enrichment(gene_list, feature_idx, output_dir):
    """Run GO enrichment on a gene list using gseapy."""
    if not HAS_GSEAPY:
        return None

    gene_sets = ['GO_Biological_Process_2021', 'GO_Molecular_Function_2021', 'GO_Cellular_Component_2021']
    all_results = []

    for gs in gene_sets:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gs,
                organism='human',
                outdir=os.path.join(output_dir, f'feature_{feature_idx}_enrichr'),
                cutoff=0.05
            )
            if enr.results is not None and not enr.results.empty:
                all_results.append(enr.results)
        except Exception as e:
            # Silent in parallel mode to avoid cluttering output
            continue

    if not all_results:
        return None

    return pd.concat(all_results, ignore_index=True)


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
    feat_idx, gene_activations, expressed_names, pr_values, delay = task_data

    # Add small delay to be courteous to API
    time.sleep(delay)

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
    result = run_go_enrichment(top_genes, feat_idx, OUTPUT_DIR)

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
    feature_gene_matrix_path = os.path.join(MATRIX_DIR, 'feature_gene_matrix.npy')
    if not os.path.exists(feature_gene_matrix_path):
        print(f"\nERROR: Feature-gene matrix not found at {feature_gene_matrix_path}")
        print("\nPlease run compute_feature_matrices.py first to generate the matrices.")
        return

    print(f"Loading feature-gene matrix from {MATRIX_DIR}...")
    feature_gene_matrix = np.load(feature_gene_matrix_path)
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

    # Load gene names
    print("\nLoading gene names...")
    gene_names = load_gene_names(RAW_DATA_FILE)
    print(f"  Loaded {len(gene_names)} gene names")

    # Get expressed genes only
    print("\nFiltering to expressed genes only...")
    _, expressed_names, expressed_mask = get_expressed_genes(
        RAW_DATA_FILE, min_mean_expr=0.01, min_pct_cells=0.5
    )
    print(f"  Filtered from {len(gene_names)} to {len(expressed_names)} expressed genes")

    # Load cell type labels
    print("\nLoading cell type labels...")

    # First try to get cell types from raw data
    raw_path = RAW_DATA_FILE
    if not os.path.isabs(raw_path):
        raw_path = os.path.join(script_dir, raw_path)
    adata_raw = ad.read_h5ad(raw_path)

    if CELLTYPE_COL in adata_raw.obs.columns:
        print(f"  Found '{CELLTYPE_COL}' column in raw data")
        cell_types = adata_raw.obs[CELLTYPE_COL]
    else:
        # Fall back to processed data
        if PROCESSED_DATA_FILE is None:
            print(f"\nERROR: Raw data does not have '{CELLTYPE_COL}' column and --processed-data was not provided.")
            print(f"  Either add cell type labels to raw data or provide --processed-data")
            return
        print(f"  '{CELLTYPE_COL}' not in raw data, loading from processed data...")
        processed_path = PROCESSED_DATA_FILE
        if not os.path.isabs(processed_path):
            processed_path = os.path.join(script_dir, processed_path)
        adata_processed = ad.read_h5ad(processed_path)
        if CELLTYPE_COL not in adata_processed.obs.columns:
            print(f"\nERROR: '{CELLTYPE_COL}' column not found in processed data either.")
            print(f"  Available columns: {list(adata_processed.obs.columns)}")
            return
        cell_types = adata_processed.obs[CELLTYPE_COL]

    print(f"  Cell types: {cell_types.value_counts().to_dict()}")

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

        gene_activations = feature_gene_matrix[feat_idx][expressed_mask]
        features_to_process.append((feat_idx, gene_activations, expressed_names, pr_values, args.delay))

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
    print("\n" + "="*60)
    print("Cell-Type Analysis")
    print("="*60)

    # Load cell-feature matrix (REQUIRED)
    cell_feature_matrix_path = os.path.join(MATRIX_DIR, 'cell_feature_matrix.npy')
    if not os.path.exists(cell_feature_matrix_path):
        print(f"\nERROR: Cell-feature matrix not found at {cell_feature_matrix_path}")
        print("\nPlease run compute_feature_matrices.py first")
        return

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

    print(f"\nNew GO enrichment results saved to {OUTPUT_DIR}")
    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()

"""Interpret SAE features via GO enrichment and cell type correlation (PARALLELIZED VERSION).

For each SAE feature:
1. Find genes with highest mean activation across cells
2. Run GO enrichment on top genes (PARALLELIZED)
3. Correlate feature activation with cell types

This version parallelizes GO enrichment API calls for significantly faster execution.

IMPORTANT: This script reads pre-computed matrices from interpretations_fixed_30_backup
but saves new GO enrichment results to interpretations_filter_zero_expressed.
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
    parser.add_argument('--layer', type=int, default=12,
                        help='AIDO.Cell layer (12 or 15)')
    parser.add_argument('--expansion', type=int, default=8,
                        help='SAE expansion factor (4 or 8)')
    parser.add_argument('--k', type=int, default=32,
                        help='Top-K sparsity')
    parser.add_argument('--n_features', type=int, default=50,
                        help='Number of top features to analyze for GO enrichment')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers for GO enrichment (default: 8)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between API calls in seconds (default: 0.1)')
    parser.add_argument('--online', action='store_true',
                        help='Use online SAE directory (sae_k_{k}_{latent_dim}_online)')
    return parser.parse_args()

args = parse_args()

# Build paths from arguments
INPUT_DIM = 640
LATENT_DIM = INPUT_DIM * args.expansion
BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
SAE_SUFFIX = "_online" if args.online else ""
SAE_DIR = f"{BASE_DIR}/sae_k_{args.k}_{LATENT_DIM}{SAE_SUFFIX}"

ACTIVATIONS_FILE = f"{BASE_DIR}/layer{args.layer}_activations.h5"
SAE_MODEL_FILE = f"{SAE_DIR}/topk_sae.pt"

# INPUT: Read pre-computed data from fixed_30_backup
INPUT_DIR = f"{SAE_DIR}/interpretations_fixed_30_backup"
# OUTPUT: Save new GO enrichments to this folder
OUTPUT_DIR = f"{SAE_DIR}/interpretations_filter_zero_expressed"

INPUT_DIR = OUTPUT_DIR

PLOT_DIR = f"../plots/sae/layer_{args.layer}"

# Static paths (don't depend on layer/SAE config)
PROCESSED_DATA_FILE = "../data/pbmc/pbmc3k_processed.h5ad"
RAW_DATA_FILE = "../data/pbmc/pbmc3k_raw.h5ad"

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
print(f"Input (pre-computed data): {INPUT_DIR}")
print(f"Output (new GO enrichments): {OUTPUT_DIR}")
print(f"Parallelization: {args.workers} workers, {args.delay}s delay")


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
        # Pure Top-K: no ReLU, sparsity comes from top-k selection
        latents = self.encoder(x)
        topk_vals, topk_idx = latents.topk(self.k, dim=-1)
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(-1, topk_idx, topk_vals)
        return sparse_latents

    def forward(self, x):
        sparse_latents = self.encode(x)
        recon = self.decoder(sparse_latents)
        return recon, sparse_latents


def load_gene_names(h5_path, raw_data_path):
    """Load gene names from original data aligned with activations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, raw_data_path)

    adata_raw = ad.read_h5ad(raw_path)
    adata_aligned, attention_mask = align_adata(adata_raw)

    # Get gene names for valid genes
    gene_names = adata_aligned.var_names[attention_mask.astype(bool)]
    return gene_names.tolist()


def get_expressed_genes(raw_data_path, min_mean_expr=0.01, min_pct_cells=0.5):
    """Return indices and names of genes with sufficient expression.

    Filters out genes with zero/low expression to avoid spurious GO enrichments
    from genes like olfactory receptors that have embeddings but no expression.

    Args:
        raw_data_path: Path to raw h5ad file
        min_mean_expr: Minimum mean expression across cells (default: 0.01)
        min_pct_cells: Minimum % of cells with nonzero expression (default: 0.5%)

    Returns:
        expressed_indices: Indices of expressed genes in the full gene list
        expressed_names: Names of expressed genes
        expressed_mask: Boolean mask for expressed genes
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

    expressed_indices = np.where(expressed_mask)[0]
    expressed_names = [gene_names[i] for i in expressed_indices]

    return expressed_indices, expressed_names, expressed_mask


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

    # Check files exist
    if not os.path.exists(ACTIVATIONS_FILE):
        print(f"ERROR: Activations file not found at {ACTIVATIONS_FILE}")
        return
    if not os.path.exists(SAE_MODEL_FILE):
        print(f"ERROR: SAE model not found at {SAE_MODEL_FILE}")
        return
    if not os.path.exists(INPUT_DIR):
        print(f"Creating output directory at {INPUT_DIR}")
        os.makedirs(INPUT_DIR, exist_ok=True)

    print("="*60)
    print("SAE Feature Interpretation (PARALLELIZED)")
    print("="*60)

    # Load SAE model
    print("\nLoading SAE model...")
    checkpoint = torch.load(SAE_MODEL_FILE, map_location=DEVICE)

    # Handle both checkpoint formats:
    # - Offline: {'model_state_dict': ..., 'input_dim': ..., 'expansion': ..., 'k': ...}
    # - Online: {'encoder.weight': ..., 'encoder.bias': ..., 'decoder.weight': ..., 'decoder.bias': ...}
    if 'model_state_dict' in checkpoint:
        # Offline format
        input_dim = checkpoint['input_dim']
        expansion = checkpoint['expansion']
        k = checkpoint['k']
        state_dict = checkpoint['model_state_dict']
    else:
        # Online format - infer parameters from weight shapes
        # decoder.weight shape: [output_dim, latent_dim] = [640, 5120]
        # encoder.weight shape: [latent_dim, input_dim] = [5120, 640]
        input_dim = checkpoint['decoder.weight'].shape[0]  # 640
        latent_dim = checkpoint['decoder.weight'].shape[1]  # 5120
        expansion = latent_dim // input_dim
        k = args.k  # Use from CLI args
        state_dict = checkpoint

    sae = TopKSAE(input_dim=input_dim, expansion=expansion, k=k).to(DEVICE)
    sae.load_state_dict(state_dict)
    sae.eval()
    print(f"Loaded SAE: {input_dim} -> {sae.latent_dim} (K={k})")

    # Load gene names
    print("\nLoading gene names...")
    gene_names = load_gene_names(ACTIVATIONS_FILE, RAW_DATA_FILE)
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

    # Load or compute feature_gene_matrix
    feature_gene_matrix_path = os.path.join(INPUT_DIR, 'feature_gene_matrix.npy')
    if os.path.exists(feature_gene_matrix_path):
        print(f"Loading pre-computed feature-gene matrix from {INPUT_DIR}...")
        feature_gene_matrix = np.load(feature_gene_matrix_path)
        print(f"Loaded feature-gene matrix shape: {feature_gene_matrix.shape}")
    else:
        print("Computing feature-gene matrix (this may take a while)...")
        feature_gene_matrix = compute_feature_gene_activations(sae, ACTIVATIONS_FILE)
        print(f"Computed feature-gene matrix shape: {feature_gene_matrix.shape}")
        print(f"Saving to {feature_gene_matrix_path}...")
        np.save(feature_gene_matrix_path, feature_gene_matrix)

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
    pr_values = None
    if USE_ADAPTIVE_SELECTION:
        pr_path = os.path.join(INPUT_DIR, 'feature_participation_ratios.npy')
        if not os.path.exists(pr_path):
            print(f"\nERROR: PR values not found at {pr_path}")
            print("\nPlease run analyze_feature_scale.py first to generate the participation ratio values:")
            online_flag = "--online" if args.online else ""
            print(f"  python scripts/analyze_feature_scale.py --layer {args.layer} --expansion {args.expansion} {online_flag}")
            return
        pr_values = np.load(pr_path)
        print(f"Loaded PR values. Using adaptive selection (factor={PR_SCALE_FACTOR}, bounds=[{MIN_GENES_PER_FEATURE},{MAX_GENES_PER_FEATURE}])")

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

    # Copy feature-gene matrix to OUTPUT_DIR if it doesn't exist there
    output_matrix_path = os.path.join(OUTPUT_DIR, 'feature_gene_matrix.npy')
    if not os.path.exists(output_matrix_path):
        print(f"\nCopying feature-gene matrix to {OUTPUT_DIR}")
        np.save(output_matrix_path, feature_gene_matrix)

    # Copy PR values to OUTPUT_DIR if it doesn't exist there
    if USE_ADAPTIVE_SELECTION:
        output_pr_path = os.path.join(OUTPUT_DIR, 'feature_participation_ratios.npy')
        if not os.path.exists(output_pr_path):
            print(f"Copying PR values to {OUTPUT_DIR}")
            np.save(output_pr_path, pr_values)

    # ==================== Cell-Type Analysis ====================
    print("\n" + "="*60)
    print("Cell-Type Analysis")
    print("="*60)

    # Try to load cell_feature_matrix from INPUT_DIR or OUTPUT_DIR
    cell_feature_matrix = None
    correlations_df = None

    for dir_path in [OUTPUT_DIR, INPUT_DIR]:
        try:
            cell_feature_matrix = np.load(os.path.join(dir_path, 'cell_feature_matrix.npy'))
            correlations_df = pd.read_csv(
                os.path.join(dir_path, 'feature_celltype_correlations.csv'),
                index_col=0
            )
            print(f"Loaded cell-feature matrix and correlations from {dir_path}")
            break
        except:
            continue

    if cell_feature_matrix is None:
        print("Computing cell-type correlations...")
        cell_feature_matrix = compute_cell_feature_activations(sae, ACTIVATIONS_FILE, expressed_mask)
        print(f"Cell-feature matrix shape: {cell_feature_matrix.shape}")

        # Compute correlations
        correlations_df = analyze_cell_type_correlations(cell_feature_matrix, cell_types)

        # Save to OUTPUT_DIR
        print(f"Saving cell-feature matrix to {OUTPUT_DIR}")
        np.save(os.path.join(OUTPUT_DIR, 'cell_feature_matrix.npy'), cell_feature_matrix)
        correlations_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_celltype_correlations.csv'))

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

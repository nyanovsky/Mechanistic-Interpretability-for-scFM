"""Data loading and expression filtering utilities for SAE analysis."""

import os
import re
from glob import glob
import numpy as np
import pandas as pd
import torch
import anndata as ad

from aido_cell.utils import align_adata


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gene_names(raw_data_path=None, gene_names_file=None):
    """Load gene names from original data aligned with AIDO.Cell attention mask.

    Supports two modes:
    - From raw h5ad: loads data, aligns with AIDO, extracts gene names
    - From pre-computed file: reads gene names from a text file (one per line)

    Args:
        raw_data_path: Path to raw h5ad file (absolute or relative to scripts/)
        gene_names_file: Path to pre-computed gene names text file
    """
    if gene_names_file is not None:
        with open(gene_names_file) as f:
            return f.read().strip().split('\n')

    if raw_data_path is None:
        raise ValueError("Either raw_data_path or gene_names_file must be provided")

    if not os.path.isabs(raw_data_path):
        # Resolve relative to repo root (two levels up from scripts/utils/)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_data_path = os.path.join(repo_root, raw_data_path)

    adata_raw = ad.read_h5ad(raw_data_path)
    adata_raw.var_names_make_unique()
    adata_aligned, attention_mask = align_adata(adata_raw)

    # Get gene names for valid genes
    gene_names = adata_aligned.var_names[attention_mask.astype(bool)]
    return gene_names.tolist()


def load_decoder_weights(sae_dir):
    """Load decoder weights as (n_features, hidden_dim) array.

    Tries loading 'sae_decoder.pt' first, then falls back to 'topk_sae.pt'.
    """
    decoder_path = os.path.join(sae_dir, 'sae_decoder.pt')
    W_dec = None

    # Try standalone decoder file
    if os.path.exists(decoder_path):
        print(f"Loading decoder from {decoder_path}")
        try:
            decoder = torch.load(decoder_path, map_location='cpu')
            if isinstance(decoder, dict) and 'weight' in decoder:
                 W_dec = decoder['weight'].numpy()
            elif isinstance(decoder, torch.Tensor):
                 W_dec = decoder.numpy()
            else:
                keys = decoder.keys() if isinstance(decoder, dict) else []
                if 'state_dict' in keys:
                     W_dec = decoder['state_dict']['decoder.weight'].numpy()
        except Exception as e:
            print(f"Warning: Failed to load {decoder_path}: {e}")

    # Fallback to checkpoint
    if W_dec is None:
        checkpoint_path = os.path.join(sae_dir, 'topk_sae.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading decoder from checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                W_dec = checkpoint['model_state_dict']['decoder.weight'].numpy()
            elif 'state_dict' in checkpoint:
                W_dec = checkpoint['state_dict']['decoder.weight'].numpy()

    if W_dec is None:
        raise FileNotFoundError(f"Could not load decoder weights from {sae_dir}")

    # Standardize shape: ensure (n_features, hidden_dim)
    if W_dec.shape[0] < W_dec.shape[1]:
         W_dec = W_dec.T

    return W_dec


def load_go_enrichment(interpretations_dir, p_threshold=0.05, mode='terms'):
    """Load GO enrichment results.

    Args:
        interpretations_dir (str): Directory containing interpretation results
        p_threshold (float): Adjusted p-value threshold
        mode (str): 'terms' (returns full term strings) or 'ids' (returns GO:XXXXX IDs)

    Returns:
        dict: {feature_id: set(terms/ids)}
    """
    feature_data = {}
    pattern = os.path.join(interpretations_dir, 'feature_*_enrichr')
    feature_dirs = glob(pattern)

    go_id_regex = re.compile(r'\(GO:(\d+)\)')

    for feature_dir in feature_dirs:
        match = re.search(r'feature_(\d+)_enrichr', feature_dir)
        if not match:
            continue
        feature_id = int(match.group(1))
        items = set()

        for go_file in glob(os.path.join(feature_dir, 'GO_*_2021.human.enrichr.reports.txt')):
            try:
                df = pd.read_csv(go_file, sep='\t')
                if 'Adjusted P-value' not in df.columns or 'Term' not in df.columns:
                    continue

                significant = df[df['Adjusted P-value'] < p_threshold]

                for term in significant['Term']:
                    if mode == 'ids':
                        id_match = go_id_regex.search(term)
                        if id_match:
                            items.add(f"GO:{id_match.group(1)}")
                    else:
                        items.add(term)
            except Exception:
                continue

        if items:
            feature_data[feature_id] = items

    print(f"Loaded GO enrichment ({mode}) for {len(feature_data)} features")
    return feature_data


def load_celltype_correlations(interpretations_dir):
    """Load feature-celltype correlation matrix."""
    corr_path = os.path.join(interpretations_dir, 'feature_celltype_correlations.csv')
    if os.path.exists(corr_path):
        df = pd.read_csv(corr_path, index_col=0)
        print(f"Loaded celltype correlations: {df.shape}")
        return df
    print(f"Warning: Celltype correlations not found at {corr_path}")
    return None


def load_activation_matrices(interpretations_dir):
    """Load feature-gene and cell-feature activation matrices.

    Returns:
        tuple: (feature_gene_matrix, cell_feature_matrix)
    """
    feature_gene_path = os.path.join(interpretations_dir, 'feature_gene_matrix.npy')
    cell_feature_path = os.path.join(interpretations_dir, 'cell_feature_matrix.npy')

    fg_mat = None
    cf_mat = None

    if os.path.exists(feature_gene_path):
        print(f"Loading feature-gene matrix from {feature_gene_path}...")
        fg_mat = np.load(feature_gene_path)
        print(f"  Shape: {fg_mat.shape}")
    else:
        print(f"Warning: {feature_gene_path} not found")

    if os.path.exists(cell_feature_path):
        print(f"Loading cell-feature matrix from {cell_feature_path}...")
        cf_mat = np.load(cell_feature_path)
        print(f"  Shape: {cf_mat.shape}")
    else:
        print(f"Warning: {cell_feature_path} not found")

    return fg_mat, cf_mat


def get_expressed_genes(raw_data_path=None, min_mean_expr=0.01, min_pct_cells=0.5,
                        gene_names_file=None, expressed_mask_file=None):
    """Return indices, names, and mask for genes with sufficient expression.

    Supports two modes:
    - From raw h5ad: loads data, aligns, computes expression statistics
    - From pre-computed files: loads gene names and expressed mask from disk

    Args:
        raw_data_path: Path to raw h5ad file (absolute or relative to scripts/)
        min_mean_expr: Minimum mean expression across cells (default: 0.01)
        min_pct_cells: Minimum % of cells with nonzero expression (default: 0.5%)
        gene_names_file: Path to pre-computed expressed gene names (one per line)
        expressed_mask_file: Path to pre-computed expressed mask (.npy boolean array)

    Returns:
        expressed_indices: Indices of expressed genes in the full gene list (None if pre-computed)
        expressed_names: Names of expressed genes
        expressed_mask: Boolean mask for expressed genes (None if pre-computed without mask file)
    """
    if gene_names_file is not None:
        with open(gene_names_file) as f:
            expressed_names = f.read().strip().split('\n')
        expressed_mask = None
        if expressed_mask_file is not None:
            expressed_mask = np.load(expressed_mask_file)
        expressed_indices = np.where(expressed_mask)[0] if expressed_mask is not None else None
        return expressed_indices, expressed_names, expressed_mask

    if raw_data_path is None:
        raise ValueError("Either raw_data_path or gene_names_file must be provided")

    if not os.path.isabs(raw_data_path):
        # Resolve relative to repo root (two levels up from scripts/utils/)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_data_path = os.path.join(repo_root, raw_data_path)

    adata_raw = ad.read_h5ad(raw_data_path)
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


def get_expressed_genes_mask(raw_data_path, min_mean_expr=0.01, min_pct_cells=0.5):
    """Return boolean mask for genes with sufficient expression.

    Convenience wrapper around get_expressed_genes() that returns only the mask.
    """
    _, _, expressed_mask = get_expressed_genes(raw_data_path, min_mean_expr, min_pct_cells)
    return expressed_mask


def load_feature_statistics(interpretation_dir: str):
    """Load feature activation statistics from feature-gene matrix.

    Args:
        interpretation_dir: Path to SAE interpretation directory

    Returns:
        Dict with 'max', 'mean', 'median', 'std', 'p90' tensors of shape (n_features,)
    """
    feature_gene_path = os.path.join(interpretation_dir, "feature_gene_matrix.npy")

    if not os.path.exists(feature_gene_path):
        raise FileNotFoundError(f"Gene-feature matrix not found at {feature_gene_path}")

    feature_gene_matrix = np.load(feature_gene_path)
    print(f"Loaded gene-feature matrix with shape: {feature_gene_matrix.shape}")

    # Handle shape: want (n_features, n_genes)
    if feature_gene_matrix.shape[0] != 5120:
        print("Detected shape (n_genes, n_features), transposing...")
        feature_gene_matrix = feature_gene_matrix.T

    n_features, n_genes = feature_gene_matrix.shape
    print(f"Working with {n_features} features across {n_genes} genes")

    # Compute statistics across cells (axis=1)
    stats = {
        'mean': torch.tensor(np.mean(feature_gene_matrix, axis=1), dtype=torch.float32),
        'median': torch.tensor(np.median(feature_gene_matrix, axis=1), dtype=torch.float32),
        'max': torch.tensor(np.max(feature_gene_matrix, axis=1), dtype=torch.float32),
        'std': torch.tensor(np.std(feature_gene_matrix, axis=1), dtype=torch.float32),
        'p90': torch.tensor(np.percentile(feature_gene_matrix, 90, axis=1), dtype=torch.float32),
    }

    print("\nFeature activation statistics:")
    print(f"  Max - range: [{stats['max'].min():.4f}, {stats['max'].max():.4f}]")
    print(f"  Mean - range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
    print(f"  Mean/Max ratio: {(stats['mean'] / (stats['max'] + 1e-8)).mean():.3f}")

    return stats

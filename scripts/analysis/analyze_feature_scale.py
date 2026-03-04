"""Analyze scale/sparsity of SAE features using Participation Ratio.

Computes PR for each feature in a feature-gene matrix and saves the results.

Usage:
    # Basic usage (all genes)
    python scripts/analyze_feature_scale.py \\
        --matrix-path /path/to/feature_gene_matrix.npy \\
        --plot-dir /path/to/plots

    # Filter to expressed genes only
    python scripts/analyze_feature_scale.py \\
        --matrix-path /path/to/feature_gene_matrix.npy \\
        --plot-dir /path/to/plots \\
        --filter-expressed \\
        --raw-data /path/to/raw.h5ad
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import get_expressed_genes_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze scale/sparsity of SAE features using Participation Ratio')
    parser.add_argument('--matrix-path', type=str, required=True,
                        help='Path to feature_gene_matrix.npy')
    parser.add_argument('--plot-dir', type=str, required=True,
                        help='Directory to save plots')
    parser.add_argument('--filter-expressed', action='store_true',
                        help='Filter to expressed genes only before computing PR')
    parser.add_argument('--raw-data', type=str, default=None,
                        help='Path to raw h5ad file (required if --filter-expressed)')
    parser.add_argument('--min-mean-expr', type=float, default=0.01,
                        help='Minimum mean expression threshold (default: 0.01)')
    parser.add_argument('--min-pct-cells', type=float, default=0.5,
                        help='Minimum %% of cells with nonzero expression (default: 0.5)')
    return parser.parse_args()


def main():
    args = parse_args()

    matrix_path = args.matrix_path
    matrix_dir = os.path.dirname(matrix_path)
    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Analyzing Feature Scale...")
    print(f"  Matrix path: {matrix_path}")
    print(f"  Plot dir: {plot_dir}")
    print(f"  Filter expressed: {args.filter_expressed}")

    if args.filter_expressed and args.raw_data is None:
        print("Error: --raw-data is required when using --filter-expressed")
        return

    if not os.path.exists(matrix_path):
        print(f"Error: Matrix not found at {matrix_path}")
        return

    # 1. Load Data
    # Matrix shape: [n_features, n_genes]
    print("Loading feature-gene matrix...")
    fg_matrix = np.load(matrix_path)
    n_features, n_genes_total = fg_matrix.shape
    print(f"Shape: {n_features} features x {n_genes_total} genes")

    # 2. Optionally filter to expressed genes
    if args.filter_expressed:
        print(f"\nFiltering to expressed genes (min_mean_expr={args.min_mean_expr}, min_pct_cells={args.min_pct_cells})...")
        expressed_mask = get_expressed_genes_mask(
            args.raw_data,
            min_mean_expr=args.min_mean_expr,
            min_pct_cells=args.min_pct_cells
        )
        fg_matrix = fg_matrix[:, expressed_mask]
        n_genes = fg_matrix.shape[1]
        print(f"Filtered: {n_genes_total} -> {n_genes} genes ({n_genes/n_genes_total*100:.1f}%)")
    else:
        n_genes = n_genes_total

    # 3. Compute Participation Ratio (PR)
    print("Computing Participation Ratio...")
    
    # Square the activations to get "energy"
    energy = fg_matrix ** 2
    
    # Normalize to get probability distribution P_i for each feature
    # Sum over genes (axis 1)
    energy_sum = energy.sum(axis=1, keepdims=True)
    
    # Avoid division by zero for dead features
    energy_sum[energy_sum == 0] = 1.0 
    
    probs = energy / energy_sum
    
    # PR = 1 / Sum(P_i^2)
    # Sum of squares of probabilities (Inverse Simpson Index)
    ipr = (probs ** 2).sum(axis=1)
    
    # Handle dead features (ipr will be 0 if energy was 0)
    pr = np.zeros_like(ipr)
    mask = ipr > 0
    pr[mask] = 1.0 / ipr[mask]
    
    # 4. Statistics
    print("\nFeature Scale Statistics (Effective Gene Count):")
    print(f"Min PR: {pr.min():.2f}")
    print(f"Max PR: {pr.max():.2f}")
    print(f"Mean PR: {pr.mean():.2f}")
    print(f"Median PR: {np.median(pr):.2f}")

    # 5. Visualization
    plt.figure(figsize=(8, 5))

    sns.histplot(pr, bins=100, log_scale=(True, False))
    plt.xlabel('Effective Gene Count (PR)')
    plt.ylabel('Count')
    title = 'Distribution of Feature Scales'
    if args.filter_expressed:
        title += f' (Expressed Genes Only, n={n_genes})'
    plt.title(title)
    plt.axvline(np.median(pr), color='r', linestyle='--', label=f'Median: {np.median(pr):.1f}')
    plt.legend()
    plt.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plot_suffix = '_expressed' if args.filter_expressed else ''
    save_path = os.path.join(plot_dir, f'feature_pr_distribution{plot_suffix}.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")

    # 6. Examples
    print("\nExamples of Specific Features (Low PR):")
    indices = np.argsort(pr)
    
    # Filter out dead features (PR=0)
    valid_indices = [i for i in indices if pr[i] >= 1.0]
    
    for i in valid_indices[:10]:
        print(f"Feature {i:<4} | PR: {pr[i]:.2f}")
        
    print("\nExamples of Pathway Features (PR ~ 20-50):")
    # Find features closest to PR=30
    mid_indices = [i for i in valid_indices if 20 < pr[i] < 50]
    for i in mid_indices[:10]:
        print(f"Feature {i:<4} | PR: {pr[i]:.2f}")

    print("\nExamples of Global Features (High PR):")
    for i in valid_indices[-10:]:
        print(f"Feature {i:<4} | PR: {pr[i]:.2f}")

    # Save PRs to file for later use
    pr_path = os.path.join(matrix_dir, 'feature_participation_ratios.npy')
    np.save(pr_path, pr)
    print(f"\nPR values saved to {pr_path}")

if __name__ == "__main__":
    main()

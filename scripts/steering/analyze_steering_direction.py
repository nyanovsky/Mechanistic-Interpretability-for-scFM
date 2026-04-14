"""
Analyze directional consistency of steering in gene expression space.

For each gene, checks whether steering moved CD4 expression toward CD8 (correct direction),
and by how much (gap fraction: 0 = no change, 1 = perfect match to CD8 mean).

Two framings:
1. Cell-type DE genes: genes significantly different between CD8 and CD4 at baseline
2. Steering DE genes: genes significantly changed by steering (paired test on CD4 cells)

Both framings use the CD8-CD4 baseline difference as the expected direction.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import get_expressed_genes_mask, compute_de_genes, compute_independent_de

import anndata as ad
from aido_cell.utils import align_adata


def load_data(args):
    """Load and align all data sources."""
    # Load baseline logits h5ad
    print("Loading baseline logits...")
    baseline_adata = ad.read_h5ad(args.baseline_file)

    # Load steering data
    print("Loading steering data...")
    cached = torch.load(args.steering_data, map_location='cpu')
    post_logits_source = cached['post_logits_source']
    cell_types = cached['cell_types']
    cell_names = cached['cell_names']

    if isinstance(post_logits_source, torch.Tensor):
        post_logits_source = post_logits_source.float().numpy()

    # Identify source (CD4) and target (CD8) cells
    source_mask = cell_types == args.source_celltype
    target_mask = cell_types == args.target_celltype

    source_names = [cell_names[i] for i in range(len(cell_names)) if source_mask[i]]
    target_names = [cell_names[i] for i in range(len(cell_names)) if target_mask[i]]

    print(f"  Source ({args.source_celltype}): {len(source_names)} cells")
    print(f"  Target ({args.target_celltype}): {len(target_names)} cells")

    # Filter to cells present in baseline
    source_in_baseline = [n for n in source_names if n in baseline_adata.obs_names]
    target_in_baseline = [n for n in target_names if n in baseline_adata.obs_names]
    print(f"  Source in baseline: {len(source_in_baseline)}, Target in baseline: {len(target_in_baseline)}")

    # Get baseline logits for source and target cells
    baseline_source = baseline_adata[source_in_baseline].X
    baseline_target = baseline_adata[target_in_baseline].X

    if hasattr(baseline_source, 'toarray'):
        baseline_source = baseline_source.toarray()
    if hasattr(baseline_target, 'toarray'):
        baseline_target = baseline_target.toarray()

    # Align steered logits to the same source cell order
    # post_logits_source is ordered by source_mask over cell_names
    source_name_to_steered_idx = {name: i for i, name in enumerate(source_names)}
    steered_indices = [source_name_to_steered_idx[n] for n in source_in_baseline]
    steered_source = post_logits_source[steered_indices]

    # Gene names and expression mask
    gene_names = np.array(baseline_adata.var_names)
    expr_mask = get_expressed_genes_mask(args.raw_data_file)

    print(f"  Genes: {len(gene_names)}, Expressed: {expr_mask.sum()}")

    return baseline_source, baseline_target, steered_source, gene_names, expr_mask


def compute_direction_metrics(baseline_source, baseline_target, steered_source, gene_mask):
    """Compute directional consistency and gap fraction for a set of genes.

    Args:
        baseline_source: [n_source, n_genes] baseline source logits (subset to gene_mask)
        baseline_target: [n_target, n_genes] baseline target logits (subset to gene_mask)
        steered_source: [n_source, n_genes] steered source logits (subset to gene_mask)
        gene_mask: boolean mask for which genes to analyze

    Returns:
        DataFrame with per-gene metrics
    """
    # Means
    cd4_mean = baseline_source[:, gene_mask].mean(axis=0)
    cd8_mean = baseline_target[:, gene_mask].mean(axis=0)
    steered_mean = steered_source[:, gene_mask].mean(axis=0)

    expected_direction = np.sign(cd8_mean - cd4_mean)
    actual_change = steered_mean - cd4_mean
    correct = np.sign(actual_change) == expected_direction

    # Gap fraction from population means: (steered_mean - CD4_mean) / (CD8_mean - CD4_mean)
    denominator = cd8_mean - cd4_mean
    denom_clamped = np.where(np.abs(denominator) < 1e-6, np.sign(denominator + 1e-12) * 1e-6, denominator)
    gap_fraction = (steered_mean - cd4_mean) / denom_clamped

    return pd.DataFrame({
        'CD4_mean': cd4_mean,
        'CD8_mean': cd8_mean,
        'steered_CD4_mean': steered_mean,
        'expected_direction': expected_direction,
        'actual_change': actual_change,
        'correct': correct,
        'gap_fraction': gap_fraction,
    })


def plot_gap_fraction_histogram(df, title, output_path, n_genes, p_value):
    """Plot gap fraction histogram (ICLR-ready)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    gap = df['gap_fraction'].values
    gap_clipped = np.clip(gap, -3, 5)

    ax.hist(gap_clipped, bins=80, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.2, label='No change')
    ax.axvline(1, color='green', linestyle='--', linewidth=1.2, label='Target mean')

    ax.set_xlabel('Gap fraction', fontsize=10)
    ax.set_ylabel('Number of genes', fontsize=10)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cumulative_direction_curves(df, rank_col, output_dir, framing_name):
    """Plot cumulative % correct and median gap fraction as separate PDFs (ICLR-ready).

    Genes are ranked by descending |rank_col|, then metrics are computed
    cumulatively as more genes are included.
    """
    # Sort by descending absolute effect size
    order = df[rank_col].abs().sort_values(ascending=False).index
    df_sorted = df.loc[order].reset_index(drop=True)

    n = len(df_sorted)
    ks = np.arange(1, n + 1)

    # Cumulative % correct
    cum_correct = df_sorted['correct'].cumsum().values / ks * 100

    # Cumulative median gap fraction (rolling median)
    gap_vals = df_sorted['gap_fraction'].values
    cum_median = np.array([np.median(gap_vals[:k]) for k in ks])

    # --- Plot 1: % correct direction ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, cum_correct, color='steelblue', linewidth=1.5)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Chance (50%)')
    ax.set_xlabel('Genes included (ranked by effect size)', fontsize=10)
    ax.set_ylabel('Cumulative % correct direction', fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path1 = os.path.join(output_dir, f'cumulative_correct_{framing_name}.pdf')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # --- Plot 2: median gap fraction ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, cum_median, color='coral', linewidth=1.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No change')
    ax.axhline(1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target mean')
    ax.set_xlabel('Genes included (ranked by effect size)', fontsize=10)
    ax.set_ylabel('Cumulative median gap fraction', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path2 = os.path.join(output_dir, f'cumulative_gap_{framing_name}.pdf')
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")


def analyze_framing(baseline_source, baseline_target, steered_source,
                    gene_names, sig_mask, framing_name, rank_col, output_dir):
    """Run directional analysis for one framing.

    Args:
        rank_col: Column name for ranking genes in cumulative curve.
            'celltype_effect' (|CD8-CD4|) or 'actual_change' (|steered-baseline|).
    """
    n_sig = sig_mask.sum()
    print(f"\n=== {framing_name} ===")
    print(f"  Significant genes: {n_sig}")

    if n_sig == 0:
        print("  No significant genes, skipping.")
        return None

    # Compute metrics
    df = compute_direction_metrics(baseline_source, baseline_target, steered_source, sig_mask)
    df.insert(0, 'gene', gene_names[sig_mask])

    # Add cell-type effect size column for ranking
    df['celltype_effect'] = df['CD8_mean'] - df['CD4_mean']

    # Binomial test (all genes)
    n_correct = int(df['correct'].sum())
    result = binomtest(n_correct, n_sig, 0.5, alternative='greater')

    # Print summary
    pct_correct = n_correct / n_sig * 100
    median_gap = df['gap_fraction'].median()
    pct_overshoot = (df['gap_fraction'] > 1).mean() * 100
    print(f"  Correct direction: {n_correct}/{n_sig} ({pct_correct:.1f}%)")
    print(f"  Median gap fraction: {median_gap:.3f}")
    print(f"  Overshoot (>1): {pct_overshoot:.1f}%")
    print(f"  Binomial test p-value: {result.pvalue:.2e}")

    # Save CSV
    csv_path = os.path.join(output_dir, f'direction_{framing_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Split histograms by direction (up in CD8 vs down in CD8)
    for direction, label, mask in [
        ('up', 'Up in target (CD8 > CD4)', df['expected_direction'] > 0),
        ('down', 'Down in target (CD8 < CD4)', df['expected_direction'] < 0),
    ]:
        df_dir = df[mask]
        if len(df_dir) == 0:
            continue
        n_dir = len(df_dir)
        n_corr_dir = int(df_dir['correct'].sum())
        p_dir = binomtest(n_corr_dir, n_dir, 0.5, alternative='greater').pvalue
        plot_path = os.path.join(output_dir, f'gap_fraction_{framing_name}_{direction}.pdf')
        plot_gap_fraction_histogram(
            df_dir, f'{framing_name}: {label}', plot_path, n_dir, p_dir
        )

    # Cumulative direction curves (separate PDFs)
    plot_cumulative_direction_curves(df, rank_col, output_dir, framing_name)

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze steering directional consistency')
    parser.add_argument('--steering_data', required=True,
                        help='Path to post_steer_embeddings.pt')
    parser.add_argument('--baseline_file', required=True,
                        help='Path to baseline logits h5ad')
    parser.add_argument('--raw_data_file', default=None,
                        help='Path to raw h5ad (for expression mask). Defaults to data/pbmc/pbmc3k_raw.h5ad')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for plots and CSVs')
    parser.add_argument('--source_celltype', default='CD4 T cells', help='Source cell type')
    parser.add_argument('--target_celltype', default='CD8 T cells', help='Target cell type')
    args = parser.parse_args()

    if args.raw_data_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.raw_data_file = 'data/pbmc/pbmc3k_raw.h5ad'

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    baseline_source, baseline_target, steered_source, gene_names, expr_mask = load_data(args)

    # Filter to expressed genes for all analyses
    baseline_src_expr = baseline_source[:, expr_mask]
    baseline_tgt_expr = baseline_target[:, expr_mask]
    steered_src_expr = steered_source[:, expr_mask]
    names_expr = gene_names[expr_mask]

    # --- Cell-type DE genes (independent t-test) ---
    print("\nComputing cell-type DE (CD8 vs CD4, independent t-test)...")
    _, _, celltype_de_stats = compute_independent_de(
        baseline_tgt_expr, baseline_src_expr, names_expr
    )
    celltype_sig_mask_expr = celltype_de_stats['sig_mask']
    celltype_sig_full = np.zeros(len(gene_names), dtype=bool)
    celltype_sig_full[np.where(expr_mask)[0][celltype_sig_mask_expr]] = True

    # --- Steering DE genes (paired t-test) ---
    print("Computing steering DE (steered vs baseline CD4, paired t-test)...")
    _, _, steering_de_stats = compute_de_genes(
        steered_src_expr, baseline_src_expr, names_expr
    )
    steering_sig_mask_expr = steering_de_stats['sig_mask']
    steering_sig_full = np.zeros(len(gene_names), dtype=bool)
    steering_sig_full[np.where(expr_mask)[0][steering_sig_mask_expr]] = True

    # --- Intersection: genes that are both celltype DE and steering DE ---
    intersection_sig = celltype_sig_full & steering_sig_full
    print(f"\nGene counts:")
    print(f"  Celltype DE: {celltype_sig_full.sum()}")
    print(f"  Steering DE: {steering_sig_full.sum()}")
    print(f"  Intersection: {intersection_sig.sum()}")

    analyze_framing(baseline_source, baseline_target, steered_source,
                    gene_names, intersection_sig, 'intersection_DE',
                    rank_col='celltype_effect', output_dir=args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()

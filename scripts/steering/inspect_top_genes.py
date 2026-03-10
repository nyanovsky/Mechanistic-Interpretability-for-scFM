"""
Inspect top genes by cell-type effect size to understand why steering
doesn't preferentially affect the largest CD8-CD4 differences.

Reads the CSV output of analyze_steering_direction.py and produces:
- Summary stats comparing top N vs rest
- Scatter plot: celltype effect vs gap fraction
- Gap fraction histogram for top N genes
- GO enrichment on top N genes
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.go_utils import run_go_enrichment


def print_comparison(df_top, df_rest, top_n):
    """Print summary stats comparing top N genes vs the rest."""
    print(f"\n{'='*60}")
    print(f"Top {top_n} genes by |celltype_effect| vs rest ({len(df_rest)} genes)")
    print(f"{'='*60}")

    for label, df in [('Top', df_top), ('Rest', df_rest)]:
        n = len(df)
        pct_correct = df['correct'].mean() * 100
        median_gap = df['gap_fraction'].median()
        mean_gap = df['gap_fraction'].mean()
        median_abs_change = df['actual_change'].abs().median()
        mean_abs_change = df['actual_change'].abs().mean()
        near_zero = (df['gap_fraction'].abs() < 0.1).mean() * 100
        overshoot = (df['gap_fraction'] > 1).mean() * 100

        print(f"\n  {label} (N={n}):")
        print(f"    Correct direction: {pct_correct:.1f}%")
        print(f"    Gap fraction — median: {median_gap:.3f}, mean: {mean_gap:.3f}")
        print(f"    |actual_change| — median: {median_abs_change:.4f}, mean: {mean_abs_change:.4f}")
        print(f"    Near zero (|gap| < 0.1): {near_zero:.1f}%")
        print(f"    Overshoot (gap > 1): {overshoot:.1f}%")

    # Median celltype effect size for context
    print(f"\n  |celltype_effect| — Top median: {df_top['celltype_effect'].abs().median():.4f}, "
          f"Rest median: {df_rest['celltype_effect'].abs().median():.4f}")


def plot_scatter(df, top_mask, output_path):
    """Scatter plot: celltype_effect vs gap_fraction, highlighting top genes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot rest first (background)
    rest = df[~top_mask]
    ax.scatter(rest['celltype_effect'], rest['gap_fraction'],
               s=3, alpha=0.3, color='gray', label=f'Rest ({(~top_mask).sum()})')

    # Plot top genes on top
    top = df[top_mask]
    ax.scatter(top['celltype_effect'], top['gap_fraction'],
               s=15, alpha=0.7, color='red', label=f'Top {top_mask.sum()}')

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(1, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='CD8 mean')
    ax.set_xlabel('Cell-type effect (CD8_mean - CD4_mean)')
    ax.set_ylabel('Gap fraction')
    ax.set_title('Cell-type Effect Size vs Gap Fraction')
    ax.legend(fontsize=8)

    # Clip y for visualization
    y_lo, y_hi = np.percentile(df['gap_fraction'], [1, 99])
    ax.set_ylim(y_lo - 0.5, y_hi + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_top_histogram(df_top, top_n, output_path):
    """Gap fraction histogram for top N genes."""
    from scipy.stats import binomtest

    fig, ax = plt.subplots(figsize=(8, 5))
    gap = df_top['gap_fraction'].values
    gap_clipped = np.clip(gap, -3, 5)

    ax.hist(gap_clipped, bins=30, color='indianred', alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No change')
    ax.axvline(1, color='green', linestyle='--', linewidth=1.5, label='CD8 mean')
    ax.axvline(np.median(gap), color='orange', linestyle='-', linewidth=2,
               label=f'Median = {np.median(gap):.3f}')

    n_correct = int(df_top['correct'].sum())
    n = len(df_top)
    p = binomtest(n_correct, n, 0.5, alternative='greater').pvalue

    annotation = (
        f"N genes = {n}\n"
        f"Median gap = {np.median(gap):.3f}\n"
        f"Correct = {n_correct/n*100:.1f}%\n"
        f"Binomial p = {p:.2e}"
    )
    ax.text(0.97, 0.97, annotation, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Gap Fraction')
    ax.set_ylabel('Number of genes')
    ax.set_title(f'Gap Fraction: Top {top_n} genes by |celltype effect|')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Inspect top genes by cell-type effect size')
    parser.add_argument('--csv', required=True, help='Path to direction_intersection_DE.csv')
    parser.add_argument('--top_n', type=int, default=100, help='Number of top genes to inspect')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--raw_data_file', default=None,
                        help='Path to raw h5ad (for expressed genes background). Defaults to ../../data/pbmc/pbmc3k_raw.h5ad')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and sort
    df = pd.read_csv(args.csv)
    df['abs_celltype_effect'] = df['celltype_effect'].abs()
    df = df.sort_values('abs_celltype_effect', ascending=False).reset_index(drop=True)

    top_mask = pd.Series(False, index=df.index)
    top_mask.iloc[:args.top_n] = True

    df_top = df[top_mask]
    df_rest = df[~top_mask]

    # Summary stats
    print_comparison(df_top, df_rest, args.top_n)

    # Print top 20 genes
    print(f"\n  Top 20 genes:")
    for _, row in df_top.head(20).iterrows():
        direction = "+" if row['expected_direction'] > 0 else "-"
        correct = "Y" if row['correct'] else "N"
        print(f"    {row['gene']:12s}  effect={row['celltype_effect']:+.4f}  "
              f"gap={row['gap_fraction']:+.3f}  correct={correct}")

    # Plots
    plot_scatter(df, top_mask, os.path.join(args.output_dir, 'top_genes_scatter.pdf'))
    plot_top_histogram(df_top, args.top_n, os.path.join(args.output_dir, 'top_genes_gap_fraction.pdf'))

    # GO enrichment (background = all expressed genes)
    print("\nRunning GO enrichment on top genes...")
    top_genes = df_top['gene'].tolist()
    from utils.data_utils import get_expressed_genes

    raw_data_path = args.raw_data_file
    if raw_data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_path = os.path.join(script_dir, '../../data/pbmc/pbmc3k_raw.h5ad')
    _, expressed_names, _ = get_expressed_genes(raw_data_path)
    run_go_enrichment(top_genes, args.output_dir, background=expressed_names,
                      identifier=f'top_{args.top_n}_by_effect')

    print("\nDone.")


if __name__ == '__main__':
    main()

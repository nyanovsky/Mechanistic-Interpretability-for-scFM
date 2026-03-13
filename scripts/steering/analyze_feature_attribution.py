"""
Analyze which SAE features are associated with top DE genes and whether
those features are being steered.

For each gene, computes a gene-centric participation ratio (PR) to determine
the effective number of features encoding it, then uses PR-adaptive thresholding
to identify the top features per gene. Cross-references with the steering alpha
vector to check whether those features are being steered.

Answers two hypotheses for why top DE genes are unmoved by steering:
1. Representation gap: top DE genes aren't strongly encoded by SAE features
2. Steering misses relevant features: features exist but aren't being steered
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.similarity import get_top_k_genes_per_feature
from utils.data_utils import load_gene_names
from interpretation.compute_feature_matrices import compute_participation_ratio


def load_data(args):
    """Load feature-gene matrix, alpha vector, gene names, and direction CSV."""
    print("Loading feature-gene matrix...")
    fg_matrix = np.load(args.feature_gene_matrix)
    if fg_matrix.shape[0] != 5120:
        fg_matrix = fg_matrix.T
    n_features, n_genes = fg_matrix.shape
    print(f"  Shape: {fg_matrix.shape} (features x genes)")

    print("Loading alpha vector...")
    alpha_data = torch.load(args.alpha_vector, map_location='cpu')
    alpha = alpha_data['alpha_vector'].numpy()
    print(f"  Non-identity features (|a-1| > 0.05): {(np.abs(alpha - 1) > 0.05).sum()}")

    print("Loading direction CSV...")
    df_direction = pd.read_csv(args.direction_csv)
    print(f"  Genes in CSV: {len(df_direction)}")

    print("Loading gene names...")
    gene_names = load_gene_names(
        raw_data_path=args.raw_data_file,
        gene_names_file=args.gene_names_file
    )
    print(f"  Total genes: {len(gene_names)}")

    return fg_matrix, alpha, df_direction, gene_names


def compute_gene_feature_sets(fg_matrix, min_features=5, max_features=200):
    """Compute top features per gene using gene-centric PR-adaptive thresholding.

    Transposes the feature-gene matrix to (n_genes, n_features), computes PR
    per gene, then uses the same adaptive thresholding as feature-gene sets.
    """
    gene_matrix = fg_matrix.T  # (n_genes, n_features)
    n_genes, n_features = gene_matrix.shape

    print("Computing gene-centric participation ratios...")
    gene_pr = compute_participation_ratio(gene_matrix, axis=1)
    print(f"  Gene PR range: [{gene_pr.min():.1f}, {gene_pr.max():.1f}], "
          f"median: {np.median(gene_pr):.1f}")

    print("Computing top features per gene (PR-adaptive)...")
    feature_sets = get_top_k_genes_per_feature(
        gene_matrix, gene_pr, pr_scale=1,
        min_genes=min_features, max_genes=max_features
    )

    set_sizes = [len(s) for s in feature_sets]
    print(f"  Feature set sizes: min={min(set_sizes)}, max={max(set_sizes)}, "
          f"median={np.median(set_sizes):.0f}, mean={np.mean(set_sizes):.1f}")

    return feature_sets, gene_pr


def compute_attribution_metrics(df_direction, gene_names, feature_sets, gene_pr, alpha,
                                fg_matrix, steered_threshold=0.05):
    """Compute per-gene feature attribution metrics.

    For each gene in the direction CSV:
    - gene_pr: participation ratio (effective number of features)
    - n_top_features: number of features in PR-adaptive top-k
    - n_steered_features: how many of those have |alpha - 1| > threshold
    - steered_fraction: n_steered / n_top
    - weighted_steered_fraction: fraction of activation mass from steered features
    """
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    steered_mask = np.abs(alpha - 1) > steered_threshold

    records = []
    for _, row in df_direction.iterrows():
        gene = row['gene']
        if gene not in gene_to_idx:
            continue

        g_idx = gene_to_idx[gene]
        feat_set = feature_sets[g_idx]
        n_top = len(feat_set)

        feat_indices = sorted(feat_set)
        n_steered = sum(1 for f in feat_indices if steered_mask[f])
        steered_frac = n_steered / n_top if n_top > 0 else 0.0

        # Activation-weighted steered fraction
        gene_activations = fg_matrix[:, g_idx]
        total_act = sum(gene_activations[f] for f in feat_indices)
        steered_act = sum(gene_activations[f] for f in feat_indices if steered_mask[f])
        weighted_steered_frac = steered_act / total_act if total_act > 0 else 0.0

        records.append({
            'gene': gene,
            'gene_pr': gene_pr[g_idx],
            'n_top_features': n_top,
            'n_steered_features': n_steered,
            'steered_fraction': steered_frac,
            'weighted_steered_fraction': weighted_steered_frac,
            'gap_fraction': row['gap_fraction'],
            'celltype_effect': row['celltype_effect'],
            'correct': row['correct'],
        })

    return pd.DataFrame(records)


def plot_comparison_boxplots(df_attr, top_n, output_dir):
    """Box plots comparing top DE genes vs rest for attribution metrics."""
    df_attr = df_attr.copy()
    df_attr['abs_effect'] = df_attr['celltype_effect'].abs()
    df_attr = df_attr.sort_values('abs_effect', ascending=False).reset_index(drop=True)

    top_mask = pd.Series(False, index=df_attr.index)
    top_mask.iloc[:top_n] = True

    metrics = [
        ('gene_pr', 'Gene PR (effective # features)'),
        ('steered_fraction', 'Fraction of top features steered'),
        ('weighted_steered_fraction', 'Activation-weighted steered fraction'),
    ]

    for col, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(4, 3.5))

        data_top = df_attr.loc[top_mask, col].dropna()
        data_rest = df_attr.loc[~top_mask, col].dropna()

        bp = ax.boxplot(
            [data_top, data_rest],
            labels=[f'Top {top_n}', f'Rest ({(~top_mask).sum()})'],
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
        )
        bp['boxes'][0].set_facecolor('indianred')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('steelblue')
        bp['boxes'][1].set_alpha(0.7)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add medians as text
        for i, data in enumerate([data_top, data_rest]):
            med = data.median()
            ax.text(i + 1, med, f'  {med:.2f}', va='center', fontsize=8, color='black')

        plt.tight_layout()
        path = os.path.join(output_dir, f'attribution_{col}.pdf')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def plot_pr_vs_gap(df_attr, output_dir):
    """Scatter: gene PR vs gap fraction."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.scatter(df_attr['gene_pr'], df_attr['gap_fraction'],
               s=3, alpha=0.3, color='steelblue')

    ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(1, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Gene PR (effective # features)', fontsize=10)
    ax.set_ylabel('Gap fraction', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Clip y
    y_lo, y_hi = np.percentile(df_attr['gap_fraction'], [1, 99])
    ax.set_ylim(y_lo - 0.3, y_hi + 0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'attribution_pr_vs_gap.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def print_top_gene_details(df_attr, fg_matrix, alpha, gene_names, top_n=20):
    """Print detailed feature breakdown for top genes."""
    df_sorted = df_attr.sort_values('celltype_effect', key=abs, ascending=False)

    gene_to_idx = {name: i for i, name in enumerate(gene_names)}

    print(f"\n{'='*90}")
    print(f"Top {top_n} genes: feature attribution details")
    print(f"{'='*90}")

    for _, row in df_sorted.head(top_n).iterrows():
        gene = row['gene']
        g_idx = gene_to_idx[gene]

        # Get top features by activation for this gene
        gene_activations = fg_matrix[:, g_idx]
        top_feat_idx = np.argsort(gene_activations)[::-1][:5]

        feat_str = ', '.join(
            f"f{f}(act={gene_activations[f]:.2f}, a={alpha[f]:.3f})"
            for f in top_feat_idx
        )

        print(f"  {gene:12s}  effect={row['celltype_effect']:+.4f}  "
              f"gap={row['gap_fraction']:+.3f}  "
              f"PR={row['gene_pr']:.1f}  "
              f"top_feat={row['n_top_features']}  "
              f"steered={row['n_steered_features']}  "
              f"frac={row['steered_fraction']:.2f}  "
              f"wfrac={row['weighted_steered_fraction']:.2f}")
        print(f"    top 5: {feat_str}")


def analyze_steered_features(fg_matrix, alpha, df_direction, gene_names, pr_path,
                             steered_threshold=0.05, output_dir=None):
    """Reverse analysis: for each steered feature, check how its top genes changed.

    For each feature with |alpha - 1| > threshold:
    - Get its top genes (using existing feature-centric PR)
    - Look up those genes' gap fractions from the direction CSV
    - Report: median gap fraction, % correct, and individual gene details
    """
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    steered_mask = np.abs(alpha - 1) > steered_threshold
    steered_indices = np.where(steered_mask)[0]

    # Load feature-centric PR and compute feature -> gene sets
    print(f"\nReverse analysis: {len(steered_indices)} steered features (|a-1| > {steered_threshold})")
    pr_values = np.load(pr_path)
    feat_gene_sets = get_top_k_genes_per_feature(
        fg_matrix, pr_values, pr_scale=1, min_genes=10, max_genes=100
    )

    # Build gene name -> gap fraction lookup from direction CSV
    gap_lookup = {}
    correct_lookup = {}
    effect_lookup = {}
    for _, row in df_direction.iterrows():
        gap_lookup[row['gene']] = row['gap_fraction']
        correct_lookup[row['gene']] = row['correct']
        effect_lookup[row['gene']] = row['celltype_effect']

    # Per-feature analysis
    records = []
    for f_idx in sorted(steered_indices, key=lambda f: abs(alpha[f] - 1), reverse=True):
        gene_set = feat_gene_sets[f_idx]
        top_gene_names = [gene_names[g] for g in sorted(gene_set)]

        # Filter to genes present in direction CSV
        matched_genes = [g for g in top_gene_names if g in gap_lookup]
        if not matched_genes:
            continue

        gaps = [gap_lookup[g] for g in matched_genes]
        corrects = [correct_lookup[g] for g in matched_genes]
        effects = [effect_lookup[g] for g in matched_genes]

        records.append({
            'feature': f_idx,
            'alpha': alpha[f_idx],
            'alpha_minus_1': alpha[f_idx] - 1,
            'n_top_genes': len(gene_set),
            'n_matched_genes': len(matched_genes),
            'median_gap_fraction': np.median(gaps),
            'mean_gap_fraction': np.mean(gaps),
            'pct_correct': np.mean(corrects) * 100,
            'median_abs_effect': np.median(np.abs(effects)),
        })

    df_features = pd.DataFrame(records)

    # Print summary
    print(f"\n{'='*90}")
    print(f"Steered features → top gene gap fractions")
    print(f"{'='*90}")

    for _, row in df_features.iterrows():
        direction = "UP" if row['alpha'] > 1 else "DN"
        print(f"  f{int(row['feature']):<5d}  a={row['alpha']:.3f} ({direction})  "
              f"top_genes={row['n_top_genes']}  matched={row['n_matched_genes']}  "
              f"median_gap={row['median_gap_fraction']:+.3f}  "
              f"correct={row['pct_correct']:.0f}%  "
              f"median_|effect|={row['median_abs_effect']:.3f}")

    # Summary stats
    print(f"\n  Overall across {len(df_features)} steered features:")
    print(f"    Median of median_gap_fraction: {df_features['median_gap_fraction'].median():.3f}")
    print(f"    Mean of pct_correct: {df_features['pct_correct'].mean():.1f}%")

    # Save CSV
    if output_dir:
        csv_path = os.path.join(output_dir, 'steered_feature_gene_gaps.csv')
        df_features.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Plot: alpha_minus_1 vs median gap fraction of top genes
    if output_dir and len(df_features) > 5:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(df_features['alpha_minus_1'], df_features['median_gap_fraction'],
                   s=20, alpha=0.7, color='steelblue')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Alpha - 1 (steering magnitude)', fontsize=10)
        ax.set_ylabel('Median gap fraction of top genes', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Label extreme features
        for _, row in df_features.iterrows():
            if abs(row['alpha_minus_1']) > 0.3:
                ax.annotate(f"f{int(row['feature'])}", (row['alpha_minus_1'], row['median_gap_fraction']),
                            fontsize=7, alpha=0.8)

        plt.tight_layout()
        path = os.path.join(output_dir, 'steered_features_vs_gene_gaps.pdf')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    return df_features


def main():
    parser = argparse.ArgumentParser(
        description='Analyze feature attribution for top DE genes')
    parser.add_argument('--feature_gene_matrix', required=True,
                        help='Path to feature_gene_matrix.npy')
    parser.add_argument('--alpha_vector', required=True,
                        help='Path to alpha vector .pt file')
    parser.add_argument('--direction_csv', required=True,
                        help='Path to direction_intersection_DE.csv')
    parser.add_argument('--raw_data_file', default=None,
                        help='Path to raw h5ad (for gene names). Defaults to ../../data/pbmc/pbmc3k_raw.h5ad')
    parser.add_argument('--gene_names_file', default=None,
                        help='Path to pre-computed gene names text file (alternative to raw_data_file)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for plots and CSVs')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top genes to compare (default: 100)')
    parser.add_argument('--steered_threshold', type=float, default=0.05,
                        help='Threshold for |alpha - 1| to count as steered (default: 0.05)')
    parser.add_argument('--feature_pr', default=None,
                        help='Path to feature_participation_ratios.npy (for reverse analysis). '
                             'If not provided, looks in same dir as feature_gene_matrix.')
    args = parser.parse_args()

    if args.raw_data_file is None and args.gene_names_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.raw_data_file = os.path.join(script_dir, '../../data/pbmc/pbmc3k_raw.h5ad')

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load data
    fg_matrix, alpha, df_direction, gene_names = load_data(args)

    # Step 2: Compute gene-centric PR and top features per gene
    feature_sets, gene_pr = compute_gene_feature_sets(fg_matrix)

    # Step 3: Compute per-gene attribution metrics
    print("\nComputing attribution metrics...")
    df_attr = compute_attribution_metrics(
        df_direction, gene_names, feature_sets, gene_pr, alpha,
        fg_matrix, steered_threshold=args.steered_threshold
    )
    print(f"  Genes with metrics: {len(df_attr)}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'feature_attribution_metrics.csv')
    df_attr.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Step 4: Print summary comparison
    df_sorted = df_attr.sort_values('celltype_effect', key=abs, ascending=False).reset_index(drop=True)
    top = df_sorted.iloc[:args.top_n]
    rest = df_sorted.iloc[args.top_n:]

    print(f"\n{'='*60}")
    print(f"Top {args.top_n} vs rest ({len(rest)} genes)")
    print(f"{'='*60}")
    for label, df_sub in [('Top', top), ('Rest', rest)]:
        print(f"\n  {label} (N={len(df_sub)}):")
        print(f"    Gene PR      — median: {df_sub['gene_pr'].median():.1f}, "
              f"mean: {df_sub['gene_pr'].mean():.1f}")
        print(f"    Top features — median: {df_sub['n_top_features'].median():.0f}, "
              f"mean: {df_sub['n_top_features'].mean():.1f}")
        print(f"    Steered feat — median: {df_sub['n_steered_features'].median():.0f}, "
              f"mean: {df_sub['n_steered_features'].mean():.1f}")
        print(f"    Steered frac — median: {df_sub['steered_fraction'].median():.3f}, "
              f"mean: {df_sub['steered_fraction'].mean():.3f}")
        print(f"    Weighted frac— median: {df_sub['weighted_steered_fraction'].median():.3f}, "
              f"mean: {df_sub['weighted_steered_fraction'].mean():.3f}")
        print(f"    % correct    — {df_sub['correct'].mean()*100:.1f}%")

    # Step 5: Plots
    print("\nGenerating plots...")
    plot_comparison_boxplots(df_attr, args.top_n, args.output_dir)
    plot_pr_vs_gap(df_attr, args.output_dir)

    # Step 6: Detailed breakdown for top genes
    print_top_gene_details(df_attr, fg_matrix, alpha, gene_names, top_n=20)

    # Step 7: Reverse analysis — steered features → top gene gaps
    pr_path = args.feature_pr
    if pr_path is None:
        pr_path = os.path.join(os.path.dirname(args.feature_gene_matrix),
                               'feature_participation_ratios.npy')
    if os.path.exists(pr_path):
        analyze_steered_features(
            fg_matrix, alpha, df_direction, gene_names, pr_path,
            steered_threshold=args.steered_threshold, output_dir=args.output_dir
        )
    else:
        print(f"\nSkipping reverse analysis: PR file not found at {pr_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()

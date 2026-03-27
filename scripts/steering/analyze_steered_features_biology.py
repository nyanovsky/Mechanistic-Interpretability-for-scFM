"""Aggregate biological annotations for steered features from optimization results.

Identifies two feature sets:
1. Top features by |alpha - 1| (most strongly steered)
2. Features associated with top DE genes (by celltype effect)

For each feature, loads GO annotations and top genes with gap fractions,
producing a structured markdown report for downstream interpretation.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import (
    load_feature_attribution_data, load_go_enrichment_detailed,
)
from utils.similarity import get_top_k_genes_per_feature, compute_gene_feature_sets


def build_direction_lookups(df_direction):
    """Build gene -> metric lookups from direction CSV."""
    df_ranked = df_direction.copy()
    df_ranked['de_rank'] = df_ranked['celltype_effect'].abs().rank(ascending=False).astype(int)

    lookups = {}
    for _, row in df_ranked.iterrows():
        lookups[row['gene']] = {
            'gap_fraction': row['gap_fraction'],
            'celltype_effect': row['celltype_effect'],
            'correct': row['correct'],
            'de_rank': row['de_rank'],
        }
    return lookups


def identify_top_alpha_features(steered_csv_path, n_top):
    """Feature Set 1: top features by |alpha - 1|."""
    df = pd.read_csv(steered_csv_path)
    df['abs_alpha_minus_1'] = df['alpha_minus_1'].abs()
    df = df.sort_values('abs_alpha_minus_1', ascending=False).head(n_top)
    print(f"Set 1: {len(df)} features by |alpha-1| (max={df['abs_alpha_minus_1'].iloc[0]:.3f})")
    return df


def identify_de_gene_features(attribution_csv_path, fg_matrix, gene_names,
                                n_top_genes, steered_feature_ids):
    """Feature Set 2: features associated with top DE genes.

    Returns per-gene feature sets, the union of all features, top genes df,
    and per-gene feature details (top features with activation values).
    """
    df_attr = pd.read_csv(attribution_csv_path)
    df_attr['abs_effect'] = df_attr['celltype_effect'].abs()
    top_genes = df_attr.nlargest(n_top_genes, 'abs_effect')

    print(f"\nComputing gene-feature associations for top {n_top_genes} DE genes...")
    feature_sets, _ = compute_gene_feature_sets(fg_matrix)

    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    all_features = set()
    gene_feature_map = {}  # gene -> list of (feature_id, activation)
    per_gene_features = {}  # gene -> set of feature ids

    for _, row in top_genes.iterrows():
        gene = row['gene']
        if gene not in gene_to_idx:
            continue
        g_idx = gene_to_idx[gene]
        feat_set = feature_sets[g_idx]
        all_features.update(feat_set)
        per_gene_features[gene] = feat_set

        activations = fg_matrix[:, g_idx]
        gene_feature_map[gene] = sorted(
            [(f, activations[f]) for f in feat_set],
            key=lambda x: x[1], reverse=True
        )[:10]

    overlap = all_features & set(steered_feature_ids)
    print(f"Set 2: {len(all_features)} unique features from top {n_top_genes} DE genes")
    print(f"  Overlap with Set 1: {len(overlap)}")
    print(f"  Unique to Set 2: {len(all_features - set(steered_feature_ids))}")

    return all_features, top_genes, gene_feature_map, per_gene_features


def get_feature_top_genes(fg_matrix, gene_names, feature_ids, pr_path,
                           direction_lookups, max_genes=20):
    """Get top genes per feature using feature-centric PR-adaptive thresholding."""
    pr_values = np.load(pr_path)
    feat_gene_sets = get_top_k_genes_per_feature(
        fg_matrix, pr_values, pr_scale=1, min_genes=10, max_genes=100
    )

    feature_genes = {}
    for fid in feature_ids:
        gene_set = feat_gene_sets[fid]
        genes_info = []
        for g_idx in sorted(gene_set):
            gname = gene_names[g_idx]
            if gname in direction_lookups:
                m = direction_lookups[gname]
                genes_info.append({
                    'gene': gname,
                    'activation': fg_matrix[fid, g_idx],
                    'celltype_effect': m['celltype_effect'],
                    'gap_fraction': m['gap_fraction'],
                    'correct': m['correct'],
                })
        genes_info.sort(key=lambda x: x['activation'], reverse=True)
        feature_genes[fid] = genes_info[:max_genes]

    return feature_genes


def _is_steered(alpha_val, threshold):
    """Check if a feature is steered (|alpha - 1| >= threshold)."""
    return abs(alpha_val - 1.0) >= threshold


def _write_feature_section(f, fid, alpha_lookup, go_data, feature_genes,
                           heading_level=3):
    """Write a single feature's GO annotations and top genes section."""
    alpha_val = alpha_lookup.get(fid, 1.0)
    direction = "UP" if alpha_val > 1 else ("DOWN" if alpha_val < 1 else "UNCHANGED")
    heading = '#' * heading_level

    f.write(f"{heading} Feature {fid} (alpha={alpha_val:.3f}, {direction})\n\n")

    # GO annotations
    if fid in go_data:
        df_go = go_data[fid]
        f.write("**GO Annotations** (adj. p < 0.05):\n\n")
        for cat in ['BP', 'MF', 'CC']:
            cat_terms = df_go[df_go['GO_category'] == cat]
            if len(cat_terms) == 0:
                continue
            cat_name = {'BP': 'Biological Process', 'MF': 'Molecular Function',
                        'CC': 'Cellular Component'}[cat]
            f.write(f"*{cat_name}* ({len(cat_terms)} terms):\n")
            for _, term_row in cat_terms.head(10).iterrows():
                genes_str = f" — genes: {term_row['Genes']}" if term_row['Genes'] else ""
                f.write(f"- {term_row['Term']} (p={term_row['Adjusted P-value']:.2e}){genes_str}\n")
            if len(cat_terms) > 10:
                f.write(f"- ... and {len(cat_terms) - 10} more\n")
            f.write("\n")

        f.write(f"**Total significant terms**: {len(df_go)}\n\n")
    else:
        f.write("**GO Annotations**: No significant terms (adj. p < 0.05)\n\n")

    # Top genes
    if fid in feature_genes and feature_genes[fid]:
        genes = feature_genes[fid]
        n_correct = sum(1 for g in genes if g['correct'])
        f.write(f"**Top genes** ({n_correct}/{len(genes)} correct direction):\n\n")
        f.write("| Gene | Activation | Celltype Effect | Gap Fraction | Correct? |\n")
        f.write("|---|---|---|---|---|\n")
        for g in genes[:15]:
            correct_str = "Y" if g['correct'] else "N"
            f.write(f"| {g['gene']} | {g['activation']:.2f} "
                    f"| {g['celltype_effect']:+.3f} | {g['gap_fraction']:.3f} "
                    f"| {correct_str} |\n")
        f.write("\n")

    f.write("[INTERPRETATION PLACEHOLDER]\n\n---\n\n")


def write_report(output_path, df_set1, set2_features, top_genes_df, gene_feature_map,
                 per_gene_features, go_data, feature_genes, alpha_lookup,
                 unsteered_threshold):
    """Write structured markdown report."""
    set1_ids = set(df_set1['feature'].astype(int))

    with open(output_path, 'w') as f:
        f.write("# Steered Features Biology Report\n\n")
        f.write(f"> Unsteered threshold: |alpha - 1| < {unsteered_threshold}\n\n")

        # --- Summary table: Set 1 ---
        f.write("## Feature Set 1: Top Features by Steering Magnitude\n\n")
        f.write("| Feature | Alpha | Direction | Top Genes | Median Gap | % Correct | Best DE Rank |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, row in df_set1.iterrows():
            direction = "UP" if row['alpha'] > 1 else "DOWN"
            f.write(f"| f{int(row['feature'])} | {row['alpha']:.3f} | {direction} "
                    f"| {int(row['n_top_genes'])} | {row['median_gap_fraction']:.3f} "
                    f"| {row['pct_correct']:.0f}% | {int(row['min_de_rank'])} |\n")

        # --- Summary table: Set 2 ---
        f.write(f"\n## Feature Set 2: Features of Top {len(top_genes_df)} DE Genes\n\n")
        f.write("| Gene | Celltype Effect | Gap Fraction | Total Features | % Steered | Top Features (by activation) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in top_genes_df.iterrows():
            gene = row['gene']
            gene_feats = per_gene_features.get(gene, set())
            n_total = len(gene_feats)
            n_steered = sum(
                1 for fid in gene_feats
                if _is_steered(alpha_lookup.get(fid, 1.0), unsteered_threshold)
            )
            pct_steered = (n_steered / n_total * 100) if n_total > 0 else 0
            if gene in gene_feature_map:
                feat_str = ", ".join(
                    f"f{fid}(a={alpha_lookup.get(fid, 1.0):.2f})"
                    for fid, act in gene_feature_map[gene][:5]
                )
            else:
                feat_str = "—"
            f.write(f"| {gene} | {row['celltype_effect']:+.3f} | {row['gap_fraction']:.3f} "
                    f"| {n_total} | {pct_steered:.0f}% | {feat_str} |\n")

        # --- Overlap ---
        overlap = set1_ids & set2_features
        f.write(f"\n### Overlap: {len(overlap)} features in both sets\n")
        if overlap:
            f.write(f"Features: {', '.join(f'f{fid}' for fid in sorted(overlap))}\n")

        # --- Per-Feature Analysis: Set 1 ---
        set1_only = sorted(set1_ids - set2_features)
        set1_and_set2 = sorted(set1_ids & set2_features)
        set1_ordered = set1_and_set2 + set1_only  # overlap first

        f.write(f"\n---\n\n## Per-Feature Analysis — Set 1: Top Steered Features ({len(set1_ids)} features)\n\n")

        for fid in set1_ordered:
            also_in_set2 = fid in set2_features
            if also_in_set2:
                f.write(f"*Also in Set 2*\n\n")
            _write_feature_section(f, fid, alpha_lookup, go_data, feature_genes)

        # --- Per-Feature Analysis: Set 2 (grouped by gene) ---
        # Collect Set 2-only features, filtered by steered threshold
        set2_only = set2_features - set1_ids
        set2_steered = {
            fid for fid in set2_only
            if _is_steered(alpha_lookup.get(fid, 1.0), unsteered_threshold)
        }
        n_skipped = len(set2_only) - len(set2_steered)

        f.write(f"## Per-Feature Analysis — Set 2: Features of Top DE Genes "
                f"({len(set2_steered)} steered features")
        if n_skipped > 0:
            f.write(f", {n_skipped} unsteered skipped")
        f.write(")\n\n")

        # Group by gene, maintaining gene order from top_genes_df
        written_set2 = set()
        for _, row in top_genes_df.iterrows():
            gene = row['gene']
            gene_feats = per_gene_features.get(gene, set())
            # Only features unique to Set 2 and steered
            gene_steered = sorted(gene_feats & set2_steered)
            if not gene_steered:
                continue

            f.write(f"### Gene: {gene} (celltype effect={row['celltype_effect']:+.3f})\n\n")

            for fid in gene_steered:
                if fid in written_set2:
                    alpha_val = alpha_lookup.get(fid, 1.0)
                    f.write(f"#### Feature {fid} (alpha={alpha_val:.3f}) — see above\n\n---\n\n")
                else:
                    _write_feature_section(f, fid, alpha_lookup, go_data, feature_genes,
                                           heading_level=4)
                    written_set2.add(fid)

        # Global summary placeholder
        f.write("## Global Summary\n\n[GLOBAL INTERPRETATION PLACEHOLDER]\n")

    print(f"\nReport written to: {output_path}")
    print(f"  Set 1: {len(set1_ids)} features")
    print(f"  Set 2: {len(set2_steered)} steered features ({n_skipped} unsteered skipped)")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate biological annotations for steered features')
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing steered_feature_gene_gaps.csv, '
                             'feature_attribution_metrics.csv, and direction_intersection_DE.csv')
    parser.add_argument('--interpretations_dir', required=True,
                        help='Path to interpretations_mean_pooling/ directory '
                             '(contains feature_*_enrichr/, feature_gene_matrix.npy, '
                             'feature_participation_ratios.npy)')
    parser.add_argument('--alpha_vector', required=True,
                        help='Path to alpha vector .pt file')
    parser.add_argument('--raw_data_file', default=None,
                        help='Path to raw h5ad (for gene names)')
    parser.add_argument('--gene_names_file', default=None,
                        help='Path to pre-computed gene names text file')
    parser.add_argument('--n_top_alpha', type=int, default=20,
                        help='Number of top features by |alpha-1| (default: 20)')
    parser.add_argument('--n_top_de_genes', type=int, default=20,
                        help='Number of top DE genes to trace (default: 20)')
    parser.add_argument('--unsteered_threshold', type=float, default=0.3,
                        help='Features with |alpha-1| < threshold are considered '
                             'unsteered and skipped in Set 2 (default: 0.1)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for the report (default: results_dir)')
    args = parser.parse_args()

    if args.raw_data_file is None and args.gene_names_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.raw_data_file = 'data/pbmc/pbmc3k_raw.h5ad'

    if args.output_dir is None:
        args.output_dir = args.results_dir

    # Derive paths
    steered_csv = os.path.join(args.results_dir, 'steered_feature_gene_gaps.csv')
    attribution_csv = os.path.join(args.results_dir, 'feature_attribution_metrics.csv')
    direction_csv = os.path.join(args.results_dir, 'direction_intersection_DE.csv')
    fg_matrix_path = os.path.join(args.interpretations_dir, 'feature_gene_matrix.npy')
    pr_path = os.path.join(args.interpretations_dir, 'feature_participation_ratios.npy')

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load core data
    fg_matrix, alpha, df_direction, gene_names = load_feature_attribution_data(
        fg_matrix_path, args.alpha_vector, direction_csv,
        raw_data_path=args.raw_data_file, gene_names_file=args.gene_names_file
    )
    alpha_lookup = {i: float(alpha[i]) for i in range(len(alpha))}
    direction_lookups = build_direction_lookups(df_direction)

    # Step 2: Feature Set 1 — top by |alpha-1|
    df_set1 = identify_top_alpha_features(steered_csv, args.n_top_alpha)
    set1_ids = set(df_set1['feature'].astype(int))

    # Step 3: Feature Set 2 — features of top DE genes
    set2_features, top_genes_df, gene_feature_map, per_gene_features = identify_de_gene_features(
        attribution_csv, fg_matrix, gene_names, args.n_top_de_genes, set1_ids
    )
    # attribution CSV already has gap_fraction from the direction analysis
    if 'gap_fraction' not in top_genes_df.columns:
        top_genes_df = top_genes_df.merge(
            df_direction[['gene', 'gap_fraction']], on='gene', how='left'
        )

    # Step 4: Load GO annotations for all features in both sets
    all_feature_ids = set1_ids | set2_features
    print(f"\nLoading GO annotations for {len(all_feature_ids)} features...")
    go_data = load_go_enrichment_detailed(args.interpretations_dir, feature_ids=all_feature_ids)

    # Step 5: Get top genes per feature
    print("\nGetting top genes per feature...")
    feature_genes = get_feature_top_genes(
        fg_matrix, gene_names, all_feature_ids, pr_path, direction_lookups
    )

    # Step 6: Write report
    output_path = os.path.join(args.output_dir, 'steered_features_biology.md')
    write_report(output_path, df_set1, set2_features, top_genes_df,
                 gene_feature_map, per_gene_features, go_data, feature_genes,
                 alpha_lookup, args.unsteered_threshold)

    print("Done.")


if __name__ == '__main__':
    main()

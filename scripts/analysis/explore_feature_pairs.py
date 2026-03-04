"""Explore feature pairs based on similarity metrics.

This script loads similarity results and provides detailed summaries of
high/medium/low similarity feature pairs for manual inspection.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob

# Import utility functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import load_go_enrichment, load_gene_names, get_expressed_genes_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Explore feature pairs')
    parser.add_argument('--layer', type=int, default=12,
                        help='AIDO.Cell layer (9, 12, or 15)')
    parser.add_argument('--expansion', type=int, default=8,
                        help='SAE expansion factor')
    parser.add_argument('--k', type=int, default=32,
                        help='Top-K sparsity')
    parser.add_argument('--metric', type=str, default='gene_overlap',
                        choices=['gene_overlap', 'spearman', 'cosine', 'lin', 'go_overlap', 'decoder_cosine'],
                        help='Similarity metric to explore')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top pairs to show (default: 20)')
    parser.add_argument('--medium-k', type=int, default=10,
                        help='Number of medium similarity pairs to show (default: 10)')
    parser.add_argument('--low-k', type=int, default=5,
                        help='Number of low similarity pairs to show (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for summary (default: print to stdout)')
    return parser.parse_args()


def load_similarity_results(layer, expansion, k, metric):
    """Load similarity results and feature indices."""
    OUTPUT_DIR = f"plots/sae/layer_{layer}/coactivation"

    # Determine file names based on metric
    if metric == 'lin':
        corr_file = f"{OUTPUT_DIR}/feature_go_lin_filtered_similarities.npy"
        indices_file = f"{OUTPUT_DIR}/feature_go_lin_filtered_indices.npy"
    elif metric == 'go_overlap':
        corr_file = f"{OUTPUT_DIR}/feature_go_overlap_filtered_similarities.npy"
        indices_file = f"{OUTPUT_DIR}/feature_go_overlap_filtered_indices.npy"
    elif metric == 'gene_overlap':
        corr_file = f"{OUTPUT_DIR}/feature_gene_overlap_filtered_correlations.npy"
        indices_file = f"{OUTPUT_DIR}/feature_gene_overlap_filtered_indices.npy"
    elif metric == 'decoder_cosine':
        corr_file = f"{OUTPUT_DIR}/feature_decoder_cosine_filtered_similarities.npy"
        indices_file = f"{OUTPUT_DIR}/feature_decoder_cosine_filtered_indices.npy"
    else:
        corr_file = f"{OUTPUT_DIR}/feature_gene_{metric}_filtered_correlations.npy"
        indices_file = f"{OUTPUT_DIR}/feature_gene_{metric}_filtered_indices.npy"

    if not os.path.exists(corr_file):
        raise FileNotFoundError(f"Similarity file not found: {corr_file}")
    if not os.path.exists(indices_file):
        raise FileNotFoundError(f"Indices file not found: {indices_file}")

    similarities = np.load(corr_file)
    feature_indices = np.load(indices_file)

    return similarities, feature_indices


def load_metric_for_comparison(layer, expansion, k, metric):
    """Load similarity results for a given metric.

    Args:
        layer: Layer number
        expansion: SAE expansion factor
        k: Top-K sparsity
        metric: Similarity metric ('gene_overlap', 'go_overlap', 'decoder_cosine')

    Returns:
        tuple: (similarities, feature_indices) or (None, None) if not available
    """
    OUTPUT_DIR = f"plots/sae/layer_{layer}/coactivation"

    # Determine file names based on metric
    if metric == 'gene_overlap':
        corr_file = f"{OUTPUT_DIR}/feature_gene_overlap_filtered_correlations.npy"
        indices_file = f"{OUTPUT_DIR}/feature_gene_overlap_filtered_indices.npy"
    elif metric == 'go_overlap':
        corr_file = f"{OUTPUT_DIR}/feature_go_overlap_filtered_similarities.npy"
        indices_file = f"{OUTPUT_DIR}/feature_go_overlap_filtered_indices.npy"
    elif metric == 'decoder_cosine':
        corr_file = f"{OUTPUT_DIR}/feature_decoder_cosine_filtered_similarities.npy"
        indices_file = f"{OUTPUT_DIR}/feature_decoder_cosine_filtered_indices.npy"
    else:
        raise ValueError(f"Unknown metric for comparison: {metric}")

    if not os.path.exists(corr_file) or not os.path.exists(indices_file):
        return None, None

    similarities = np.load(corr_file)
    feature_indices = np.load(indices_file)

    return similarities, feature_indices


def condensed_to_pairs(condensed_idx, n_features):
    """Convert condensed distance matrix index to (i, j) pair indices."""
    # Formula for converting condensed index to pair indices
    i = 0
    while condensed_idx >= n_features - i - 1:
        condensed_idx -= n_features - i - 1
        i += 1
    j = condensed_idx + i + 1
    return i, j


def get_condensed_index(feat_i, feat_j, feature_to_idx):
    """Compute condensed distance matrix index from two feature IDs.

    Args:
        feat_i: First feature ID
        feat_j: Second feature ID
        feature_to_idx: Mapping from feature ID to index

    Returns:
        Condensed index, or None if features not in mapping
    """
    if feat_i not in feature_to_idx or feat_j not in feature_to_idx:
        return None

    i = feature_to_idx[feat_i]
    j = feature_to_idx[feat_j]

    # Ensure i < j for condensed format
    if i > j:
        i, j = j, i

    # Compute condensed index
    n_features = len(feature_to_idx)
    idx = 0
    for k in range(i):
        idx += n_features - k - 1
    idx += j - i - 1

    return idx


def get_top_genes_for_feature(feature_id, feature_gene_matrix, gene_names, top_n=20):
    """Get top activating genes for a feature."""
    activations = feature_gene_matrix[feature_id]
    top_indices = np.argsort(activations)[::-1][:top_n]
    top_genes = [(gene_names[i], activations[i]) for i in top_indices]
    return top_genes


def load_feature_go_terms(feature_id, interpretations_dir):
    """Load GO enrichment terms for a feature."""
    feature_dir = os.path.join(interpretations_dir, f'feature_{feature_id}_enrichr')

    if not os.path.exists(feature_dir):
        return []

    terms = []
    for go_file in glob(os.path.join(feature_dir, 'GO_*_2021.human.enrichr.reports.txt')):
        try:
            df = pd.read_csv(go_file, sep='\t')
            if 'Adjusted P-value' not in df.columns or 'Term' not in df.columns:
                continue
            significant = df[df['Adjusted P-value'] < 0.05]
            for _, row in significant.iterrows():
                terms.append({
                    'term': row['Term'],
                    'p_value': row['Adjusted P-value'],
                    'namespace': os.path.basename(go_file).split('_')[1]  # BP, CC, or MF
                })
        except Exception:
            continue

    # Sort by p-value
    terms = sorted(terms, key=lambda x: x['p_value'])
    return terms


def get_go_term_strings(feature_id, interpretations_dir):
    """Get GO term strings (not just IDs) for a feature."""
    feature_dir = os.path.join(interpretations_dir, f'feature_{feature_id}_enrichr')

    if not os.path.exists(feature_dir):
        return set()

    terms = set()
    for go_file in glob(os.path.join(feature_dir, 'GO_*_2021.human.enrichr.reports.txt')):
        try:
            df = pd.read_csv(go_file, sep='\t')
            if 'Adjusted P-value' not in df.columns or 'Term' not in df.columns:
                continue
            significant = df[df['Adjusted P-value'] < 0.05]
            for term in significant['Term']:
                terms.add(term)
        except Exception:
            continue

    return terms


def format_feature_summary(feature_id, top_genes, go_terms, pr_value=None, go_term_count=None):
    """Format a readable summary for a feature."""
    lines = []
    lines.append(f"  Feature {feature_id}")
    if pr_value is not None:
        lines.append(f"    Participation Ratio: {pr_value:.1f}")
    if go_term_count is not None:
        lines.append(f"    GO terms: {go_term_count}")

    # Top genes
    lines.append(f"    Top genes ({len(top_genes)}):")
    for gene, activation in top_genes[:10]:
        lines.append(f"      {gene}: {activation:.4f}")
    if len(top_genes) > 10:
        lines.append(f"      ... and {len(top_genes) - 10} more")

    # GO terms
    if go_terms:
        lines.append(f"    GO enrichment (top 5 terms):")
        for term_info in go_terms[:5]:
            lines.append(f"      [{term_info['namespace']}] {term_info['term']} (p={term_info['p_value']:.2e})")
        if len(go_terms) > 5:
            lines.append(f"      ... and {len(go_terms) - 5} more significant terms")
    else:
        lines.append(f"    GO enrichment: None found")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Paths
    INPUT_DIM = 640
    LATENT_DIM = INPUT_DIM * args.expansion
    BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
    SAE_DIR = f"{BASE_DIR}/sae_k_{args.k}_{LATENT_DIM}"
    INTERPRETATION_DIR = f"{SAE_DIR}/interpretations_filter_zero_expressed"

    print(f"\n{'='*80}")
    print(f"Exploring Feature Pairs (Layer {args.layer}, {args.metric} similarity)")
    print(f"{'='*80}\n")

    # Load similarity results
    print("Loading similarity results...")
    similarities, feature_indices = load_similarity_results(args.layer, args.expansion, args.k, args.metric)
    n_features = len(feature_indices)
    print(f"  {n_features} features, {len(similarities):,} pairs")

    # Load feature-gene matrix and gene names
    print("Loading feature-gene matrix and gene names...")
    feature_gene_matrix_full = np.load(f"{INTERPRETATION_DIR}/feature_gene_matrix.npy")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gene_names_full = load_gene_names("../../data/pbmc/pbmc3k_raw.h5ad")

    # Filter to expressed genes (matching compute_feature_correlations.py filtering)
    print("Filtering to expressed genes (matching overlap computation)...")
    RAW_DATA_FILE = os.path.join(script_dir, "../../data/pbmc/pbmc3k_raw.h5ad")
    expressed_mask = get_expressed_genes_mask(
        RAW_DATA_FILE,
        min_mean_expr=0.01,
        min_pct_cells=0.5
    )
    feature_gene_matrix = feature_gene_matrix_full[:, expressed_mask]
    gene_names = [gene_names_full[i] for i in range(len(gene_names_full)) if expressed_mask[i]]
    print(f"  Filtered from {len(gene_names_full)} to {len(gene_names)} expressed genes")

    # Load PR values
    pr_file = f"{INTERPRETATION_DIR}/feature_participation_ratios.npy"
    if os.path.exists(pr_file):
        pr_values_full = np.load(pr_file)
        # Map PR values to filtered features
        pr_dict = {feat_id: pr_values_full[feat_id] for feat_id in feature_indices}
    else:
        pr_dict = None

    # Get indices sorted by similarity
    sorted_indices = np.argsort(similarities)[::-1]  # High to low

    # Load ALL comparison metrics (for metric-agnostic display)
    print("\nLoading all available similarity metrics...")

    # Load gene overlap
    print("  Gene overlap...", end=" ")
    gene_overlaps, gene_feature_ids = load_metric_for_comparison(args.layer, args.expansion, args.k, 'gene_overlap')
    if gene_overlaps is not None:
        gene_feature_to_idx = {fid: idx for idx, fid in enumerate(gene_feature_ids)}
        print(f"loaded ({len(gene_feature_ids)} features)")
    else:
        gene_feature_to_idx = None
        print("not available")

    # Load GO overlap
    print("  GO overlap...", end=" ")
    go_overlaps, go_feature_ids = load_metric_for_comparison(args.layer, args.expansion, args.k, 'go_overlap')
    if go_overlaps is not None:
        go_feature_to_idx = {fid: idx for idx, fid in enumerate(go_feature_ids)}
        print(f"loaded ({len(go_feature_ids)} features)")
    else:
        go_feature_to_idx = None
        print("not available")

    # Load decoder cosine similarity
    print("  Decoder cosine...", end=" ")
    decoder_sims, decoder_feature_ids = load_metric_for_comparison(args.layer, args.expansion, args.k, 'decoder_cosine')
    if decoder_sims is not None:
        decoder_feature_to_idx = {fid: idx for idx, fid in enumerate(decoder_feature_ids)}
        print(f"loaded ({len(decoder_feature_ids)} features)")
    else:
        decoder_feature_to_idx = None
        print("not available")

    # Define thresholds for medium/low
    if args.metric in ['spearman', 'cosine', 'lin', 'decoder_cosine']:
        high_threshold = 0.5
        medium_min = 0.2
        medium_max = 0.5
    elif args.metric == 'go_overlap':
        high_threshold = 0.5
        medium_min = 0.2
        medium_max = 0.5
    else:  # gene_overlap or other gene-based metrics
        high_threshold = 0.3
        medium_min = 0.1
        medium_max = 0.3

    # Find high, medium, low pairs
    high_pairs = []
    medium_pairs = []
    low_pairs = []

    for idx in sorted_indices:
        sim_val = similarities[idx]
        if sim_val >= high_threshold and len(high_pairs) < args.top_k:
            high_pairs.append((idx, sim_val))
        elif medium_min <= sim_val < medium_max and len(medium_pairs) < args.medium_k:
            medium_pairs.append((idx, sim_val))
        elif sim_val < medium_min and len(low_pairs) < args.low_k:
            low_pairs.append((idx, sim_val))

        if len(high_pairs) >= args.top_k and len(medium_pairs) >= args.medium_k and len(low_pairs) >= args.low_k:
            break

    # Random low pairs if needed
    if len(low_pairs) < args.low_k:
        low_indices = sorted_indices[len(sorted_indices) - args.low_k:]
        low_pairs = [(idx, similarities[idx]) for idx in low_indices]

    # Prepare output
    output_lines = []

    # Summary statistics
    output_lines.append(f"\n{'='*80}")
    output_lines.append("SUMMARY STATISTICS")
    output_lines.append(f"{'='*80}")
    output_lines.append(f"Metric: {args.metric}")
    output_lines.append(f"Total pairs: {len(similarities):,}")
    output_lines.append(f"Mean similarity: {similarities.mean():.4f}")
    output_lines.append(f"Std similarity: {similarities.std():.4f}")
    output_lines.append(f"Min similarity: {similarities.min():.4f}")
    output_lines.append(f"Max similarity: {similarities.max():.4f}")
    output_lines.append(f"Median similarity: {np.median(similarities):.4f}")

    # Process pairs
    for category, pairs in [("HIGH", high_pairs), ("MEDIUM", medium_pairs), ("LOW", low_pairs)]:
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"{category} SIMILARITY PAIRS (n={len(pairs)})")
        output_lines.append(f"{'='*80}\n")

        for pair_idx, sim_val in pairs:
            i, j = condensed_to_pairs(pair_idx, n_features)
            feat_i = feature_indices[i]
            feat_j = feature_indices[j]

            output_lines.append(f"\nPair: Feature {feat_i} <-> Feature {feat_j}")
            output_lines.append(f"Similarity: {sim_val:.4f}\n")

            # Load data for both features
            genes_i = get_top_genes_for_feature(feat_i, feature_gene_matrix, gene_names, top_n=20)
            genes_j = get_top_genes_for_feature(feat_j, feature_gene_matrix, gene_names, top_n=20)

            go_i = load_feature_go_terms(feat_i, INTERPRETATION_DIR)
            go_j = load_feature_go_terms(feat_j, INTERPRETATION_DIR)

            pr_i = pr_dict[feat_i] if pr_dict else None
            pr_j = pr_dict[feat_j] if pr_dict else None

            # Always get GO term strings (for metric-agnostic display)
            go_terms_i = get_go_term_strings(feat_i, INTERPRETATION_DIR)
            go_terms_j = get_go_term_strings(feat_j, INTERPRETATION_DIR)
            go_count_i = len(go_terms_i)
            go_count_j = len(go_terms_j)

            # Format summaries
            output_lines.append(format_feature_summary(feat_i, genes_i, go_i, pr_i, go_count_i))
            output_lines.append("")
            output_lines.append(format_feature_summary(feat_j, genes_j, go_j, pr_j, go_count_j))

            # Check gene overlap using PR-based top-k selection (like overlap coefficient)
            # Get top genes based on PR for each feature
            if pr_i is not None and pr_j is not None:
                # PR-based selection: k = clip(PR * 0.6, 10, 100)
                k_i = int(np.clip(pr_i*0.6, 10, 100))
                k_j = int(np.clip(pr_j*0.6, 10, 100))

                # Get top-k genes for each feature
                genes_i_topk = get_top_genes_for_feature(feat_i, feature_gene_matrix, gene_names, top_n=k_i)
                genes_j_topk = get_top_genes_for_feature(feat_j, feature_gene_matrix, gene_names, top_n=k_j)

                genes_i_set = set([g[0] for g in genes_i_topk])
                genes_j_set = set([g[0] for g in genes_j_topk])
                gene_overlap = genes_i_set & genes_j_set

                if gene_overlap:
                    output_lines.append(f"\n  Overlapping genes in top-k (k_i={k_i}, k_j={k_j}): {len(gene_overlap)}")
                    output_lines.append(f"    {', '.join(sorted(gene_overlap))}")
                else:
                    output_lines.append(f"\n  No overlapping genes in top-k (k_i={k_i}, k_j={k_j})")
            else:
                output_lines.append(f"\n  Gene overlap: PR values not available")

            # Show GO term overlap (always)
            go_overlap_terms = go_terms_i & go_terms_j
            if go_overlap_terms:
                output_lines.append(f"\n  Overlapping GO terms ({len(go_overlap_terms)}):")
                for term in sorted(list(go_overlap_terms))[:10]:
                    output_lines.append(f"    - {term}")
                if len(go_overlap_terms) > 10:
                    output_lines.append(f"    ... and {len(go_overlap_terms) - 10} more")
            else:
                output_lines.append(f"\n  No overlapping GO terms")

            # Show ALL similarity metrics (metric-agnostic display)
            output_lines.append(f"\n  Similarity metrics:")

            comparison_metrics = [
                ('Gene overlap', gene_overlaps, gene_feature_to_idx),
                ('GO overlap', go_overlaps, go_feature_to_idx),
                ('Decoder cosine', decoder_sims, decoder_feature_to_idx)
            ]

            for metric_name, metric_values, metric_feature_to_idx in comparison_metrics:
                if metric_values is not None:
                    metric_idx = get_condensed_index(feat_i, feat_j, metric_feature_to_idx)
                    if metric_idx is not None and metric_idx < len(metric_values):
                        output_lines.append(f"    {metric_name}: {metric_values[metric_idx]:.4f}")
                    else:
                        output_lines.append(f"    {metric_name}: N/A")
                else:
                    output_lines.append(f"    {metric_name}: not computed")

            output_lines.append("\n" + "-"*80)

    # Output
    output_text = "\n".join(output_lines)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"\nSummary written to {args.output}")
    else:
        print(output_text)

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

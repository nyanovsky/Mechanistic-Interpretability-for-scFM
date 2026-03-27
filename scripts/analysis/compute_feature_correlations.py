"""Compute pairwise feature correlations from activation matrices.

This script computes various similarity metrics between SAE features based on their
activation patterns across genes or cells.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import utility functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import load_go_enrichment, get_expressed_genes_mask
from utils.go_utils import load_go_dag_and_associations, compute_pairwise_go_similarity, compute_go_term_overlap
from utils.similarity import compute_pairwise_correlations

def parse_args():
    parser = argparse.ArgumentParser(description='Compute feature pairwise correlations')
    parser.add_argument('--layer', type=int, default=12,
                        help='AIDO.Cell layer (9, 12, or 15)')
    parser.add_argument('--expansion', type=int, default=8,
                        help='SAE expansion factor')
    parser.add_argument('--k', type=int, default=32,
                        help='Top-K sparsity')
    parser.add_argument('--matrix', type=str, default='gene', choices=['gene', 'cell'],
                        help='Matrix to use: gene (feature-gene, default) or cell (cell-feature)')
    parser.add_argument('--metric', type=str, default='spearman',
                        choices=['pearson', 'cosine', 'spearman', 'overlap', 'overlap_sqrt', 'lin', 'go_overlap', 'decoder_cosine'],
                        help='Similarity metric: spearman (default), pearson, cosine (continuous), overlap (gene set-based), overlap_sqrt (sqrt-weighted overlap/Ochiai), lin (GO semantic similarity), go_overlap (GO term overlap coefficient), decoder_cosine (cosine similarity of decoder weights)')
    parser.add_argument('--pr-scale', type=float, default=1,
                        help='For overlap metric: scale factor for PR-based top-k selection (default: 0.6)')
    parser.add_argument('--min-genes', type=int, default=10,
                        help='For overlap metric: minimum genes per feature (default: 10)')
    parser.add_argument('--max-genes', type=int, default=100,
                        help='For overlap metric: maximum genes per feature (default: 100)')
    parser.add_argument('--sample-pairs', type=int, default=None,
                        help='Randomly sample N pairs instead of computing all (useful for large datasets)')
    parser.add_argument('--interp_dir', type=str)
    parser.add_argument('--plot_suffix', type=str, default='')
    return parser.parse_args()


def plot_correlation_distribution(correlations, output_path, metric='spearman'):
    """Plot histogram of correlation distribution."""
    # Count highly correlated pairs
    high_corr_threshold = 0.9
    moderate_corr_threshold = 0.5
    n_high_corr = np.sum(correlations > high_corr_threshold)
    n_pairs = len(correlations)

    fig, ax = plt.subplots(figsize=(12, 6))

    # For overlap metrics, filter out zero-valued pairs and annotate
    if metric in ('overlap', 'overlap_sqrt'):
        n_total = len(correlations)
        n_zeros = np.sum(correlations == 0)
        pct_zeros = 100.0 * n_zeros / n_total if n_total > 0 else 0.0
        correlations = correlations[correlations > 0]
        n_pairs = len(correlations)

    # Plot histogram
    ax.hist(correlations, bins=100, edgecolor='black', alpha=0.7, color='steelblue')

    # Add threshold lines (only for continuous metrics)
    if metric in ['pearson', 'cosine', 'spearman', 'lin', 'go_overlap', 'decoder_cosine']:
        ax.axvline(high_corr_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'High correlation (>{high_corr_threshold})')
        ax.axvline(moderate_corr_threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Moderate correlation (>{moderate_corr_threshold})')

    # Labels and title
    metric_names = {
        'pearson': 'Pearson Correlation',
        'spearman': 'Spearman Correlation',
        'cosine': 'Cosine Similarity',
        'overlap': 'Overlap Coefficient',
        'overlap_sqrt': 'Sqrt-Weighted Overlap (Ochiai)',
        'lin': 'Lin Semantic Similarity',
        'go_overlap': 'GO Term Overlap Coefficient',
        'decoder_cosine': 'Decoder Cosine Similarity'
    }
    metric_label = metric_names.get(metric, metric.capitalize())

    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

    if metric in ['pearson', 'cosine', 'spearman', 'lin', 'go_overlap', 'decoder_cosine']:
        title = f'Feature Pairwise {metric_label}\n(n={n_pairs:,} pairs, {n_high_corr} highly correlated)'
    else:
        title = f'Feature Pairwise {metric_label}\n(n={n_pairs:,} pairs)'

    ax.set_title(title, fontsize=14, fontweight='bold')

    if metric in ['pearson', 'cosine', 'spearman', 'lin', 'go_overlap', 'decoder_cosine']:
        ax.legend(fontsize=11)

    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    # Annotate zero-filtered overlap metrics
    if metric in ('overlap', 'overlap_sqrt'):
        ax.text(0.95, 0.95, f'{pct_zeros:.1f}% pairs = 0 (omitted)',
                transform=ax.transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

    # Print statistics
    print(f"\n{metric_label} statistics:")
    print(f"  Mean: {correlations.mean():.4f}")
    print(f"  Std: {correlations.std():.4f}")
    print(f"  Min: {correlations.min():.4f}")
    print(f"  Max: {correlations.max():.4f}")
    print(f"  Median: {np.median(correlations):.4f}")

    if metric in ['pearson', 'cosine', 'spearman', 'lin', 'go_overlap', 'decoder_cosine']:
        print(f"  Pairs > {moderate_corr_threshold}: {np.sum(correlations > moderate_corr_threshold)}")
        print(f"  Pairs > {high_corr_threshold}: {n_high_corr}")


def main():
    args = parse_args()

    # Paths
    INPUT_DIM = 640
    LATENT_DIM = INPUT_DIM * args.expansion
    BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
    SAE_DIR = f"{BASE_DIR}/sae_k_{args.k}_{LATENT_DIM}"
    INTERPRETATION_DIR = args.interp_dir
    PLOT_SUFFIX = args.plot_suffix
    OUTPUT_DIR = f"plots/sae/layer_{args.layer}{PLOT_SUFFIX}/coactivation"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Handle decoder cosine similarity (uses decoder weights, not activation matrices)
    if args.metric == 'decoder_cosine':
        print(f"\n{'='*60}")
        print(f"Computing DECODER COSINE SIMILARITY")
        print(f"{'='*60}\n")

        # Load decoder weights
        print(f"Loading decoder weights from {SAE_DIR}")
        from utils.data_utils import load_decoder_weights
        W_dec = load_decoder_weights(SAE_DIR)
        print(f"Decoder weights shape: {W_dec.shape} [n_features, hidden_dim]")

        # Filter to significantly enriched features
        print("\nFiltering to significantly enriched features...")
        go_enrichments = load_go_enrichment(INTERPRETATION_DIR, p_threshold=0.05, mode='ids')
        enriched_features = sorted(go_enrichments.keys())
        print(f"  Total features: {W_dec.shape[0]}")
        print(f"  Enriched features: {len(enriched_features)}")

        if len(enriched_features) == 0:
            print("ERROR: No features with GO enrichment found!")
            return

        # Filter decoder weights to enriched features only
        W_dec_filtered = W_dec[enriched_features]
        print(f"  Filtered decoder shape: {W_dec_filtered.shape}")

        # Compute pairwise cosine similarity
        print(f"\nComputing pairwise cosine similarity...")
        from utils.similarity import compute_cosine_similarity

        # Compute full similarity matrix
        similarity_matrix = compute_cosine_similarity(W_dec_filtered)
        print(f"Similarity matrix shape: {similarity_matrix.shape}")

        # Extract upper triangle (condensed form for consistency with other metrics)
        n_features = len(enriched_features)
        similarities = similarity_matrix[np.triu_indices(n_features, k=1)]
        print(f"Extracted {len(similarities):,} pairwise similarities")

        # Save results
        suffix = "_filtered"
        corr_output_file = f"{OUTPUT_DIR}/feature_decoder_cosine{suffix}_similarities.npy"
        np.save(corr_output_file, similarities)
        print(f"\nSaved similarities to {corr_output_file}")

        indices_file = f"{OUTPUT_DIR}/feature_decoder_cosine{suffix}_indices.npy"
        feature_indices = np.array(enriched_features)
        np.save(indices_file, feature_indices)
        print(f"Saved feature indices to {indices_file}")

        # Plot distribution
        plot_output = f"{OUTPUT_DIR}/decoder_cosine{suffix}_hist.png"
        plot_correlation_distribution(similarities, plot_output, metric='decoder_cosine')

        print(f"\n{'='*60}")
        print("Done!")
        print(f"{'='*60}\n")
        return

    # Handle GO-based metrics separately (don't use activation matrices)
    if args.metric in ['lin', 'go_overlap']:
        metric_name = 'LIN semantic similarity' if args.metric == 'lin' else 'GO term overlap'
        print(f"\n{'='*60}")
        print(f"Computing {metric_name.upper()}")
        print(f"{'='*60}\n")

        # Load GO enrichment (as GO IDs)
        print(f"Loading GO enrichment from {INTERPRETATION_DIR}")
        go_enrichments = load_go_enrichment(INTERPRETATION_DIR, p_threshold=0.05, mode='ids')
        print(f"Loaded {len(go_enrichments)} features with significant GO enrichment")

        if len(go_enrichments) == 0:
            print("ERROR: No features with GO enrichment found!")
            return

        # Print GO term count distribution for go_overlap
        if args.metric == 'go_overlap':
            term_counts_dist = [len(terms) for terms in go_enrichments.values()]
            print(f"\nGO term count statistics:")
            print(f"  Mean: {np.mean(term_counts_dist):.1f}")
            print(f"  Std: {np.std(term_counts_dist):.1f}")
            print(f"  Min: {np.min(term_counts_dist)}")
            print(f"  Max: {np.max(term_counts_dist)}")
            print(f"  Median: {np.median(term_counts_dist):.0f}")

        # Load GO DAG and associations (only for lin)
        if args.metric == 'lin':
            print("\nLoading GO DAG and associations...")
            godag, term_counts = load_go_dag_and_associations()
        else:
            godag, term_counts = None, None

        # Compute pairwise semantic similarity
        feature_ids = sorted(go_enrichments.keys())
        n_features = len(feature_ids)

        # Generate all pairs or sample
        total_pairs = n_features * (n_features - 1) // 2

        if args.sample_pairs is not None and args.sample_pairs < total_pairs:
            print(f"\nRandomly sampling {args.sample_pairs:,} pairs out of {total_pairs:,} total pairs...")
            # Generate all possible pairs
            all_pairs = [(feature_ids[i], feature_ids[j])
                        for i in range(n_features)
                        for j in range(i + 1, n_features)]
            # Random sample
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(all_pairs), size=args.sample_pairs, replace=False)
            feature_pairs = [all_pairs[i] for i in sampled_indices]
            print(f"Sampled {len(feature_pairs):,} pairs")
        else:
            print(f"\nComputing pairwise {metric_name}...")
            print(f"This will compute ~{total_pairs:,} pairs")
            feature_pairs = None

        print("This may take a while...\n")

        # Compute similarities based on metric type
        if args.metric == 'lin':
            similarities_dict = compute_pairwise_go_similarity(
                go_enrichments,
                godag,
                term_counts,
                metric='lin',
                feature_pairs=feature_pairs,
                verbose=True,
                n_workers=20  # Use all available CPUs
            )
        else:  # go_overlap
            similarities_dict = compute_go_term_overlap(
                go_enrichments,
                feature_pairs=feature_pairs,
                metric='overlap',
                verbose=True,
                n_workers=20  # Use all available CPUs
            )

        print(f"\nComputed {len(similarities_dict)} pairwise similarities")

        # Convert to array
        if args.sample_pairs is not None and args.sample_pairs < total_pairs:
            # For sampling, just extract values directly
            print("\nExtracting sampled similarity values...")
            correlations = np.array(list(similarities_dict.values()))
            # Store the pairs that were sampled
            sampled_pairs_array = np.array(list(similarities_dict.keys()))
        else:
            # For full computation, convert to condensed array (consistent with other metrics)
            n_pairs = (n_features * (n_features - 1)) // 2
            correlations = np.zeros(n_pairs)

            print("\nConverting to condensed array...")
            idx = 0
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    feat_i = feature_ids[i]
                    feat_j = feature_ids[j]

                    # Look up similarity (handle both orderings)
                    if (feat_i, feat_j) in similarities_dict:
                        correlations[idx] = similarities_dict[(feat_i, feat_j)]
                    elif (feat_j, feat_i) in similarities_dict:
                        correlations[idx] = similarities_dict[(feat_j, feat_i)]
                    else:
                        correlations[idx] = 0.0

                    idx += 1
            sampled_pairs_array = None

        feature_indices = np.array(feature_ids)

        # Save results
        suffix = "_filtered"
        if args.sample_pairs is not None:
            suffix += f"_sampled{args.sample_pairs}"

        # File prefix based on metric
        file_prefix = 'go_lin' if args.metric == 'lin' else 'go_overlap'
        plot_prefix = 'semantic_lin' if args.metric == 'lin' else 'go_overlap'

        corr_output_file = f"{OUTPUT_DIR}/feature_{file_prefix}{suffix}_similarities.npy"
        np.save(corr_output_file, correlations)
        print(f"\nSaved similarities to {corr_output_file}")

        indices_file = f"{OUTPUT_DIR}/feature_{file_prefix}{suffix}_indices.npy"
        np.save(indices_file, feature_indices)
        print(f"Saved feature indices to {indices_file}")

        # Save sampled pairs if applicable
        if sampled_pairs_array is not None:
            pairs_file = f"{OUTPUT_DIR}/feature_{file_prefix}{suffix}_pairs.npy"
            np.save(pairs_file, sampled_pairs_array)
            print(f"Saved sampled pairs to {pairs_file}")

        # Plot distribution
        plot_output = f"{OUTPUT_DIR}/{plot_prefix}{suffix}_hist.png"
        plot_correlation_distribution(correlations, plot_output, metric=args.metric)

        print(f"\n{'='*60}")
        print("Done!")
        print(f"{'='*60}\n")
        return

    # Check if using overlap metric
    if args.metric in ['overlap', 'overlap_sqrt'] and args.matrix != 'gene':
        raise ValueError(f"{args.metric} metric requires --matrix=gene")

    # Select matrix file based on argument
    if args.matrix == 'gene':
        MATRIX_FILE = f"{INTERPRETATION_DIR}/feature_gene_matrix.npy"
        matrix_name = "feature-gene"
    else:  # cell
        MATRIX_FILE = f"{INTERPRETATION_DIR}/cell_feature_matrix.npy"
        matrix_name = "cell-feature"

    print(f"\n{'='*60}")
    print(f"Computing {args.metric.upper()} correlations")
    print(f"{'='*60}\n")

    print(f"Loading {matrix_name} matrix from {MATRIX_FILE}")
    feature_matrix = np.load(MATRIX_FILE)
    print(f"Matrix shape: {feature_matrix.shape}")

    # For cell-feature matrix, transpose to get features as rows
    if args.matrix == 'cell':
        feature_matrix = feature_matrix.T
        print(f"Transposed shape: {feature_matrix.shape} [n_features, n_cells]")

    # HARDCODED FILTERING PIPELINE
    feature_indices = np.arange(feature_matrix.shape[0])  # Track which features we keep

    # Filter 1: Always filter to significantly enriched features
    print("\n[FILTER 1] Filtering to significantly enriched features...")
    go_enrichments = load_go_enrichment(INTERPRETATION_DIR, p_threshold=0.05, mode='ids')
    enriched_features = set(go_enrichments.keys())
    enriched_mask = np.array([i in enriched_features for i in range(feature_matrix.shape[0])])

    print(f"  Before: {feature_matrix.shape[0]} features")
    feature_matrix = feature_matrix[enriched_mask]
    feature_indices = feature_indices[enriched_mask]
    print(f"  After: {feature_matrix.shape[0]} features ({enriched_mask.sum()} enriched)")

    # Filter 2: Always filter to expressed genes (gene-based metrics only)
    if args.matrix == 'gene':
        print("\n[FILTER 2] Filtering to expressed genes...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        RAW_DATA_FILE = "data/pbmc/pbmc3k_raw.h5ad"
        expressed_mask = get_expressed_genes_mask(
            RAW_DATA_FILE,
            min_mean_expr=0.01,
            min_pct_cells=0.5
        )

        print(f"  Before: {feature_matrix.shape[1]} genes")
        feature_matrix = feature_matrix[:, expressed_mask]
        print(f"  After: {feature_matrix.shape[1]} genes ({expressed_mask.sum()} expressed)")

    print(f"\nFinal matrix shape: {feature_matrix.shape}")

    # Load PR values if needed for overlap metric
    pr_values = None
    if args.metric in ['overlap', 'overlap_sqrt']:
        PR_FILE = f"{INTERPRETATION_DIR}/feature_participation_ratios.npy"
        print(f"\nLoading participation ratios from {PR_FILE}")
        pr_values_full = np.load(PR_FILE)
        # Filter PR values to match filtered features
        pr_values = pr_values_full[enriched_mask]
        print(f"PR values shape: {pr_values.shape}")

    # Compute correlations
    print(f"\nComputing {args.metric} correlations...")
    correlations = compute_pairwise_correlations(
        feature_matrix,
        metric=args.metric,
        pr_values=pr_values,
        pr_scale=args.pr_scale,
        min_genes=args.min_genes,
        max_genes=args.max_genes
    )

    # Save correlations and feature indices
    suffix = "_filtered"
    corr_output_file = f"{OUTPUT_DIR}/feature_{args.matrix}_{args.metric}{suffix}_correlations.npy"
    np.save(corr_output_file, correlations)
    print(f"\nSaved correlations to {corr_output_file}")

    indices_file = f"{OUTPUT_DIR}/feature_{args.matrix}_{args.metric}{suffix}_indices.npy"
    np.save(indices_file, feature_indices)
    print(f"Saved feature indices to {indices_file}")

    # Plot distribution
    plot_output = f"{OUTPUT_DIR}/coactivation_{args.matrix}_{args.metric}{suffix}_hist.png"
    plot_correlation_distribution(correlations, plot_output, metric=args.metric)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

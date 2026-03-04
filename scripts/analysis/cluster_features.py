"""Hierarchical clustering of SAE features based on gene overlap."""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster SAE features')
    parser.add_argument('--layer', type=int, default=12,
                        help='AIDO.Cell layer (9, 12, or 15)')
    parser.add_argument('--metric', type=str, default='overlap_sqrt',
                        choices=['overlap_sqrt', 'overlap', 'go_overlap'],
                        help='Distance metric for clustering')
    parser.add_argument('--linkage-method', type=str, default='ward',
                        choices=['ward', 'average', 'complete', 'single'],
                        help='Hierarchical clustering linkage method')
    parser.add_argument('--max-clusters', type=int, default=100,
                        help='Maximum clusters to test for optimal selection')
    return parser.parse_args()


def plot_dendrogram(linkage_matrix, output_path, max_display=30):
    """Plot dendrogram with truncation for large datasets."""
    fig, ax = plt.subplots(figsize=(15, 8))

    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode='lastp',
        p=max_display,
        show_leaf_counts=True,
        leaf_font_size=10
    )

    ax.set_xlabel('Cluster Size', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title(f'Hierarchical Clustering Dendrogram (truncated to {max_display} clusters)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved dendrogram to {output_path}")
    plt.close()


def determine_optimal_clusters(distance_matrix, linkage_matrix, max_clusters=100):
    """Determine optimal number of clusters using silhouette score."""
    from sklearn.metrics import silhouette_score

    n_features = distance_matrix.shape[0]
    max_k = min(max_clusters, n_features // 2)

    print(f"\nTesting cluster counts from 2 to {max_k}...")
    print("(This may take a while...)\n")

    scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        score = silhouette_score(distance_matrix, labels, metric='precomputed')
        scores.append(score)

        if k <= 10 or k % 10 == 0:
            print(f"  k={k}: silhouette={score:.4f}")

    scores = np.array(scores)
    optimal_idx = np.argmax(scores)
    optimal_k = list(k_range)[optimal_idx]

    print(f"\nOptimal clusters: {optimal_k} (silhouette={scores[optimal_idx]:.4f})")

    # Check if silhouette scores are meaningful
    if scores.max() < 0.1:
        print("\nWARNING: Very low silhouette scores suggest weak or no cluster structure.")
        print("The data may be too sparse for meaningful clustering.")

    return optimal_k, scores, list(k_range)


def plot_silhouette_scores(k_range, scores, optimal_k, output_path):
    """Plot silhouette scores vs number of clusters."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(k_range, scores, 'o-', linewidth=2, markersize=4, color='steelblue')
    ax.axvline(optimal_k, color='red', linestyle='--', linewidth=2,
               label=f'Optimal k={optimal_k} (score={scores[k_range.index(optimal_k)]:.4f})')

    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    # Add reference line at 0
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved silhouette plot to {output_path}")
    plt.close()


def main():
    args = parse_args()

    OUTPUT_DIR = f"plots/sae/layer_{args.layer}/clustering"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"HIERARCHICAL CLUSTERING OF SAE FEATURES")
    print(f"{'='*80}\n")
    print(f"Layer: {args.layer}")
    print(f"Metric: {args.metric}")
    print(f"Linkage: {args.linkage_method}")

    # Load similarity metric
    print(f"\nLoading {args.metric} similarities...")
    COACTIVATION_DIR = f"plots/sae/layer_{args.layer}/coactivation"

    if args.metric == 'overlap_sqrt':
        corr_file = f"{COACTIVATION_DIR}/feature_gene_overlap_sqrt_filtered_correlations.npy"
        indices_file = f"{COACTIVATION_DIR}/feature_gene_overlap_sqrt_filtered_indices.npy"
    elif args.metric == 'overlap':
        corr_file = f"{COACTIVATION_DIR}/feature_gene_overlap_filtered_correlations.npy"
        indices_file = f"{COACTIVATION_DIR}/feature_gene_overlap_filtered_indices.npy"
    elif args.metric == 'go_overlap':
        corr_file = f"{COACTIVATION_DIR}/feature_go_overlap_filtered_similarities.npy"
        indices_file = f"{COACTIVATION_DIR}/feature_go_overlap_filtered_indices.npy"
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    if not os.path.exists(corr_file) or not os.path.exists(indices_file):
        raise FileNotFoundError(f"Similarity files not found. Run compute_feature_correlations.py first.")

    similarities = np.load(corr_file)
    feature_indices = np.load(indices_file)
    n_features = len(feature_indices)

    print(f"  Loaded {len(similarities):,} pairwise similarities for {n_features} features")
    print(f"  Similarity statistics:")
    print(f"    Mean: {similarities.mean():.4f}")
    print(f"    Median: {np.median(similarities):.4f}")
    print(f"    Max: {similarities.max():.4f}")
    print(f"    Pairs > 0.1: {np.sum(similarities > 0.1)} ({100*np.sum(similarities > 0.1)/len(similarities):.2f}%)")

    # Convert similarities to distance matrix
    print(f"\nConverting to distance matrix...")
    distances = 1 - similarities  # Distance = 1 - similarity
    distance_matrix = squareform(distances, checks=False)
    print(f"  Distance matrix shape: {distance_matrix.shape}")

    # Perform hierarchical clustering
    print(f"\nPerforming hierarchical clustering (method={args.linkage_method})...")
    linkage_matrix = linkage(distances, method=args.linkage_method)
    print(f"  Linkage matrix shape: {linkage_matrix.shape}")

    # Plot dendrogram
    dendrogram_path = f"{OUTPUT_DIR}/dendrogram_{args.metric}.png"
    plot_dendrogram(linkage_matrix, dendrogram_path)

    # Determine optimal number of clusters
    optimal_k, scores, k_range = determine_optimal_clusters(
        distance_matrix, linkage_matrix, max_clusters=args.max_clusters
    )

    # Plot silhouette scores
    silhouette_path = f"{OUTPUT_DIR}/silhouette_{args.metric}.png"
    plot_silhouette_scores(k_range, scores, optimal_k, silhouette_path)

    # Cut tree to get cluster assignments
    print(f"\nCutting tree at k={optimal_k}...")
    cluster_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust')

    # Print cluster size distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster size distribution (top 20 by size):")
    for cluster_id, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:20]:
        print(f"  Cluster {cluster_id}: {count} features")
    if len(unique) > 20:
        print(f"  ... and {len(unique) - 20} more clusters")

    # Count singleton clusters
    n_singletons = np.sum(counts == 1)
    print(f"\nSingleton clusters (size=1): {n_singletons}/{len(unique)} ({100*n_singletons/len(unique):.1f}%)")

    # Save results
    results_file = f"{OUTPUT_DIR}/cluster_assignments_{args.metric}_k{optimal_k}.npz"
    np.savez(
        results_file,
        cluster_labels=cluster_labels,
        feature_indices=feature_indices,
        linkage_matrix=linkage_matrix,
        n_clusters=optimal_k,
        silhouette_scores=scores,
        k_range=k_range
    )
    print(f"\nSaved cluster assignments to {results_file}")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

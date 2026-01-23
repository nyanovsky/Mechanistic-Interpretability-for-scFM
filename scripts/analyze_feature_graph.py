"""Graph-based analysis of feature relationships using overlap coefficients.

This script builds a feature similarity graph and analyzes connected components
to find biologically coherent groups of features.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict
from scipy.spatial.distance import squareform

# Import utility functions
sys.path.insert(0, os.path.dirname(__file__))
from sae_analysis_utils import load_go_enrichment


def parse_args():
    parser = argparse.ArgumentParser(description='Graph-based feature clustering')
    parser.add_argument('--layer', type=int, default=12,
                        help='AIDO.Cell layer (9, 12, or 15)')
    parser.add_argument('--expansion', type=int, default=8,
                        help='SAE expansion factor')
    parser.add_argument('--k', type=int, default=32,
                        help='Top-K sparsity')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Overlap threshold for edge creation (default: 0.05)')
    parser.add_argument('--min-cc-size', type=int, default=3,
                        help='Minimum connected component size to analyze (default: 3)')
    parser.add_argument('--metric', type=str, default='overlap_sqrt',
                        choices=['overlap', 'overlap_sqrt'],
                        help='Overlap metric to use (default: overlap_sqrt)')
    return parser.parse_args()


def load_similarity_data(layer, expansion, k, metric):
    """Load similarity matrix and feature indices."""
    OUTPUT_DIR = f"plots/sae/layer_{layer}/coactivation"

    corr_file = f"{OUTPUT_DIR}/feature_gene_{metric}_filtered_correlations.npy"
    indices_file = f"{OUTPUT_DIR}/feature_gene_{metric}_filtered_indices.npy"

    if not os.path.exists(corr_file):
        raise FileNotFoundError(f"Similarity file not found: {corr_file}")
    if not os.path.exists(indices_file):
        raise FileNotFoundError(f"Indices file not found: {indices_file}")

    similarities = np.load(corr_file)
    feature_indices = np.load(indices_file)

    return similarities, feature_indices


def load_go_overlap_data(layer, expansion, k):
    """Load pre-computed GO overlap coefficients."""
    OUTPUT_DIR = f"plots/sae/layer_{layer}/coactivation"

    go_file = f"{OUTPUT_DIR}/feature_go_overlap_filtered_similarities.npy"
    indices_file = f"{OUTPUT_DIR}/feature_go_overlap_filtered_indices.npy"

    if not os.path.exists(go_file):
        print(f"Warning: GO overlap file not found: {go_file}")
        return None, None

    go_overlaps = np.load(go_file)
    go_indices = np.load(indices_file)

    return go_overlaps, go_indices


def build_feature_graph(similarities, feature_indices, threshold):
    """Build graph from similarity matrix with edges above threshold.

    Args:
        similarities: Condensed distance matrix of pairwise similarities
        feature_indices: Array of feature IDs
        threshold: Minimum similarity for edge creation

    Returns:
        G: NetworkX graph
        edge_weights: Dict of edge weights {(i,j): weight}
    """
    n_features = len(feature_indices)

    # Convert condensed to square matrix for easier indexing
    print(f"Converting condensed matrix to square matrix...")
    sim_matrix = squareform(similarities)

    # Build graph
    print(f"Building graph with threshold >= {threshold}...")
    G = nx.Graph()

    # Add all features as nodes
    G.add_nodes_from(feature_indices)

    # Add edges for pairs above threshold
    edge_weights = {}
    n_edges = 0

    for i in range(n_features):
        for j in range(i + 1, n_features):
            weight = sim_matrix[i, j]
            if weight >= threshold:
                feat_i = feature_indices[i]
                feat_j = feature_indices[j]
                G.add_edge(feat_i, feat_j, weight=weight)
                edge_weights[(feat_i, feat_j)] = weight
                n_edges += 1

    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, edge_weights


def analyze_connected_components(G, min_size=3):
    """Find and analyze connected components.

    Args:
        G: NetworkX graph
        min_size: Minimum component size to return

    Returns:
        ccs: List of connected components (each a set of feature IDs)
        cc_stats: Dict with statistics about CCs
    """
    print(f"\nFinding connected components...")
    all_ccs = list(nx.connected_components(G))

    # Filter by size
    ccs = [cc for cc in all_ccs if len(cc) >= min_size]

    # Compute statistics
    all_sizes = [len(cc) for cc in all_ccs]
    filtered_sizes = [len(cc) for cc in ccs]

    cc_stats = {
        'n_total_components': len(all_ccs),
        'n_filtered_components': len(ccs),
        'size_distribution_all': all_sizes,
        'size_distribution_filtered': filtered_sizes,
        'largest_cc_size': max(all_sizes) if all_sizes else 0,
        'n_isolated_nodes': sum(1 for s in all_sizes if s == 1),
    }

    print(f"  Total components: {cc_stats['n_total_components']}")
    print(f"  Components >= {min_size}: {cc_stats['n_filtered_components']}")
    print(f"  Largest component size: {cc_stats['largest_cc_size']}")
    print(f"  Isolated nodes: {cc_stats['n_isolated_nodes']}")

    return ccs, cc_stats


def compute_cc_coherence(cc, G, go_enrichments, go_overlap_matrix, feature_to_idx):
    """Compute biological coherence metrics for a connected component.

    Args:
        cc: Set of feature IDs in the component
        G: NetworkX graph
        go_enrichments: Dict {feature_id: set(GO_terms)}
        go_overlap_matrix: Square matrix of GO overlap coefficients
        feature_to_idx: Dict mapping feature_id to matrix index

    Returns:
        dict: Coherence metrics
    """
    cc_list = list(cc)

    # Edge density
    subgraph = G.subgraph(cc)
    n_nodes = len(cc)
    n_edges = subgraph.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) / 2
    edge_density = n_edges / max_edges if max_edges > 0 else 0

    # GO term overlap using pre-computed coefficients
    if go_overlap_matrix is not None:
        pairwise_go_overlaps = []
        for i, feat_i in enumerate(cc_list):
            for feat_j in cc_list[i+1:]:
                if feat_i in feature_to_idx and feat_j in feature_to_idx:
                    idx_i = feature_to_idx[feat_i]
                    idx_j = feature_to_idx[feat_j]
                    overlap = go_overlap_matrix[idx_i, idx_j]
                    pairwise_go_overlaps.append(overlap)

        mean_go_overlap = np.mean(pairwise_go_overlaps) if pairwise_go_overlaps else 0
    else:
        mean_go_overlap = None

    # Count GO term frequencies across features in CC
    go_term_counts = defaultdict(int)
    for feat in cc_list:
        terms = go_enrichments.get(feat, set())
        for term in terms:
            go_term_counts[term] += 1

    # Shared GO terms (appear in >= 50% of features)
    min_freq = len(cc) * 0.5
    shared_go_terms = {term: count for term, count in go_term_counts.items() if count >= min_freq}

    return {
        'edge_density': edge_density,
        'shared_go_terms': shared_go_terms,
        'mean_pairwise_go_overlap': mean_go_overlap,
        'n_features_with_go': sum(1 for feat in cc_list if go_enrichments.get(feat))
    }


def plot_cc_size_distribution(cc_stats, output_dir, metric):
    """Plot histogram of connected component sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # All components
    sizes = cc_stats['size_distribution_all']
    ax1.hist(sizes, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Component Size', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'All Connected Components (n={cc_stats["n_total_components"]})',
                  fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.axvline(1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Isolated nodes')
    ax1.legend()

    # Filtered components (>= min_size)
    if cc_stats['size_distribution_filtered']:
        sizes_filtered = cc_stats['size_distribution_filtered']
        ax2.hist(sizes_filtered, bins=30, edgecolor='black', alpha=0.7, color='darkgreen')
        ax2.set_xlabel('Component Size', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Filtered Components (n={cc_stats["n_filtered_components"]})',
                      fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'No components meet filter criteria',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Filtered Components (n=0)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    output_path = f"{output_dir}/cc_size_distribution_{metric}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved component size distribution to {output_path}")
    plt.close()


def plot_go_overlap_distribution(go_overlaps, output_dir, metric, threshold):
    """Plot histogram of mean pairwise GO overlap coefficients across CCs."""
    valid_overlaps = [x for x in go_overlaps if x is not None]

    if not valid_overlaps:
        print("No GO overlap data available for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(valid_overlaps, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('Mean Pairwise GO Overlap', fontsize=12)
    ax.set_ylabel('Number of Components', fontsize=12)
    ax.set_title(f'Distribution of GO Overlap Across Components (n={len(valid_overlaps)})',
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    # Add statistics
    mean_val = np.mean(valid_overlaps)
    median_val = np.median(valid_overlaps)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    ax.legend()

    plt.tight_layout()
    output_path = f"{output_dir}/cc_go_overlap_hist_{metric}_t{threshold}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved GO overlap histogram to {output_path}")
    plt.close()


def visualize_component(cc, G, go_enrichments, output_dir, cc_id, metric):
    """Visualize a single connected component as a network."""
    subgraph = G.subgraph(cc)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # Draw nodes
    node_colors = ['lightblue' if go_enrichments.get(node) else 'lightgray'
                   for node in subgraph.nodes()]

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                          node_size=500, alpha=0.9, ax=ax)

    # Draw edges with weights
    edges = subgraph.edges()
    weights = [subgraph[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_edges(subgraph, pos, alpha=0.6, width=2,
                          edge_color=weights, edge_cmap=plt.cm.YlOrRd,
                          edge_vmin=0, edge_vmax=max(weights) if weights else 1,
                          ax=ax)

    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(f'Connected Component {cc_id}\n{len(cc)} features, '
                f'{subgraph.number_of_edges()} edges',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    output_path = f"{output_dir}/cc_{cc_id}_{metric}_network.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved network visualization to {output_path}")
    plt.close()


def main():
    args = parse_args()

    # Paths
    INPUT_DIM = 640
    LATENT_DIM = INPUT_DIM * args.expansion
    BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
    SAE_DIR = f"{BASE_DIR}/sae_k_{args.k}_{LATENT_DIM}"
    INTERPRETATION_DIR = f"{SAE_DIR}/interpretations_filter_zero_expressed"
    OUTPUT_DIR = f"plots/sae/layer_{args.layer}/graph_analysis"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"GRAPH-BASED FEATURE ANALYSIS")
    print(f"{'='*80}")
    print(f"Layer: {args.layer}")
    print(f"Metric: {args.metric}")
    print(f"Edge threshold: {args.threshold}")
    print(f"Min CC size: {args.min_cc_size}")
    print(f"{'='*80}\n")

    # Load gene overlap similarity data (for graph construction)
    print("Loading gene overlap similarity data...")
    similarities, feature_indices = load_similarity_data(
        args.layer, args.expansion, args.k, args.metric
    )
    print(f"Loaded {len(feature_indices)} features, {len(similarities):,} pairwise similarities")

    # Load GO overlap data (for coherence analysis)
    print("\nLoading GO overlap coefficients...")
    go_overlaps, go_indices = load_go_overlap_data(args.layer, args.expansion, args.k)

    go_overlap_matrix = None
    feature_to_idx = None

    if go_overlaps is not None:
        print(f"Loaded {len(go_overlaps):,} GO overlap coefficients")
        # Convert to square matrix
        go_overlap_matrix = squareform(go_overlaps)
        # Create feature ID to index mapping
        feature_to_idx = {feat_id: idx for idx, feat_id in enumerate(go_indices)}
    else:
        print("GO overlap coefficients not available - coherence analysis will be limited")

    # Load GO enrichment
    print("\nLoading GO enrichment...")
    go_enrichments = load_go_enrichment(INTERPRETATION_DIR, p_threshold=0.05, mode='terms')
    print(f"Loaded GO enrichment for {len(go_enrichments)} features")

    # Build graph
    G, edge_weights = build_feature_graph(similarities, feature_indices, args.threshold)

    # Analyze connected components
    ccs, cc_stats = analyze_connected_components(G, min_size=args.min_cc_size)

    # Plot size distribution
    plot_cc_size_distribution(cc_stats, OUTPUT_DIR, args.metric)

    # Analyze individual components
    if ccs:
        print(f"\n{'='*80}")
        print(f"ANALYZING TOP CONNECTED COMPONENTS")
        print(f"{'='*80}\n")

        # Sort by size (descending)
        ccs_sorted = sorted(ccs, key=lambda x: len(x), reverse=True)

        # Analyze top 10 largest components
        n_analyze = min(25, len(ccs_sorted))

        report_lines = []
        report_lines.append(f"CONNECTED COMPONENTS ANALYSIS")
        report_lines.append(f"{'='*80}")
        report_lines.append(f"Threshold: {args.threshold}")
        report_lines.append(f"Metric: {args.metric}")
        report_lines.append(f"Min size: {args.min_cc_size}")
        report_lines.append(f"Total components: {len(ccs)}")
        report_lines.append(f"{'='*80}\n")

        # Collect GO overlap values for all components
        all_go_overlaps = []

        for i, cc in enumerate(ccs_sorted[:n_analyze], 1):
            print(f"\nComponent {i}: {len(cc)} features")

            # Compute coherence
            coherence = compute_cc_coherence(cc, G, go_enrichments, go_overlap_matrix, feature_to_idx)
            all_go_overlaps.append(coherence['mean_pairwise_go_overlap'])

            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"Component {i}")
            report_lines.append(f"{'='*80}")
            report_lines.append(f"Size: {len(cc)} features")
            report_lines.append(f"Features: {sorted(list(cc))}")
            report_lines.append(f"\nCoherence metrics:")
            report_lines.append(f"  Edge density: {coherence['edge_density']:.3f}")

            if coherence['mean_pairwise_go_overlap'] is not None:
                report_lines.append(f"  Mean pairwise GO overlap: {coherence['mean_pairwise_go_overlap']:.3f}")

            report_lines.append(f"  Features with GO annotation: {coherence['n_features_with_go']}/{len(cc)}")

            if coherence['shared_go_terms']:
                report_lines.append(f"\nShared GO terms (>= 50% of features):")
                # Sort by frequency
                shared_sorted = sorted(coherence['shared_go_terms'].items(),
                                      key=lambda x: x[1], reverse=True)
                for term, count in shared_sorted[:20]:
                    report_lines.append(f"  [{count}/{len(cc)}] {term}")
                if len(shared_sorted) > 20:
                    report_lines.append(f"  ... and {len(shared_sorted) - 20} more")
            else:
                report_lines.append(f"\nNo shared GO terms (appearing in >= 50% of features)")

            report_lines.append("")

            # Visualize top 5 largest components
            if i <= 5:
                visualize_component(cc, G, go_enrichments, OUTPUT_DIR, i, args.metric)

        # Write report
        report_path = f"{OUTPUT_DIR}/cc_analysis_{args.metric}_t{args.threshold}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\nSaved detailed report to {report_path}")

        # Plot GO overlap distribution
        plot_go_overlap_distribution(all_go_overlaps, OUTPUT_DIR, args.metric, args.threshold)

    else:
        print(f"\nNo connected components with size >= {args.min_cc_size} found.")
        print(f"Consider:")
        print(f"  - Lowering --threshold (current: {args.threshold})")
        print(f"  - Lowering --min-cc-size (current: {args.min_cc_size})")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

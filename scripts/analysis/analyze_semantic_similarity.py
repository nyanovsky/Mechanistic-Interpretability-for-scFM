import os
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import load_decoder_weights, load_go_enrichment
from utils.go_utils import (
    NAMESPACES, NS_ABBREV,
    load_go_dag_and_associations, separate_by_namespace, compute_max_sim_within_namespace
)
from utils.similarity import compute_cosine_similarity, find_nearest_neighbors


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic similarity analysis of SAE features (Lin/Resnik)')
    parser.add_argument('--layer', type=int, default=12, help='Layer to analyze')
    parser.add_argument('--expansion', type=int, default=8, help='Expansion factor')
    parser.add_argument('--k_neighbors', type=int, default=10, help='Number of neighbors to check')
    parser.add_argument('--n_random_pairs', type=int, default=5000, help='Number of random pairs for baseline')
    return parser.parse_args()


def plot_similarity_comparison(nn_scores, rand_scores, metric_name, output_path):
    """Plot histogram comparing NN vs random similarity distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 50)
    ax.hist(rand_scores, bins=bins, alpha=0.5, label=f'Random pairs (n={len(rand_scores)})',
            density=True, color='gray')
    ax.hist(nn_scores, bins=bins, alpha=0.5, label=f'NN pairs (n={len(nn_scores)})',
            density=True, color='steelblue')

    ax.axvline(np.mean(rand_scores), color='gray', linestyle='--', linewidth=2,
               label=f'Random mean: {np.mean(rand_scores):.3f}')
    ax.axvline(np.mean(nn_scores), color='steelblue', linestyle='--', linewidth=2,
               label=f'NN mean: {np.mean(nn_scores):.3f}')

    ax.set_xlabel(f'{metric_name} Similarity')
    ax.set_ylabel('Density')
    ax.set_title(f'Semantic Similarity: Nearest Neighbors vs Random Pairs ({metric_name})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_namespace_breakdown(ns_scores, metric_name, output_path):
    """Plot per-namespace similarity breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ns in enumerate(NAMESPACES):
        ax = axes[idx]
        abbrev = NS_ABBREV[ns]

        if ns in ns_scores and ns_scores[ns]:
            scores = ns_scores[ns]
            ax.hist(scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(scores):.3f}')
            ax.set_title(f'{abbrev} (n={len(scores)})')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{abbrev}')

        ax.set_xlabel(f'{metric_name} Similarity')
        ax.set_ylabel('Count')

    plt.suptitle(f'Semantic Similarity by GO Namespace ({metric_name})', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    latent_dim = 640 * args.expansion
    sae_dir = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}/sae_k_32_{latent_dim}"
    interpretations_dir = f"{sae_dir}/interpretations_filter_zero_expressed"

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, f"../../plots/sae/layer_{args.layer}")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Load GO Data (using refactored utility)
    godag, term_counts = load_go_dag_and_associations()

    # 2. Load Features & Annotations
    print("Loading features and annotations...")
    weights = load_decoder_weights(sae_dir)
    feature_go_ids = load_go_enrichment(interpretations_dir, mode='ids')
    annotated_feats = list(feature_go_ids.keys())
    print(f"Annotated features: {len(feature_go_ids)}")

    # Pre-compute namespace separation for all features
    print("Separating GO terms by namespace...")
    feature_go_by_ns = {
        feat_id: separate_by_namespace(go_ids, godag)
        for feat_id, go_ids in feature_go_ids.items()
    }

    # 3. Nearest Neighbors Analysis
    print("Finding nearest neighbors...")
    sim_matrix = compute_cosine_similarity(weights)
    nn_indices, nn_similarities = find_nearest_neighbors(sim_matrix, k=args.k_neighbors)

    print("Calculating similarity for Nearest Neighbors...")
    lin_scores = []
    resnik_scores = []
    lin_by_ns = defaultdict(list)
    resnik_by_ns = defaultdict(list)

    for feat_id in tqdm(annotated_feats, desc="NN pairs"):
        feat_gos_ns = feature_go_by_ns[feat_id]
        for rank in range(args.k_neighbors):
            neighbor_id = nn_indices[feat_id, rank]
            if neighbor_id in feature_go_by_ns:
                neighbor_gos_ns = feature_go_by_ns[neighbor_id]

                # Lin similarity
                lin_result = compute_max_sim_within_namespace(
                    feat_gos_ns, neighbor_gos_ns, godag, term_counts, 'lin'
                )
                lin_scores.append(lin_result['max_overall'])
                for ns, score in lin_result['by_namespace'].items():
                    lin_by_ns[ns].append(score)

                # Resnik similarity
                resnik_result = compute_max_sim_within_namespace(
                    feat_gos_ns, neighbor_gos_ns, godag, term_counts, 'resnik'
                )
                resnik_scores.append(resnik_result['max_overall'])
                for ns, score in resnik_result['by_namespace'].items():
                    resnik_by_ns[ns].append(score)

    if not lin_scores:
        print("No annotated neighbor pairs found.")
        return

    # 4. Random Baseline
    print("Calculating Random Baseline...")
    rand_lin = []
    rand_resnik = []
    
    np.random.seed(42)
    n_pairs = min(args.n_random_pairs, len(annotated_feats) * (len(annotated_feats) - 1) // 2)

    pairs_generated = 0
    attempts = 0
    max_attempts = n_pairs * 10

    while pairs_generated < n_pairs and attempts < max_attempts:
        i, j = np.random.choice(len(annotated_feats), size=2, replace=False)
        attempts += 1

        feat_i = annotated_feats[i]
        feat_j = annotated_feats[j]

        gos_a_ns = feature_go_by_ns[feat_i]
        gos_b_ns = feature_go_by_ns[feat_j]

        # Lin
        lin_result = compute_max_sim_within_namespace(
            gos_a_ns, gos_b_ns, godag, term_counts, 'lin'
        )
        rand_lin.append(lin_result['max_overall'])

        # Resnik
        resnik_result = compute_max_sim_within_namespace(
            gos_a_ns, gos_b_ns, godag, term_counts, 'resnik'
        )
        rand_resnik.append(resnik_result['max_overall'])

        pairs_generated += 1
        if pairs_generated % 1000 == 0:
            print(f"  Generated {pairs_generated}/{n_pairs} random pairs...")

    # 5. Generate plots
    print("Generating plots...")
    plot_similarity_comparison(
        lin_scores, rand_lin, 'Lin',
        os.path.join(plot_dir, 'semantic_similarity_lin.png')
    )
    plot_similarity_comparison(
        resnik_scores, rand_resnik, 'Resnik',
        os.path.join(plot_dir, 'semantic_similarity_resnik.png')
    )
    plot_namespace_breakdown(
        lin_by_ns, 'Lin',
        os.path.join(plot_dir, 'semantic_similarity_lin_by_namespace.png')
    )
    plot_namespace_breakdown(
        resnik_by_ns, 'Resnik',
        os.path.join(plot_dir, 'semantic_similarity_resnik_by_namespace.png')
    )

    # Print summary
    print("\n" + "="*70)
    print(f"SEMANTIC SIMILARITY RESULTS (Layer {args.layer}, K={args.k_neighbors})")
    print("="*70)
    print(f"Pairs Analyzed: NN={len(lin_scores)}, Random={len(rand_lin)}")
    print("-" * 70)
    print(f"{ 'Metric':<10} | { 'NN Mean':<10} | { 'Rand Mean':<10} | { 'NN Median':<10} | { 'Rand Median':<10}")
    print("-" * 70)
    print(f"{ 'Lin':<10} | {np.mean(lin_scores):<10.4f} | {np.mean(rand_lin):<10.4f} | {np.median(lin_scores):<10.4f} | {np.median(rand_lin):<10.4f}")
    print(f"{ 'Resnik':<10} | {np.mean(resnik_scores):<10.4f} | {np.mean(rand_resnik):<10.4f} | {np.median(resnik_scores):<10.4f} | {np.median(rand_resnik):<10.4f}")
    print("="*70)

    print(f"\nPlots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
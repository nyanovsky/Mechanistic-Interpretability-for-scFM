#!/usr/bin/env python
"""Find features most similar to a target feature based on GO term similarity.

Compares a source feature from one interpretation directory against all features
in one or more target interpretation directories.

Usage:
    python find_similar_features_cross_layer.py 4367 \
        --source-dir /path/to/source/interpretations \
        --target-dir trained /path/to/trained/interpretations \
        --target-dir random /path/to/random/interpretations
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae_analysis_utils import (
    load_go_enrichment,
    load_go_dag_and_associations,
    separate_by_namespace,
)


def compute_jaccard(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_lin_bma(go_terms_a, go_terms_b, godag, term_counts):
    """
    Compute Best Match Average (BMA) Lin similarity between two GO term sets.

    For each term in A, find best match in B. For each term in B, find best match in A.
    Return the average of all best matches.
    """
    from goatools.semantic import lin_sim

    if not go_terms_a or not go_terms_b:
        return 0.0

    # Separate by namespace (Lin similarity only works within same namespace)
    gos_a_ns = separate_by_namespace(go_terms_a, godag)
    gos_b_ns = separate_by_namespace(go_terms_b, godag)

    best_matches = []

    # For each term in A, find best match in B (within same namespace)
    for ns, terms_a in gos_a_ns.items():
        terms_b = gos_b_ns.get(ns, set())
        if not terms_b:
            continue
        for go_a in terms_a:
            best = 0.0
            for go_b in terms_b:
                try:
                    s = lin_sim(go_a, go_b, godag, term_counts)
                    if s is not None and s > best:
                        best = s
                except Exception:
                    continue
            best_matches.append(best)

    # For each term in B, find best match in A (within same namespace)
    for ns, terms_b in gos_b_ns.items():
        terms_a = gos_a_ns.get(ns, set())
        if not terms_a:
            continue
        for go_b in terms_b:
            best = 0.0
            for go_a in terms_a:
                try:
                    s = lin_sim(go_b, go_a, godag, term_counts)
                    if s is not None and s > best:
                        best = s
                except Exception:
                    continue
            best_matches.append(best)

    if not best_matches:
        return 0.0

    return sum(best_matches) / len(best_matches)


def main():
    parser = argparse.ArgumentParser(
        description="Find similar features based on GO term similarity"
    )
    parser.add_argument(
        "target_feature",
        type=int,
        help="Target feature ID in the source directory"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Interpretation directory containing the target feature"
    )
    parser.add_argument(
        "--target-dir",
        nargs=2,
        action='append',
        metavar=('LABEL', 'PATH'),
        required=True,
        help="Target interpretation directory: LABEL PATH (can be repeated)"
    )
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        default=10,
        help="Number of top similar features to show per target (default: 10)"
    )
    parser.add_argument(
        "-m", "--metric",
        type=str,
        choices=['jaccard', 'lin'],
        default='jaccard',
        help="Similarity metric: 'jaccard' (set overlap) or 'lin' (BMA semantic similarity) (default: jaccard)"
    )
    parser.add_argument(
        "-p", "--p-threshold",
        type=float,
        default=0.05,
        help="Adjusted p-value threshold for GO enrichment (default: 0.05)"
    )
    args = parser.parse_args()

    # Load source feature GO enrichment
    print(f"Loading source GO enrichments from:\n  {args.source_dir}")
    source_enrichments = load_go_enrichment(
        args.source_dir, p_threshold=args.p_threshold, mode='ids'
    )

    if args.target_feature not in source_enrichments:
        print(f"\nError: Feature {args.target_feature} not found in source GO enrichments.")
        print(f"Available features with GO enrichment: {len(source_enrichments)}")
        sys.exit(1)

    target_go_terms = source_enrichments[args.target_feature]
    print(f"\n{'='*70}")
    print(f"Target: Feature {args.target_feature} ({len(target_go_terms)} GO terms)")
    print(f"Metric: {args.metric}")
    print(f"GO terms: {sorted(target_go_terms)[:10]}{'...' if len(target_go_terms) > 10 else ''}")
    print(f"{'='*70}")

    # Load target directories
    target_dirs = {label: path for label, path in args.target_dir}
    target_enrichments = {}
    for label, path in target_dirs.items():
        print(f"\nLoading '{label}' GO enrichments from:\n  {path}")
        target_enrichments[label] = load_go_enrichment(
            path, p_threshold=args.p_threshold, mode='ids'
        )

    # Load GO DAG if using Lin similarity
    godag, term_counts = None, None
    if args.metric == 'lin':
        print("\nLoading GO DAG for Lin similarity...")
        godag, term_counts = load_go_dag_and_associations()

    # Compute similarity against each target directory
    from tqdm import tqdm

    for label in target_dirs:
        print(f"\n--- {label} ---")

        similarities = []
        items = target_enrichments[label].items()
        if args.metric == 'lin':
            items = tqdm(items, desc=f"Computing BMA Lin similarity")

        for feat_id, go_terms in items:
            if args.metric == 'jaccard':
                sim = compute_jaccard(target_go_terms, go_terms)
            else:  # lin
                sim = compute_lin_bma(target_go_terms, go_terms, godag, term_counts)
            similarities.append((feat_id, sim, len(go_terms)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Print top-n
        print(f"Top {args.top_n} most similar features:")
        print("-" * 60)

        for feat_id, sim, n_terms in similarities[:args.top_n]:
            feat_go_terms = target_enrichments[label][feat_id]
            overlap = target_go_terms & feat_go_terms

            print(f"\nFeature {feat_id} | {args.metric}: {sim:.4f} | #GO terms: {n_terms} | #Overlap: {len(overlap)}")

            # Print GO terms: if more than 20, just show overlapping ones
            if n_terms <= 20:
                print(f"  All GO terms: {sorted(feat_go_terms)}")
            else:
                print(f"  Overlapping GO terms ({len(overlap)}): {sorted(overlap)}")


if __name__ == "__main__":
    main()

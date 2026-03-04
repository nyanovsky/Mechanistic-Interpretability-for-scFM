"""Compute and cache SAE quality metrics for later comparison.

Saves all metrics to {interp_dir}/summary/ so they can be loaded
by plot_sae_comparison.py without recomputation.

Metrics computed:
1. Max IC per enriched feature
2. GO term frequency counts across features (+ exclusive terms)
3. Pairwise GO Ochiai coefficients (sampled)
4. Pairwise gene set Ochiai coefficients (sampled)

Usage:
    python scripts/compute_sae_summary.py \
        --interp-dir /path/to/interpretations_dir \
        --n-pairs 500000
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import load_go_enrichment, get_expressed_genes_mask
from utils.go_utils import load_go_dag_and_associations, compute_all_go_dag_ics, compute_go_term_overlap
from utils.similarity import get_top_k_genes_per_feature

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_FILE = os.path.join(SCRIPT_DIR, "../../data/pbmc/pbmc3k_raw.h5ad")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SAE quality summary metrics")
    parser.add_argument('--interp-dir', type=str, required=True,
                        help="Path to interpretations directory")
    parser.add_argument('--n-pairs', type=int, default=500_000,
                        help="Number of pairs to sample for pairwise metrics (0 = all)")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pr-scale', type=float, default=1.0)
    parser.add_argument('--min-genes', type=int, default=10)
    parser.add_argument('--max-genes', type=int, default=100)
    parser.add_argument('--force', action='store_true',
                        help="Recompute all metrics even if files exist")
    return parser.parse_args()


def exists(summary_dir, filename):
    return os.path.exists(os.path.join(summary_dir, filename))


def sample_pairs(feature_ids, n, seed=42):
    """Sample n unique pairs from feature_ids."""
    rng = np.random.default_rng(seed)
    ids = np.array(feature_ids)
    n_feat = len(ids)
    i_idx = rng.integers(0, n_feat, size=n)
    j_idx = rng.integers(0, n_feat, size=n)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    pairs = [(int(ids[min(a, b)]), int(ids[max(a, b)])) for a, b in zip(i_idx, j_idx)]
    return list(set(pairs))[:n]


def main():
    args = parse_args()
    interp_dir = args.interp_dir
    summary_dir = os.path.join(interp_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    # 1. Load GO enrichments
    print("=== Loading GO enrichments ===")
    go_ids = load_go_enrichment(interp_dir, p_threshold=0.05, mode='ids')
    enriched_features = sorted(go_ids.keys())
    n_enriched = len(enriched_features)

    # Count total features from feature_gene_matrix
    fg_path = os.path.join(interp_dir, 'feature_gene_matrix.npy')
    n_total = np.load(fg_path, mmap_mode='r').shape[0] if os.path.exists(fg_path) else n_enriched
    print(f"  {n_enriched}/{n_total} features enriched ({100*n_enriched/n_total:.1f}%)")

    # 2. Load GO DAG + compute ICs (needed for max ICs)
    need_ics = args.force or not exists(summary_dir, 'max_ics.npy')
    need_terms = args.force or not exists(summary_dir, 'term_counts.json') or not exists(summary_dir, 'exclusive_counts.npy')
    need_go_ochiai = args.force or not exists(summary_dir, 'go_ochiai.npy')
    need_gene_ochiai = args.force or not exists(summary_dir, 'gene_ochiai.npy')

    godag, term_counts_obj, all_term_ics = None, None, None
    if need_ics:
        print("\n=== Loading GO DAG and computing ICs ===")
        godag, term_counts_obj = load_go_dag_and_associations()
        all_term_ics = compute_all_go_dag_ics(godag, term_counts_obj)

    # 3. Max IC per feature
    if need_ics:
        print("\n=== Computing max IC per feature ===")
        max_ics = {}
        for fid, terms in go_ids.items():
            ics = [all_term_ics[t] for t in terms if t in all_term_ics]
            if ics:
                max_ics[fid] = max(ics)
        max_ic_array = np.array([max_ics.get(f, 0.0) for f in enriched_features])
        np.save(os.path.join(summary_dir, 'max_ics.npy'), max_ic_array)
        np.save(os.path.join(summary_dir, 'feature_indices.npy'), np.array(enriched_features))
        print(f"  Median max IC: {np.median(max_ic_array):.3f}")
    else:
        print("\n=== Max IC: already computed, skipping ===")
        max_ic_array = np.load(os.path.join(summary_dir, 'max_ics.npy'))

    # 4. GO term frequency counts + exclusive counts
    if need_terms:
        print("\n=== Computing GO term frequencies ===")
        term_freq = {}
        for terms in go_ids.values():
            for t in terms:
                term_freq[t] = term_freq.get(t, 0) + 1
        with open(os.path.join(summary_dir, 'term_counts.json'), 'w') as f:
            json.dump(term_freq, f)
        n_exclusive = sum(1 for c in term_freq.values() if c == 1)
        print(f"  {len(term_freq)} unique terms, {n_exclusive} exclusive (count=1)")

        exclusive_counts = np.array([
            sum(1 for t in go_ids[f] if term_freq[t] == 1)
            for f in enriched_features
        ])
        np.save(os.path.join(summary_dir, 'exclusive_counts.npy'), exclusive_counts)
        print(f"  Features with >=1 exclusive term: {(exclusive_counts > 0).sum()}/{n_enriched}")
    else:
        print("\n=== Term frequencies + exclusive counts: already computed, skipping ===")
        with open(os.path.join(summary_dir, 'term_counts.json')) as f:
            term_freq = json.load(f)
        n_exclusive = sum(1 for c in term_freq.values() if c == 1)

    # 5. Sample pairs (needed for both pairwise metrics)
    pairs = None
    if need_go_ochiai or need_gene_ochiai:
        if args.n_pairs > 0:
            total_possible = n_enriched * (n_enriched - 1) // 2
            if args.n_pairs >= total_possible:
                pairs = None
                print(f"\n  Computing all {total_possible:,} pairs")
            else:
                pairs = sample_pairs(enriched_features, args.n_pairs)
                print(f"\n  Sampled {len(pairs):,} pairs")
        else:
            pairs = None
            print(f"\n  Computing all pairs")

    # 6. Pairwise GO Ochiai
    if need_go_ochiai:
        print("\n=== Pairwise GO Ochiai ===")
        go_ochiai = compute_go_term_overlap(go_ids, feature_pairs=pairs, metric='ochiai',
                                             verbose=True, n_workers=args.workers)
        go_scores = np.array(list(go_ochiai.values()), dtype=np.float32)
        np.save(os.path.join(summary_dir, 'go_ochiai.npy'), go_scores)
        print(f"  {len(go_scores):,} pairs, median={np.median(go_scores):.4f}")
    else:
        print("\n=== GO Ochiai: already computed, skipping ===")
        go_scores = np.load(os.path.join(summary_dir, 'go_ochiai.npy'))

    # 7. Pairwise gene set Ochiai
    if need_gene_ochiai:
        print("\n=== Pairwise gene set Ochiai ===")
        feature_gene_matrix = np.load(fg_path)
        pr_path = os.path.join(interp_dir, 'feature_participation_ratios.npy')
        pr_values = np.load(pr_path)

        print("  Filtering to expressed genes...")
        expressed_mask = get_expressed_genes_mask(RAW_DATA_FILE, min_mean_expr=0.01, min_pct_cells=0.5)
        feature_gene_matrix = feature_gene_matrix[:, expressed_mask]
        print(f"  {expressed_mask.sum()} expressed genes (of {len(expressed_mask)})")

        all_gene_sets = get_top_k_genes_per_feature(
            feature_gene_matrix, pr_values,
            pr_scale=args.pr_scale, min_genes=args.min_genes, max_genes=args.max_genes
        )
        gene_sets_dict = {f: all_gene_sets[f] for f in enriched_features}

        gene_ochiai = compute_go_term_overlap(gene_sets_dict, feature_pairs=pairs, metric='ochiai',
                                               verbose=True, n_workers=args.workers)
        gene_scores = np.array(list(gene_ochiai.values()), dtype=np.float32)
        np.save(os.path.join(summary_dir, 'gene_ochiai.npy'), gene_scores)
        print(f"  {len(gene_scores):,} pairs, median={np.median(gene_scores):.4f}")
    else:
        print("\n=== Gene Ochiai: already computed, skipping ===")
        gene_scores = np.load(os.path.join(summary_dir, 'gene_ochiai.npy'))

    # 8. Metadata
    metadata = {
        'interp_dir': os.path.abspath(interp_dir),
        'n_features_total': int(n_total),
        'n_features_enriched': int(n_enriched),
        'enrichment_rate': round(n_enriched / n_total, 4),
        'n_unique_terms': len(term_freq),
        'n_exclusive_terms': n_exclusive,
        'n_pairs_sampled': len(go_scores),
        'pr_scale': args.pr_scale,
        'min_genes': args.min_genes,
        'max_genes': args.max_genes,
    }
    with open(os.path.join(summary_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Summary saved to {summary_dir} ===")
    print(f"  Enrichment rate: {metadata['enrichment_rate']*100:.1f}%")
    print(f"  Median max IC: {np.median(max_ic_array):.3f}")
    print(f"  Exclusive terms: {n_exclusive}")
    print(f"  Median GO Ochiai: {np.median(go_scores):.4f}")
    print(f"  Median gene Ochiai: {np.median(gene_scores):.4f}")


if __name__ == '__main__':
    main()

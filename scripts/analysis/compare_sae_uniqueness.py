"""Compare feature uniqueness across SAEs (e.g. trained vs random).

Computes for each SAE:
1. Per-feature exclusive GO terms (terms not shared with any other feature)
2. Pairwise GO term Ochiai over sampled pairs
3. Pairwise gene set Ochiai over sampled pairs

Expects feature_gene_matrix.npy and feature_participation_ratios.npy
to live directly inside each --interp-dir.

Usage:
    python scripts/compare_sae_uniqueness.py \
        --interp-dirs /path/to/trained/interp /path/to/random/interp \
        --labels "Trained" "Random" \
        --output-dir plots/sae/layer_12_bm/uniqueness_comparison
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import load_go_enrichment
from utils.go_utils import compute_go_term_overlap
from utils.similarity import get_top_k_genes_per_feature


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interp-dirs', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--n-pairs', type=int, default=500_000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pr-scale', type=float, default=1.0)
    parser.add_argument('--min-genes', type=int, default=10)
    parser.add_argument('--max-genes', type=int, default=100)
    return parser.parse_args()


def exclusive_go_counts(go_enrichments):
    term_freq = {}
    for terms in go_enrichments.values():
        for t in terms:
            term_freq[t] = term_freq.get(t, 0) + 1
    return {fid: sum(1 for t in terms if term_freq[t] == 1)
            for fid, terms in go_enrichments.items()}


def sample_pairs(feature_ids, n, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.array(feature_ids)
    n_feat = len(ids)
    i_idx = rng.integers(0, n_feat, size=n)
    j_idx = rng.integers(0, n_feat, size=n)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    # Ensure i < j for consistency
    pairs = [(int(ids[min(a, b)]), int(ids[max(a, b)])) for a, b in zip(i_idx, j_idx)]
    return list(set(pairs))[:n]


def main():
    args = parse_args()
    assert len(args.interp_dirs) == len(args.labels)
    os.makedirs(args.output_dir, exist_ok=True)
    colors = ['#E8836A', '#5BA89E', '#7B68C8', '#4DAF7C'][:len(args.labels)]

    # Load GO enrichments
    print("=== Loading GO enrichments ===")
    all_go = [load_go_enrichment(d, p_threshold=0.05, mode='ids') for d in args.interp_dirs]
    for label, go in zip(args.labels, all_go):
        print(f"  {label}: {len(go)} features annotated")

    common = sorted(set.intersection(*[set(go.keys()) for go in all_go]))
    print(f"Common annotated features: {len(common)}")
    pairs = sample_pairs(common, args.n_pairs)
    print(f"Sampled {len(pairs):,} pairs\n")

    # Analysis 1: exclusive GO terms
    print("=== Exclusive GO terms ===")
    excl_data = []
    for label, go in zip(args.labels, all_go):
        counts = np.array(list(exclusive_go_counts(go).values()))
        excl_data.append(counts)
        print(f"  {label}: median={np.median(counts):.1f}  mean={np.mean(counts):.1f}"
              f"  %>0={100*(counts>0).mean():.1f}%")

    # Analysis 2: pairwise GO Ochiai
    print("\n=== Pairwise GO Ochiai ===")
    go_och = []
    for label, go in zip(args.labels, all_go):
        res = compute_go_term_overlap(go, feature_pairs=pairs, metric='ochiai',
                                      verbose=True, n_workers=args.workers)
        scores = np.array(list(res.values()))
        go_och.append(scores)
        print(f"  {label}: mean={np.mean(scores):.4f}  median={np.median(scores):.4f}")

    # Analysis 3: pairwise gene set Ochiai
    print("\n=== Pairwise gene set Ochiai ===")
    gene_och = []
    for label, idir in zip(args.labels, args.interp_dirs):
        gene_sets_list = get_top_k_genes_per_feature(
            np.load(os.path.join(idir, 'feature_gene_matrix.npy')),
            np.load(os.path.join(idir, 'feature_participation_ratios.npy')),
            pr_scale=args.pr_scale, min_genes=args.min_genes, max_genes=args.max_genes
        )
        gene_sets_dict = {i: s for i, s in enumerate(gene_sets_list)}
        res = compute_go_term_overlap(gene_sets_dict, feature_pairs=pairs, metric='ochiai',
                                      verbose=True, n_workers=args.workers)
        scores = np.array(list(res.values()))
        gene_och.append(scores)
        print(f"  {label}: mean={np.mean(scores):.4f}  median={np.median(scores):.4f}")

    def violin(ax, data, title, ylabel):
        pct_zeros = [100 * (d == 0).mean() for d in data]
        filtered = [d[d > 0] for d in data]
        parts = ax.violinplot(filtered, positions=range(len(args.labels)),
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts['bodies'], colors):
            pc.set_facecolor(c); pc.set_edgecolor('black'); pc.set_alpha(0.8); pc.set_linewidth(0.8)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        ax.set_xticks(range(len(args.labels))); ax.set_xticklabels(args.labels, fontsize=10)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='black', lw=0.8)
                   for c in colors[:len(args.labels)]]
        legend_labels = [f"{l} ({p:.1f}% zeros)"
                         for l, p in zip(args.labels, pct_zeros)]
        ax.legend(handles, legend_labels, fontsize=8, loc='upper right')

    # Plot
    datasets = [('Exclusive GO Terms per Feature', 'Exclusive GO term count', excl_data),
                ('Pairwise GO Ochiai', 'Ochiai coefficient', go_och),
                ('Pairwise Gene Set Ochiai', 'Ochiai coefficient', gene_och)]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (title, ylabel, data) in zip(axes, datasets):
        violin(ax, data, title, ylabel)

    plt.tight_layout()
    out = os.path.join(args.output_dir, 'sae_uniqueness_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()

"""Plot SAE quality comparison from precomputed summaries.

Accepts 1 or 2 summary directories (produced by compute_sae_summary.py).
With 1 directory: plots single distributions.
With 2 directories: plots side-by-side violins for comparison.

Usage:
    # Single SAE
    python scripts/plot_sae_comparison.py \
        --summary-dirs /path/to/interp/summary \
        --labels "Trained" \
        --output-dir plots/sae/comparison

    # Two SAEs side by side
    python scripts/plot_sae_comparison.py \
        --summary-dirs /path/to/sae1/summary /path/to/sae2/summary \
        --labels "No b_pre" "With b_pre" \
        --output-dir plots/sae/comparison
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SAE quality comparison")
    parser.add_argument('--summary-dirs', nargs='+', required=True,
                        help="1 or 2 summary directories")
    parser.add_argument('--labels', nargs='+', required=True,
                        help="Labels for each SAE")
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--title', type=str, default=None,
                        help="Optional suptitle for the figure")
    return parser.parse_args()


def load_summary(summary_dir):
    """Load all precomputed summary data."""
    data = {}
    data['max_ics'] = np.load(os.path.join(summary_dir, 'max_ics.npy'))
    data['feature_indices'] = np.load(os.path.join(summary_dir, 'feature_indices.npy'))
    data['exclusive_counts'] = np.load(os.path.join(summary_dir, 'exclusive_counts.npy'))
    data['go_ochiai'] = np.load(os.path.join(summary_dir, 'go_ochiai.npy'))
    data['gene_ochiai'] = np.load(os.path.join(summary_dir, 'gene_ochiai.npy'))
    with open(os.path.join(summary_dir, 'term_counts.json')) as f:
        data['term_counts'] = json.load(f)
    with open(os.path.join(summary_dir, 'metadata.json')) as f:
        data['metadata'] = json.load(f)
    return data


def gini_coefficient(values):
    """Compute Gini coefficient for a distribution."""
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def violin_plot(ax, datasets, labels, colors, title, ylabel, filter_zeros=False):
    """Create violin plot, optionally filtering zeros and reporting their %."""
    if filter_zeros:
        pct_zeros = [100 * (d == 0).mean() for d in datasets]
        filtered = [d[d > 0] for d in datasets]
    else:
        pct_zeros = [0.0] * len(datasets)
        filtered = datasets

    parts = ax.violinplot(filtered, positions=range(len(labels)),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
        pc.set_linewidth(0.8)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if filter_zeros:
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='black', lw=0.8)
                   for c in colors[:len(labels)]]
        legend_labels = [f"{l} ({p:.1f}% zeros)" for l, p in zip(labels, pct_zeros)]
        ax.legend(handles, legend_labels, fontsize=8, loc='upper right')


def main():
    args = parse_args()
    assert len(args.summary_dirs) == len(args.labels), "Need same number of dirs and labels"
    assert len(args.summary_dirs) in (1, 2), "Accepts 1 or 2 summary directories"
    os.makedirs(args.output_dir, exist_ok=True)

    colors = ['#E8836A', '#5BA89E'][:len(args.labels)]
    summaries = [load_summary(d) for d in args.summary_dirs]

    # Print summary table
    print("\n=== Summary Statistics ===")
    print(f"{'Metric':<30s}", end="")
    for l in args.labels:
        print(f"  {l:>15s}", end="")
    print()
    print("-" * (30 + 17 * len(args.labels)))

    rows = [
        ("Enrichment rate", [f"{s['metadata']['enrichment_rate']*100:.1f}%" for s in summaries]),
        ("Median max IC", [f"{np.median(s['max_ics']):.3f}" for s in summaries]),
        ("Unique GO terms", [f"{s['metadata']['n_unique_terms']}" for s in summaries]),
        ("Exclusive GO terms", [f"{s['metadata']['n_exclusive_terms']}" for s in summaries]),
        ("Median GO Ochiai (>0)", [f"{np.median(s['go_ochiai'][s['go_ochiai']>0]):.4f}"
                                    if (s['go_ochiai']>0).any() else "N/A" for s in summaries]),
        ("Median gene Ochiai (>0)", [f"{np.median(s['gene_ochiai'][s['gene_ochiai']>0]):.4f}"
                                      if (s['gene_ochiai']>0).any() else "N/A" for s in summaries]),
    ]
    for name, values in rows:
        print(f"{name:<30s}", end="")
        for v in values:
            print(f"  {v:>15s}", end="")
        print()

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))

    # 1. Max IC violin
    violin_plot(axes[0],
                [s['max_ics'] for s in summaries],
                args.labels, colors,
                'Max IC per Feature', 'Information Content')

    # 2. Term frequency violin
    freq_data = []
    gini_values = []
    for s in summaries:
        freqs = np.array(list(s['term_counts'].values()))
        freq_data.append(freqs)
        gini_values.append(gini_coefficient(freqs))
    legend_labels = [f"{l} (Gini={g:.3f})" for l, g in zip(args.labels, gini_values)]
    violin_plot(axes[1], freq_data, args.labels, colors,
                'GO Term Frequency', 'Features per term')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='black', lw=0.8)
               for c in colors[:len(args.labels)]]
    axes[1].legend(handles, legend_labels, fontsize=8, loc='upper right')

    # 3. GO Ochiai violin
    violin_plot(axes[2],
                [s['go_ochiai'] for s in summaries],
                args.labels, colors,
                'Pairwise GO Ochiai', 'Ochiai coefficient',
                filter_zeros=True)

    # 4. Exclusive GO terms per feature violin
    violin_plot(axes[3],
                [s['exclusive_counts'] for s in summaries],
                args.labels, colors,
                'Exclusive GO Terms per Feature', 'Exclusive GO term count',
                filter_zeros=True)

    # 5. Gene Ochiai violin
    violin_plot(axes[4],
                [s['gene_ochiai'] for s in summaries],
                args.labels, colors,
                'Pairwise Gene Set Ochiai', 'Ochiai coefficient',
                filter_zeros=True)

    if args.title:
        fig.suptitle(args.title, fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(args.output_dir, 'sae_quality_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()

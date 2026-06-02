"""Committee directionality analyzer (D statistic).

Computes the paired discriminator
    D = (Δ_A − Δ_A_rand) − (Δ_B − Δ_B_rand)
per cell, per celltype, where:
  - Δ_A      = logit(IFNG  | committee_A inputs overridden, IFNG  masked) − baseline
  - Δ_A_rand = logit(IFNG  | A_random   inputs overridden,  IFNG  masked) − baseline
  - Δ_B      = logit(TBX21 | committee_B inputs overridden, TBX21 masked) − baseline
  - Δ_B_rand = logit(TBX21 | B_random   inputs overridden,  TBX21 masked) − baseline

D > 0 → directional encoding (forward dominates).
D ≈ 0 → symmetric coupling.
D < 0 → reverse dominant.

Reports per-celltype CSV rows for each Δ and D (mean, bootstrap 95% CI,
sign-flip permutation p, Wilcoxon signed-rank p, n_cells) plus a 2-panel
PNG (bar chart + dA_corr-vs-dB_corr scatter).

Usage:
    python analyze_committee_directionality.py \\
        --cd4-results results/mlm_perturbation/committee_ifng_tbx21_cd4.pt \\
        --cd8-results results/mlm_perturbation/committee_ifng_tbx21_cd8.pt \\
        --baseline-catalog results/masked_baselines/tbx21_cd4_cd8.pt \\
        --output-dir plots/pbmc/mlm_pert/committee
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


ANCHOR_TF = 'TBX21'
ANCHOR_TARGET = 'IFNG'

CELLTYPE_SUFFIX = {
    'CD4 T cells': 'cd4',
    'CD8 T cells': 'cd8',
}


def sign_flip_permutation(a, n_perm=10000, seed=0):
    observed = float(a.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(a)))
    null = (signs * a).mean(axis=1)
    pval = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, float(pval)


def bootstrap_ci(a, n_boot=5000, seed=0, ci=95):
    rng = np.random.default_rng(seed)
    n = len(a)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = a[idx].mean()
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def wilcoxon_pval(x):
    """Two-sided Wilcoxon signed-rank on x vs zero."""
    try:
        _, p = wilcoxon(x, alternative='two-sided', zero_method='wilcox')
        return float(p)
    except ValueError:
        return float('nan')


def summarize(name, arr, n_perm, n_boot, seed):
    arr = np.asarray(arr, dtype=np.float64)
    mean, perm_p = sign_flip_permutation(arr, n_perm=n_perm, seed=seed)
    lo, hi = bootstrap_ci(arr, n_boot=n_boot, seed=seed)
    wp = wilcoxon_pval(arr)
    return {
        'quantity': name,
        'mean': mean,
        'ci_lo': lo,
        'ci_hi': hi,
        'perm_p': perm_p,
        'wilcoxon_p': wp,
        'n_cells': int(len(arr)),
    }


def analyze_celltype(results_path, catalog, n_perm, n_boot):
    payload = torch.load(results_path, map_location='cpu', weights_only=False)
    meta = payload.get('metadata', {})
    if meta.get('experiment_kind') != 'committee_perturbation':
        raise SystemExit(f"{results_path}: unexpected experiment_kind "
                         f"{meta.get('experiment_kind')}")
    if meta.get('anchor_tf') != ANCHOR_TF or meta.get('anchor_target') != ANCHOR_TARGET:
        raise SystemExit(f"{results_path}: anchor mismatch "
                         f"({meta.get('anchor_tf')}, {meta.get('anchor_target')})")
    celltype = payload['celltype']
    if celltype not in CELLTYPE_SUFFIX:
        raise SystemExit(f"{results_path}: unknown celltype '{celltype}'")
    cell_names = list(payload['cell_names'])

    # Pair to catalog by cell_names
    cat_name_to_idx = {n: i for i, n in enumerate(catalog['cell_names'])}
    cat_idx = np.array([cat_name_to_idx[n] for n in cell_names])
    cat_celltypes = [catalog['cell_celltype'][i] for i in cat_idx]
    if any(ct != celltype for ct in cat_celltypes):
        raise SystemExit(f"{results_path}: celltype mismatch with catalog rows")
    gene_order = list(catalog['gene_order'])
    col_ifng = gene_order.index(ANCHOR_TARGET)
    col_tbx21 = gene_order.index(ANCHOR_TF)
    b_ifng = catalog['logits'][ANCHOR_TARGET][cat_idx, col_ifng].astype(np.float64)
    b_tbx21 = catalog['logits'][ANCHOR_TF][cat_idx, col_tbx21].astype(np.float64)

    logits = payload['logits']
    for k in ('A_committee', 'A_random', 'B_committee', 'B_random'):
        if k not in logits:
            raise SystemExit(f"{results_path}: missing pass '{k}'")
        if len(logits[k]) != len(cell_names):
            raise SystemExit(f"{results_path}: pass '{k}' has "
                             f"{len(logits[k])} logits, expected {len(cell_names)}")

    dA = logits['A_committee'].astype(np.float64) - b_ifng
    dA_rand = logits['A_random'].astype(np.float64) - b_ifng
    dB = logits['B_committee'].astype(np.float64) - b_tbx21
    dB_rand = logits['B_random'].astype(np.float64) - b_tbx21
    dA_corr = dA - dA_rand
    dB_corr = dB - dB_rand
    D = dA_corr - dB_corr

    quantities = [
        ('dA', dA, 0),
        ('dA_rand', dA_rand, 1),
        ('dB', dB, 2),
        ('dB_rand', dB_rand, 3),
        ('dA_corr', dA_corr, 4),
        ('dB_corr', dB_corr, 5),
        ('D', D, 6),
    ]
    rows = [summarize(name, arr, n_perm, n_boot, seed)
            for name, arr, seed in quantities]
    df = pd.DataFrame(rows)
    arrays = {name: arr for name, arr, _ in quantities}
    return celltype, df, arrays


def make_plot(celltype, df, arrays, output_dir, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axL = axes[0]
    order = ['dA', 'dA_rand', 'dB', 'dB_rand', 'dA_corr', 'dB_corr', 'D']
    df_indexed = df.set_index('quantity').loc[order]
    means = df_indexed['mean'].values
    lo = df_indexed['ci_lo'].values
    hi = df_indexed['ci_hi'].values
    perm_p = df_indexed['perm_p'].values
    yerr = np.vstack([means - lo, hi - means])
    colors = ['#3498db', '#7f8c8d', '#e67e22', '#7f8c8d',
              '#2ecc71', '#f39c12', '#9b59b6']
    x = np.arange(len(order))
    axL.bar(x, means, color=colors)
    axL.errorbar(x, means, yerr=yerr, fmt='none', ecolor='k',
                 capsize=2, lw=0.8)
    for i, (m, p) in enumerate(zip(means, perm_p)):
        if p < 0.05:
            axL.text(x[i], m, '*', ha='center',
                     va='bottom' if m >= 0 else 'top', fontsize=12)
    axL.axhline(0, color='k', lw=0.5)
    axL.set_xticks(x)
    axL.set_xticklabels(order, rotation=30, ha='right')
    axL.set_ylabel('per-cell paired mean (logit Δ)')
    axL.set_title(f'{celltype}: Δ components + D')
    axL.grid(axis='y', alpha=0.3)

    axR = axes[1]
    a = arrays['dA_corr']
    b = arrays['dB_corr']
    axR.scatter(a, b, s=8, alpha=0.4, color='#34495e')
    lim_lo = float(min(a.min(), b.min()))
    lim_hi = float(max(a.max(), b.max()))
    pad = 0.05 * (lim_hi - lim_lo + 1e-9)
    lo_lim = lim_lo - pad
    hi_lim = lim_hi + pad
    axR.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color='k', lw=0.8,
             label='y = x')
    D_mean = float(df.set_index('quantity').loc['D', 'mean'])
    # D = dA_corr - dB_corr  =>  dB_corr = dA_corr - D_mean
    axR.plot([lo_lim, hi_lim], [lo_lim - D_mean, hi_lim - D_mean],
             color='#9b59b6', lw=1.2, ls='--',
             label=f'y = x − mean(D)  (D={D_mean:+.3f})')
    axR.set_xlim(lo_lim, hi_lim)
    axR.set_ylim(lo_lim, hi_lim)
    axR.set_xlabel('dA_corr  (Δ_A − Δ_A_rand)')
    axR.set_ylabel('dB_corr  (Δ_B − Δ_B_rand)')
    axR.set_title(f'{celltype}: per-cell dA_corr vs dB_corr')
    axR.legend(fontsize=8, loc='best')
    axR.grid(alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(output_dir, f'D_{suffix}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--cd4-results', default=None)
    parser.add_argument('--cd8-results', default=None)
    parser.add_argument('--baseline-catalog', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-perm', type=int, default=10000)
    parser.add_argument('--n-boot', type=int, default=5000)
    args = parser.parse_args()

    if not args.cd4_results and not args.cd8_results:
        raise SystemExit("Provide at least one of --cd4-results / --cd8-results")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading baseline catalog: {args.baseline_catalog}")
    catalog = torch.load(args.baseline_catalog, map_location='cpu',
                         weights_only=False)

    todo = []
    if args.cd4_results:
        todo.append(args.cd4_results)
    if args.cd8_results:
        todo.append(args.cd8_results)

    all_summaries = []
    for path in todo:
        print(f"\n--- Analyzing {path} ---")
        celltype, df, arrays = analyze_celltype(
            path, catalog, n_perm=args.n_perm, n_boot=args.n_boot,
        )
        suffix = CELLTYPE_SUFFIX[celltype]
        csv_path = os.path.join(args.output_dir, f'asymmetry_{suffix}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")
        print(df.to_string(index=False,
                           formatters={
                               'mean': '{:+.4f}'.format,
                               'ci_lo': '{:+.4f}'.format,
                               'ci_hi': '{:+.4f}'.format,
                               'perm_p': '{:.4g}'.format,
                               'wilcoxon_p': '{:.4g}'.format,
                           }))
        make_plot(celltype, df, arrays, args.output_dir, suffix)
        df_tagged = df.copy()
        df_tagged.insert(0, 'celltype', celltype)
        all_summaries.append(df_tagged)

    if len(all_summaries) >= 2:
        combined = pd.concat(all_summaries, ignore_index=True)
        out = combined[combined['quantity'] == 'D'][
            ['celltype', 'mean', 'ci_lo', 'ci_hi', 'perm_p',
             'wilcoxon_p', 'n_cells']]
        print("\n=== D summary across celltypes ===")
        print(out.to_string(index=False,
                            formatters={
                                'mean': '{:+.4f}'.format,
                                'ci_lo': '{:+.4f}'.format,
                                'ci_hi': '{:+.4f}'.format,
                                'perm_p': '{:.4g}'.format,
                                'wilcoxon_p': '{:.4g}'.format,
                            }))


if __name__ == '__main__':
    main()

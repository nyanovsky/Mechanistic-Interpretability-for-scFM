"""Analysis of subset / recruitment steering (run_subset_steering.py output).

Spec: reports/pbmc/subset_steering_analysis_spec.md

Readout 1 — singles driver matrix (14x15 heatmap per celltype):
  rows = steered single gene, cols = 14 readout module genes (greedy order) + 1
  control-pool column. Diagonal = self-effect, off-diagonal = recruitment,
  control col = off-target null.

Readout 2 — prefix recruitment (per celltype):
  (a) recruitment-vs-size curve: held-out trio probes (GZMA/GNLY/HOPX, never
      steered) + in-S mean (sanity) + control-pool null + random-seed band
      (seed-specificity null, pooled across sizes).
  (b) staircase heatmap: rows = 14 module genes (greedy order), cols = prefix
      size; a per-row box marks where each gene enters the steered set. A warm
      cell ABOVE the staircase = a gene recruited before it was added.

Effect = stat over cells of (steered_readout - baseline_readout), per celltype,
single --alpha. AIDO.Cell MLM logit shift vs the additive alpha=0 baseline.

Subset taxonomy is reconstructed from meta['order']/['module'] (NOT meta['subsets'],
which the orchestrator leaves frozen from the first invocation on resume).

Usage:
    python scripts/steering/analyze_subset_steering.py \
        --sweep-file results/steering/feat3079_subset/sweep.pt \
        --alpha 3 \
        --out-dir plots/sae/pbmc/layer_12/steering_analysis/3079/subset
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu

# Global greedy co-expression order (from feat3079_module_coexpr_heatmap.py).
GREEDY_ORDER = ["NKG7", "CST7", "GZMA", "PRF1", "GZMB", "FGFBP2", "GNLY", "GZMH",
                "CCL4", "CCL5", "KLRD1", "HOPX", "CMC1", "SSR2"]
TRIO_COLORS = {"GZMA": "tab:green", "GNLY": "tab:orange", "HOPX": "tab:red"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-file', required=True)
    p.add_argument('--alpha', type=float, default=None,
                   help='Alpha to analyze (default: the only alpha in the store).')
    p.add_argument('--stat', default='mean', choices=['mean', 'median'])
    p.add_argument('--out-dir', default='plots/sae/pbmc/layer_12/steering_analysis/3079/subset')
    return p.parse_args()


def resolve_alpha(store, alpha):
    keys = sorted({a for r in store['results'].values() for a in r})
    if not keys:
        raise SystemExit("No results in store yet.")
    if alpha is None:
        if len(keys) != 1:
            raise SystemExit(f"Multiple alphas {keys}; pass --alpha explicitly.")
        return keys[0]
    for k in keys:
        if np.isclose(float(k), alpha):
            return k
    raise SystemExit(f"alpha {alpha} not in store (have {keys}).")


def order_genes(genes):
    present = [g for g in GREEDY_ORDER if g in genes]
    return present + [g for g in genes if g not in present]


def agg(x, stat, axis=0):
    return np.median(x, axis=axis) if stat == 'median' else np.mean(x, axis=axis)


def classify(store, alpha):
    """Reconstruct subset taxonomy from order/module (meta['subsets'] is stale)."""
    m = store['meta']
    res = store['results']
    module, order = list(m['module']), list(m['order'])
    have = lambda k: k in res and alpha in res[k]

    singles = [g for g in module if have(g)]                      # single-gene keys
    prefixes = {}                                                 # size n -> skey
    for n in range(1, len(order) + 1):
        k = "|".join(sorted(order[:n]))
        if have(k):
            prefixes[n] = k
    modset = set(module)
    randoms = []                                                  # (size, skey)
    for sk in res:
        if alpha not in res[sk]:
            continue
        if not set(sk.split("|")) <= modset:
            randoms.append((len(sk.split("|")), sk))
    trio = [g for g in module if g not in order]                  # held-out probes
    return singles, prefixes, randoms, trio


def delta_of(store, skey, alpha, baseline):
    return np.asarray(store['results'][skey][alpha]['readout'], np.float32) - baseline


# ----------------------------- Readout 1 ---------------------------------------
def singles_matrix(store, alpha, stat, singles):
    m = store['meta']
    readout = list(m['readout_genes'])
    col = {g: i for i, g in enumerate(readout)}
    module = order_genes([g for g in m['module'] if g in col])
    ctrl_cols = np.array([col[g] for g in m['control_pool']])
    cl = np.asarray(m['cell_labels'])
    baseline = np.asarray(store['baseline']['readout'], np.float32)
    steered = order_genes(singles)
    cols = module + ['control']
    out = {ct: np.full((len(steered), len(cols)), np.nan, np.float32) for ct in m['celltypes']}
    for ri, g in enumerate(steered):
        d = delta_of(store, g, alpha, baseline)
        for ct in m['celltypes']:
            dct = d[cl == ct]
            vals = [agg(dct[:, col[r]], stat) for r in module]
            vals.append(agg(dct[:, ctrl_cols].mean(1), stat))
            out[ct][ri] = vals
    return steered, cols, out


def plot_singles(steered, cols, mats, m, alpha, stat, out_path):
    cts = m['celltypes']
    cl = np.asarray(m['cell_labels'])
    vmax = max(np.nanmax(np.abs(x)) for x in mats.values()) or 1.0
    n_row, n_col = len(steered), len(cols)
    fig, axes = plt.subplots(1, len(cts), figsize=(0.7 * n_col * len(cts) + 2, 0.55 * n_row + 1.5))
    axes = np.atleast_1d(axes)
    for ax, ct in zip(axes, cts):
        M = mats[ct]
        im = ax.imshow(M, vmin=-vmax, vmax=vmax, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(n_col)); ax.set_xticklabels(cols[:-1] + ['ctrl\n(200)'], rotation=90, fontsize=7)
        ax.set_yticks(range(n_row)); ax.set_yticklabels(steered, fontsize=7)
        ax.axvline(n_col - 1.5, color='k', lw=1.2)
        for ri, g in enumerate(steered):
            if g in cols:
                ax.add_patch(Rectangle((cols.index(g) - .5, ri - .5), 1, 1, fill=False, ec='k', lw=1.2))
        for ri in range(n_row):
            for ci in range(n_col):
                v = M[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.2f}", ha='center', va='center', fontsize=5,
                            color='black' if abs(v) < 0.6 * vmax else 'white')
        ax.set_title(f"{ct}  (n={int((cl == ct).sum())})", fontsize=10)
        ax.set_xlabel("readout gene")
    axes[0].set_ylabel("steered gene")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=f"{stat} Δlogit")
    fig.suptitle(f"Feature {m['feature']} singles driver matrix (alpha={alpha}, {m['steering_mode']})  "
                 f"diagonal=self, off-diagonal=recruitment", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved {out_path}")


# ----------------------------- Readout 2 ---------------------------------------
def recruitment_data(store, alpha, stat, prefixes, randoms, trio):
    """Per-celltype recruitment curve arrays + random-seed band + MWU per size."""
    m = store['meta']
    col = {g: i for i, g in enumerate(m['readout_genes'])}
    ctrl_cols = np.array([col[g] for g in m['control_pool']])
    trio_cols = np.array([col[g] for g in trio])
    cl = np.asarray(m['cell_labels'])
    baseline = np.asarray(store['baseline']['readout'], np.float32)
    sizes = sorted(prefixes)
    D = {ct: {'sizes': sizes, 'in_S': [], 'control': [], 'mwu_p': [],
              'trio': {g: [] for g in trio}} for ct in m['celltypes']}

    for n in sizes:
        d = delta_of(store, prefixes[n], alpha, baseline)
        in_S = [col[g] for g in m['order'][:n]]
        for ct in m['celltypes']:
            dct = d[cl == ct]
            D[ct]['in_S'].append(float(agg(dct[:, in_S].mean(1), stat)))
            D[ct]['control'].append(float(agg(dct[:, ctrl_cols].mean(1), stat)))
            for g in trio:
                D[ct]['trio'][g].append(float(agg(dct[:, col[g]], stat)))
            # one-sided MWU: trio per-cell deltas > control per-cell deltas
            tv = dct[:, trio_cols].ravel(); cv = dct[:, ctrl_cols].ravel()
            D[ct]['mwu_p'].append(float(mannwhitneyu(tv, cv, alternative='greater').pvalue))

    # random-seed null: per random subset, mean trio Δ over cells (pooled across sizes)
    band = {ct: [] for ct in m['celltypes']}
    for _, sk in randoms:
        d = delta_of(store, sk, alpha, baseline)
        for ct in m['celltypes']:
            band[ct].append(float(agg(d[cl == ct][:, trio_cols].mean(1), stat)))
    for ct in m['celltypes']:
        D[ct]['random_band'] = band[ct]
    return D


def plot_recruitment_curve(D, m, alpha, stat, out_path):
    cts = m['celltypes']
    fig, axes = plt.subplots(1, len(cts), figsize=(6.4 * len(cts), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ct in zip(axes, cts):
        d = D[ct]; x = d['sizes']
        rb = np.array(d['random_band'])
        lo, hi = np.percentile(rb, [10, 90])
        ax.axhspan(lo, hi, color='gray', alpha=0.18, label='random-seed null (10–90%)')
        ax.axhline(np.median(rb), color='gray', ls='--', lw=1)
        ax.axhline(0, color='k', lw=0.6)
        ax.plot(x, d['in_S'], color='0.4', ls=':', marker='s', ms=4, label='in-S (steered) mean')
        ax.plot(x, d['control'], color='k', ls='-.', marker='.', label='control pool mean')
        for g in d['trio']:
            ax.plot(x, d['trio'][g], color=TRIO_COLORS.get(g), marker='o', ms=4, label=f'{g} (held-out)')
        for xi, p in zip(x, d['mwu_p']):                       # significance star on trio recruitment
            if p < 0.05:
                ax.text(xi, max(d['trio'][g][x.index(xi)] for g in d['trio']) + 0.01, '*',
                        ha='center', fontsize=11, color='darkgreen')
        ax.set_title(ct); ax.set_xlabel("prefix size |S|"); ax.set_xticks(x)
    axes[0].set_ylabel(f"{stat} Δlogit vs baseline")
    axes[-1].legend(fontsize=7, loc='upper left')
    fig.suptitle(f"Feature {m['feature']} recruitment vs size (alpha={alpha}, {m['steering_mode']})  "
                 f"* = trio>control MWU p<0.05", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved {out_path}")


def staircase_data(store, alpha, stat, prefixes):
    m = store['meta']
    col = {g: i for i, g in enumerate(m['readout_genes'])}
    cl = np.asarray(m['cell_labels'])
    baseline = np.asarray(store['baseline']['readout'], np.float32)
    rows = [g for g in GREEDY_ORDER if g in m['module']]
    sizes = sorted(prefixes)
    M = {ct: np.full((len(rows), len(sizes)), np.nan, np.float32) for ct in m['celltypes']}
    for ci, n in enumerate(sizes):
        d = delta_of(store, prefixes[n], alpha, baseline)
        for ct in m['celltypes']:
            dct = d[cl == ct]
            for ri, g in enumerate(rows):
                M[ct][ri, ci] = agg(dct[:, col[g]], stat)
    return rows, sizes, M


def plot_staircase(rows, sizes, M, m, alpha, stat, out_path):
    cts = m['celltypes']
    order = list(m['order'])
    vmax = max(np.nanmax(np.abs(x)) for x in M.values()) or 1.0
    fig, axes = plt.subplots(1, len(cts), figsize=(0.55 * len(sizes) * len(cts) + 2, 0.5 * len(rows) + 1.5))
    axes = np.atleast_1d(axes)
    for ax, ct in zip(axes, cts):
        im = ax.imshow(M[ct], vmin=-vmax, vmax=vmax, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
        for ri, g in enumerate(rows):                          # box the in-S (steered) cells
            if g in order:
                c0 = sizes.index(order.index(g) + 1) if (order.index(g) + 1) in sizes else None
                if c0 is not None:
                    ax.add_patch(Rectangle((c0 - .5, ri - .5), len(sizes) - c0, 1,
                                           fill=False, ec='k', lw=1.4))
            else:
                ax.text(-0.9, ri, '●', va='center', ha='center', fontsize=7, color='purple')
        for ri in range(len(rows)):
            for ci in range(len(sizes)):
                v = M[ct][ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.2f}", ha='center', va='center', fontsize=5,
                            color='black' if abs(v) < 0.6 * vmax else 'white')
        ax.set_title(ct); ax.set_xlabel("prefix size |S|")
    axes[0].set_ylabel("module gene (greedy order; ● = held-out trio)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=f"{stat} Δlogit")
    fig.suptitle(f"Feature {m['feature']} prefix recruitment (alpha={alpha})  "
                 f"box = steered (in-S); above-box warm = recruited before added", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    store = torch.load(args.sweep_file, weights_only=False)
    m = store['meta']
    alpha = resolve_alpha(store, args.alpha)
    singles, prefixes, randoms, trio = classify(store, alpha)
    print(f"feature {m['feature']}  alpha={alpha}  stat={args.stat}  celltypes={m['celltypes']}")
    print(f"singles={len(singles)}  prefixes={sorted(prefixes)}  randoms={len(randoms)}  trio={trio}")
    os.makedirs(args.out_dir, exist_ok=True)
    a = f"a{alpha:g}"

    # Readout 1
    steered, cols, mats = singles_matrix(store, alpha, args.stat, singles)
    plot_singles(steered, cols, mats, m, alpha, args.stat,
                 os.path.join(args.out_dir, f"singles_driver_matrix_{a}.png"))
    with open(os.path.join(args.out_dir, f"singles_driver_{a}.json"), 'w') as f:
        json.dump({'feature': m['feature'], 'alpha': float(alpha), 'stat': args.stat,
                   'steered_genes': steered, 'readout_cols': cols,
                   'matrix': {ct: {steered[ri]: {cols[ci]: float(mats[ct][ri, ci])
                                                 for ci in range(len(cols))}
                                   for ri in range(len(steered))} for ct in m['celltypes']}},
                  f, indent=2)

    # Readout 2 — needs the prefix (n>=2) + random families; skip on a singles-only
    # store (the recruitment curve/staircase are meaningless and the empty random
    # band would crash np.percentile).
    if len(prefixes) <= 1 or not randoms:
        print(f"Skipping Readout 2: needs prefixes (n>=2) + random draws "
              f"(have prefixes={sorted(prefixes)}, randoms={len(randoms)}). "
              f"Run the prefixes/random part, then re-run.")
        print(f"Saved Readout 1 to {args.out_dir}")
        return

    D = recruitment_data(store, alpha, args.stat, prefixes, randoms, trio)
    plot_recruitment_curve(D, m, alpha, args.stat,
                           os.path.join(args.out_dir, f"prefix_recruitment_curve_{a}.png"))
    rows, sizes, Mst = staircase_data(store, alpha, args.stat, prefixes)
    plot_staircase(rows, sizes, Mst, m, alpha, args.stat,
                   os.path.join(args.out_dir, f"prefix_recruitment_heatmap_{a}.png"))
    with open(os.path.join(args.out_dir, f"recruitment_{a}.json"), 'w') as f:
        json.dump({'feature': m['feature'], 'alpha': float(alpha), 'stat': args.stat,
                   'trio': trio, 'curve': D}, f, indent=2)
    print(f"Saved JSON summaries to {args.out_dir}")


if __name__ == '__main__':
    main()

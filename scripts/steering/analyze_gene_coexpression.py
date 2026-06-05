"""
Gene-wise overlap + co-expression analysis for a single steered feature.

Standalone follow-up to analyze_steering_results.py (Script 1). Reads the ranked DE genes
saved by Script 1 (de_genes_*.csv) and the feature's top-PR genes, then for each comparison/
celltype group:

  1. Picks a DE direction (up if alpha > threshold, else down; CLI-overridable) and trims to
     the top --n-de-top genes by |logFC|.
  2. Overlaps the top-PR gene set with the DE set -> inclusion (% of top-PR genes in the DE set)
     and reverse coverage.
  3. Quantifies co-expression of each of the three gene sets (top-PR, DE, intersection) in the
     raw data as signed pairwise Pearson r across cells, vs a pooled null of random matched-size
     gene sets (Mann-Whitney U). By default two views are reported side by side:
       - single_cell_global   : r across all cells (captures the celltype-identity axis)
       - single_cell_celltype : r within the steered celltype (strict within-state test)
     The contrast between them separates "co-marker of a celltype/state program" from genuine
     within-celltype co-regulation. Pass --pseudobulk to additionally compute dropout-corrected
     Leiden-metacell views (only worth it when chasing within-celltype co-regulation of sparse
     genes, where single-cell dropout would hide it). One boxplot per set (spans all views).

Re-runnable with different --n-de-top without touching DE/GO (only co-expression recomputes).
"""

import os
import re
import sys
import json
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import get_expressed_genes_mask, load_activation_matrices
from utils.similarity import (
    get_top_k_genes_per_feature,
    gene_set_pairwise_corr,
    sample_null_coexpression,
)

from aido_cell.utils import align_adata


SET_NAMES = ['top_pr', 'de', 'intersection']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--de-dir', dest='de_dir', required=True,
                   help='Directory from Script 1 with de_genes_*.csv (searched recursively).')
    p.add_argument('--interp-dir', dest='interp_dir', required=True,
                   help='SAE interpretations dir with feature_gene_matrix.npy + '
                        'feature_participation_ratios.npy')
    p.add_argument('--feature-id', dest='feature_id', type=int, default=None,
                   help='Steered feature id (default: read from feature_pr_*.json in --de-dir).')
    p.add_argument('--output-dir', dest='output_dir', default=None,
                   help='Where to write JSON + boxplots (default: --de-dir).')
    p.add_argument('--raw-file', dest='raw_file', default='data/pbmc/pbmc3k_raw.h5ad')
    p.add_argument('--processed-file', dest='processed_file',
                   default='data/pbmc/pbmc3k_processed.h5ad',
                   help='Only used when --celltypes is given (for celltype annotations).')
    p.add_argument('--n-de-top', dest='n_de_top', type=int, default=50,
                   help='Number of DE genes (by |logFC|) to keep in the DE set.')
    p.add_argument('--de-direction', dest='de_direction', choices=['auto', 'up', 'down'],
                   default='auto', help="DE direction; 'auto' = up if alpha>threshold else down.")
    p.add_argument('--de-direction-threshold', dest='de_threshold', type=float, default=0.5)
    p.add_argument('--n-null-sets', dest='n_null_sets', type=int, default=50,
                   help='Number of random matched-size gene sets pooled for the null.')
    p.add_argument('--celltypes', nargs='+', default=None,
                   help='Optional filter: only build the within-celltype views for these celltype '
                        'label(s) (must match the DE subdir names, e.g. "CD4 T cells"). Default: '
                        'derive the celltype automatically from each DE subdir.')
    p.add_argument('--pseudobulk', action='store_true',
                   help='Additionally compute dropout-corrected metacell co-expression (Leiden '
                        'over-clustering): metacell_global + metacell_celltype views. Optional; the '
                        'two single-cell views already resolve identity-vs-within-celltype for most features.')
    p.add_argument('--cells-per-metacell', dest='cells_per_mc', type=int, default=20,
                   help='Target avg cells per metacell; the Leiden resolution is auto-tuned per '
                        'scope to hit this (adapts to celltype size). Default 20.')
    p.add_argument('--min-metacells', dest='min_metacells', type=int, default=20,
                   help='Floor on the target metacell count when auto-tuning resolution.')
    p.add_argument('--leiden-resolution', dest='leiden_resolution', type=float, default=None,
                   help='Override the auto-tuned Leiden resolution with a fixed value (advanced).')
    p.add_argument('--min-cells-per-metacell', dest='min_cells_per_mc', type=int, default=10,
                   help='Drop Leiden communities with fewer cells than this when aggregating.')
    p.add_argument('--coexpr-seed', dest='coexpr_seed', type=int, default=0)
    return p.parse_args()


def load_feature_matrices(interp_dir, feature_id, de_dir, expr_mask):
    """Load fg matrix + PR, resolve feature id, return top-PR gene index set + PR scalar.

    Top-PR selection is restricted to expressed genes (expr_mask over the AIDO gene
    axis); returned indices are in full-gene space for downstream expression lookups.
    """
    if feature_id is None:
        pr_jsons = glob.glob(os.path.join(de_dir, '**', 'feature_pr_*.json'), recursive=True)
        if not pr_jsons:
            raise ValueError("No --feature-id and no feature_pr_*.json found under --de-dir")
        with open(pr_jsons[0]) as f:
            feature_id = int(json.load(f)['feature_id'])
        print(f"Resolved feature id {feature_id} from {pr_jsons[0]}")

    fg_matrix, _ = load_activation_matrices(interp_dir)
    if fg_matrix is None:
        raise FileNotFoundError(f"feature_gene_matrix.npy not found in {interp_dir}")
    pr_values = np.load(os.path.join(interp_dir, 'feature_participation_ratios.npy'))
    # Standardize to (n_features, n_genes): genes always outnumber features
    if fg_matrix.shape[0] > fg_matrix.shape[1]:
        fg_matrix = fg_matrix.T
    assert fg_matrix.shape[0] == len(pr_values), \
        f"fg_matrix {fg_matrix.shape} vs PR {pr_values.shape} mismatch"

    expr_indices = np.where(expr_mask)[0]
    gene_sets = get_top_k_genes_per_feature(fg_matrix[:, expr_mask], pr_values, pr_scale=1,
                                            min_genes=10, max_genes=100)
    top_pr_idx = expr_indices[list(gene_sets[feature_id])]  # map back to full-gene space
    pr_value = float(pr_values[feature_id])
    k = int(np.clip(round(pr_value), 10, 100))
    print(f"Feature {feature_id}: PR={pr_value:.2f} -> k={k}, |top-PR set|={len(top_pr_idx)}")
    return feature_id, top_pr_idx, pr_value, k


def load_expression(raw_file):
    """Return (X_raw [cells x genes counts], gene_names, expr_mask, cell_names) aligned to AIDO genes.

    Raw counts are returned (not log1p) so metacell aggregation can sum/average counts before
    normalizing; the single-cell view applies log1p in `main`.
    """
    adata_raw = ad.read_h5ad(raw_file)
    adata_raw.var_names_make_unique()
    adata_aligned, attention_mask = align_adata(adata_raw)
    mask = attention_mask.astype(bool)
    gene_names = adata_aligned.var_names[mask].to_numpy()

    X = adata_aligned.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X_raw = np.asarray(X)[:, mask].astype(np.float32)

    expr_mask = get_expressed_genes_mask(raw_file)
    cell_names = adata_aligned.obs_names.to_numpy()
    print(f"Expression: {X_raw.shape[0]} cells x {X_raw.shape[1]} genes (raw counts); "
          f"{int(expr_mask.sum())} expressed genes")
    return X_raw, gene_names, expr_mask, cell_names


def load_louvain_labels(processed_file, cell_names):
    """Return per-cell celltype labels (aligned to cell_names) + sanitized->label map.

    The sanitized map keys celltype labels the same way the DE subdirs were named
    (spaces -> underscores), so a group/subdir can be matched back to its celltype.
    """
    obs = ad.read_h5ad(processed_file).obs
    col = 'celltype' if 'celltype' in obs.columns else ('louvain' if 'louvain' in obs.columns else None)
    if col is None:
        raise ValueError("No celltype/louvain column in processed file")
    labels = obs[col].reindex(pd.Index(cell_names)).astype(str).to_numpy()
    sanitized_map = {lab.replace(' ', '_'): lab for lab in pd.unique(labels) if lab != 'nan'}
    return labels, sanitized_map


def _tune_leiden_resolution(adata, target_n, seed, iters=12):
    """Binary-search the Leiden resolution so the *total* community count ~= target_n.

    Total community count is monotonic in resolution (unlike the surviving-metacell count,
    which is unimodal once the min-cells floor is applied), so it is safe to bisect. Returns
    (best_labels, best_resolution). The neighbor graph in `adata` is reused across iterations.
    """
    import scanpy as sc
    lo, hi = 0.05, 50.0
    best = None  # (abs_err, n, resolution, labels)
    for _ in range(iters):
        mid = (lo + hi) / 2
        sc.tl.leiden(adata, resolution=mid, random_state=seed, flavor='igraph',
                     n_iterations=2, directed=False, key_added='_tune')
        labels = adata.obs['_tune'].astype(int).to_numpy()
        n = int(labels.max() + 1)
        err = abs(n - target_n)
        if best is None or err < best[0]:
            best = (err, n, mid, labels)
        if n == target_n:
            break
        if n < target_n:
            lo = mid
        else:
            hi = mid
    return best[3], best[2]


def compute_metacells(counts_raw, min_cells, seed, cells_per_metacell=20,
                      min_metacells=20, resolution=None):
    """Leiden over-cluster the given cells and return a log-normalized pseudobulk matrix.

    The Leiden resolution is **auto-tuned per call** so communities average ~cells_per_metacell
    cells (target community count = n_cells / cells_per_metacell, floored at min_metacells). This
    adapts to each scope automatically — a single fixed resolution over-shatters small/homogeneous
    populations (e.g. one celltype) while under-clustering the full dataset. Pass `resolution` to
    override with a fixed value. Communities with < min_cells are dropped; metacell profiles are
    the mean raw counts per community, library-normalized and log1p'd.

    Returns ([n_metacells x n_genes] or None, n_metacells, resolution_used).
    """
    import scanpy as sc
    n_cells = counts_raw.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = ad.AnnData(counts_raw.astype(np.float32).copy())
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars))
        a = a[:, a.var.highly_variable].copy()
        sc.pp.scale(a, max_value=10)
        sc.pp.pca(a, n_comps=min(50, a.n_obs - 1, a.n_vars - 1))
        sc.pp.neighbors(a, n_neighbors=15)
        if resolution is not None:
            sc.tl.leiden(a, resolution=resolution, random_state=seed,
                         flavor='igraph', n_iterations=2, directed=False)
            labels = a.obs['leiden'].astype(int).to_numpy()
            res_used = resolution
        else:
            target_n = max(min_metacells, n_cells // cells_per_metacell)
            labels, res_used = _tune_leiden_resolution(a, target_n, seed)

    profiles = []
    for c in range(labels.max() + 1):
        m = labels == c
        if m.sum() >= min_cells:
            profiles.append(counts_raw[m].mean(0))
    if len(profiles) < 3:
        return None, len(profiles), res_used
    mc = np.vstack(profiles)
    totals = mc.sum(1, keepdims=True)
    totals[totals == 0] = 1.0
    mc = np.log1p(mc / totals * np.median(mc.sum(1)))
    return mc, mc.shape[0], round(res_used, 3)


def parse_de_filename(path):
    """de_genes_{label}.csv -> (label, alpha or None)."""
    label = re.sub(r'^de_genes_|\.csv$', '', os.path.basename(path))
    m = re.search(r'Alpha([-\d.]+)', label)
    alpha = float(m.group(1)) if m else None
    return label, alpha


def choose_direction(alpha, args):
    if args.de_direction != 'auto':
        return args.de_direction
    if alpha is None:
        return 'up'
    return 'up' if alpha > args.de_threshold else 'down'


def summarize_corrs(corrs, n_genes_used, null, run_mwu=True):
    out = {
        'median': None, 'mean_abs': None, 'n_pairs': int(corrs.size),
        'n_genes_used': int(n_genes_used), 'null_median': None, 'mwu_p': None,
    }
    if corrs.size:
        out['median'] = float(np.median(corrs))
        out['mean_abs'] = float(np.mean(np.abs(corrs)))
    if null is not None and null.size:
        out['null_median'] = float(np.median(null))
    if run_mwu and corrs.size and null is not None and null.size:
        try:
            _, p = mannwhitneyu(corrs, null, alternative='two-sided')
            out['mwu_p'] = float(p)
        except ValueError:
            out['mwu_p'] = None
    return out


def coexpression_boxplot(view_corrs, set_name, label, group_name, out_dir):
    """One figure per gene set; for each view a (set, null) pair of boxes side by side.

    view_corrs: list of (view_name, corrs, null) tuples.
    """
    fig, ax = plt.subplots(figsize=(2.5 + 2.2 * max(1, len(view_corrs)), 5))
    ticks, tick_labels, pos = [], [], 1
    for vname, corrs, null in view_corrs:
        ax.boxplot([corrs if corrs.size else [np.nan]], positions=[pos], widths=0.6,
                   showmeans=True, patch_artist=True, boxprops=dict(facecolor='#7fa8d6'))
        ax.boxplot([null if null.size else [np.nan]], positions=[pos + 1], widths=0.6,
                   showmeans=True, patch_artist=True, boxprops=dict(facecolor='#d0d0d0'))
        ticks += [pos, pos + 1]
        tick_labels += [f"{vname}\nset", "null"]
        pos += 3
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.axhline(0, color='grey', linewidth=0.8, alpha=0.6)
    ax.set_ylabel('pairwise Pearson r')
    ax.set_title(f"{set_name} co-expression\n{label} [{group_name}]")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"coexpr_box_{set_name}_{label}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def analyze_one(de_csv, group_name, views, top_pr_idx, pr_value, k, feature_id,
                gene_names, expressed_indices, args, out_dir):
    """views: list of dicts {name, matrix [samples x genes], scope, n_metacells (or None)}."""
    label, alpha = parse_de_filename(de_csv)
    direction = choose_direction(alpha, args)
    print(f"\n--- {group_name} | {label} | direction={direction} ---")

    df = pd.read_csv(de_csv)
    de_sub = df[df['direction'] == direction].sort_values('abs_logFC', ascending=False)
    de_genes = de_sub['gene'].head(args.n_de_top).tolist()

    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    de_idx = {name_to_idx[g] for g in de_genes if g in name_to_idx}
    pr_set = set(int(i) for i in top_pr_idx)
    inter = pr_set & de_idx

    inclusion = len(inter) / len(pr_set) if pr_set else None
    reverse = len(inter) / len(de_idx) if de_idx else None
    incl_str = f"{inclusion:.3f}" if inclusion is not None else "n/a"
    print(f"  |top-PR|={len(pr_set)} |DE|={len(de_idx)} |inter|={len(inter)} inclusion={incl_str}")

    rng = np.random.default_rng(args.coexpr_seed)
    set_indices = {'top_pr': pr_set, 'de': de_idx, 'intersection': inter}
    coexpr = {}
    for name in SET_NAMES:
        idx = set_indices[name]
        coexpr[name] = {}
        view_corrs = []
        for v in views:
            corrs, n_used = gene_set_pairwise_corr(v['matrix'], idx)
            null = sample_null_coexpression(v['matrix'], n_used, expressed_indices,
                                            n_sets=args.n_null_sets, rng=rng)
            summary = summarize_corrs(corrs, n_used, null)
            summary['scope'] = v['scope']
            summary['n_metacells'] = v['n_metacells']
            summary['leiden_resolution'] = v.get('resolution')
            coexpr[name][v['name']] = summary
            view_corrs.append((v['name'], corrs, null))
            print(f"  {name}/{v['name']}: median_r={summary['median']}, "
                  f"null={summary['null_median']}, mwu_p={summary['mwu_p']}")
        coexpr[name]['boxplot'] = os.path.basename(
            coexpression_boxplot(view_corrs, name, label, group_name, out_dir))

    result = {
        'feature_id': feature_id, 'group': group_name, 'label': label, 'alpha': alpha,
        'de_direction': direction, 'PR': pr_value, 'k': k,
        'n_de_top': args.n_de_top, 'views': [v['name'] for v in views],
        'top_pr_size': len(pr_set), 'de_set_size': len(de_idx),
        'intersection_size': len(inter),
        'inclusion': inclusion, 'reverse_coverage': reverse,
        'intersection_genes': sorted(gene_names[i] for i in inter),
        'coexpression': coexpr,
    }
    out_json = os.path.join(out_dir, f"gene_coexpression_feature_{feature_id}_{label}.json")
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote {out_json}")


def main():
    args = parse_args()
    output_dir = args.output_dir or args.de_dir
    os.makedirs(output_dir, exist_ok=True)

    X_raw, gene_names, expr_mask, cell_names = load_expression(args.raw_file)
    X_log = np.log1p(X_raw)
    expressed_indices = np.where(expr_mask)[0]

    feature_id, top_pr_idx, pr_value, k = load_feature_matrices(
        args.interp_dir, args.feature_id, args.de_dir, expr_mask)

    de_csvs = sorted(glob.glob(os.path.join(args.de_dir, '**', 'de_genes_*.csv'), recursive=True))
    if not de_csvs:
        raise ValueError(f"No de_genes_*.csv found under {args.de_dir}")
    print(f"Found {len(de_csvs)} DE gene file(s).")

    # Celltype labels for the within-celltype views (derived from each DE subdir).
    labels, sanitized_map = (None, {})
    try:
        labels, sanitized_map = load_louvain_labels(args.processed_file, cell_names)
    except Exception as e:
        print(f"WARNING: could not load celltype labels ({e}); only the global view will be produced.")
    # Optional filter: restrict within-celltype views to these (sanitized) celltype subdirs.
    celltype_filter = {ct.replace(' ', '_') for ct in args.celltypes} if args.celltypes else None

    metacell_cache = {}  # 'global' or celltype label -> (matrix, n_metacells, resolution)

    def get_metacells(key, cell_mask):
        if key not in metacell_cache:
            res_note = f"fixed res={args.leiden_resolution}" if args.leiden_resolution \
                else f"auto-tuned to ~{args.cells_per_mc} cells/metacell"
            print(f"\nBuilding metacells [{key}] from {int(cell_mask.sum())} cells ({res_note})...")
            mat, n, res = compute_metacells(
                X_raw[cell_mask], args.min_cells_per_mc, args.coexpr_seed,
                cells_per_metacell=args.cells_per_mc, min_metacells=args.min_metacells,
                resolution=args.leiden_resolution)
            metacell_cache[key] = (mat, n, res)
            if mat is None:
                print(f"  -> only {n} metacells (<3; view skipped)")
            else:
                print(f"  -> {n} metacells (Leiden res={res})"
                      + ("  WARNING: <15 metacells, correlation underpowered" if n < 15 else ""))
        return metacell_cache[key]

    for de_csv in de_csvs:
        # Group = subdir name relative to de_dir, or 'all' if directly in de_dir
        rel = os.path.relpath(os.path.dirname(de_csv), args.de_dir)
        group_name = 'all' if rel in ('.', '') else rel.replace(os.sep, '_')
        out_dir = output_dir if group_name == 'all' else os.path.join(output_dir, group_name)
        os.makedirs(out_dir, exist_ok=True)

        # Which celltype (if any) does this group/subdir map to?
        ct = sanitized_map.get(group_name)
        use_celltype = ct is not None and (celltype_filter is None or group_name in celltype_filter)

        # --- default single-cell views: global (all cells) + within the steered celltype
        views = [{'name': 'single_cell_global', 'matrix': X_log, 'scope': 'all', 'n_metacells': None}]
        if use_celltype:
            views.append({'name': 'single_cell_celltype', 'matrix': X_log[labels == ct],
                          'scope': ct, 'n_metacells': None})

        # --- optional dropout-corrected metacell views (global + within the group's celltype)
        if args.pseudobulk:
            mg, ng, rg = get_metacells('global', np.ones(len(X_raw), dtype=bool))
            if mg is not None:
                views.append({'name': 'metacell_global', 'matrix': mg, 'scope': 'all',
                              'n_metacells': ng, 'resolution': rg})
            if use_celltype:
                mc, nc, rc = get_metacells(ct, labels == ct)
                if mc is not None:
                    views.append({'name': 'metacell_celltype', 'matrix': mc, 'scope': ct,
                                  'n_metacells': nc, 'resolution': rc})

        analyze_one(de_csv, group_name, views, top_pr_idx, pr_value, k, feature_id,
                    gene_names, expressed_indices, args, out_dir)

    print("\nGene co-expression analysis complete.")


if __name__ == '__main__':
    main()

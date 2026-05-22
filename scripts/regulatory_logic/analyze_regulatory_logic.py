"""
Analyze per-target directionality of position-selective SAE steering.

Design:
  - TF-steered (1 condition): steer at TF only.
  - Target-steered (N conditions, one per control gene T_i): steer at T_i only.
  - Random-steered (M conditions): steer at feature-member non-regulon non-regulator genes.

Primary test, per target T_i, per alpha:
  F_i^c = logit(T_i)_c^{TF-steered}     - logit(T_i)_c^{baseline}
  R_i^c = logit(TF)_c^{T_i-steered}     - logit(TF)_c^{baseline}
  Paired permutation (sign-flip) on (|F_i^c| - |R_i^c|), one-sided H1: |F| > |R|.

Secondary test: paired Wilcoxon signed-rank on the same pairs.

Random baselines (descriptive, no test):
  Random → TF: mean |logFC(TF)| averaged over cells and random genes under each
               random-steered condition.
  TF → random: mean |logFC(G_j)| averaged over cells and random genes under
               the TF-steered condition.

Within-feature vs rest (per steering condition):
  DE genes (paired t-test, Bonferroni) split by feature membership; report
  mean |logFC| and count per subset.

Usage:
    python analyze_regulatory_logic.py \
        --results-file results/regulatory_logic/regulatory_logic_TBX21_feat3092_layer12.pt \
        --baseline-file /biodata/nyanovsky/datasets/pbmc3k/pbmc3k_logits.h5ad \
        --data-file data/pbmc/pbmc3k_raw.h5ad \
        --output-dir plots/regulatory_logic/TBX21_feat3092
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import anndata as ad
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import compute_de_genes, get_expressed_genes_mask


# --- Core logFC ---

def per_cell_logfc(steered_logits, baseline_logits):
    """Return (n_cells, n_genes) per-cell logFC matrix."""
    return steered_logits.float().numpy() - baseline_logits.float().numpy()


# --- Paired tests ---

def paired_permutation_test(f_abs, r_abs, n_perm=10000, seed=0):
    """Sign-flip permutation test, one-sided H1: mean(|F| - |R|) > 0.

    Args:
        f_abs: (n_cells,) |F_i^c|
        r_abs: (n_cells,) |R_i^c|
        n_perm: number of sign-flip permutations

    Returns:
        dict with observed_diff, pval, null_mean, null_std
    """
    diff = f_abs - r_abs
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(diff)))
    null = (signs * diff).mean(axis=1)
    # One-sided: P(null >= observed)
    pval = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return {
        'observed_diff': float(observed),
        'pval': float(pval),
        'null_mean': float(null.mean()),
        'null_std': float(null.std()),
        'n_cells': len(diff),
    }


def paired_wilcoxon_test(f_abs, r_abs):
    """Wilcoxon signed-rank, one-sided: |F| > |R|."""
    diff = f_abs - r_abs
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return {'stat': np.nan, 'pval': np.nan, 'n_cells': len(diff), 'n_nonzero': 0}
    try:
        stat, pval = wilcoxon(f_abs, r_abs, alternative='greater', zero_method='wilcox')
        return {'stat': float(stat), 'pval': float(pval),
                'n_cells': len(diff), 'n_nonzero': len(nonzero)}
    except ValueError as e:
        return {'stat': np.nan, 'pval': np.nan,
                'n_cells': len(diff), 'n_nonzero': len(nonzero), 'error': str(e)}


def effect_sizes(f_abs, r_abs):
    """Per-cell effect sizes: median difference, fraction |F|>|R|, mean ratio."""
    diff = f_abs - r_abs
    eps = 1e-12
    return {
        'median_diff': float(np.median(diff)),
        'mean_diff': float(diff.mean()),
        'frac_F_gt_R': float((f_abs > r_abs).mean()),
        'mean_log_ratio': float(np.mean(np.log((f_abs + eps) / (r_abs + eps)))),
    }


# --- Within-feature vs rest ---

def within_feature_vs_rest(logfc_matrix, gene_names, feature_gene_set, baseline_logits,
                            expr_mask):
    """For a condition, split DE genes into in-feature vs rest and report mean |logFC|.

    DE is computed over expressed genes only (expr_mask applied before t-test).

    Args:
        logfc_matrix: (n_cells, n_genes) per-cell logFC (full gene axis)
        gene_names: gene name array (full)
        feature_gene_set: set of gene names in the feature's top-gene set
        baseline_logits: (n_cells, n_genes) used by compute_de_genes
        expr_mask: boolean mask over gene axis (True = expressed, use for DE)
    """
    steered = logfc_matrix + baseline_logits.float().numpy()
    baseline_np = baseline_logits.float().numpy()

    gene_names_expr = gene_names[expr_mask]
    top_up, top_down, de_stats = compute_de_genes(
        steered[:, expr_mask], baseline_np[:, expr_mask], gene_names_expr
    )
    de_genes = list(top_up) + list(top_down)
    if not de_genes:
        return {'n_de': 0, 'n_in_feature': 0, 'n_rest': 0,
                'mean_abs_logfc_in': np.nan, 'mean_abs_logfc_rest': np.nan,
                'median_abs_logfc_in': np.nan, 'median_abs_logfc_rest': np.nan}

    # mean_diff is aligned to gene_names_expr (the expressed subset)
    name_to_idx_expr = {n: i for i, n in enumerate(gene_names_expr)}
    mean_diff = de_stats['mean_diff']

    in_feature = [g for g in de_genes if g in feature_gene_set]
    rest = [g for g in de_genes if g not in feature_gene_set]

    in_signed = np.array([mean_diff[name_to_idx_expr[g]] for g in in_feature]) \
        if in_feature else np.array([])
    rest_signed = np.array([mean_diff[name_to_idx_expr[g]] for g in rest]) \
        if rest else np.array([])
    in_vals = np.abs(in_signed)
    rest_vals = np.abs(rest_signed)

    return {
        'n_de': len(de_genes),
        'n_in_feature': len(in_feature),
        'n_rest': len(rest),
        'mean_abs_logfc_in': float(in_vals.mean()) if in_vals.size else np.nan,
        'mean_abs_logfc_rest': float(rest_vals.mean()) if rest_vals.size else np.nan,
        'median_abs_logfc_in': float(np.median(in_vals)) if in_vals.size else np.nan,
        'median_abs_logfc_rest': float(np.median(rest_vals)) if rest_vals.size else np.nan,
        'mean_logfc_in': float(in_signed.mean()) if in_signed.size else np.nan,
        'mean_logfc_rest': float(rest_signed.mean()) if rest_signed.size else np.nan,
        'median_logfc_in': float(np.median(in_signed)) if in_signed.size else np.nan,
        'median_logfc_rest': float(np.median(rest_signed)) if rest_signed.size else np.nan,
    }


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description='Per-target directionality analysis of regulatory logic steering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--results-file', type=str, required=True)
    parser.add_argument('--baseline-file', type=str, required=True,
                        help='Path to baseline (unsteered) logits .h5ad')
    parser.add_argument('--data-file', type=str, default='data/pbmc/pbmc3k_raw.h5ad',
                        help='Raw .h5ad used to derive expressed-gene mask for DE analysis')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-perm', type=int, default=10000,
                        help='Number of sign-flip permutations')

    args = parser.parse_args()

    # --- Load ---
    print("Loading results...")
    data = torch.load(args.results_file, map_location='cpu')
    conditions = data['conditions']
    metadata = data['metadata']

    tf = metadata['tf']
    feature = metadata['feature']
    alphas = metadata['alphas']
    gene_names = np.array(metadata['gene_names'])
    control_genes = metadata['control_genes']
    random_genes = metadata.get('random_genes', [])
    feature_gene_set = set(metadata['feature_gene_set'])

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.results_file),
            f"analysis_{tf}_feat{feature}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"TF: {tf}, Feature: {feature}")
    print(f"Alphas: {alphas}")
    print(f"Control (target) genes: {control_genes}")
    print(f"Random-pool genes: {random_genes}")

    # --- Baseline alignment (name-based) ---
    print("\nLoading baseline logits...")
    baseline_adata = ad.read_h5ad(args.baseline_file)
    if 'cell_names' not in metadata:
        raise ValueError("Results file missing 'cell_names' in metadata.")
    cell_names_steered = np.array(metadata['cell_names'])

    baseline_name_to_idx = {name: i for i, name in enumerate(baseline_adata.obs_names)}
    common_mask = np.array([name in baseline_name_to_idx for name in cell_names_steered])
    if not common_mask.all():
        print(f"  WARNING: {(~common_mask).sum()} cells dropped (not in baseline)")

    baseline_indices = np.array([baseline_name_to_idx[name]
                                  for name in cell_names_steered[common_mask]])
    baseline_all = baseline_adata.X
    if hasattr(baseline_all, 'toarray'):
        baseline_all = baseline_all.toarray()
    baseline_logits = torch.tensor(baseline_all[baseline_indices], dtype=torch.float32)
    print(f"  Baseline aligned: {baseline_logits.shape}")

    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    if tf not in gene_to_idx:
        raise ValueError(f"TF {tf} not in active gene list")
    tf_idx = gene_to_idx[tf]

    # --- Compute per-cell logFC for each condition × alpha ---
    def get_logfc(entry):
        steered = entry['logits']
        if common_mask is not None:
            steered = steered[common_mask]
        return per_cell_logfc(steered, baseline_logits)

    logfc = {
        'tf_steered': {a: get_logfc(conditions['tf_steered'][a]) for a in alphas},
        'target_steered': {
            g: {a: get_logfc(conditions['target_steered'][g][a]) for a in alphas}
            for g in control_genes
        },
        'random_steered': {
            g: {a: get_logfc(conditions['random_steered'][g][a]) for a in alphas}
            for g in random_genes
        },
    }

    # ================================================================
    # PRIMARY: paired permutation per target, per alpha
    # ================================================================
    print("\n" + "=" * 70)
    print("PRIMARY TESTS: paired permutation (|F_i| vs |R_i|) per target")
    print("=" * 70)

    primary_rows = []
    for target in control_genes:
        if target not in gene_to_idx:
            print(f"  SKIP {target}: not in gene_names")
            continue
        t_idx = gene_to_idx[target]

        for alpha in alphas:
            # F_i^c: effect at target T_i under TF-steering
            F = logfc['tf_steered'][alpha][:, t_idx]
            # R_i^c: effect at TF under T_i-steered
            R = logfc['target_steered'][target][alpha][:, tf_idx]

            F_abs, R_abs = np.abs(F), np.abs(R)
            perm = paired_permutation_test(F_abs, R_abs, n_perm=args.n_perm)
            wil = paired_wilcoxon_test(F_abs, R_abs)
            eff = effect_sizes(F_abs, R_abs)

            primary_rows.append({
                'target': target, 'alpha': alpha,
                'mean_F': float(F.mean()),
                'mean_R': float(R.mean()),
                'mean_F_abs': float(F_abs.mean()),
                'mean_R_abs': float(R_abs.mean()),
                'median_diff': eff['median_diff'],
                'frac_F_gt_R': eff['frac_F_gt_R'],
                'mean_log_ratio': eff['mean_log_ratio'],
                'perm_pval': perm['pval'],
                'perm_observed': perm['observed_diff'],
                'wilcoxon_pval': wil['pval'],
                'wilcoxon_stat': wil['stat'],
                'n_cells': perm['n_cells'],
            })

            verdict = "F > R" if perm['pval'] < 0.05 else "ns"
            print(f"  {target} α={alpha}: "
                  f"|F|={F_abs.mean():.4f} vs |R|={R_abs.mean():.4f}, "
                  f"frac(F>R)={eff['frac_F_gt_R']:.2f}, "
                  f"perm p={perm['pval']:.2e}, "
                  f"wilcoxon p={wil['pval']:.2e} [{verdict}]")

    primary_df = pd.DataFrame(primary_rows)
    primary_csv = os.path.join(args.output_dir, 'primary_tests.csv')
    primary_df.to_csv(primary_csv, index=False)
    print(f"\n  Saved {primary_csv}")

    # ================================================================
    # Random baselines (descriptive)
    # ================================================================
    print("\n" + "=" * 70)
    print("RANDOM BASELINES (descriptive)")
    print("=" * 70)

    baseline_rows = []
    for alpha in alphas:
        # Random → TF: |logFC(TF)| under each random-steered condition
        rand_to_tf_per_gene = []
        for g in random_genes:
            rand_to_tf_per_gene.append(np.abs(logfc['random_steered'][g][alpha][:, tf_idx]).mean())

        # TF → random: |logFC(G_j)| under TF-steered
        tf_to_rand_per_gene = []
        for g in random_genes:
            if g not in gene_to_idx:
                continue
            g_idx = gene_to_idx[g]
            tf_to_rand_per_gene.append(np.abs(logfc['tf_steered'][alpha][:, g_idx]).mean())

        rand_to_tf_mean = float(np.mean(rand_to_tf_per_gene)) if rand_to_tf_per_gene else np.nan
        rand_to_tf_std = float(np.std(rand_to_tf_per_gene)) if rand_to_tf_per_gene else np.nan
        tf_to_rand_mean = float(np.mean(tf_to_rand_per_gene)) if tf_to_rand_per_gene else np.nan
        tf_to_rand_std = float(np.std(tf_to_rand_per_gene)) if tf_to_rand_per_gene else np.nan

        baseline_rows.append({
            'alpha': alpha,
            'n_random': len(random_genes),
            'random_to_TF_mean_abs_logfc': rand_to_tf_mean,
            'random_to_TF_std': rand_to_tf_std,
            'TF_to_random_mean_abs_logfc': tf_to_rand_mean,
            'TF_to_random_std': tf_to_rand_std,
        })
        print(f"  α={alpha}: random→TF = {rand_to_tf_mean:.4f} ± {rand_to_tf_std:.4f}, "
              f"TF→random = {tf_to_rand_mean:.4f} ± {tf_to_rand_std:.4f}")

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_csv = os.path.join(args.output_dir, 'random_baselines.csv')
    baseline_df.to_csv(baseline_csv, index=False)
    print(f"\n  Saved {baseline_csv}")

    # ================================================================
    # Within-feature vs rest of DE genes, per condition
    # ================================================================
    print("\n" + "=" * 70)
    print("WITHIN-FEATURE vs REST (DE genes, per condition)")
    print("=" * 70)

    expr_mask = get_expressed_genes_mask(args.data_file)

    def run_wf_vs_rest(name, logfc_mat, alpha):
        stats = within_feature_vs_rest(logfc_mat, gene_names, feature_gene_set,
                                        baseline_logits, expr_mask)
        stats.update({'condition': name, 'alpha': alpha})
        return stats

    wf_rows = []
    for alpha in alphas:
        wf_rows.append(run_wf_vs_rest('tf_steered', logfc['tf_steered'][alpha], alpha))
        for g in control_genes:
            wf_rows.append(run_wf_vs_rest(f'target_steered::{g}',
                                          logfc['target_steered'][g][alpha], alpha))
        for g in random_genes:
            wf_rows.append(run_wf_vs_rest(f'random_steered::{g}',
                                          logfc['random_steered'][g][alpha], alpha))

    wf_df = pd.DataFrame(wf_rows)
    wf_df = wf_df[['condition', 'alpha', 'n_de', 'n_in_feature', 'n_rest',
                   'mean_logfc_in', 'mean_logfc_rest',
                   'median_logfc_in', 'median_logfc_rest',
                   'mean_abs_logfc_in', 'mean_abs_logfc_rest',
                   'median_abs_logfc_in', 'median_abs_logfc_rest']]
    print(wf_df.to_string(index=False))
    wf_csv = os.path.join(args.output_dir, 'within_feature_vs_rest.csv')
    wf_df.to_csv(wf_csv, index=False)
    print(f"\n  Saved {wf_csv}")

    # ================================================================
    # Plots
    # ================================================================
    print("\nGenerating plots...")
    plot_primary(primary_df, alphas, args.output_dir)
    plot_paired_per_cell(logfc, control_genes, gene_to_idx, tf_idx, alphas, args.output_dir)
    plot_wf_vs_rest(wf_df, alphas, args.output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


# --- Plots ---

def plot_primary(primary_df, alphas, output_dir):
    """Bar plot of signed mean F vs mean R per target per alpha, with p-values."""
    if len(primary_df) == 0:
        return
    targets = primary_df['target'].unique()
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), squeeze=False)
    for j, alpha in enumerate(alphas):
        ax = axes[0, j]
        sub = primary_df[primary_df['alpha'] == alpha].set_index('target').reindex(targets)
        x = np.arange(len(targets))
        ax.bar(x - 0.2, sub['mean_F'], 0.4, label='F (TF→target)', color='#e74c3c')
        ax.bar(x + 0.2, sub['mean_R'], 0.4, label='R (target→TF)', color='#3498db')
        for i, t in enumerate(targets):
            p = sub.loc[t, 'perm_pval']
            vals = [sub.loc[t, 'mean_F'], sub.loc[t, 'mean_R']]
            y = max(vals, key=abs)
            offset = 1.02 if y >= 0 else 0.98
            va = 'bottom' if y >= 0 else 'top'
            ax.text(i, y * offset, f"p={p:.1e}", ha='center', va=va, fontsize=8)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.set_title(f'α = {alpha}')
        ax.set_ylabel('Mean logFC across cells (signed)')
        ax.grid(axis='y', alpha=0.3)
        if j == 0:
            ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'primary_F_vs_R.png'), dpi=150)
    plt.close()
    print("  Saved primary_F_vs_R.png")


def plot_paired_per_cell(logfc, control_genes, gene_to_idx, tf_idx, alphas, output_dir):
    """Scatter of per-cell |F_i^c| vs |R_i^c| for each target, at the max alpha."""
    if not control_genes:
        return
    alpha = alphas[-1]
    fig, axes = plt.subplots(1, len(control_genes), figsize=(5 * len(control_genes), 5),
                              squeeze=False)
    for j, target in enumerate(control_genes):
        ax = axes[0, j]
        if target not in gene_to_idx:
            continue
        t_idx = gene_to_idx[target]
        F = np.abs(logfc['tf_steered'][alpha][:, t_idx])
        R = np.abs(logfc['target_steered'][target][alpha][:, tf_idx])
        lim = max(F.max(), R.max()) * 1.05
        ax.scatter(R, F, alpha=0.4, s=10)
        ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y=x')
        ax.set_xlabel(f'|R| (|logFC(TF)| when {target} steered)')
        ax.set_ylabel(f'|F| (|logFC({target})| when TF steered)')
        ax.set_title(f'{target}, α={alpha}')
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paired_per_cell_scatter.png'), dpi=150)
    plt.close()
    print("  Saved paired_per_cell_scatter.png")


def plot_wf_vs_rest(wf_df, alphas, output_dir):
    """For each alpha, bar of in-feature vs rest mean |logFC| across conditions."""
    for alpha in alphas:
        sub = wf_df[wf_df['alpha'] == alpha].copy()
        if len(sub) == 0:
            continue
        conds = sub['condition'].tolist()
        x = np.arange(len(conds))
        fig, ax = plt.subplots(figsize=(max(8, len(conds) * 0.6), 5))
        ax.bar(x - 0.2, sub['mean_logfc_in'], 0.4,
               label='in-feature DE', color='#2ecc71')
        ax.bar(x + 0.2, sub['mean_logfc_rest'], 0.4,
               label='rest DE', color='#95a5a6')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=60, ha='right', fontsize=8)
        ax.set_ylabel('Mean logFC (signed)')
        ax.set_title(f'In-feature vs rest of DE genes (α={alpha})')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'wf_vs_rest_alpha{alpha}.png'), dpi=150)
        plt.close()
    print(f"  Saved wf_vs_rest_alpha*.png")


if __name__ == "__main__":
    main()

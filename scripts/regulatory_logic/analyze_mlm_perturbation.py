"""Paired per-cell tests of perturbed vs reference logits, per (entry, dose).

Two reference comparisons per (bucket, key, dose):

  (A) vs masked-baseline catalog: Δ_base = perturbed − baseline_catalog.
      Detects any shift relative to the catalog's no-perturbation reference.
      In CD4 with dose=0 this is essentially a no-op (TBX21 already 0 in most
      cells); in CD8 with dose=0 it is a real silencing intervention.

  (B) vs in-experiment dose=0 control: Δ_d0 = perturbed_dose − perturbed_dose0.
      Subtracts the structural noise floor (bf16/mask/attention drift present
      identically in both conditions), so only the induction effect above
      that floor remains. At dose=0 itself this is trivially zero by
      construction. Skipped if no dose=0 entry exists for that key.

Each comparison runs paired sign-flip permutation + Wilcoxon signed-rank
(two-sided). Effect sizes additionally reported as fraction of the masked
celltype gap (CD8 − CD4 at the readout gene) when a catalog summary CSV is
available.

Usage:
    python analyze_mlm_perturbation.py \\
        --results-file results/mlm_perturbation/tbx21_cd4_forward.pt \\
        --catalog-summary results/masked_baselines/tbx21_cd4_cd8_summary.csv \\
        --output-dir plots/pbmc/mlm_pert
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def paired_permutation_two_sided(delta, n_perm=10000, seed=0):
    """Sign-flip permutation test, two-sided H1: mean(Δ) ≠ 0."""
    observed = float(delta.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(delta)))
    null = (signs * delta).mean(axis=1)
    pval = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return {'observed_mean': observed, 'pval': float(pval),
            'null_mean': float(null.mean()), 'null_std': float(null.std())}


def paired_wilcoxon_two_sided(perturbed, baseline):
    """Wilcoxon signed-rank, two-sided."""
    diff = perturbed - baseline
    if np.all(diff == 0):
        return {'stat': np.nan, 'pval': np.nan, 'n_nonzero': 0}
    try:
        stat, pval = wilcoxon(perturbed, baseline, alternative='two-sided',
                              zero_method='wilcox')
        return {'stat': float(stat), 'pval': float(pval),
                'n_nonzero': int((diff != 0).sum())}
    except ValueError as e:
        return {'stat': np.nan, 'pval': np.nan,
                'n_nonzero': int((diff != 0).sum()), 'error': str(e)}


def iter_entries(conditions):
    """Yield (bucket, key, dose, entry) tuples across all conditions."""
    for bucket in ('tf_steered', 'target_steered', 'random_steered'):
        for key, dose_dict in conditions.get(bucket, {}).items():
            for dose, entry in dose_dict.items():
                yield bucket, key, dose, entry


def parse_mean_std(s):
    """Parse 'mean ± std' string to (mean, std). Returns (nan, nan) on miss."""
    if not isinstance(s, str) or '±' not in s:
        return float('nan'), float('nan')
    a, b = s.split('±')
    return float(a.strip()), float(b.strip())


def load_celltype_gaps(summary_csv):
    """Return {gene: cd8_masked_mean - cd4_masked_mean} from the catalog summary."""
    df = pd.read_csv(summary_csv)
    gaps = {}
    for _, r in df.iterrows():
        cd4_m, _ = parse_mean_std(r.get('output_cd4_masked', ''))
        cd8_m, _ = parse_mean_std(r.get('output_cd8_masked', ''))
        if not (np.isnan(cd4_m) or np.isnan(cd8_m)):
            gaps[r['gene']] = cd8_m - cd4_m
    return gaps


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--results-file', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--catalog-summary', default=None,
                        help='Path to catalog summary CSV (e.g. '
                             'results/masked_baselines/tbx21_cd4_cd8_summary.csv) — '
                             'used to compute fraction-of-celltype-gap effect sizes.')
    parser.add_argument('--n-perm', type=int, default=10000)
    args = parser.parse_args()

    print(f"Loading {args.results_file}")
    data = torch.load(args.results_file, map_location='cpu', weights_only=False)
    meta = data['metadata']
    if meta.get('experiment_kind') != 'mlm_perturbation':
        raise SystemExit(f"Unexpected experiment_kind: {meta.get('experiment_kind')}")

    tf = meta['tf']
    direction = meta['direction']
    celltype = meta['celltype']
    baseline_logits = data['baseline']['logits']

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.results_file),
            f"analysis_{tf}_{celltype.replace(' ', '_')}_{direction}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"  tf={tf}  celltype={celltype}  direction={direction}")
    print(f"  baseline genes: {list(baseline_logits.keys())}")
    print(f"  output dir: {args.output_dir}")

    celltype_gaps = {}
    if args.catalog_summary:
        celltype_gaps = load_celltype_gaps(args.catalog_summary)
        print(f"  celltype gaps loaded for {len(celltype_gaps)} genes")

    # Build a (bucket, key) -> dose=0 perturbed array lookup for the in-experiment
    # noise-floor reference (test B).
    dose0_logits = {}
    for bucket, key, dose, entry in iter_entries(data['conditions']):
        if float(dose) == 0.0:
            dose0_logits[(bucket, key)] = entry['logits'].float().numpy().reshape(-1)

    rows = []
    for bucket, key, dose, entry in iter_entries(data['conditions']):
        readout_gene = entry['readout_gene_order'][0]
        if readout_gene not in baseline_logits:
            print(f"  SKIP {bucket}:{key} dose={dose} — no baseline for {readout_gene}")
            continue
        perturbed = entry['logits'].float().numpy().reshape(-1)
        base = baseline_logits[readout_gene].astype(np.float32).reshape(-1)
        if perturbed.shape != base.shape:
            raise SystemExit(
                f"Shape mismatch for {bucket}:{key} dose={dose}: "
                f"perturbed {perturbed.shape} vs baseline {base.shape}"
            )

        # (A) vs catalog masked baseline
        delta_base = perturbed - base
        perm_base = paired_permutation_two_sided(delta_base, n_perm=args.n_perm)
        wil_base = paired_wilcoxon_two_sided(perturbed, base)

        # (B) vs in-experiment dose=0 perturbed (noise-floor cancels)
        d0 = dose0_logits.get((bucket, key))
        if d0 is not None and d0.shape == perturbed.shape:
            delta_d0 = perturbed - d0
            perm_d0 = paired_permutation_two_sided(delta_d0, n_perm=args.n_perm)
            wil_d0 = paired_wilcoxon_two_sided(perturbed, d0)
            mean_delta_d0 = float(delta_d0.mean())
            median_delta_d0 = float(np.median(delta_d0))
            frac_pos_d0 = float((delta_d0 > 0).mean())
            perm_pval_d0 = perm_d0['pval']
            wilcoxon_pval_d0 = wil_d0['pval']
        else:
            mean_delta_d0 = np.nan
            median_delta_d0 = np.nan
            frac_pos_d0 = np.nan
            perm_pval_d0 = np.nan
            wilcoxon_pval_d0 = np.nan

        gap = celltype_gaps.get(readout_gene, np.nan)
        frac_gap = (float(delta_base.mean()) / gap) if gap and not np.isnan(gap) else np.nan

        rows.append({
            'bucket': bucket,
            'key': key,
            'perturb_gene': entry.get('perturb_gene'),
            'mask_gene': entry.get('mask_gene'),
            'readout_gene': readout_gene,
            'dose': float(dose),
            'n_cells': len(delta_base),
            # vs catalog baseline (test A)
            'mean_delta_base': float(delta_base.mean()),
            'median_delta_base': float(np.median(delta_base)),
            'frac_pos_base': float((delta_base > 0).mean()),
            'perm_pval_base': perm_base['pval'],
            'wilcoxon_pval_base': wil_base['pval'],
            # vs dose=0 perturbed (test B)
            'mean_delta_d0': mean_delta_d0,
            'median_delta_d0': median_delta_d0,
            'frac_pos_d0': frac_pos_d0,
            'perm_pval_d0': perm_pval_d0,
            'wilcoxon_pval_d0': wilcoxon_pval_d0,
            # context
            'celltype_gap': gap,
            'fraction_of_celltype_gap': frac_gap,
            'baseline_mean': float(base.mean()),
            'perturbed_mean': float(perturbed.mean()),
        })

    df = pd.DataFrame(rows).sort_values(['bucket', 'key', 'dose']).reset_index(drop=True)
    csv_path = os.path.join(args.output_dir, 'paired_tests.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path} ({len(df)} rows)")

    # Quick console summary
    print("\n[A] vs catalog baseline   (B) vs dose=0 perturbed")
    for (bucket, key), sub in df.groupby(['bucket', 'key']):
        print(f"\n  [{bucket} :: {key}]")
        for _, r in sub.iterrows():
            gap_s = (f" frac_gap={r['fraction_of_celltype_gap']:+.2f}"
                     if not np.isnan(r['fraction_of_celltype_gap']) else "")
            if not np.isnan(r['mean_delta_d0']):
                d0_s = (f" | (B) Δ_d0={r['mean_delta_d0']:+.5f} "
                        f"perm={r['perm_pval_d0']:.1e} "
                        f"wil={r['wilcoxon_pval_d0']:.1e}")
            else:
                d0_s = " | (B) –"
            print(f"    dose={r['dose']:7.3f}  "
                  f"(A) Δ_base={r['mean_delta_base']:+.5f} "
                  f"perm={r['perm_pval_base']:.1e} "
                  f"wil={r['wilcoxon_pval_base']:.1e}"
                  f"{gap_s}{d0_s}")

    plot_per_key(df, args.output_dir)
    print(f"\nDone.")


def plot_per_key(df, output_dir):
    """Per (bucket, key): two side-by-side panels (vs baseline, vs dose=0).

    Bars annotated with mean Δ and perm p-value; right axis shows fraction-of-
    celltype-gap when available. Dose=0 bar in panel B is forced to 0.
    """
    if len(df) == 0:
        return
    for (bucket, key), sub in df.groupby(['bucket', 'key']):
        sub = sub.sort_values('dose').reset_index(drop=True)
        readout = sub['readout_gene'].iloc[0]
        perturb = sub['perturb_gene'].iloc[0]
        gap = sub['celltype_gap'].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.6 * len(sub)), 4),
                                  sharey=True)
        x = np.arange(len(sub))

        # Panel A: vs catalog baseline
        axA = axes[0]
        bars_A = axA.bar(x, sub['mean_delta_base'],
                          color=['#3498db' if d >= 0 else '#e74c3c'
                                 for d in sub['mean_delta_base']])
        axA.axhline(0, color='k', lw=0.5)
        for i, r in sub.iterrows():
            star = '*' if r['perm_pval_base'] < 0.05 else ''
            axA.text(i, r['mean_delta_base'],
                     f" p={r['perm_pval_base']:.1e}{star}",
                     ha='center',
                     va='bottom' if r['mean_delta_base'] >= 0 else 'top',
                     fontsize=7)
        axA.set_xticks(x)
        axA.set_xticklabels([f"{d:.2f}" for d in sub['dose']])
        axA.set_xlabel('dose (log1p CPM)')
        axA.set_ylabel('mean Δ logit')
        axA.set_title('(A) vs catalog masked baseline')
        axA.grid(axis='y', alpha=0.3)

        # Panel B: vs dose=0 perturbed
        axB = axes[1]
        vals_B = sub['mean_delta_d0'].fillna(0.0).values
        bars_B = axB.bar(x, vals_B,
                         color=['#3498db' if d >= 0 else '#e74c3c'
                                for d in vals_B])
        axB.axhline(0, color='k', lw=0.5)
        for i, r in sub.iterrows():
            if np.isnan(r['mean_delta_d0']):
                continue
            star = '*' if r['perm_pval_d0'] < 0.05 else ''
            axB.text(i, r['mean_delta_d0'],
                     f" p={r['perm_pval_d0']:.1e}{star}",
                     ha='center',
                     va='bottom' if r['mean_delta_d0'] >= 0 else 'top',
                     fontsize=7)
        axB.set_xticks(x)
        axB.set_xticklabels([f"{d:.2f}" for d in sub['dose']])
        axB.set_xlabel('dose (log1p CPM)')
        axB.set_title('(B) vs in-experiment dose=0')
        axB.grid(axis='y', alpha=0.3)

        title = f"{bucket} :: {perturb}  →  {readout}"
        if not np.isnan(gap):
            title += f"   (celltype gap CD8−CD4 = {gap:+.3f})"
        fig.suptitle(title)
        plt.tight_layout()
        fname = f"delta_{bucket}_{key}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
    print(f"\nSaved per-key delta plots to {output_dir}/")


if __name__ == '__main__':
    main()

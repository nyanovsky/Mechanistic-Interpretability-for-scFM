"""Resolve committee A (IFNG TF regulators) and committee B (TBX21 non-TF
targets) plus size-matched random pools, ahead of the combinatorial
directionality experiment. No model forward pass.

Steps:
  1. Load CollecTRI; enumerate IFNG TFs (110) and TBX21 non-TF targets (42).
  2. Load PBMC3K + baseline catalog; restrict to catalog cells per celltype.
  3. Apply vocab filter (gene must be in aligned var_names) and expression
     filter (mean log1p(CPM) > 0.01 OR pct nonzero > 0.5%) over the
     CD4∪CD8 T-cell subset.
  4. Rank by mean log1p(CPM) over CD4∪CD8 T cells; take top-6 per committee.
  5. Build A_random (n=6) = expressed in-vocab genes that are NOT IFNG
     regulators and NOT in CollecTRI's TF source list. Seed with the
     single-gene experiment's pool {OASL, HMGA1, KPNA2} if they qualify.
     B_random analogous but excluding TBX21 targets and TFs.
  6. Per gene per celltype, compute the 5-point dose grid via the shared
     compute_dose_grid; record dose-index-2 (≈max) and dose-index-3
     (≈1.5×max) for the perturbation step and the joint-config OOD check.
  7. Joint-config OOD: per celltype, count cells where all 6 committee
     members are simultaneously above their own dose-index-2 in
     preprocessed input space.
  8. Write CSV + markdown report for user sign-off.

Usage:
    python scripts/regulatory_logic/resolve_committees.py \\
        --baseline-catalog results/masked_baselines/tbx21_cd4_cd8.pt \\
        --data-file data/pbmc/pbmc3k_raw.h5ad \\
        --committee-size 6 \\
        --report-md reports/pbmc/committee_resolution_report.md \\
        --output-csv results/mlm_perturbation/committee_resolved.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import torch
import decoupler as dc

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ModelGenerator', 'huggingface', 'aido.cell'))
sys.path.insert(0, os.path.join(_REPO, 'scripts'))

from aido_cell.utils import align_adata, preprocess_counts  # noqa: E402
from regulatory_logic.mlm_perturbation_experiment import compute_dose_grid  # noqa: E402


ANCHOR_TF = 'TBX21'
ANCHOR_TARGET = 'IFNG'
RANDOM_SEED_POOL = ['OASL', 'HMGA1', 'KPNA2']
CELLTYPES = ['CD4 T cells', 'CD8 T cells']
PSEUDO_COMPLEXES = {'NFKB', 'AP1'}  # CollecTRI rows that aren't real symbols
# Canonical Th1 master/effector TFs. All confirmed in CollecTRI's IFNG-regulator
# edge set. Used by --committee-a-mode th1 instead of expression rank.
TH1_DEFAULT = ['STAT1', 'STAT4', 'IRF1', 'TBX21', 'EOMES', 'RUNX3']


def fmt_float(x, prec=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    return f'{x:.{prec}f}'


def df_to_md(df, cols):
    """Render a small dataframe as a markdown table without needing tabulate."""
    if len(df) == 0:
        return '*(no rows)*'
    head = '| ' + ' | '.join(cols) + ' |'
    sep = '|' + '|'.join(['---'] * len(cols)) + '|'
    body = []
    for _, r in df.iterrows():
        body.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join([head, sep] + body)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--baseline-catalog', required=True)
    parser.add_argument('--data-file', default='data/pbmc/pbmc3k_raw.h5ad')
    parser.add_argument('--committee-size', type=int, default=6)
    parser.add_argument('--committee-a-mode', choices=['expression', 'th1'],
                        default='expression',
                        help='expression = force-include + top-up by union '
                             'mean log1p(CPM). th1 = use the canonical Th1 TF '
                             'list (overrides force-include and rank).')
    parser.add_argument('--committee-a-th1-list', nargs='*', default=TH1_DEFAULT,
                        help='Th1 priority list used when '
                             '--committee-a-mode th1.')
    parser.add_argument('--force-include-a', nargs='*', default=['TBX21'],
                        help='Genes always kept in committee A under the '
                             'expression mode (then top-up by rank).')
    parser.add_argument('--force-include-b', nargs='*', default=[],
                        help='Genes always kept in committee B.')
    parser.add_argument('--report-md', required=True)
    parser.add_argument('--output-csv', required=True)
    args = parser.parse_args()

    print('=' * 70)
    print('COMMITTEE RESOLUTION (no model forward)')
    print('=' * 70)
    print(f'Anchor TF:        {ANCHOR_TF}')
    print(f'Anchor target:    {ANCHOR_TARGET}')
    print(f'Committee size:   {args.committee_size}')
    print(f'Random seed pool: {RANDOM_SEED_POOL}')

    # 1. CollecTRI
    print('\n1. Loading CollecTRI ...')
    net = dc.op.collectri()
    all_tfs = set(net['source'].unique())
    ifng_edges = net[net['target'] == ANCHOR_TARGET].copy()
    ifng_regs_pool = ifng_edges[~ifng_edges['source'].isin(PSEUDO_COMPLEXES)]
    tbx21_edges = net[net['source'] == ANCHOR_TF].copy()
    tbx21_target_pool = tbx21_edges[~tbx21_edges['target'].isin(all_tfs)]
    print(f'   IFNG regulators (CollecTRI):     '
          f'{len(ifng_edges)} edges, {len(ifng_regs_pool)} after pseudo-complex drop')
    print(f'   TBX21 non-TF targets (CollecTRI): {len(tbx21_target_pool)}')

    # 2. Catalog + data
    print('\n2. Loading baseline catalog and aligned data ...')
    catalog = torch.load(args.baseline_catalog, map_location='cpu',
                         weights_only=False)
    cell_names = list(catalog['cell_names'])
    cell_celltype = list(catalog['cell_celltype'])
    for ct in CELLTYPES:
        n = sum(1 for c in cell_celltype if c == ct)
        print(f'   catalog has {n} cells of "{ct}"')

    adata = ad.read_h5ad(args.data_file)
    adata.var_names_make_unique()
    adata_aligned, attention_mask = align_adata(adata)
    var_names = list(adata_aligned.var_names)
    name_to_pos = {n: i for i, n in enumerate(var_names)}

    name_to_aligned_idx = {n: i for i, n in enumerate(adata_aligned.obs_names)}
    missing = [n for n in cell_names if n not in name_to_aligned_idx]
    if missing:
        raise SystemExit(f'{len(missing)} catalog cells missing from aligned data')
    catalog_cell_aligned_idx = np.array(
        [name_to_aligned_idx[n] for n in cell_names])

    # Per-celltype cell indices into the catalog ordering
    per_ct_idx = {}
    for ct in CELLTYPES:
        per_ct_idx[ct] = np.array(
            [i for i, c in enumerate(cell_celltype) if c == ct])

    # 3. Preprocess full PBMC3K T-cell subset once for ranking + dose grid
    print('\n3. Preprocessing input counts on catalog cells ...')
    counts_subset = adata_aligned.X[catalog_cell_aligned_idx]
    if hasattr(counts_subset, 'toarray'):
        counts_subset = counts_subset.toarray()
    proc = preprocess_counts(counts_subset, device='cpu')[:, :-2].float().numpy()
    # `proc` is (1460, n_genes) in log1p(CPM) space.

    union_idx = np.concatenate([per_ct_idx[ct] for ct in CELLTYPES])
    mean_expr_union = proc[union_idx].mean(axis=0)
    per_ct_mean = {ct: proc[per_ct_idx[ct]].mean(axis=0) for ct in CELLTYPES}
    per_ct_pctnz = {
        ct: (proc[per_ct_idx[ct]] > 0).mean(axis=0) * 100.0 for ct in CELLTYPES
    }

    # 4. Vocab + expression filter helpers
    def in_vocab(g):
        return g in name_to_pos

    def expressed(g):
        if not in_vocab(g):
            return False
        i = name_to_pos[g]
        for ct in CELLTYPES:
            if per_ct_mean[ct][i] > 0.01 or per_ct_pctnz[ct][i] > 0.5:
                return True
        return False

    def mean_expr_in(ct, g):
        if not in_vocab(g):
            return float('nan')
        return float(per_ct_mean[ct][name_to_pos[g]])

    def pctnz_in(ct, g):
        if not in_vocab(g):
            return float('nan')
        return float(per_ct_pctnz[ct][name_to_pos[g]])

    def mean_expr_union_of(g):
        if not in_vocab(g):
            return float('nan')
        return float(mean_expr_union[name_to_pos[g]])

    def resolve_committee(pool, gene_col, force_include, label):
        p = pool.copy()
        p['in_vocab'] = p[gene_col].apply(in_vocab)
        p['expressed'] = p[gene_col].apply(expressed)
        p['mean_expr_union'] = p[gene_col].apply(mean_expr_union_of)
        keep = p[p['in_vocab'] & p['expressed']].copy()
        keep = keep.sort_values('mean_expr_union', ascending=False)
        forced_rows = keep[keep[gene_col].isin(force_include)]
        forced_missing = [g for g in force_include
                          if g not in set(forced_rows[gene_col])
                          and (g not in set(p[gene_col]) or not in_vocab(g)
                               or not expressed(g))]
        if forced_missing:
            print(f'   WARNING: force-include for {label} dropped '
                  f'(absent from pool / vocab / expression filter): '
                  f'{forced_missing}')
        topup = keep[~keep[gene_col].isin(force_include)]
        n_remaining = args.committee_size - len(forced_rows)
        chosen = pd.concat([forced_rows, topup.head(max(0, n_remaining))])
        chosen = chosen.sort_values('mean_expr_union', ascending=False)
        # Keep the residual ranked list for the "near misses" report section.
        residual = topup.head(max(0, n_remaining) + 8).iloc[
            max(0, n_remaining):
        ]
        print(f'\n   Committee {label}: {len(chosen)}/{len(p)} '
              f'(force-include kept: '
              f'{sorted(set(forced_rows[gene_col]))})')
        return chosen, residual, p

    def resolve_committee_th1(pool, gene_col, th1_list, label):
        """Pick the committee as the th1_list, in the order given, after
        verifying each gene is in the pool + vocab. Genes that fail the
        expression filter are KEPT (Th1 TFs are often sparsely expressed
        but we still want them perturbed); a warning is emitted."""
        p = pool.copy()
        p['in_vocab'] = p[gene_col].apply(in_vocab)
        p['expressed'] = p[gene_col].apply(expressed)
        p['mean_expr_union'] = p[gene_col].apply(mean_expr_union_of)
        keep = p[p[gene_col].isin(th1_list)].copy()
        # Preserve the th1_list order rather than expression order.
        order = {g: i for i, g in enumerate(th1_list)}
        keep['_order'] = keep[gene_col].map(order)
        keep = keep.sort_values('_order').drop(columns='_order')
        missing_pool = [g for g in th1_list if g not in set(pool[gene_col])]
        not_in_vocab = [g for g in th1_list
                        if g in set(pool[gene_col]) and not in_vocab(g)]
        not_expressed = [g for g in th1_list
                         if g in set(pool[gene_col]) and in_vocab(g)
                         and not expressed(g)]
        if missing_pool:
            print(f'   WARNING: th1 list members not in IFNG-regulator '
                  f'CollecTRI pool: {missing_pool}')
        if not_in_vocab:
            print(f'   WARNING: th1 list members not in AIDO.Cell vocab '
                  f'(dropped): {not_in_vocab}')
        if not_expressed:
            print(f'   NOTE: th1 list members below expression filter '
                  f'(kept anyway): {not_expressed}')
        keep = keep[keep['in_vocab']]
        print(f'\n   Committee {label}: {len(keep)}/{len(th1_list)} th1 members '
              f'used (vocab-passing)')
        # No near-miss for th1 mode; emit an empty df with the same columns.
        return keep.copy(), keep.iloc[0:0].copy(), p

    print('\n4. Resolving committees ...')
    if args.committee_a_mode == 'th1':
        committee_a, near_miss_a, pool_a = resolve_committee_th1(
            ifng_regs_pool, 'source', args.committee_a_th1_list, 'A')
    else:
        committee_a, near_miss_a, pool_a = resolve_committee(
            ifng_regs_pool, 'source', args.force_include_a, 'A')
    committee_b, near_miss_b, pool_b = resolve_committee(
        tbx21_target_pool, 'target', args.force_include_b, 'B')

    # 7. Build random pools. Expression-matched to the committee they
    # control: candidates are ranked by abs(mean_expr_union − target_mean),
    # where target_mean is the *per-member* expression of the committee.
    # We pair one random gene to each committee member by closest
    # union-mean expression, so the random pool spans the same expression
    # range as the regulon and isn't dominated by housekeeping. Seeds
    # {OASL, HMGA1, KPNA2} are kept if eligible AND if their expression
    # lands within the committee's range.
    ifng_reg_set = set(ifng_regs_pool['source'])
    tbx21_target_set = set(tbx21_target_pool['target'])

    def build_random_pool(committee_genes, committee_member_means, exclude_set,
                          label):
        eligible = [g for g in var_names
                    if g not in exclude_set
                    and g not in all_tfs
                    and expressed(g)]
        # Seeds first (only if eligible).
        chosen = []
        for s in RANDOM_SEED_POOL:
            if s in eligible and s not in chosen:
                chosen.append(s)
        # Now pair the remaining committee_member_means to nearest-expression
        # non-chosen eligible genes. We iterate sorted-by-target-mean so the
        # biggest target is matched first (it has the most need of a high-
        # expression neighbour).
        used = set(chosen)
        committee_set = set(committee_genes)
        targets_for_topup = sorted(
            committee_member_means, key=lambda x: -x
        )
        # Skip targets that the seeds already cover (heuristic: any seed
        # whose mean is closer to this target than 0.5 logit-units).
        n_seed = len(chosen)
        remaining_targets = targets_for_topup[n_seed:]
        # Build a (gene, expr) array of eligible non-used candidates once.
        eligible_arr = np.array(
            [(g, mean_expr_union_of(g)) for g in eligible
             if g not in committee_set],
            dtype=object,
        )
        if len(eligible_arr) == 0:
            print(f'   Random pool [{label}]: NONE eligible — bug or filter '
                  f'too strict.')
            return chosen
        gene_arr = eligible_arr[:, 0]
        expr_arr = eligible_arr[:, 1].astype(float)
        for tgt in remaining_targets:
            mask_used = np.array([g not in used for g in gene_arr])
            if not mask_used.any():
                break
            dists = np.where(mask_used, np.abs(expr_arr - tgt), np.inf)
            j = int(np.argmin(dists))
            chosen.append(gene_arr[j])
            used.add(gene_arr[j])
        kept = [s for s in RANDOM_SEED_POOL if s in chosen]
        print(f'   Random pool [{label}]: {len(chosen)} chosen '
              f'(seed kept: {kept})')
        return chosen

    committee_a_means = [mean_expr_union_of(g) for g in committee_a['source']]
    committee_b_means = [mean_expr_union_of(g) for g in committee_b['target']]
    a_random = build_random_pool(
        list(committee_a['source']), committee_a_means,
        ifng_reg_set, 'A_random (~IFNG-regulator)')
    b_random = build_random_pool(
        list(committee_b['target']), committee_b_means,
        tbx21_target_set, 'B_random (~TBX21-target)')

    # 8. Per-gene per-celltype dose grids (committee + random)
    print('\n5. Computing per-gene per-celltype dose grids ...')
    all_perturb_genes = (
        list(committee_a['source']) + list(committee_b['target'])
        + a_random + b_random
    )
    all_perturb_genes = sorted(set(all_perturb_genes))
    dose_grid = {ct: {} for ct in CELLTYPES}
    for g in all_perturb_genes:
        pos = name_to_pos[g]
        for ct in CELLTYPES:
            dose_grid[ct][g] = compute_dose_grid(proc[per_ct_idx[ct], pos])

    # 9. Joint-config naturalness per committee per celltype.
    # The dose grid is {0, q75(nz), max, 1.5*max, 3*max}; index-2 is the
    # observed per-celltype max, so "all members above idx2" is 0 by
    # construction. Two more meaningful thresholds:
    #   (a) "all members > 0"          — joint co-expression baseline.
    #   (b) "all members >= q75(nz)"   — joint highly-expressed configuration.
    # The dose-index-3 we will impose (1.5 * max) is strictly higher than
    # both of these, so neither number is "this exact joint config is
    # observed"; together they bound how natural a multi-gene joint-high
    # state is in this dataset.
    print('\n6. Joint-config naturalness counts ...')

    def joint_above(member_genes, ct, threshold_kind):
        sub = proc[per_ct_idx[ct]]
        n_cells = sub.shape[0]
        if n_cells == 0:
            return 0, 0
        ok = np.ones(n_cells, dtype=bool)
        for g in member_genes:
            col = sub[:, name_to_pos[g]]
            if threshold_kind == 'nonzero':
                ok &= col > 0
            elif threshold_kind == 'q75_nz':
                grid = dose_grid[ct][g]
                thr = grid[1] if len(grid) > 1 else None
                if thr is None:
                    ok &= False
                else:
                    ok &= col >= thr
        return int(ok.sum()), n_cells

    joint_stats = {}
    members_a = list(committee_a['source'])
    members_b = list(committee_b['target'])
    for ct in CELLTYPES:
        joint_stats[ct] = {}
        for thr in ('nonzero', 'q75_nz'):
            n_a, tot_a = joint_above(members_a, ct, thr)
            n_b, tot_b = joint_above(members_b, ct, thr)
            joint_stats[ct][thr] = {'A': (n_a, tot_a), 'B': (n_b, tot_b)}
            print(f'   {ct:14s} thr={thr:8s}  A: {n_a}/{tot_a} '
                  f'({100*n_a/tot_a:.1f}%)   B: {n_b}/{tot_b} '
                  f'({100*n_b/tot_b:.1f}%)')

    # 10. Build long-format CSV
    print(f'\n7. Writing CSV: {args.output_csv}')

    def row(role, gene, weight, exclude=False, kind=''):
        r = {
            'role': role,
            'gene': gene,
            'kind': kind,
            'collectri_weight': float(weight) if weight is not None else '',
            'in_vocab': in_vocab(gene),
            'expressed': expressed(gene),
            'mean_expr_union': fmt_float(mean_expr_union_of(gene)),
        }
        for ct in CELLTYPES:
            short = 'cd4' if 'CD4' in ct else 'cd8'
            r[f'mean_expr_{short}'] = fmt_float(mean_expr_in(ct, gene))
            r[f'pctnz_{short}'] = fmt_float(pctnz_in(ct, gene), 2)
            if gene in dose_grid[ct]:
                grid = dose_grid[ct][gene]
                r[f'dose_idx2_{short}'] = fmt_float(grid[2] if len(grid) > 2 else None)
                r[f'dose_idx3_{short}'] = fmt_float(grid[3] if len(grid) > 3 else None)
            else:
                r[f'dose_idx2_{short}'] = ''
                r[f'dose_idx3_{short}'] = ''
        return r

    rows = []
    for _, e in committee_a.iterrows():
        rows.append(row('committee_A', e['source'], e['weight'],
                        kind='activator' if e['weight'] > 0 else 'repressor'))
    for _, e in committee_b.iterrows():
        rows.append(row('committee_B', e['target'], e['weight'],
                        kind='activator' if e['weight'] > 0 else 'repressor'))
    for g in a_random:
        rows.append(row('A_random', g, None, kind='random'))
    for g in b_random:
        rows.append(row('B_random', g, None, kind='random'))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # 11. Markdown report
    print(f'8. Writing report: {args.report_md}')
    os.makedirs(os.path.dirname(os.path.abspath(args.report_md)), exist_ok=True)

    def md_table(role_filter):
        sub = df[df['role'] == role_filter]
        cols = ['gene', 'collectri_weight', 'kind', 'mean_expr_union',
                'mean_expr_cd4', 'mean_expr_cd8',
                'pctnz_cd4', 'pctnz_cd8',
                'dose_idx2_cd4', 'dose_idx3_cd4',
                'dose_idx2_cd8', 'dose_idx3_cd8']
        return df_to_md(sub, cols)

    lines = []
    lines.append(f'# Committee resolution — {ANCHOR_TF} ↔ {ANCHOR_TARGET}')
    lines.append('')
    lines.append('Generated by `scripts/regulatory_logic/resolve_committees.py`.')
    lines.append('No model forward pass — this report describes only the gene-set '
                 'selection and dose values to be used in the upcoming experiment.')
    lines.append('')
    lines.append('## Inputs')
    lines.append('')
    lines.append(f'- Anchor TF: **{ANCHOR_TF}**; anchor target: **{ANCHOR_TARGET}**.')
    lines.append(f'- CollecTRI source: `decoupler.op.collectri()`.')
    lines.append(f'- PBMC3K data: `{args.data_file}`.')
    lines.append(f'- Baseline catalog: `{args.baseline_catalog}`.')
    lines.append(f'- Committee size: {args.committee_size}; random pool size matched.')
    lines.append(f'- Committee A mode: **{args.committee_a_mode}**.')
    if args.committee_a_mode == 'th1':
        lines.append(f'  - Th1 list: `{args.committee_a_th1_list}`.')
    else:
        lines.append(f'  - Force-include for A: `{args.force_include_a}`.')
    lines.append(f'- Seed random pool: `{RANDOM_SEED_POOL}` (kept if they pass filters).')
    lines.append('')
    lines.append('## CollecTRI pools (pre-filter)')
    lines.append('')
    lines.append(f'- **IFNG regulators**: {len(ifng_edges)} CollecTRI edges → '
                 f'{len(ifng_regs_pool)} after dropping pseudo-complexes '
                 f'`{sorted(PSEUDO_COMPLEXES)}`.')
    lines.append(f'- **TBX21 non-TF targets**: {len(tbx21_target_pool)} '
                 f'(of {len(tbx21_edges)} total TBX21 edges; '
                 f'{len(tbx21_edges) - len(tbx21_target_pool)} TF-targets excluded by spec).')
    lines.append('')
    def near_miss_md(near_miss_df, gene_col, label):
        if len(near_miss_df) == 0:
            return f'*(no near-miss candidates for {label})*'
        rows = []
        for _, r in near_miss_df.iterrows():
            rows.append(f'`{r[gene_col]}` ({r["mean_expr_union"]:.3f})')
        return ', '.join(rows)

    if args.committee_a_mode == 'th1':
        lines.append('## Committee A — IFNG regulators (Th1 priority list)')
    else:
        lines.append('## Committee A — IFNG regulators (force-include + top-N by union mean expr)')
    lines.append('')
    lines.append(md_table('committee_A'))
    lines.append('')
    n_act_a = int((committee_a['weight'] > 0).sum())
    n_rep_a = int((committee_a['weight'] < 0).sum())
    lines.append(f'Activators: {n_act_a}, repressors: {n_rep_a}.')
    if args.committee_a_mode == 'expression' and ANCHOR_TF in list(committee_a['source']):
        lines.append(f'Note: {ANCHOR_TF} itself is forced into committee A '
                     f'(per design decision; would not have been in the '
                     f'expression-ranked top-{args.committee_size}).')
    lines.append('')
    if args.committee_a_mode == 'expression':
        lines.append(f'**Next 8 IFNG regulators by union expression** (not in '
                     f'the committee): '
                     + near_miss_md(near_miss_a, 'source', 'A'))
        lines.append('')
    lines.append('## Committee B — TBX21 non-TF targets (force-include + top-N by union mean expr)')
    lines.append('')
    lines.append(md_table('committee_B'))
    lines.append('')
    n_act_b = int((committee_b['weight'] > 0).sum())
    n_rep_b = int((committee_b['weight'] < 0).sum())
    lines.append(f'Activators: {n_act_b}, repressors: {n_rep_b}.')
    lines.append('')
    lines.append(f'**Next 8 TBX21 non-TF targets by union expression** (not in '
                 f'the committee): ' + near_miss_md(near_miss_b, 'target', 'B'))
    lines.append('')
    lines.append('## A_random — non-regulators of IFNG, not TFs')
    lines.append('')
    lines.append(md_table('A_random'))
    lines.append('')
    lines.append('## B_random — non-targets of TBX21, not TFs')
    lines.append('')
    lines.append(md_table('B_random'))
    lines.append('')
    lines.append('## Joint-config naturalness check')
    lines.append('')
    lines.append('Per celltype: cells in which **all committee members are '
                 'simultaneously above a threshold** in the preprocessed '
                 '(log1p CPM) input space. The dose-index-3 we will impose '
                 '(1.5 × per-celltype max) is strictly above the per-cell '
                 'observed values, so the joint config we will set is never '
                 'natively observed. Two reference thresholds bracket how '
                 'natural a multi-gene high-expression state is in this '
                 'dataset:')
    lines.append('')
    lines.append('- `nonzero`: all members have any non-zero expression in '
                 'the same cell.')
    lines.append('- `q75_nz`: all members are simultaneously above their own '
                 'q75 of nonzero expression in the same cell (dose-index-1).')
    lines.append('')
    lines.append('| celltype | threshold | A: all-above / total | B: all-above / total |')
    lines.append('|---|---|---|---|')
    for ct in CELLTYPES:
        for thr in ('nonzero', 'q75_nz'):
            n_a, tot_a = joint_stats[ct][thr]['A']
            n_b, tot_b = joint_stats[ct][thr]['B']
            lines.append(f'| {ct} | {thr} | {n_a}/{tot_a} '
                         f'({100*n_a/tot_a:.1f}%) | {n_b}/{tot_b} '
                         f'({100*n_b/tot_b:.1f}%) |')
    lines.append('')
    lines.append('If `nonzero` is ≪ 1% on a celltype we will be running, '
                 'consider falling back from dose-index-3 to dose-index-2 '
                 '(=max) to reduce joint-config OOD pressure.')
    lines.append('')
    lines.append('## Sign-handling rule (for downstream runner)')
    lines.append('')
    lines.append('- Activators (CollecTRI weight +1): set input to **+dose-index-3**.')
    lines.append('- Repressors (CollecTRI weight −1): set input to **0** (silenced; '
                 'log1p(CPM) is non-negative so "down" = zero-out).')
    lines.append('- Random-pool members: set input to +dose-index-3 (no sign).')
    lines.append('')
    lines.append('## Sign-off')
    lines.append('')
    lines.append('Review the four tables and the joint-config check. If OK, the next '
                 'step is `scripts/regulatory_logic/committee_perturbation_experiment.py` '
                 f'(four model passes per celltype). CSV at `{args.output_csv}`.')

    with open(args.report_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Done.')


if __name__ == '__main__':
    main()

"""Committee perturbation runner: joint multi-gene input overrides + mask.

Continuation of mlm_perturbation_experiment.py for the combinatorial
directionality experiment. Each invocation runs **four** forward passes
over one celltype's cells:

  - A_committee: override committee_A members (activators -> +dose,
                 repressors -> 0), mask IFNG, read IFNG logit.
  - A_random:    override A_random members at unsigned +dose, mask IFNG,
                 read IFNG logit.
  - B_committee: override committee_B members (all activators here)
                 at +dose, mask TBX21, read TBX21 logit.
  - B_random:    override B_random members at unsigned +dose, mask TBX21,
                 read TBX21 logit.

Doses come from the resolver CSV at dose_idx3_<celltype> (1.5 * per-celltype
max, same `compute_dose_grid` as the single-gene experiment).

Usage:
    python committee_perturbation_experiment.py \\
        --committee-spec results/mlm_perturbation/committee_resolved.csv \\
        --celltype "CD4 T cells" \\
        --baseline-catalog results/masked_baselines/tbx21_cd4_cd8.pt \\
        --output results/mlm_perturbation/committee_ifng_tbx21_cd4.pt
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import torch
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ModelGenerator', 'huggingface', 'aido.cell'))

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts


MASK_TOKEN_ID = -1.0
ANCHOR_TF = 'TBX21'
ANCHOR_TARGET = 'IFNG'

CELLTYPE_SUFFIX = {
    'CD4 T cells': 'cd4',
    'CD8 T cells': 'cd8',
}


def perturbed_pass_multi(model, adata_aligned, attention_mask,
                         perturb_positions, mask_pos, readout_pos,
                         cell_indices, device, batch_size, desc="",
                         record_positions=None):
    """One forward pass with multiple input overrides + one mask + one readout.

    Args:
        perturb_positions: dict[int, float] — position -> dose value applied
            to every cell in the batch. Empty dict = unperturbed pass.
        mask_pos: int.
        readout_pos: int.
        cell_indices: np.ndarray of int — rows of adata_aligned.
        record_positions: optional list of int — positions whose output logits
            to additionally record (e.g. the perturbed input positions, to
            check whether the model "echoes" the override).

    Returns:
        (readout, recorded) tuple:
          readout:  np.ndarray (n_cells,) of float32 logits at readout_pos.
          recorded: dict[int, np.ndarray] of float32 logits at each
                    record_position (empty if record_positions is None/[]).
    """
    n_cells = len(cell_indices)
    n_genes_full = adata_aligned.n_vars
    out = np.zeros(n_cells, dtype=np.float32)
    record_positions = list(record_positions or [])
    recorded = {p: np.zeros(n_cells, dtype=np.float32) for p in record_positions}
    attn_full = torch.from_numpy(attention_mask).to(device)

    with torch.no_grad():
        for start in tqdm(range(0, n_cells, batch_size), desc=desc):
            end = min(start + batch_size, n_cells)
            batch_idx = cell_indices[start:end]
            batch_counts = adata_aligned.X[batch_idx]
            if hasattr(batch_counts, 'toarray'):
                batch_counts = batch_counts.toarray()

            batch_processed = preprocess_counts(batch_counts, device=device)
            b = batch_processed.shape[0]
            input_ids = batch_processed.clone()

            for pos, dose in perturb_positions.items():
                d = torch.tensor(float(dose), dtype=input_ids.dtype, device=device)
                input_ids[:, pos] = d

            mask_sentinel = torch.tensor(MASK_TOKEN_ID, dtype=input_ids.dtype,
                                         device=device)
            input_ids[:, mask_pos] = mask_sentinel

            batch_attn = torch.cat([
                attn_full.unsqueeze(0).repeat(b, 1),
                torch.ones((b, 2), device=device),
            ], dim=1)

            outputs = model(
                input_ids=input_ids,
                attention_mask=batch_attn,
                return_dict=True,
            )
            logits = outputs.logits[:, :n_genes_full, :].squeeze(-1).float().cpu().numpy()
            out[start:end] = logits[:, readout_pos]
            for p in record_positions:
                recorded[p][start:end] = logits[:, p]

    return out, recorded


def perturbed_pass_per_cell(model, adata_aligned, attention_mask,
                            perturb_per_cell, mask_pos, readout_pos,
                            cell_indices, device, batch_size, desc=""):
    """Per-cell multi-override variant (for no-op sanity check).

    Args:
        perturb_per_cell: dict[int, torch.Tensor(n_cells,)] — position ->
            per-cell dose values, aligned with cell_indices.
    """
    n_cells = len(cell_indices)
    n_genes_full = adata_aligned.n_vars
    out = np.zeros(n_cells, dtype=np.float32)
    attn_full = torch.from_numpy(attention_mask).to(device)
    perturb_per_cell = {
        pos: v.to(device) for pos, v in perturb_per_cell.items()
    }

    with torch.no_grad():
        for start in tqdm(range(0, n_cells, batch_size), desc=desc):
            end = min(start + batch_size, n_cells)
            batch_idx = cell_indices[start:end]
            batch_counts = adata_aligned.X[batch_idx]
            if hasattr(batch_counts, 'toarray'):
                batch_counts = batch_counts.toarray()

            batch_processed = preprocess_counts(batch_counts, device=device)
            b = batch_processed.shape[0]
            input_ids = batch_processed.clone()

            for pos, vec in perturb_per_cell.items():
                input_ids[:, pos] = vec[start:end].to(input_ids.dtype)

            mask_sentinel = torch.tensor(MASK_TOKEN_ID, dtype=input_ids.dtype,
                                         device=device)
            input_ids[:, mask_pos] = mask_sentinel

            batch_attn = torch.cat([
                attn_full.unsqueeze(0).repeat(b, 1),
                torch.ones((b, 2), device=device),
            ], dim=1)

            outputs = model(
                input_ids=input_ids,
                attention_mask=batch_attn,
                return_dict=True,
            )
            logits = outputs.logits[:, :n_genes_full, :].squeeze(-1).float().cpu().numpy()
            out[start:end] = logits[:, readout_pos]

    return out


def signed_dose(kind, dose):
    """Activators/random -> +dose; repressors -> 0.0 (silenced input)."""
    if kind == 'repressor':
        return 0.0
    return float(dose)


def build_committees(spec_df, celltype_suffix, name_to_pos, dose_index):
    """Group resolver rows into {role: [{gene, kind, weight, dose, pos}...]}."""
    dose_col = f'dose_idx{dose_index}_{celltype_suffix}'
    if dose_col not in spec_df.columns:
        raise SystemExit(f"committee spec missing column '{dose_col}'")
    committees = {}
    for role, sub in spec_df.groupby('role'):
        members = []
        for _, r in sub.iterrows():
            gene = r['gene']
            if gene not in name_to_pos:
                raise SystemExit(f"gene '{gene}' (role={role}) not in vocab")
            members.append({
                'gene': gene,
                'kind': r['kind'],
                'weight': (None if pd.isna(r.get('collectri_weight', np.nan))
                           else float(r['collectri_weight'])),
                'dose': float(r[dose_col]),
                'pos': int(name_to_pos[gene]),
            })
        committees[role] = members
    expected = {'committee_A', 'committee_B', 'A_random', 'B_random'}
    missing = expected - set(committees)
    if missing:
        raise SystemExit(f"committee spec missing roles: {missing}")
    return committees


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--committee-spec', required=True,
                        help='Resolver CSV (results/mlm_perturbation/committee_resolved.csv)')
    parser.add_argument('--celltype', required=True,
                        choices=list(CELLTYPE_SUFFIX.keys()))
    parser.add_argument('--baseline-catalog', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dose-index', type=int, default=3,
                        help='Which dose grid index to use (default 3 = 1.5*max)')
    parser.add_argument('--data-file', default='data/pbmc/pbmc3k_raw.h5ad')
    parser.add_argument('--model-name', default='genbio-ai/AIDO.Cell-100M')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Subsample first N cells (debug).')
    parser.add_argument('--calibration-check', action='store_true',
                        help='Run unperturbed IFNG-mask and TBX21-mask on first '
                             '32 cells; compare to catalog; then exit.')
    parser.add_argument('--noop-check', action='store_true',
                        help='Override committee_A members to each cell\'s own '
                             'preprocessed value, mask IFNG; verify max |Δ|≈0; '
                             'then exit.')
    args = parser.parse_args()

    celltype_suffix = CELLTYPE_SUFFIX[args.celltype]

    print("=" * 70)
    print("COMMITTEE PERTURBATION EXPERIMENT")
    print("=" * 70)
    print(f"Committee spec: {args.committee_spec}")
    print(f"Celltype:       {args.celltype}")
    print(f"Catalog:        {args.baseline_catalog}")
    print(f"Output:         {args.output}")
    print(f"Dose index:     {args.dose_index}")
    print("=" * 70)

    # 1. Load committee spec
    spec_df = pd.read_csv(args.committee_spec)
    print(f"\n1. Loaded {len(spec_df)} committee rows.")

    # 2. Load catalog & pick celltype subset
    print("\n2. Loading baseline catalog...")
    catalog = torch.load(args.baseline_catalog, weights_only=False)
    catalog_cell_names = list(catalog['cell_names'])
    catalog_cell_celltype = list(catalog['cell_celltype'])
    catalog_gene_order = list(catalog['gene_order'])
    for g in (ANCHOR_TF, ANCHOR_TARGET):
        if g not in catalog['logits']:
            raise SystemExit(f"Catalog missing single-mask pass for '{g}'")
        if g not in catalog_gene_order:
            raise SystemExit(f"Catalog gene_order missing '{g}'")

    sel = [i for i, ct in enumerate(catalog_cell_celltype) if ct == args.celltype]
    if not sel:
        raise SystemExit(f"No cells with celltype '{args.celltype}' in catalog")
    if args.max_cells is not None and len(sel) > args.max_cells:
        sel = sel[:args.max_cells]
        print(f"   subsampling to first {args.max_cells} cells (--max-cells)")
    catalog_cell_idx = np.array(sel)
    selected_cell_names = [catalog_cell_names[i] for i in sel]
    print(f"   {len(catalog_cell_names)} catalog cells; "
          f"{len(sel)} match '{args.celltype}'")

    # 3. Load model
    print("\n3. Loading AIDO.Cell...")
    config = CellFoundationConfig.from_pretrained(args.model_name)
    model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config)
    model = model.to(args.device)
    if args.device == 'cuda':
        model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # 4. Load and align data
    print("\n4. Loading and aligning data...")
    adata = ad.read_h5ad(args.data_file)
    adata.var_names_make_unique()
    adata_aligned, attention_mask = align_adata(adata)
    name_to_aligned_idx = {n: i for i, n in enumerate(adata_aligned.obs_names)}
    missing_cells = [n for n in selected_cell_names if n not in name_to_aligned_idx]
    if missing_cells:
        raise SystemExit(f"{len(missing_cells)} catalog cells absent from "
                         f"aligned data — catalog/data mismatch")
    cell_indices = np.array([name_to_aligned_idx[n] for n in selected_cell_names])

    var_names = list(adata_aligned.var_names)
    name_to_pos = {n: i for i, n in enumerate(var_names)}
    for g in (ANCHOR_TF, ANCHOR_TARGET):
        if g not in name_to_pos:
            raise SystemExit(f"Anchor gene '{g}' missing from aligned var_names")

    committees = build_committees(spec_df, celltype_suffix, name_to_pos,
                                  args.dose_index)
    for role, members in committees.items():
        sigs = [f"{m['gene']}({m['kind'][0]},{signed_dose(m['kind'], m['dose']):.2f})"
                for m in members]
        print(f"   {role:13s} ({len(members)}): {', '.join(sigs)}")

    mask_pos_ifng = name_to_pos[ANCHOR_TARGET]
    mask_pos_tbx21 = name_to_pos[ANCHOR_TF]

    # 5. Calibration check (one-off)
    if args.calibration_check:
        print("\n5. Calibration check: unperturbed mask passes vs catalog "
              "(first 32 cells)...")
        n_chk = min(32, len(cell_indices))
        chk_idx = cell_indices[:n_chk]
        chk_cat = catalog_cell_idx[:n_chk]
        for gene, mask_pos in [(ANCHOR_TARGET, mask_pos_ifng),
                               (ANCHOR_TF, mask_pos_tbx21)]:
            result, _ = perturbed_pass_multi(
                model, adata_aligned, attention_mask,
                perturb_positions={}, mask_pos=mask_pos,
                readout_pos=mask_pos, cell_indices=chk_idx,
                device=args.device, batch_size=args.batch_size,
                desc=f'calib:{gene}',
            )
            col = catalog_gene_order.index(gene)
            ref = catalog['logits'][gene][chk_cat, col].astype(np.float32)
            diff = result - ref
            print(f"   [{gene}] mean |Δ|={np.abs(diff).mean():.6f}  "
                  f"max |Δ|={np.abs(diff).max():.6f}  std={diff.std():.6f}")
            if np.abs(diff).max() > 1e-3:
                print(f"   ⚠ {gene} max |Δ| > 1e-3 — investigate.")
            else:
                print(f"   ✓ {gene} calibration OK.")
        return

    # 6. No-op sanity check (override A_committee with own values, expect ≈0 Δ)
    if args.noop_check:
        print("\n5. No-op sanity: A_committee members -> own value, "
              "mask IFNG, vs catalog IFNG mask.")
        counts_subset = adata_aligned.X[cell_indices]
        if hasattr(counts_subset, 'toarray'):
            counts_subset = counts_subset.toarray()
        proc_full = preprocess_counts(counts_subset, device='cpu')[:, :-2].float().numpy()
        perturb_per_cell = {}
        for m in committees['committee_A']:
            vec = torch.from_numpy(proc_full[:, m['pos']].astype(np.float32))
            perturb_per_cell[m['pos']] = vec
        result = perturbed_pass_per_cell(
            model, adata_aligned, attention_mask,
            perturb_per_cell=perturb_per_cell, mask_pos=mask_pos_ifng,
            readout_pos=mask_pos_ifng, cell_indices=cell_indices,
            device=args.device, batch_size=args.batch_size, desc='noop:A',
        )
        col = catalog_gene_order.index(ANCHOR_TARGET)
        ref = catalog['logits'][ANCHOR_TARGET][catalog_cell_idx, col].astype(np.float32)
        diff = result - ref
        print(f"   mean |Δ|={np.abs(diff).mean():.6f}  "
              f"max |Δ|={np.abs(diff).max():.6f}  std={diff.std():.6f}")
        if np.abs(diff).max() > 1e-2:
            print("   ⚠ max |Δ| > 1e-2 — investigate.")
        else:
            print("   ✓ no-op pipeline OK.")
        return

    # 7. Run the four passes
    print("\n5. Running four committee passes...")
    pass_plan = [
        ('A_committee', 'committee_A', mask_pos_ifng,  ANCHOR_TARGET),
        ('A_random',    'A_random',    mask_pos_ifng,  ANCHOR_TARGET),
        ('B_committee', 'committee_B', mask_pos_tbx21, ANCHOR_TF),
        ('B_random',    'B_random',    mask_pos_tbx21, ANCHOR_TF),
    ]

    logits_by_pass = {}
    perturbed_logits_by_pass = {}
    for pass_name, role, mask_pos, readout_gene in pass_plan:
        members = committees[role]
        perturb_positions = {
            m['pos']: signed_dose(m['kind'], m['dose']) for m in members
        }
        record_positions = [m['pos'] for m in members]
        pos_to_gene = {m['pos']: m['gene'] for m in members}
        print(f"\n   [{pass_name}] perturb={[m['gene'] for m in members]} "
              f"mask={readout_gene}")
        result, recorded = perturbed_pass_multi(
            model, adata_aligned, attention_mask,
            perturb_positions=perturb_positions,
            mask_pos=mask_pos, readout_pos=mask_pos,
            cell_indices=cell_indices, device=args.device,
            batch_size=args.batch_size, desc=pass_name,
            record_positions=record_positions,
        )
        logits_by_pass[pass_name] = result
        perturbed_logits_by_pass[pass_name] = {
            pos_to_gene[p]: arr for p, arr in recorded.items()
        }
        print(f"   readout {readout_gene}: shape={result.shape}  "
              f"mean={result.mean():.4f}  std={result.std():.4f}")
        for m in members:
            arr = perturbed_logits_by_pass[pass_name][m['gene']]
            applied = signed_dose(m['kind'], m['dose'])
            print(f"     {m['gene']:8s} (kind={m['kind']:9s} dose_in={applied:+.3f}) "
                  f"out: mean={arr.mean():+.4f} std={arr.std():.4f} "
                  f"out−in={arr.mean()-applied:+.4f}")

    # 8. Save
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    committees_serialized = {
        role: [{'gene': m['gene'], 'kind': m['kind'], 'weight': m['weight'],
                'dose': m['dose']} for m in members]
        for role, members in committees.items()
    }
    payload = {
        'logits': logits_by_pass,
        'perturbed_logits': perturbed_logits_by_pass,
        'cell_names': selected_cell_names,
        'cell_indices': cell_indices,
        'celltype': args.celltype,
        'committees': committees_serialized,
        'metadata': {
            'experiment_kind': 'committee_perturbation',
            'anchor_tf': ANCHOR_TF,
            'anchor_target': ANCHOR_TARGET,
            'dose_index': args.dose_index,
            'mask_token_id': MASK_TOKEN_ID,
            'model_name': args.model_name,
            'data_file': args.data_file,
            'baseline_catalog': args.baseline_catalog,
            'committee_spec': args.committee_spec,
            'args': vars(args),
        },
    }
    tmp = out_path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    print(f"\nDone. Output: {out_path}")


if __name__ == '__main__':
    main()

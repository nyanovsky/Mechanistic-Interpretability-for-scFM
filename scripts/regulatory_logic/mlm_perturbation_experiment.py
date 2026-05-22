"""MLM perturbation runner: input-level perturbation + masking.

For a TF and target list:
  - forward (TF -> target): perturb TF input value, mask target individually,
    read target logit. One pass per (target, dose).
  - reverse (target -> TF): perturb each target's input value, mask TF, read
    TF logit. One pass per (perturb_gene, dose) for perturb_gene in
    targets ∪ random_pool.

Per-gene per-celltype 5-dose grid is computed inline in preprocess_counts
(model input) space:
    {0, q75(nonzero), max, 1.5*max, min(3*max, 18)}.
The masked-baseline catalog (build_masked_baselines.py output) provides the
F/R reference logits — single-mask pass at the readout gene.

Resumes from a pre-existing --output file at per-(perturb_gene, dose)
granularity.

Usage:
    python mlm_perturbation_experiment.py \\
        --tf TBX21 \\
        --targets CD8A KLRG1 CCL4 SOCS3 CCL3 IL2RB IFNG TNF \\
        --celltype "CD4 T cells" \\
        --direction forward \\
        --baseline-catalog results/masked_baselines/tbx21_cd4_cd8.pt \\
        --output results/mlm_perturbation/tbx21_cd4_forward.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import anndata as ad
import torch
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ModelGenerator', 'huggingface', 'aido.cell'))

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts


MASK_TOKEN_ID = -1.0
DOSE_HARD_CAP = 18.0  # just under AIDO.Cell's 20-clip pretraining envelope


def compute_dose_grid(input_values):
    """5-point dose grid in preprocess_counts (log1p CPM) space.

    Args:
        input_values: (n_cells,) np.ndarray of preprocessed values at one gene.

    Returns:
        sorted list of unique doses (rounded to 6 decimals).
    """
    nz = input_values[input_values > 0]
    q75 = float(np.quantile(nz, 0.75)) if len(nz) > 0 else 0.0
    mx = float(input_values.max())
    raw = [0.0, q75, mx, 1.5 * mx, min(3 * mx, DOSE_HARD_CAP)]
    return sorted({round(d, 6) for d in raw})


def perturbed_pass(model, adata_aligned, attention_mask, perturb_pos, dose,
                   readout_pos, mask_pos, cell_indices, device, batch_size,
                   desc=""):
    """One perturbed forward pass.

    Set perturb_pos -> dose, set mask_pos -> MASK, read logit at readout_pos.

    Args:
        perturb_pos: int.
        dose: float OR torch tensor of shape (n_cells,) for per-cell doses.
        readout_pos: int.
        mask_pos: int.
        cell_indices: np.ndarray of int — rows of adata_aligned.

    Returns:
        np.ndarray (len(cell_indices),) of float32 logits at readout_pos.
    """
    n_cells = len(cell_indices)
    n_genes_full = adata_aligned.n_vars
    out = np.zeros(n_cells, dtype=np.float32)
    attn_full = torch.from_numpy(attention_mask).to(device)
    per_cell = torch.is_tensor(dose) and dose.dim() == 1
    if per_cell:
        dose = dose.to(device)

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

            if per_cell:
                input_ids[:, perturb_pos] = dose[start:end].to(input_ids.dtype)
            else:
                d = torch.tensor(float(dose), dtype=input_ids.dtype, device=device)
                input_ids[:, perturb_pos] = d

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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--tf', required=True)
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--random-pool', nargs='*',
                        default=['KPNA2', 'OASL', 'HMGA1'],
                        help='Random-pool genes (reverse direction only). '
                             'Pass with no values to disable.')
    parser.add_argument('--celltype', required=True,
                        help='e.g. "CD4 T cells" or "CD8 T cells"')
    parser.add_argument('--direction', choices=['forward', 'reverse'],
                        required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-catalog', required=True,
                        help='Path to masked-baseline .pt from build_masked_baselines.py')
    parser.add_argument('--data-file', default='data/pbmc/pbmc3k_raw.h5ad')
    parser.add_argument('--processed-file', default='data/pbmc/pbmc3k_processed.h5ad')
    parser.add_argument('--model-name', default='genbio-ai/AIDO.Cell-100M')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--noop-check', action='store_true',
                        help='Run TBX21 own-value no-op vs catalog IFNG mask, then exit')
    parser.add_argument('--doses-override', type=str, default=None,
                        help='JSON {gene: [doses]} to override computed grid')
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Subsample to first N cells (debug / sanity check only)')
    args = parser.parse_args()

    print("=" * 70)
    print("MLM PERTURBATION EXPERIMENT")
    print("=" * 70)
    print(f"TF:        {args.tf}")
    print(f"Targets:   {args.targets}")
    if args.direction == 'reverse' and args.random_pool:
        print(f"Random:    {args.random_pool}")
    print(f"Celltype:  {args.celltype}")
    print(f"Direction: {args.direction}")
    print(f"Catalog:   {args.baseline_catalog}")
    print(f"Output:    {args.output}")
    print("=" * 70)

    # 1. Load catalog & validate
    print("\n1. Loading masked-baseline catalog...")
    catalog = torch.load(args.baseline_catalog, weights_only=False)
    cmeta = catalog.get('metadata', {})
    if cmeta.get('tf') != args.tf:
        raise SystemExit(f"Catalog TF '{cmeta.get('tf')}' != --tf '{args.tf}'")
    catalog_targets = list(cmeta.get('targets', []))
    missing = [g for g in args.targets if g not in catalog_targets]
    if missing:
        raise SystemExit(f"Targets missing from catalog: {missing}")
    catalog_gene_order = list(catalog['gene_order'])
    catalog_cell_names = list(catalog['cell_names'])
    catalog_cell_celltype = list(catalog['cell_celltype'])

    sel = [i for i, ct in enumerate(catalog_cell_celltype) if ct == args.celltype]
    if not sel:
        raise SystemExit(f"No cells with celltype '{args.celltype}' in catalog")
    if args.max_cells is not None and len(sel) > args.max_cells:
        sel = sel[:args.max_cells]
        print(f"   subsampling to first {args.max_cells} cells (--max-cells)")
    catalog_cell_idx = np.array(sel)
    selected_cell_names = [catalog_cell_names[i] for i in sel]
    print(f"   {len(catalog_cell_names)} cells in catalog; "
          f"{len(sel)} match '{args.celltype}'")

    # Slice baseline (single-mask) logits per readout gene
    baseline_logits = {}
    for g in [args.tf] + list(args.targets):
        if g not in catalog['logits']:
            raise SystemExit(f"Catalog missing single-mask pass for '{g}'")
        col = catalog_gene_order.index(g)
        baseline_logits[g] = catalog['logits'][g][catalog_cell_idx, col].astype(np.float32)

    # 2. Model
    print("\n2. Loading AIDO.Cell...")
    config = CellFoundationConfig.from_pretrained(args.model_name)
    model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config)
    model = model.to(args.device)
    if args.device == 'cuda':
        model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # 3. Data + cell indices
    print("\n3. Loading and aligning data...")
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

    # Determine which genes need positions
    if args.direction == 'forward':
        perturb_genes = [args.tf]
        readout_genes = list(args.targets)
    else:
        perturb_genes = list(args.targets) + list(args.random_pool)
        readout_genes = [args.tf]
    needed = set(perturb_genes) | set(readout_genes) | {args.tf}
    missing_g = [g for g in needed if g not in name_to_pos]
    if missing_g:
        raise SystemExit(f"Genes missing from aligned var_names: {missing_g}")

    # 4. Dose grids
    print("\n4. Computing per-gene dose grids on celltype subset...")
    counts_subset = adata_aligned.X[cell_indices]
    if hasattr(counts_subset, 'toarray'):
        counts_subset = counts_subset.toarray()
    proc_full = preprocess_counts(counts_subset, device='cpu')[:, :-2].float().numpy()

    user_doses = {}
    if args.doses_override:
        with open(args.doses_override) as f:
            user_doses = {k: [float(x) for x in v] for k, v in json.load(f).items()}

    doses_per_gene = {}
    for g in set(perturb_genes):
        if g in user_doses:
            doses_per_gene[g] = sorted({round(d, 6) for d in user_doses[g]})
        else:
            doses_per_gene[g] = compute_dose_grid(proc_full[:, name_to_pos[g]])
        print(f"   {g:8s} doses = {doses_per_gene[g]}")

    # 5. No-op sanity
    if args.noop_check:
        print(f"\n5. No-op sanity: {args.tf} -> own value, mask IFNG, "
              f"compare to catalog IFNG mask...")
        if 'IFNG' not in name_to_pos or 'IFNG' not in catalog['logits']:
            raise SystemExit("--noop-check requires IFNG present in data and catalog")
        per_cell_dose = torch.from_numpy(
            proc_full[:, name_to_pos[args.tf]].astype(np.float32)
        )
        result = perturbed_pass(
            model, adata_aligned, attention_mask,
            perturb_pos=name_to_pos[args.tf], dose=per_cell_dose,
            readout_pos=name_to_pos['IFNG'], mask_pos=name_to_pos['IFNG'],
            cell_indices=cell_indices, device=args.device,
            batch_size=args.batch_size, desc='noop',
        )
        col = catalog_gene_order.index('IFNG')
        ref = catalog['logits']['IFNG'][catalog_cell_idx, col].astype(np.float32)
        diff = result - ref
        print(f"   mean |Δ|: {np.abs(diff).mean():.6f}")
        print(f"   max  |Δ|: {np.abs(diff).max():.6f}")
        print(f"   std    Δ: {diff.std():.6f}")
        if np.abs(diff).max() > 1e-2:
            print("   ⚠ max |Δ| > 1e-2 — investigate before running experiment.")
        else:
            print("   ✓ pipeline OK (max |Δ| < 1e-2).")
        return

    # 6. Resume support
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"\n5. Resuming from existing file: {out_path}")
        payload = torch.load(out_path, weights_only=False)
        meta = payload.get('metadata', {})
        if (meta.get('tf') != args.tf
                or list(meta.get('targets', [])) != list(args.targets)
                or meta.get('celltype') != args.celltype
                or list(payload.get('cell_names', [])) != selected_cell_names):
            raise SystemExit(
                "Existing output disagrees on TF/targets/celltype/cells. "
                "Pick a new --output path."
            )
    else:
        print(f"\n5. Starting fresh. Output: {out_path}")
        payload = {
            'conditions': {
                'tf_steered': {},
                'target_steered': {},
                'random_steered': {},
            },
            'baseline': {
                'logits': baseline_logits,
                'source_catalog': args.baseline_catalog,
                'baseline_kind': 'single_mask',
            },
            'cell_names': selected_cell_names,
            'cell_celltype': [args.celltype] * len(selected_cell_names),
            'cell_indices': cell_indices.tolist(),
            'metadata': {
                'experiment_kind': 'mlm_perturbation',
                'tf': args.tf,
                'targets': list(args.targets),
                'random_pool': list(args.random_pool),
                'celltype': args.celltype,
                'direction': args.direction,
                'doses_per_gene': {},
                'mask_token_id': MASK_TOKEN_ID,
                'model_name': args.model_name,
                'data_file': args.data_file,
                'processed_file': args.processed_file,
                'baseline_catalog': args.baseline_catalog,
                'args': vars(args),
            },
        }

    payload['conditions'].setdefault('tf_steered', {})
    payload['conditions'].setdefault('target_steered', {})
    payload['conditions'].setdefault('random_steered', {})
    payload['metadata'].setdefault('doses_per_gene', {})
    for g, ds in doses_per_gene.items():
        payload['metadata']['doses_per_gene'][g] = ds

    # Build pass plan
    plan = []
    if args.direction == 'forward':
        for target in args.targets:
            for dose in doses_per_gene[args.tf]:
                plan.append({
                    'bucket': 'tf_steered',
                    'key': target,
                    'dose': dose,
                    'perturb_gene': args.tf,
                    'readout_gene': target,
                    'mask_gene': target,
                })
    else:
        random_set = set(args.random_pool)
        for perturb_gene in list(args.targets) + list(args.random_pool):
            bucket = 'random_steered' if perturb_gene in random_set else 'target_steered'
            for dose in doses_per_gene[perturb_gene]:
                plan.append({
                    'bucket': bucket,
                    'key': perturb_gene,
                    'dose': dose,
                    'perturb_gene': perturb_gene,
                    'readout_gene': args.tf,
                    'mask_gene': args.tf,
                })

    def is_done(p):
        b = payload['conditions'][p['bucket']]
        return p['key'] in b and p['dose'] in b[p['key']]

    todo = [p for p in plan if not is_done(p)]
    print(f"\n6. {len(todo)}/{len(plan)} passes remaining "
          f"({len(cell_indices)} cells each)")

    for p in todo:
        desc = f"{p['bucket']}:{p['key']} dose={p['dose']}"
        print(f"\n   [{desc}]  perturb={p['perturb_gene']} mask={p['mask_gene']}")
        result = perturbed_pass(
            model, adata_aligned, attention_mask,
            perturb_pos=name_to_pos[p['perturb_gene']],
            dose=float(p['dose']),
            readout_pos=name_to_pos[p['readout_gene']],
            mask_pos=name_to_pos[p['mask_gene']],
            cell_indices=cell_indices, device=args.device,
            batch_size=args.batch_size, desc=desc,
        )
        bucket = payload['conditions'][p['bucket']]
        bucket.setdefault(p['key'], {})
        bucket[p['key']][p['dose']] = {
            'logits': torch.from_numpy(result.reshape(-1, 1)),
            'cell_indices': cell_indices,
            'dose': p['dose'],
            'readout_gene_order': [p['readout_gene']],
            'perturb_gene': p['perturb_gene'],
            'mask_gene': p['mask_gene'],
        }
        tmp = out_path + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, out_path)
        print(f"   saved -> {out_path}  (logits shape {result.shape})")

    n_done = sum(
        len(payload['conditions'][b][k])
        for b in payload['conditions']
        for k in payload['conditions'][b]
    )
    print(f"\nDone. Output: {out_path}")
    print(f"  total stored entries: {n_done}")


if __name__ == '__main__':
    main()

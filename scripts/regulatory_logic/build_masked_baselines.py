"""
Build masked-baseline logit catalog for input-perturbation experiments.

Given a TF and a list of target genes, runs N+1 forward passes of AIDO.Cell:
  - one pass per gene with that gene masked (single-mask)
  - one joint pass with all (TF + targets) masked together
Each pass records logits at all (TF + targets) positions for every cell in
the chosen celltype subset and writes results to disk before the next pass
starts. Re-running with the same --output skips passes already on disk.

Usage:
    python build_masked_baselines.py \
        --tf TBX21 \
        --targets CD8A KLRG1 CCL4 SOCS3 CDKN1B CCL3 IL2RB CXCR3 SOCS1 IFNG TNF \
        --celltypes "CD4 T cells" "CD8 T cells" \
        --output results/masked_baselines/tbx21_cd4_cd8.pt
"""

import os
import sys
import argparse
import torch
import anndata as ad
import numpy as np
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'ModelGenerator', 'huggingface', 'aido.cell'))

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts


MASK_TOKEN_ID = -1.0  # AIDO.Cell input-space sentinel for [MASK]


def run_masked_pass(model, adata_aligned, attention_mask, masked_positions,
                    readout_positions, cell_indices, device, batch_size, desc=""):
    """One masked forward pass; return logits at readout positions for each cell.

    Args:
        masked_positions: list[int] — positions in [0, n_genes_full) to set to MASK.
        readout_positions: list[int] — positions to extract logits from.
        cell_indices: np.ndarray of int — rows of adata_aligned to process.

    Returns:
        np.ndarray of shape (len(cell_indices), len(readout_positions)).
    """
    n_cells = len(cell_indices)
    n_genes_full = adata_aligned.n_vars
    n_readout = len(readout_positions)
    out = np.zeros((n_cells, n_readout), dtype=np.float32)

    attn_full = torch.from_numpy(attention_mask).to(device)
    masked_pos_tensor = torch.tensor(masked_positions, dtype=torch.long, device=device)
    readout_pos = np.array(readout_positions, dtype=np.int64)

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
            sentinel = torch.tensor(MASK_TOKEN_ID, dtype=input_ids.dtype, device=device)
            input_ids[:, masked_pos_tensor] = sentinel

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
    parser.add_argument('--tf', type=str, required=True,
                        help='TF gene symbol (e.g., TBX21)')
    parser.add_argument('--targets', type=str, nargs='+', required=True,
                        help='Target gene symbols')
    parser.add_argument('--celltypes', type=str, nargs='+',
                        default=['CD4 T cells', 'CD8 T cells'],
                        help='Cell types to include')
    parser.add_argument('--data-file', type=str, default='data/pbmc/pbmc3k_raw.h5ad')
    parser.add_argument('--processed-file', type=str,
                        default='data/pbmc/pbmc3k_processed.h5ad')
    parser.add_argument('--model-name', type=str, default='genbio-ai/AIDO.Cell-100M')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Subsample for debugging')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .pt path; resumes if file already exists')
    args = parser.parse_args()

    print("=" * 70)
    print("MASKED-BASELINE CATALOG")
    print("=" * 70)
    print(f"TF: {args.tf}")
    print(f"Targets ({len(args.targets)}): {args.targets}")
    print(f"Celltypes: {args.celltypes}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # 1. Model
    print("\n1. Loading AIDO.Cell...")
    config = CellFoundationConfig.from_pretrained(args.model_name)
    model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config)
    model = model.to(args.device)
    if args.device == 'cuda':
        model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # 2. Data + alignment
    print("\n2. Loading and aligning data...")
    adata = ad.read_h5ad(args.data_file)
    adata.var_names_make_unique()
    adata_aligned, attention_mask = align_adata(adata)
    print(f"   Aligned shape: {adata_aligned.shape}")

    # 3. Cell filter
    print("\n3. Filtering cells by celltype...")
    adata_p = ad.read_h5ad(args.processed_file)
    if 'celltype' in adata_p.obs.columns:
        cell_type_col = 'celltype'
    elif 'louvain' in adata_p.obs.columns:
        cell_type_col = 'louvain'
    else:
        raise ValueError("No celltype column found in processed file")
    common = [c for c in adata_aligned.obs_names if c in adata_p.obs_names]
    types = adata_p.obs.loc[common, cell_type_col]
    selected = types[types.isin(args.celltypes)].index.tolist()
    name_to_idx = {n: i for i, n in enumerate(adata_aligned.obs_names)}
    cell_indices = np.array([name_to_idx[n] for n in selected])
    if args.max_cells is not None and len(cell_indices) > args.max_cells:
        cell_indices = np.random.default_rng(0).choice(
            cell_indices, args.max_cells, replace=False
        )
    print(f"   {len(cell_indices)} cells selected")
    for ct in args.celltypes:
        n = (types.loc[selected] == ct).sum()
        print(f"     {ct}: {n}")

    # 4. Resolve gene positions
    print("\n4. Resolving gene positions...")
    var_names = list(adata_aligned.var_names)
    name_to_pos = {n: i for i, n in enumerate(var_names)}
    gene_order = [args.tf] + list(args.targets)
    missing = [g for g in gene_order if g not in name_to_pos]
    if missing:
        raise SystemExit(f"Genes not found in aligned var_names: {missing}")
    readout_positions = [name_to_pos[g] for g in gene_order]
    for g, p in zip(gene_order, readout_positions):
        active = bool(attention_mask[p])
        print(f"   {g:8s}  pos={p}  active={active}")

    cell_names = [adata_aligned.obs_names[i] for i in cell_indices]
    cell_celltype = [types.loc[adata_aligned.obs_names[i]] for i in cell_indices]

    # 5. Mask configs: one per gene + one joint
    mask_configs = [(g, [g]) for g in gene_order]
    mask_configs.append(('all_tf_and_targets', list(gene_order)))

    # 6. Resume support: load existing payload if present
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"\n6. Resuming from existing file: {out_path}")
        payload = torch.load(out_path, weights_only=False)
        existing = set(payload.get('logits', {}).keys())
        # Sanity-check that the cached run matches current invocation
        meta = payload.get('metadata', {})
        if (meta.get('tf') != args.tf or
                list(meta.get('targets', [])) != list(args.targets) or
                list(meta.get('celltypes', [])) != list(args.celltypes) or
                payload.get('cell_names') != cell_names):
            raise SystemExit(
                "Existing output has different TF/targets/celltypes/cells. "
                "Refusing to overwrite. Pick a new --output path."
            )
        print(f"   Already-completed configs: {sorted(existing)}")
    else:
        print(f"\n6. Starting fresh. Output: {out_path}")
        payload = {
            'logits': {},
            'gene_order': gene_order,
            'masked_in_pass': {n: g for n, g in mask_configs},
            'cell_names': cell_names,
            'cell_celltype': cell_celltype,
            'metadata': {
                'tf': args.tf,
                'targets': list(args.targets),
                'celltypes': args.celltypes,
                'data_file': args.data_file,
                'processed_file': args.processed_file,
                'model_name': args.model_name,
                'mask_token_id': MASK_TOKEN_ID,
                'args': vars(args),
            },
        }
        existing = set()

    # 7. Run remaining passes, saving after each
    print(f"\n7. Running {len(mask_configs) - len(existing)} of "
          f"{len(mask_configs)} passes ({len(cell_indices)} cells each)...")

    for name, masked_genes in mask_configs:
        if name in existing:
            print(f"\n   [{name}] SKIP (already on disk)")
            continue

        masked_pos = [name_to_pos[g] for g in masked_genes]
        print(f"\n   [{name}] masking {masked_genes}")

        result = run_masked_pass(
            model, adata_aligned, attention_mask,
            masked_pos, readout_positions, cell_indices,
            args.device, args.batch_size,
            desc=f"  {name}",
        )
        payload['logits'][name] = result

        # Atomic-ish write: write to .tmp then rename
        tmp_path = out_path + '.tmp'
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
        print(f"   saved {name} -> {out_path} "
              f"(shape {result.shape}, configs done: {len(payload['logits'])}/"
              f"{len(mask_configs)})")

    print(f"\nDone. Output: {out_path}")
    print(f"  configs: {list(payload['logits'].keys())}")


if __name__ == '__main__':
    main()

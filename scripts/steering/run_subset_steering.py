"""Orchestrator for subset / recruitment steering of a single SAE feature.

Loads AIDO.Cell + SAE + data ONCE, then sweeps many gene-token subsets by
swapping the steering token_mask (see SAESteeringModel.token_mask). For each
subset it runs the alpha sweep over CD4+CD8 cells in a single forward pass and
records logits for a restricted readout (module genes + a fixed random control
pool) plus per-celltype mean logits over all genes.

Subset families (--modes):
  singles  : each module gene steered alone           -> driver scan (incl. the
             decoupled members HOPX/CMC1/SSR2 as module-internal negative seeds)
  prefixes : order[:n] for n = 1..len(order)          -> recruitment-vs-size curve
  random   : matched-size random NON-module gene sets -> seed-specificity null

Controls computed later from this output:
  - random held-out readout : module-minus-S vs matched random control-pool genes
  - module-internal negatives: HOPX/CMC1/SSR2 singles vs core singles
  - random non-module seeds : the 'random' family above

Baseline = additive alpha=0 (exact no-op identity), computed once.

Example:
    python scripts/steering/run_subset_steering.py \\
        --feature 3079 \\
        --order NKG7 GZMH GZMB CST7 FGFBP2 PRF1 CCL4 CCL5 GZMA KLRD1 GNLY HOPX CMC1 SSR2 \\
        --alphas 2 -2 \\
        --celltypes "CD4 T cells" "CD8 T cells" \\
        --data-file data/pbmc/pbmc3k_raw.h5ad \\
        --out-file results/steering/feat3079_subset/sweep.pt
"""

import os
import sys
import argparse

import numpy as np
import torch
import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_sae
from utils.steering import SteeringExperiment
from utils.data_utils import get_expressed_genes_mask

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata

# Feature 3079 top-PR module (k=14). Override with --module-genes.
DEFAULT_MODULE = [
    "NKG7", "GNLY", "GZMB", "GZMH", "CST7", "PRF1", "CMC1",
    "GZMA", "KLRD1", "CCL5", "FGFBP2", "CCL4", "HOPX", "SSR2",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feature', type=int, required=True, help='SAE feature id to steer.')
    p.add_argument('--module-genes', nargs='+', default=DEFAULT_MODULE,
                   help='Full module gene set (default: feature 3079 top-PR 14).')
    p.add_argument('--order', nargs='+', default=None,
                   help='Gene ordering for the growing-prefix family (default: --module-genes order).')
    p.add_argument('--alphas', nargs='+', type=float, default=[2.0, -2.0],
                   help='Alpha values per subset (default: 2 -2).')
    p.add_argument('--modes', nargs='+', default=['singles', 'prefixes', 'random'],
                   choices=['singles', 'prefixes', 'random'])
    p.add_argument('--prefix-start', type=int, default=1,
                   help='Smallest prefix size to run (default 1). Use 2 to skip the '
                        'n=1 prefix when the matching single-gene steer already exists.')
    p.add_argument('--steering-mode', default='multiplicative',
                   choices=['multiplicative', 'additive'])
    p.add_argument('--mask-readout', nargs='+', default=None,
                   help='Masked (leave-one-out) variant: input-mask these probe '
                        'genes (set to mask_token_id=-1 after preprocess) on EVERY '
                        'pass, incl. the alpha=0 baseline -> within-masked contrast. '
                        'Probes must be disjoint from every steered set (use with '
                        '--modes prefixes random; the trio = module-minus-order is '
                        'never steered). Run once per probe-set config into its own '
                        '--out-file. See reports/pbmc/subset_steering_analysis_spec.md.')
    p.add_argument('--mask-whole-module', action='store_true',
                   help='Masked variant: input-mask the ENTIRE --module-genes set on '
                        'every pass instead of only --mask-readout. Steered genes are '
                        'then themselves masked (fine: token_mask edits the hidden '
                        'state, the input mask only removes the own-input anchor). '
                        '--mask-readout still defines the held-out probes that singles '
                        'skip and that the analysis targets, so singles steer '
                        'module-minus-readout (each masked). Needs --mask-readout; use '
                        'a fresh --out-file.')
    # random control / seed pools
    p.add_argument('--n-control', type=int, default=200,
                   help='Size of the fixed random non-module readout control pool.')
    p.add_argument('--random-sizes', nargs='+', type=int, default=[1, 2, 4, 8, 13],
                   help='Sizes for random non-module seed sets.')
    p.add_argument('--n-random', type=int, default=3, help='Random seed draws per size.')
    p.add_argument('--rng-seed', type=int, default=0)
    # model / data
    p.add_argument('--model-name', default='genbio-ai/AIDO.Cell-100M')
    p.add_argument('--data-file', required=True)
    p.add_argument('--processed-file', default='data/pbmc/pbmc3k_processed.h5ad')
    p.add_argument('--celltypes', nargs='+', default=["CD4 T cells", "CD8 T cells"])
    p.add_argument('--layer', type=int, default=12)
    p.add_argument('--sae-dir', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-cells', type=int, default=None)
    p.add_argument('--out-file', required=True)
    # split a long sweep across multiple runs (resumes into the same --out-file)
    p.add_argument('--nparts', type=int, default=1,
                   help='Split the unique gene-sets into this many contiguous parts.')
    p.add_argument('--part', type=int, default=1,
                   help='Which 1-indexed part to run this invocation (1..nparts).')
    return p.parse_args()


def build_subsets(module, order, modes, random_sizes, n_random, rng, seed_pool,
                  prefix_start=1, mask_set=None):
    """Return list of (subset_id, mode, [genes]).

    mask_set: probe genes that are input-masked on every pass (masked variant).
    Singles that would steer a masked probe are skipped (steered must stay
    unmasked); the remaining singles give the masked off-diagonal driver scan.
    Prefixes/random never contain the trio (= module-minus-order), so they are
    unaffected.
    """
    mask_set = set(mask_set or [])
    subsets = []
    if 'singles' in modes:
        for g in module:
            if g in mask_set:
                continue
            subsets.append((f"single__{g}", "single", [g]))
    if 'prefixes' in modes:
        for n in range(max(prefix_start, 1), len(order) + 1):
            subsets.append((f"prefix__{n:02d}", "prefix", list(order[:n])))
    if 'random' in modes:
        pool = np.asarray(seed_pool)
        for size in random_sizes:
            k = int(min(size, pool.size))
            for d in range(n_random):
                pick = sorted(rng.choice(pool, size=k, replace=False).tolist())
                subsets.append((f"random__n{size:02d}__d{d}", "random", pick))
    return subsets


def main():
    args = parse_args()
    order = args.order if args.order is not None else list(args.module_genes)
    rng = np.random.default_rng(args.rng_seed)

    if args.sae_dir is None:
        args.sae_dir = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}/sae_k_32_5120_1e-3lr"

    print("=" * 70)
    print(f"SUBSET STEERING  feature={args.feature}  layer={args.layer}")
    print(f"alphas={args.alphas}  mode={args.steering_mode}  modes={args.modes}")
    print("=" * 70)

    # --- model + SAE (once) ---
    print("\n1. Loading model + SAE ...")
    config = CellFoundationConfig.from_pretrained(args.model_name)
    model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config).to(args.device)
    if args.device == 'cuda':
        model = model.to(torch.bfloat16)
    for prm in model.parameters():
        prm.requires_grad = False
    model.eval()
    sae = load_sae(args.sae_dir, args.device)

    # --- data (once) ---
    print("\n2. Loading + aligning data ...")
    adata = ad.read_h5ad(args.data_file)
    adata_aligned, attention_mask = align_adata(adata)
    attn_bool = attention_mask.astype(bool)
    var_names = adata_aligned.var_names
    full_pos = {g: i for i, g in enumerate(var_names)}            # steering token space
    gene_names_filt = var_names[attn_bool].to_numpy()             # readout column space
    filt_col = {g: i for i, g in enumerate(gene_names_filt)}
    n_seq = adata_aligned.n_vars + 2                              # + 2 depth tokens

    # --- cells: CD4 + CD8 ---
    print("\n3. Selecting cells ...")
    obs = ad.read_h5ad(args.processed_file).obs
    ct_col = 'celltype' if 'celltype' in obs.columns else 'louvain'
    name_to_idx = {n: i for i, n in enumerate(adata_aligned.obs_names)}
    keep_names = [n for n in adata_aligned.obs_names
                  if n in obs.index and obs.loc[n, ct_col] in args.celltypes]
    cell_indices = np.array([name_to_idx[n] for n in keep_names])
    cell_labels = np.array([str(obs.loc[n, ct_col]) for n in keep_names])
    if args.max_cells is not None and args.max_cells < len(cell_indices):
        sel = rng.choice(len(cell_indices), args.max_cells, replace=False)
        cell_indices, cell_labels = cell_indices[sel], cell_labels[sel]
    for ct in args.celltypes:
        print(f"   {ct}: {(cell_labels == ct).sum()}")

    # --- readout columns: module + fixed random non-module control pool ---
    module = [g for g in args.module_genes if g in filt_col]
    expr_mask = get_expressed_genes_mask(args.data_file)          # filtered space
    expr_names = set(gene_names_filt[np.where(expr_mask)[0]].tolist())
    nonmodule_pool = sorted(expr_names - set(module))
    perm = rng.permutation(len(nonmodule_pool))
    control_pool = [nonmodule_pool[i] for i in perm[:args.n_control]]
    seed_pool = [nonmodule_pool[i] for i in perm[args.n_control:]]  # disjoint from control
    readout_genes = module + control_pool
    readout_cols = np.array([filt_col[g] for g in readout_genes])
    print(f"\n   module={len(module)}  control_pool={len(control_pool)}  "
          f"seed_pool={len(seed_pool)}  readout={len(readout_genes)}")

    # --- masked (leave-one-out) variant: input mask over the probe set ---
    # mask_readout = held-out probes (singles-skip + analysis target).
    # mask_input_genes = positions actually set to -1 in the input: the probes, or
    # the WHOLE module when --mask-whole-module (steered genes then masked too).
    input_mask = None
    mask_readout = None
    mask_input_genes = None
    if args.mask_readout is not None:
        mask_readout = list(args.mask_readout)
        missing = [g for g in mask_readout if g not in full_pos]
        if missing:
            raise ValueError(f"--mask-readout genes not in aligned var space: {missing}")
        not_read = [g for g in mask_readout if g not in filt_col]
        if not_read:
            print(f"   WARNING: masked probes not in readout columns: {not_read} "
                  f"(deltas for them won't be analyzable).")
        if args.mask_whole_module:
            mask_input_genes = [g for g in module if g in full_pos]
        else:
            mask_input_genes = list(mask_readout)
        if 'singles' in args.modes:
            overlap = [g for g in module if g in mask_readout]
            print(f"   singles family: skipping {overlap} (held-out probes); "
                  f"remaining singles give the masked off-diagonal driver scan"
                  + (" (steered gene itself masked)." if args.mask_whole_module else "."))
        input_mask = torch.zeros(n_seq, dtype=torch.bool, device=args.device)
        for g in mask_input_genes:
            input_mask[full_pos[g]] = True
        scope = f"WHOLE module ({len(mask_input_genes)} genes)" if args.mask_whole_module \
            else f"{len(mask_input_genes)} probe(s) {mask_input_genes}"
        print(f"   MASKED variant: input-masking {scope} on every pass "
              f"(incl. alpha=0 baseline); held-out probes = {mask_readout}.")

    # --- subsets (dedup identical gene-sets -> compute once) ---
    subsets = build_subsets(module, order, args.modes, args.random_sizes,
                            args.n_random, rng, seed_pool, args.prefix_start,
                            mask_set=mask_readout)
    key_of = {sid: "|".join(sorted(genes)) for sid, _, genes in subsets}
    unique_sets = {}
    for sid, _, genes in subsets:
        unique_sets.setdefault(key_of[sid], sorted(genes))
    print(f"   {len(subsets)} subset ids -> {len(unique_sets)} unique gene-sets "
          f"x {len(args.alphas)} alphas")

    exp = SteeringExperiment(model, sae, attention_mask, args.layer,
                             args.device, args.batch_size)

    def run_one(genes, alpha, steering_mode, token_mask):
        res = exp.run_steered(adata_aligned, [args.feature], alpha, cell_indices,
                              token_mask=token_mask, steering_mode=steering_mode,
                              input_mask=input_mask)
        logits = res['logits'].float().numpy()                    # (n_cells, n_filt)
        return (logits[:, readout_cols].astype(np.float32),
                {ct: logits[cell_labels == ct].mean(0).astype(np.float32)
                 for ct in args.celltypes})

    # --- resume or init ---
    if os.path.exists(args.out_file):
        store = torch.load(args.out_file, weights_only=False)
        print(f"\nResuming: {len(store['results'])} unique sets already done.")
        # meta is frozen from the first run; merge this run's subset defs + alphas
        # so the taxonomy isn't lost when new --modes are added on resume.
        meta = store['meta']
        # Guard: the masked baseline is tied to which positions are masked, so the
        # resumed store must share the same input mask AND held-out probes (or both
        # be unmasked). Old stores predate mask_input_genes -> fall back to readout.
        stored_input = meta.get('mask_input_genes', meta.get('mask_readout'))
        if stored_input != mask_input_genes or meta.get('mask_readout') != mask_readout:
            raise ValueError(
                f"mask mismatch on resume: store has input={stored_input} "
                f"readout={meta.get('mask_readout')}, this run has "
                f"input={mask_input_genes} readout={mask_readout}. The stored "
                f"baseline is masked for the store's config; use a separate "
                f"--out-file per masking config.")
        meta.setdefault('subsets', {})
        for sid, m_, g_ in subsets:
            meta['subsets'][sid] = {'mode': m_, 'genes': g_, 'key': key_of[sid]}
        meta['alphas'] = sorted(set(meta.get('alphas', [])) | set(args.alphas))
        torch.save(store, args.out_file)
    else:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        bdesc = "masked alpha=0 (within-masked contrast)" if mask_readout \
            else "additive alpha=0 no-op identity"
        print(f"\n4. Baseline ({bdesc}) ...")
        cap_pos = [full_pos[g] for g in module]
        b_res = exp.run_steered(adata_aligned, [args.feature], 0.0, cell_indices,
                                token_mask=None, steering_mode='additive',
                                input_mask=input_mask, capture_feature=args.feature,
                                capture_positions=cap_pos)
        b_logits = b_res['logits'].float().numpy()
        b_read = b_logits[:, readout_cols].astype(np.float32)
        b_means = {ct: b_logits[cell_labels == ct].mean(0).astype(np.float32)
                   for ct in args.celltypes}

        # Gate table: feature activation at each module-gene position, cached from
        # the baseline forward pass (no extra pass). Multiplicative steering scales
        # exactly this value -> ~0 means steering is a no-op at that position. With
        # --mask-whole-module the steered genes are masked, so this is the check for
        # whether multiplicative steering does anything at all.
        fa = b_res['feature_acts'].float().numpy()  # (n_cells, n_module)
        print(f"\n   Feature {args.feature} mean activation at module-gene positions "
              f"(under this run's input mask):")
        print("   " + f"{'gene':10s}" + "".join(f"{ct:>16s}" for ct in args.celltypes)
              + f"{'frac>0 (all)':>16s}")
        for j, g in enumerate(module):
            per_ct = "".join(f"{fa[cell_labels == ct, j].mean():>16.4f}"
                             for ct in args.celltypes)
            frac = float((fa[:, j] > 0).mean())
            print(f"   {g:10s}{per_ct}{frac:>16.3f}")
        print("   (multiplicative steering scales this; ~0 => no-op at that gene)\n")

        store = {
            'meta': {
                'feature': args.feature, 'module': module, 'order': order,
                'alphas': args.alphas, 'celltypes': args.celltypes,
                'steering_mode': args.steering_mode,
                'readout_genes': readout_genes, 'control_pool': control_pool,
                'mask_readout': mask_readout, 'mask_input_genes': mask_input_genes,
                'seed_pool_size': len(seed_pool), 'rng_seed': args.rng_seed,
                'gene_names_filt': gene_names_filt,
                'cell_labels': cell_labels, 'cell_indices': cell_indices,
                'subsets': {sid: {'mode': m, 'genes': g, 'key': key_of[sid]}
                            for sid, m, g in subsets},
            },
            'baseline': {'readout': b_read, 'allgene_means': b_means},
            'results': {},  # skey -> {alpha: {'readout', 'allgene_means'}}
        }
        torch.save(store, args.out_file)

    # --- sweep unique sets (optionally a contiguous part of them) ---
    if not 1 <= args.part <= args.nparts:
        raise ValueError(f"--part {args.part} must be in 1..{args.nparts}")
    all_items = list(unique_sets.items())
    bounds = np.linspace(0, len(all_items), args.nparts + 1).astype(int)
    lo, hi = bounds[args.part - 1], bounds[args.part]
    part_items = all_items[lo:hi]
    print(f"\n5. Sweeping subsets "
          f"(part {args.part}/{args.nparts}: sets {lo}..{hi - 1} of {len(all_items)}) ...")
    for ki, (skey, genes) in enumerate(part_items, lo + 1):
        done = store['results'].get(skey, {})
        if all(a in done for a in args.alphas):
            continue
        store['results'].setdefault(skey, {})
        for alpha in args.alphas:
            if alpha in store['results'][skey]:
                continue
            print(f"   [{ki}/{len(unique_sets)}] |S|={len(genes)} alpha={alpha} : "
                  f"{genes[:6]}" + (" ..." if len(genes) > 6 else ""))
            tmask = torch.zeros(n_seq, dtype=torch.bool, device=args.device)
            for g in genes:
                tmask[full_pos[g]] = True
            read, means = run_one(genes, alpha, args.steering_mode, tmask)
            store['results'][skey][alpha] = {'readout': read, 'allgene_means': means}
        torch.save(store, args.out_file)

    print(f"\nDone. Saved {args.out_file}")
    print(f"  unique sets: {len(store['results'])}  | baseline + meta stored")


if __name__ == '__main__':
    main()

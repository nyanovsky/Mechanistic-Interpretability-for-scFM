"""
Position-selective steering experiment for testing regulatory logic.

Tests whether AIDO.Cell encodes TF→target directionality by steering an SAE
feature at specific token positions (TF only vs target only) and measuring
asymmetric propagation. Ground-truth TF-target relationships come from
CollecTRI (via decoupleR).

Usage:
    python regulatory_logic_experiment.py \
        --tf TBX21 \
        --feature 3731 \
        --alphas 2 5 10 \
        --celltypes "CD8 T cells" \
        --data-file data/pbmc/pbmc3k_raw.h5ad \
        --output-dir results/regulatory_logic
"""

import os
import sys
import argparse
import torch
import anndata as ad
import numpy as np
import pandas as pd
import decoupler as dc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_sae, load_activation_matrices, compute_participation_ratio, get_expressed_genes_mask
from utils.similarity import get_top_k_genes_per_feature
from utils.steering import SteeringExperiment

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata


# --- TF-target characterization (Section 4 of plan) ---

def get_regulon(tf_name):
    """Pull TF regulon from CollecTRI."""
    net = dc.op.collectri()
    regulon = net[net['source'] == tf_name][['target', 'weight']].copy()
    regulon = regulon[regulon['target'] != tf_name]  # Exclude self-regulation
    return regulon, net


def get_feedback_tfs(tf_name, net):
    """Find targets that also regulate the input TF back (feedback loops)."""
    tf_targets = set(net[net['source'] == tf_name]['target'])
    tf_regulators = set(net[net['target'] == tf_name]['source'])
    return tf_targets & tf_regulators


def get_all_tfs(net):
    """Get set of all genes that act as TFs (appear as source in CollecTRI)."""
    return set(net['source'].unique())


def compute_feature_rank_per_gene(feature_id, gene_indices, feature_gene_matrix):
    """For each gene, compute the feature's rank among all features by activation strength.

    Returns dict of {gene_idx: rank} where rank 1 = strongest feature for that gene.
    """
    # Gene-centric view: for each gene, rank features by activation
    ranks = {}
    for gene_idx in gene_indices:
        gene_activations = feature_gene_matrix[:, gene_idx]
        sorted_features = np.argsort(gene_activations)[::-1]
        rank = np.where(sorted_features == feature_id)[0][0] + 1  # 1-indexed
        ranks[gene_idx] = rank
    return ranks


def characterize_intersection(tf_name, feature_id, feature_gene_matrix, gene_names,
                              expr_mask, pr_scale=1, min_genes=10, max_genes=100):
    """Compute TF-target ∩ feature gene set with feature rank info.

    Returns:
        DataFrame with columns: gene, in_regulon, regulon_weight, in_feature,
                                feature_rank_in_gene, is_tf, is_feedback_tf
        regulon: DataFrame of all TF targets from CollecTRI
        feature_gene_set: set of gene names in the feature
        control_genes: list of non-TF target genes in both regulon and feature (for control steering)
    """
    # 1. Pull regulon
    regulon, net = get_regulon(tf_name)
    regulon_targets = set(regulon['target'])
    regulon_weight = dict(zip(regulon['target'], regulon['weight']))

    # 2. Compute feature's gene set (over expressed genes only, then map the
    #    expressed-space indices back to full-gene indices for downstream use)
    expr_indices = np.where(expr_mask)[0]
    fg_expr = feature_gene_matrix[:, expr_mask]
    pr_values = compute_participation_ratio(fg_expr)
    feature_gene_sets = get_top_k_genes_per_feature(
        fg_expr, pr_values,
        pr_scale=pr_scale, min_genes=min_genes, max_genes=max_genes
    )
    feature_gene_indices = expr_indices[list(feature_gene_sets[feature_id])]
    feature_gene_names = set(gene_names[feature_gene_indices])

    # 3. Data-driven TF/feedback identification
    all_tfs = get_all_tfs(net)
    feedback_tfs = get_feedback_tfs(tf_name, net)

    # 4. Build intersection info
    # Consider all genes in regulon OR feature
    all_genes_of_interest = regulon_targets | feature_gene_names
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}

    rows = []
    for gene in sorted(all_genes_of_interest):
        if gene not in gene_name_to_idx:
            continue
        gene_idx = gene_name_to_idx[gene]

        # Check expressed
        if not expr_mask[gene_idx]:
            continue

        in_regulon = gene in regulon_targets
        in_feature = gene in feature_gene_names
        is_tf = gene in all_tfs
        is_feedback = gene in feedback_tfs

        # Feature rank (only meaningful if gene is in model)
        feature_rank = compute_feature_rank_per_gene(
            feature_id, [gene_idx], feature_gene_matrix
        )[gene_idx]

        rows.append({
            'gene': gene,
            'gene_idx': gene_idx,
            'in_regulon': in_regulon,
            'regulon_weight': regulon_weight.get(gene, None),
            'in_feature': in_feature,
            'feature_rank_in_gene': feature_rank,
            'is_tf': is_tf,
            'is_feedback_tf': is_feedback,
        })

    info_df = pd.DataFrame(rows)

    # Control genes: in regulon AND in feature, but not TFs
    control_genes = info_df[
        info_df['in_regulon'] & info_df['in_feature'] & ~info_df['is_tf']
    ]['gene'].tolist()

    print(f"\n{'='*70}")
    print(f"TF-TARGET CHARACTERIZATION: {tf_name} x Feature {feature_id}")
    print(f"{'='*70}")
    print(f"  CollecTRI regulon targets: {len(regulon_targets)}")
    print(f"  Feature gene set size: {len(feature_gene_names)}")
    print(f"  Intersection (regulon ∩ feature, expressed): "
          f"{len(info_df[info_df['in_regulon'] & info_df['in_feature']])}")
    print(f"  Feedback TFs excluded: {sorted(feedback_tfs & regulon_targets)}")
    print(f"  Control genes (non-TF, in regulon ∩ feature): {len(control_genes)}")
    if control_genes:
        print(f"    {control_genes}")

    # Print intersection detail
    intersection = info_df[info_df['in_regulon'] & info_df['in_feature']].copy()
    if len(intersection) > 0:
        print(f"\n  Intersection detail (sorted by feature rank):")
        intersection = intersection.sort_values('feature_rank_in_gene')
        for _, row in intersection.iterrows():
            tf_flag = " [TF]" if row['is_tf'] else ""
            fb_flag = " [FEEDBACK]" if row['is_feedback_tf'] else ""
            print(f"    {row['gene']:12s}  rank={row['feature_rank_in_gene']:4d}  "
                  f"weight={row['regulon_weight']:+.0f}{tf_flag}{fb_flag}")

    return info_df, regulon, feature_gene_names, control_genes


# --- Token mask construction ---

def build_token_mask(gene_positions, total_genes, device):
    """Build a boolean token mask for the full model sequence (genes + 2 depth tokens).

    Args:
        gene_positions: list/array of gene indices to steer (0-indexed into 19264 genes)
        total_genes: total number of genes in aligned data (19264)
        device: torch device

    Returns:
        Boolean tensor of shape (total_genes + 2,) with True at gene_positions
    """
    seq_len = total_genes + 2  # genes + depth tokens
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[gene_positions] = True
    return mask


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description='Position-selective steering for regulatory logic testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Experiment configuration
    parser.add_argument('--tf', type=str, required=True,
                        help='Transcription factor gene name (e.g., TBX21)')
    parser.add_argument('--feature', type=int, required=True,
                        help='SAE feature ID')
    parser.add_argument('--alphas', type=float, nargs='+', default=[2, 5, 10],
                        help='Alpha values (multiplicative) or k values (additive).')
    parser.add_argument('--steering-mode', type=str, default='multiplicative',
                        choices=['multiplicative', 'additive'],
                        help='Steering mode. multiplicative: features *= alpha (SAE '
                             'encode/decode + error). additive: h += alpha * decoder_row '
                             '(direct hidden-state edit, independent of natural feat_act).')
    parser.add_argument('--n-random', type=int, default=5,
                        help='Number of random control replicates (default: 5)')

    # Model and data paths
    parser.add_argument('--model-name', type=str, default='genbio-ai/AIDO.Cell-100M',
                        help='AIDO.Cell model name or path')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to input .h5ad file')
    parser.add_argument('--layer', type=int, default=12,
                        help='Layer to steer (0-indexed, default: 12)')
    parser.add_argument('--sae-dir', type=str, default=None,
                        help='Path to SAE directory (default: auto-detect from layer)')
    parser.add_argument('--interpretations-dir', type=str, default=None,
                        help='Path to interpretations dir with feature_gene_matrix.npy')

    # Cell type filtering
    parser.add_argument('--celltypes', type=str, nargs='+', default=None,
                        help='Cell types to include (default: all cells)')
    parser.add_argument('--processed-file', type=str,
                        default='data/pbmc/pbmc3k_processed.h5ad',
                        help='Processed .h5ad with celltype annotations')

    # Processing
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-cells', type=int, default=None)

    # Output
    parser.add_argument('--output-dir', type=str, default='results/regulatory_logic')

    args = parser.parse_args()

    # Default paths
    if args.sae_dir is None:
        BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
        args.sae_dir = f"{BASE_DIR}/sae_k_32_5120"

    if args.interpretations_dir is None:
        args.interpretations_dir = os.path.join(args.sae_dir, 'interpretations_mean_pooling')

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("REGULATORY LOGIC EXPERIMENT")
    print("=" * 70)
    print(f"TF: {args.tf}")
    print(f"Feature: {args.feature}")
    print(f"Alphas: {args.alphas}")
    print(f"Random controls: {args.n_random}")
    print(f"Cell types: {args.celltypes if args.celltypes else 'all'}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # 1. Load model
    print("\n1. Loading AIDO.Cell model...")
    config = CellFoundationConfig.from_pretrained(args.model_name)
    model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config)
    model = model.to(args.device)
    if args.device == "cuda":
        model = model.to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("  Model loaded")

    # 2. Load SAE
    print("\n2. Loading SAE...")
    sae = load_sae(args.sae_dir, args.device)
    print("  SAE loaded")

    # 3. Load and align data
    print("\n3. Loading and aligning data...")
    adata = ad.read_h5ad(args.data_file)
    adata_aligned, attention_mask = align_adata(adata)
    print(f"  Aligned: {adata_aligned.shape}")

    # Gene names for active genes (attention_mask == 1)
    active_mask = attention_mask.astype(bool)
    gene_names_all = adata_aligned.var_names.to_numpy()  # All 19264 genes
    gene_names = gene_names_all[active_mask]  # Active genes only
    gene_name_to_all_idx = {name: idx for idx, name in enumerate(gene_names_all)}

    # Expression mask (over active genes)
    expr_mask = get_expressed_genes_mask(args.data_file)

    # Cell type filtering
    cell_indices = None
    if args.celltypes is not None:
        print(f"\n  Filtering to cell types: {args.celltypes}")
        adata_processed = ad.read_h5ad(args.processed_file)
        if 'celltype' in adata_processed.obs.columns:
            cell_type_col = 'celltype'
        elif 'louvain' in adata_processed.obs.columns:
            cell_type_col = 'louvain'
        else:
            raise ValueError("No celltype column found in processed file")

        common_cells = [name for name in adata_aligned.obs_names if name in adata_processed.obs_names]
        processed_types = adata_processed.obs.loc[common_cells, cell_type_col]
        matching_cells = processed_types[processed_types.isin(args.celltypes)].index.tolist()
        name_to_idx = {name: i for i, name in enumerate(adata_aligned.obs_names)}
        cell_indices = np.array([name_to_idx[name] for name in matching_cells])
        print(f"  Found {len(cell_indices)} cells matching requested types")

    if args.max_cells is not None:
        if cell_indices is None:
            if args.max_cells < adata_aligned.n_obs:
                cell_indices = np.random.choice(adata_aligned.n_obs, args.max_cells, replace=False)
        elif args.max_cells < len(cell_indices):
            cell_indices = np.random.choice(cell_indices, args.max_cells, replace=False)

    # 4. Characterize TF-target intersection
    print("\n4. Characterizing TF-target ∩ feature gene set...")
    fg_matrix, _ = load_activation_matrices(args.interpretations_dir)

    # The feature-gene matrix is over active genes (matching gene_names)
    info_df, regulon, feature_gene_set, control_genes = characterize_intersection(
        args.tf, args.feature, fg_matrix, gene_names, expr_mask
    )

    # 5. Build token masks (one position per condition)
    print("\n5. Building token masks...")

    # TF position (in the full 19264 gene space)
    if args.tf not in gene_name_to_all_idx:
        raise ValueError(f"TF {args.tf} not found in model gene list")
    tf_all_idx = gene_name_to_all_idx[args.tf]
    print(f"  TF {args.tf}: position {tf_all_idx} in model gene order")

    total_genes = len(gene_names_all)
    tf_mask = build_token_mask([tf_all_idx], total_genes, args.device)

    # Target conditions: one per control gene (non-TF regulon ∩ feature)
    if not control_genes:
        print("  WARNING: No control genes found (regulon ∩ feature, non-TF). "
              "Cannot run paired directionality test.")
    target_masks = {}
    for g in control_genes:
        target_masks[g] = build_token_mask([gene_name_to_all_idx[g]], total_genes, args.device)
    print(f"  Target-steered conditions: {list(target_masks.keys())}")

    # Random pool: feature top-genes \ regulon targets \ regulators-of-(TF ∪ regulon),
    # restricted to expressed genes in this celltype.
    regulon_targets_set = set(regulon['target'])
    nodes_to_protect = regulon_targets_set | {args.tf}
    # Any CollecTRI source whose target is in nodes_to_protect is a "regulator"
    _, net = get_regulon(args.tf)
    regulators_to_exclude = set(net[net['target'].isin(nodes_to_protect)]['source'])

    # Gene name → active-gene index (for expr_mask lookup)
    gene_name_to_active_idx = {name: i for i, name in enumerate(gene_names)}

    feature_only = [g for g in feature_gene_set if g not in regulon_targets_set]
    random_candidates = [
        g for g in feature_only
        if g != args.tf
        and g not in regulators_to_exclude
        and g in gene_name_to_all_idx
        and g in gene_name_to_active_idx
        and expr_mask[gene_name_to_active_idx[g]]
    ]
    print(f"  Random pool construction:")
    print(f"    feature top-genes: {len(feature_gene_set)}")
    print(f"    after removing regulon targets: {len(feature_only)}")
    print(f"    after removing regulators + non-expressed: {len(random_candidates)}")
    print(f"    candidates: {sorted(random_candidates)}")

    n_random_actual = min(args.n_random, len(random_candidates))
    if n_random_actual < args.n_random:
        print(f"  NOTE: only {n_random_actual} random genes available (requested {args.n_random})")
    random_genes = sorted(np.random.choice(
        random_candidates, size=n_random_actual, replace=False
    ).tolist()) if n_random_actual > 0 else []

    random_masks = {}
    for g in random_genes:
        random_masks[g] = build_token_mask([gene_name_to_all_idx[g]], total_genes, args.device)
    print(f"  Random-steered conditions: {list(random_masks.keys())}")

    # 6. Run steering experiment
    print("\n6. Running steering conditions...")
    experiment = SteeringExperiment(
        model=model, sae=sae, attention_mask=attention_mask,
        layer_idx=args.layer, device=args.device, batch_size=args.batch_size
    )

    # Nested structure:
    #   conditions['tf_steered'][alpha] = result
    #   conditions['target_steered'][gene][alpha] = result
    #   conditions['random_steered'][gene][alpha] = result
    results = {
        'conditions': {
            'tf_steered': {},
            'target_steered': {g: {} for g in target_masks},
            'random_steered': {g: {} for g in random_masks},
        },
        'metadata': {},
    }
    feature_list = [args.feature]

    for alpha in args.alphas:
        print(f"\n--- Alpha = {alpha} ({args.steering_mode}) ---")

        # TF-steered
        print(f"\n  [TF-steered] Steering at {args.tf}")
        tf_result = experiment.run_steered(
            adata_aligned, feature_list, alpha,
            cell_indices=cell_indices, token_mask=tf_mask,
            steering_mode=args.steering_mode,
        )
        results['conditions']['tf_steered'][alpha] = tf_result

        # Per-target steered (one condition per control gene)
        for g, mask in target_masks.items():
            print(f"\n  [Target-steered: {g}] Steering at {g}")
            target_result = experiment.run_steered(
                adata_aligned, feature_list, alpha,
                cell_indices=cell_indices, token_mask=mask,
                steering_mode=args.steering_mode,
            )
            results['conditions']['target_steered'][g][alpha] = target_result

        # Per-random steered (one condition per random feature-member)
        for g, mask in random_masks.items():
            print(f"\n  [Random-steered: {g}] Steering at {g}")
            rand_result = experiment.run_steered(
                adata_aligned, feature_list, alpha,
                cell_indices=cell_indices, token_mask=mask,
                steering_mode=args.steering_mode,
            )
            results['conditions']['random_steered'][g][alpha] = rand_result


    # 7. Save metadata
    # Determine cell names for the experiment (same across all conditions)
    if cell_indices is not None:
        cell_names = adata_aligned.obs_names[cell_indices].tolist()
    else:
        cell_names = adata_aligned.obs_names.tolist()

    results['metadata'] = {
        'tf': args.tf,
        'feature': args.feature,
        'alphas': args.alphas,
        'steering_mode': args.steering_mode,
        'regulon_targets': regulon['target'].tolist(),
        'regulon_weights': dict(zip(regulon['target'], regulon['weight'])),
        'feature_gene_set': sorted(feature_gene_set),
        'control_genes': control_genes,        # per-target steered gene list
        'random_genes': random_genes,           # per-random steered gene list
        'regulators_excluded': sorted(regulators_to_exclude & set(feature_gene_set)),
        'intersection_info': info_df,
        'gene_names': gene_names.tolist(),
        'cell_names': cell_names,
        'args': vars(args),
    }

    # 8. Save results
    mode_suffix = '_additive' if args.steering_mode == 'additive' else ''
    output_file = os.path.join(
        args.output_dir,
        f"regulatory_logic_{args.tf}_feat{args.feature}_layer{args.layer}{mode_suffix}.pt"
    )
    print(f"\n7. Saving results to {output_file}...")
    torch.save(results, output_file)

    # Also save intersection info as CSV for easy inspection
    csv_file = os.path.join(
        args.output_dir,
        f"intersection_{args.tf}_feat{args.feature}.csv"
    )
    info_df.to_csv(csv_file, index=False)
    print(f"  Intersection info saved to {csv_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

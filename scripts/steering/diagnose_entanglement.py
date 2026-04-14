"""Diagnose feature entanglement in CD4->CD8 steering.

Checks whether steered SAE features encode both up-regulated and down-regulated
DEGs between CD4 and CD8, which would create conflicting steering effects.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import h5py
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.sae_model import load_sae
from utils.data_utils import (
    load_gene_names, get_expressed_genes_mask, compute_participation_ratio,
    compute_de_genes, compute_independent_de,
)
from utils.similarity import get_top_k_genes_per_feature, compute_gene_feature_sets


# =============================================================================
# Cell-type-specific feature-gene matrix
# =============================================================================

def compute_celltype_feature_gene_matrix(sae, h5_path, cell_indices, device, batch_size=8):
    """Compute feature-gene matrix using mean pooling over a subset of cells.

    Adapted from compute_feature_gene_activations_mean() in
    scripts/interpretation/compute_feature_matrices.py.

    Args:
        sae: The SAE model
        h5_path: Path to HDF5 file with activations [n_cells, n_genes, hidden_dim]
        cell_indices: Array of cell indices to include
        device: 'cuda' or 'cpu'
        batch_size: Number of cells to process at once

    Returns:
        [n_features, n_genes] matrix of mean activations across selected cells
    """
    cell_indices = np.sort(cell_indices)
    n_subset = len(cell_indices)

    with h5py.File(h5_path, 'r') as f:
        _, n_genes, hidden_dim = f['activations'].shape
        n_features = sae.latent_dim

        feature_gene_sum = np.zeros((n_features, n_genes), dtype=np.float32)

        n_batches = (n_subset + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(n_batches), desc="Processing cells"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_subset)
            batch_cell_ids = cell_indices[batch_start:batch_end]
            current_batch_size = len(batch_cell_ids)

            # Load selected cells (HDF5 fancy indexing)
            batch_activations = torch.from_numpy(
                f['activations'][batch_cell_ids, :, :].astype(np.float32)
            ).to(device)

            batch_flat = batch_activations.reshape(-1, hidden_dim)

            with torch.no_grad():
                sparse_latents_flat = sae.encode(batch_flat)

            sparse_latents = sparse_latents_flat.reshape(current_batch_size, n_genes, n_features)
            feature_gene_sum += sparse_latents.permute(0, 2, 1).sum(dim=0).cpu().numpy()

        feature_gene_mean = feature_gene_sum / n_subset

    return feature_gene_mean


def get_celltype_indices(h5_path, processed_path, celltype):
    """Get HDF5 cell indices for a given cell type.

    Maps cell_names in the HDF5 to obs_names in the processed h5ad,
    then filters by the celltype column.

    Returns:
        Array of integer indices into the HDF5 activations array.
    """
    adata = ad.read_h5ad(processed_path, backed='r')
    celltype_cells = set(adata.obs_names[adata.obs['louvain'] == celltype])
    print(f"  {celltype}: {len(celltype_cells)} cells in processed data")

    with h5py.File(h5_path, 'r') as f:
        h5_cell_names = [name.decode() if isinstance(name, bytes) else name
                         for name in f['cell_names'][:]]

    indices = [i for i, name in enumerate(h5_cell_names) if name in celltype_cells]
    print(f"  Matched {len(indices)} cells in HDF5")
    return np.array(indices)


# =============================================================================
# DEG computation
# =============================================================================

def compute_degs(baseline_source, baseline_target, steered_source,
                 gene_names, expr_mask):
    """Compute DEGs as intersection of celltype DE and steering DE.

    Filters to expressed genes, then:
    1. Celltype DE: independent t-test (target vs source baseline)
    2. Steering DE: paired t-test (steered vs baseline source)
    3. Intersection of both

    Args:
        baseline_source: [n_source, n_genes] baseline logits for source cells
        baseline_target: [n_target, n_genes] baseline logits for target cells
        steered_source: [n_source, n_genes] steered logits for source cells
        gene_names: Array of all gene names
        expr_mask: Boolean mask for expressed genes

    Returns:
        dict with gene_names, mean_diff, up_ranked, down_ranked
        Ranked indices are into the full gene space.
    """
    # Filter to expressed genes
    src_expr = baseline_source[:, expr_mask]
    tgt_expr = baseline_target[:, expr_mask]
    steered_expr = steered_source[:, expr_mask]
    names_expr = gene_names[expr_mask]
    expr_indices = np.where(expr_mask)[0]

    # Celltype DE (independent t-test): target - source
    _, _, ct_stats = compute_independent_de(tgt_expr, src_expr, names_expr)
    ct_sig = ct_stats['sig_mask']
    ct_diff = ct_stats['mean_diff']

    # Steering DE (paired t-test): steered - baseline source
    _, _, st_stats = compute_de_genes(steered_expr, src_expr, names_expr)
    st_sig = st_stats['sig_mask']

    # Intersection
    both_sig = ct_sig & st_sig

    print(f"  Celltype DE: {ct_sig.sum()}")
    print(f"  Steering DE: {st_sig.sum()}")
    print(f"  Intersection (celltype ∩ steering): {both_sig.sum()}")

    # Map back to full gene space, ranked by celltype effect size
    mean_diff_full = np.zeros(len(gene_names), dtype=np.float32)
    mean_diff_full[expr_indices] = ct_diff

    up_mask = both_sig & (ct_diff > 0)
    down_mask = both_sig & (ct_diff < 0)

    up_expr_idx = np.where(up_mask)[0]
    down_expr_idx = np.where(down_mask)[0]
    up_ranked = expr_indices[up_expr_idx[np.argsort(ct_diff[up_expr_idx])[::-1]]]
    down_ranked = expr_indices[down_expr_idx[np.argsort(ct_diff[down_expr_idx])]]

    print(f"  Intersection up: {len(up_ranked)}, down: {len(down_ranked)}")

    return {
        'gene_names': gene_names,
        'mean_diff': mean_diff_full,
        'up_ranked': up_ranked,
        'down_ranked': down_ranked,
        'mean_diff_by_gene': dict(zip(gene_names, mean_diff_full)),
    }


# =============================================================================
# Entanglement analysis
# =============================================================================

def compute_entanglement(fg_matrix, steered_features, alphas, gene_names,
                         deg_info, expr_mask, top_n_degs=None,
                         pr_scale=1, min_genes=10, max_genes=100):
    """Compute count-based entanglement metrics for steered features.

    Args:
        fg_matrix: [n_features, n_genes] feature-gene matrix (CD4-specific)
        steered_features: Array of feature IDs
        alphas: Corresponding alpha values
        gene_names: Array of gene names aligned with fg_matrix columns
        deg_info: Dict from compute_degs()
        expr_mask: Boolean mask for expressed genes
        top_n_degs: If set, only use the top N up-DEGs and top N down-DEGs
                    ranked by effect size. None uses all significant DEGs.
        pr_scale, min_genes, max_genes: PR-adaptive thresholding params

    Returns:
        DataFrame with entanglement metrics per feature
    """
    # Filter to expressed genes for PR and gene set computation
    fg_expr = fg_matrix[:, expr_mask]
    expr_gene_names = gene_names[expr_mask]

    # Compute PR for the steered features on expressed genes only
    steered_fg = fg_expr[steered_features]
    pr_values = compute_participation_ratio(steered_fg, axis=1)

    # Get gene sets per feature using PR-adaptive thresholding
    gene_sets = get_top_k_genes_per_feature(
        steered_fg, pr_values, pr_scale=pr_scale,
        min_genes=min_genes, max_genes=max_genes
    )

    up_ranked = deg_info['up_ranked']
    down_ranked = deg_info['down_ranked']

    if top_n_degs is not None:
        up_ranked = up_ranked[:top_n_degs]
        down_ranked = down_ranked[:top_n_degs]

    up_genes = set(gene_names[up_ranked])
    down_genes = set(gene_names[down_ranked])

    rows = []
    for i, (feat_id, alpha) in enumerate(zip(steered_features, alphas)):
        gene_indices = gene_sets[i]
        feat_gene_names = [expr_gene_names[g] for g in gene_indices]

        feat_up = [g for g in feat_gene_names if g in up_genes]
        feat_down = [g for g in feat_gene_names if g in down_genes]
        n_up = len(feat_up)
        n_down = len(feat_down)

        if max(n_up, n_down) > 0:
            entanglement_ratio = min(n_up, n_down) / max(n_up, n_down)
        else:
            entanglement_ratio = 0.0

        rows.append({
            'feature_id': feat_id,
            'alpha': alpha,
            'n_top_genes': len(gene_indices),
            'n_up_degs': n_up,
            'n_down_degs': n_down,
            'entanglement_ratio': entanglement_ratio,
            'de_fraction': (n_up + n_down) / len(gene_indices) if len(gene_indices) > 0 else 0.0,
            'top_up_genes': ','.join(sorted(feat_up)),
            'top_down_genes': ','.join(sorted(feat_down)),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Minimum feature cover
# =============================================================================

def compute_min_cover(fg_matrix, gene_names, deg_info, expr_mask,
                      annotated_mask=None, top_n_degs=50,
                      pr_scale=1, min_genes=10, max_genes=100):
    """Find minimum set of features whose top-PR genes cover the top N DEGs.

    Uses greedy set cover: repeatedly pick the feature covering the most
    uncovered DEGs until all are covered.

    Args:
        fg_matrix: [n_features, n_genes] feature-gene matrix
        gene_names: Array of gene names
        deg_info: Dict from compute_degs()
        expr_mask: Boolean mask for expressed genes
        top_n_degs: Number of top DEGs to cover
        pr_scale, min_genes, max_genes: PR-adaptive thresholding params

    Returns:
        List of (feature_id, n_new_covered, covered_genes) tuples in selection order
    """
    fg_expr = fg_matrix[:, expr_mask]
    expr_gene_names = gene_names[expr_mask]

    # Compute PR and gene sets for ALL features
    pr_values = compute_participation_ratio(fg_expr, axis=1)
    gene_sets = get_top_k_genes_per_feature(
        fg_expr, pr_values, pr_scale=pr_scale,
        min_genes=min_genes, max_genes=max_genes
    )

    # Build universe: top N DEGs (up + down combined, ranked by |logFC|)
    up_ranked = deg_info['up_ranked'][:top_n_degs]
    down_ranked = deg_info['down_ranked'][:top_n_degs]
    all_ranked = np.concatenate([up_ranked, down_ranked])
    mean_diff = deg_info['mean_diff']
    all_logfc = np.abs(mean_diff[all_ranked])
    order = np.argsort(all_logfc)[::-1]
    all_ranked_sorted = all_ranked[order][:top_n_degs]
    universe = set(gene_names[all_ranked_sorted])

    # Build per-feature DEG sets (restrict to annotated features if provided)
    candidate_features = np.where(annotated_mask)[0] if annotated_mask is not None else range(len(fg_expr))
    feature_deg_sets = {}
    for feat_id in candidate_features:
        feat_genes = set(expr_gene_names[g] for g in gene_sets[feat_id])
        overlap = feat_genes & universe
        if overlap:
            feature_deg_sets[feat_id] = overlap

    # Greedy set cover
    uncovered = set(universe)
    selected = []
    while uncovered:
        best_feat = max(feature_deg_sets, key=lambda f: len(feature_deg_sets[f] & uncovered),
                        default=None)
        if best_feat is None:
            break
        newly_covered = feature_deg_sets[best_feat] & uncovered
        if not newly_covered:
            break
        selected.append((best_feat, len(newly_covered), sorted(newly_covered)))
        uncovered -= newly_covered

    # Compute PRs for selected features
    selected_pr = {}
    for feat_id, _, _ in selected:
        selected_pr[feat_id] = pr_values[feat_id]

    return selected, universe, uncovered, selected_pr


# =============================================================================
# Gene-centric feature analysis
# =============================================================================

def compute_gene_centric_cover(fg_matrix, gene_names, deg_info, expr_mask,
                               annotated_mask=None, top_n_degs=50):
    """For each top DEG, find its top-PR features. Report per-gene and union.

    Args:
        fg_matrix: [n_features, n_genes] feature-gene matrix
        gene_names: Array of gene names
        deg_info: Dict from compute_degs()
        expr_mask: Boolean mask for expressed genes
        annotated_mask: Boolean mask for annotated features
        top_n_degs: Number of top DEGs to analyze

    Returns:
        gene_feature_map: dict gene -> list of (feature_id, activation)
        union_features: set of all features across all DEGs
    """
    # Get top N DEGs by |logFC|
    up_ranked = deg_info['up_ranked'][:top_n_degs]
    down_ranked = deg_info['down_ranked'][:top_n_degs]
    all_ranked = np.concatenate([up_ranked, down_ranked])
    mean_diff = deg_info['mean_diff']
    all_logfc = np.abs(mean_diff[all_ranked])
    order = np.argsort(all_logfc)[::-1]
    top_deg_indices = all_ranked[order][:top_n_degs]
    top_deg_names = gene_names[top_deg_indices]

    # Compute gene-centric feature sets
    feature_sets, gene_pr = compute_gene_feature_sets(fg_matrix)

    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    gene_feature_map = {}
    union_features = set()
    per_gene_feat_sets = {}

    for gene in top_deg_names:
        g_idx = gene_to_idx[gene]
        feat_set = set(feature_sets[g_idx])

        # Restrict to annotated features
        if annotated_mask is not None:
            feat_set = {f for f in feat_set if annotated_mask[f]}

        union_features.update(feat_set)
        per_gene_feat_sets[gene] = feat_set

        # Get top features with activation values
        activations = fg_matrix[:, g_idx]
        top_feats = sorted(
            [(f, activations[f]) for f in feat_set],
            key=lambda x: x[1], reverse=True
        )
        gene_feature_map[gene] = {
            'n_features': len(feat_set),
            'gene_pr': gene_pr[g_idx],
            'logfc': mean_diff[g_idx],
            'top_features': top_feats[:10],
        }

    return gene_feature_map, union_features, per_gene_feat_sets


# =============================================================================
# Plotting
# =============================================================================

def plot_entanglement_sweep(sweep_results, output_dir):
    """5 histograms of entanglement ratio, one per DEG cutoff."""
    n = len(sweep_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, sweep_results):
        values = df['entanglement_ratio']
        n_entangled = ((df['n_up_degs'] > 0) & (df['n_down_degs'] > 0)).sum()
        pct = 100 * n_entangled / len(df)

        ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(values.median(), color='red', linestyle='--',
                   label=f'Median: {values.median():.2f}')
        ax.set_xlabel('Entanglement ratio')
        ax.set_ylabel('Count')
        ax.set_title(f'{label}\n({pct:.0f}% entangled)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'entanglement_sweep.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_de_fraction(sweep_results, output_dir):
    """Histogram of DE fraction per feature, one per DEG cutoff."""
    n = len(sweep_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, sweep_results):
        has_degs = df[(df['n_up_degs'] > 0) | (df['n_down_degs'] > 0)]
        values = has_degs['de_fraction']
        n_active = len(has_degs)
        n_total = len(df)
        n_offensive = (values > 0.5).sum()
        n_mixed = ((values >= 0.1) & (values <= 0.5)).sum()
        n_defensive = (values < 0.1).sum()

        ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
        if len(values) > 0:
            ax.axvline(values.median(), color='red', linestyle='--',
                       label=f'Median: {values.median():.2f}')
        ax.set_xlabel('DE fraction (of top-PR genes)')
        ax.set_ylabel('Count')
        pct_off = 100 * n_offensive / max(n_active, 1)
        pct_mix = 100 * n_mixed / max(n_active, 1)
        pct_def = 100 * n_defensive / max(n_active, 1)
        pct_off_total = 100 * n_offensive / n_total
        pct_mix_total = 100 * n_mixed / n_total
        pct_def_total = 100 * n_defensive / n_total
        ax.set_title(
            f'{label} ({n_active}/{n_total} with DEG overlap)\n'
            f'Off: {pct_off:.0f}% ({pct_off_total:.0f}%) | '
            f'Mix: {pct_mix:.0f}% ({pct_mix_total:.0f}%) | '
            f'Def: {pct_def:.0f}% ({pct_def_total:.0f}%)',
            fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'de_fraction_sweep.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_deg_logfc_curve(deg_info, deg_cutoffs, output_dir):
    """Scatter of DEG |logFC| ranked from highest to lowest, with cutoff lines."""
    mean_diff = deg_info['mean_diff']
    up_ranked = deg_info['up_ranked']
    down_ranked = deg_info['down_ranked']

    # Combine up and down, sorted by |logFC| descending
    all_ranked = np.concatenate([up_ranked, down_ranked])
    all_logfc = np.abs(mean_diff[all_ranked])
    order = np.argsort(all_logfc)[::-1]
    all_logfc_sorted = all_logfc[order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(np.arange(len(all_logfc_sorted)), all_logfc_sorted,
               s=3, alpha=0.5, color='steelblue')

    for cutoff in deg_cutoffs:
        if cutoff is not None and cutoff <= len(all_logfc_sorted):
            x = cutoff - 1
            ax.axvline(x, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
            ax.text(x + 2, all_logfc_sorted[0] * 0.95, f'top {cutoff}',
                    fontsize=8, color='red', rotation=90, va='top')

    ax.set_xlabel('DEG rank')
    ax.set_ylabel('|logFC| (CD8 vs CD4)')
    ax.set_title('DEG effect sizes ranked by magnitude')

    plt.tight_layout()
    path = os.path.join(output_dir, 'deg_logfc_curve.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# Main
# =============================================================================

def print_summary(label, df):
    """Print entanglement summary for a single DEG cutoff."""
    entangled = df[(df['n_up_degs'] > 0) & (df['n_down_degs'] > 0)]
    clean_up = df[(df['n_up_degs'] > 0) & (df['n_down_degs'] == 0)]
    clean_down = df[(df['n_up_degs'] == 0) & (df['n_down_degs'] > 0)]
    no_degs = df[(df['n_up_degs'] == 0) & (df['n_down_degs'] == 0)]

    print(f"\n--- {label} ---")
    print(f"  Entangled (both up+down): {len(entangled)} ({100*len(entangled)/len(df):.1f}%)")
    print(f"  Clean up only:            {len(clean_up)} ({100*len(clean_up)/len(df):.1f}%)")
    print(f"  Clean down only:          {len(clean_down)} ({100*len(clean_down)/len(df):.1f}%)")
    print(f"  No DEGs:                  {len(no_degs)} ({100*len(no_degs)/len(df):.1f}%)")
    if len(entangled) > 0:
        print(f"  Entanglement ratio — mean: {entangled['entanglement_ratio'].mean():.3f}, "
              f"median: {entangled['entanglement_ratio'].median():.3f}")

    # Offensive vs defensive breakdown
    de_frac = df['de_fraction']
    has_degs = df[(df['n_up_degs'] > 0) | (df['n_down_degs'] > 0)]
    de_frac_active = has_degs['de_fraction']
    n_no_degs = len(df) - len(has_degs)
    n_defensive = (de_frac_active < 0.1).sum()
    n_mixed = ((de_frac_active >= 0.1) & (de_frac_active <= 0.5)).sum()
    n_offensive = (de_frac_active > 0.5).sum()
    n_active = len(has_degs)
    print(f"  No DEG overlap:            {n_no_degs} ({100*n_no_degs/len(df):.1f}%)")
    print(f"  Of {n_active} features with DEG overlap:")
    print(f"    Offensive (DE frac > 0.5): {n_offensive} ({100*n_offensive/n_active:.1f}%)")
    print(f"    Mixed (0.1–0.5):           {n_mixed} ({100*n_mixed/n_active:.1f}%)")
    print(f"    Defensive (DE frac < 0.1): {n_defensive} ({100*n_defensive/n_active:.1f}%)")
    print(f"    DE fraction — mean: {de_frac_active.mean():.3f}, median: {de_frac_active.median():.3f}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose feature entanglement in steering')
    parser.add_argument('--alpha_file', required=True, help='Path to alpha vector .pt file')
    parser.add_argument('--h5_path', required=True, help='Path to HDF5 activations file')
    parser.add_argument('--sae_dir', required=True, help='Path to SAE directory')
    parser.add_argument('--baseline_file', required=True, help='Path to baseline logits h5ad')
    parser.add_argument('--steering_data', required=True, help='Path to post_steer_embeddings.pt')
    parser.add_argument('--processed_file', required=True, help='Path to processed h5ad with cell types')
    parser.add_argument('--raw_data_file', required=True, help='Path to raw h5ad for expression mask')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--source_celltype', default='CD4 T cells')
    parser.add_argument('--target_celltype', default='CD8 T cells')
    parser.add_argument('--steered_threshold', type=float, default=0.05,
                        help='|alpha - 1| threshold for steered features')
    parser.add_argument('--device', default=None, help='cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load alpha vector ---
    print("1. Loading alpha vector...")
    alpha_data = torch.load(args.alpha_file, map_location='cpu')
    alpha_vector = alpha_data['alpha_vector'].numpy()
    steered_mask = np.abs(alpha_vector - 1.0) > args.steered_threshold
    steered_features = np.where(steered_mask)[0]
    steered_alphas = alpha_vector[steered_features]
    print(f"  Steered features (|alpha-1| > {args.steered_threshold}): {len(steered_features)}")

    # --- Load gene names ---
    print("\n2. Loading gene names...")
    gene_names = np.array(load_gene_names(raw_data_path=args.raw_data_file))
    print(f"  {len(gene_names)} genes")

    # --- Compute cell-type-specific feature-gene matrices ---
    print("\n3. Computing cell-type-specific feature-gene matrices...")

    sae = load_sae(args.sae_dir, device=device)

    cd4_cache = os.path.join(args.output_dir, 'fg_matrix_cd4.npy')
    cd8_cache = os.path.join(args.output_dir, 'fg_matrix_cd8.npy')

    if os.path.exists(cd4_cache) and os.path.exists(cd8_cache):
        print("  Loading cached matrices...")
        fg_cd4 = np.load(cd4_cache)
        fg_cd8 = np.load(cd8_cache)
    else:
        cd4_indices = get_celltype_indices(args.h5_path, args.processed_file, args.source_celltype)
        cd8_indices = get_celltype_indices(args.h5_path, args.processed_file, args.target_celltype)

        print(f"\n  Computing CD4 feature-gene matrix ({len(cd4_indices)} cells)...")
        fg_cd4 = compute_celltype_feature_gene_matrix(
            sae, args.h5_path, cd4_indices, device, batch_size=args.batch_size
        )

        print(f"\n  Computing CD8 feature-gene matrix ({len(cd8_indices)} cells)...")
        fg_cd8 = compute_celltype_feature_gene_matrix(
            sae, args.h5_path, cd8_indices, device, batch_size=args.batch_size
        )

        np.save(cd4_cache, fg_cd4)
        np.save(cd8_cache, fg_cd8)
        print(f"  Cached matrices to {args.output_dir}")

    print(f"  CD4 matrix shape: {fg_cd4.shape}")
    print(f"  CD8 matrix shape: {fg_cd8.shape}")

    # --- Load baseline and steering logits ---
    print("\n4. Loading baseline and steering logits...")
    baseline_adata = ad.read_h5ad(args.baseline_file)
    baseline_gene_names = np.array(baseline_adata.var_names)

    cached = torch.load(args.steering_data, map_location='cpu')
    post_logits_source = cached['post_logits_source']
    cell_types = cached['cell_types']
    cell_names = cached['cell_names']
    if isinstance(post_logits_source, torch.Tensor):
        post_logits_source = post_logits_source.float().numpy()

    # Identify source and target cells
    source_mask_cells = cell_types == args.source_celltype
    target_mask_cells = cell_types == args.target_celltype
    source_names = [cell_names[i] for i in range(len(cell_names)) if source_mask_cells[i]]
    target_names = [cell_names[i] for i in range(len(cell_names)) if target_mask_cells[i]]

    # Filter to cells present in baseline
    baseline_obs = set(baseline_adata.obs_names)
    source_in_baseline = [n for n in source_names if n in baseline_obs]
    target_in_baseline = [n for n in target_names if n in baseline_obs]
    print(f"  Source ({args.source_celltype}): {len(source_in_baseline)} cells")
    print(f"  Target ({args.target_celltype}): {len(target_in_baseline)} cells")

    baseline_source = baseline_adata[source_in_baseline].X
    baseline_target = baseline_adata[target_in_baseline].X
    if hasattr(baseline_source, 'toarray'):
        baseline_source = baseline_source.toarray()
    if hasattr(baseline_target, 'toarray'):
        baseline_target = baseline_target.toarray()

    # Align steered logits to same source cell order
    source_name_to_idx = {name: i for i, name in enumerate(source_names)}
    steered_indices = [source_name_to_idx[n] for n in source_in_baseline]
    steered_source = post_logits_source[steered_indices]

    expr_mask = get_expressed_genes_mask(args.raw_data_file)

    # --- Compute DEGs (celltype ∩ steering) ---
    print("\n5. Computing DEGs (celltype ∩ steering DE)...")
    deg_info = compute_degs(
        baseline_source, baseline_target, steered_source,
        baseline_gene_names, expr_mask
    )

    # --- Sweep over DEG cutoffs ---
    deg_cutoffs = [50, 100, 500, 1000, None]
    sweep_results = []

    print("\n6. Computing entanglement across DEG cutoffs...")
    print("=" * 60)

    for top_n in deg_cutoffs:
        label = f"Top {top_n}" if top_n is not None else "All DEGs"
        df = compute_entanglement(
            fg_cd4, steered_features, steered_alphas, gene_names, deg_info,
            expr_mask, top_n_degs=top_n
        )
        sweep_results.append((label, df))
        print_summary(label, df)

    # Save CSV for the "all DEGs" run
    all_degs_df = sweep_results[-1][1]
    csv_path = os.path.join(args.output_dir, 'feature_entanglement_metrics.csv')
    all_degs_df.to_csv(csv_path, index=False)
    print(f"\n  Saved full metrics to {csv_path}")

    # --- Minimum feature cover ---
    print("\n7. Computing minimum feature cover for top 50 DEGs...")
    annotated_mask = alpha_data['annotated_mask']
    if isinstance(annotated_mask, torch.Tensor):
        annotated_mask = annotated_mask.numpy()

    def print_min_cover(label, mask):
        selected, universe, uncovered, prs = compute_min_cover(
            fg_cd4, gene_names, deg_info, expr_mask,
            annotated_mask=mask, top_n_degs=50
        )
        n_candidates = mask.sum()
        print(f"\n  {label} (from {n_candidates} candidates):")
        print(f"  Minimum features to cover: {len(selected)}")
        if uncovered:
            print(f"  Uncovered DEGs ({len(uncovered)}): {sorted(uncovered)}")
        cumulative = 0
        for feat_id, n_new, genes in selected:
            cumulative += n_new
            pr = prs[feat_id]
            print(f"    Feature {feat_id} (PR={pr:.1f}): +{n_new} DEGs ({cumulative}/{len(universe)} total) — {', '.join(genes)}")

    print_min_cover("Feature-centric (all annotated)", annotated_mask)

    # --- Gene-centric feature analysis ---
    print("\n8. Gene-centric feature analysis for top 50 DEGs...")
    gene_feature_map, union_features, per_gene_feat_sets = compute_gene_centric_cover(
        fg_cd4, gene_names, deg_info, expr_mask,
        annotated_mask=annotated_mask, top_n_degs=50
    )
    print(f"  Union of all gene-centric features: {len(union_features)}")

    # Greedy set cover from gene-centric perspective:
    # pick the feature that appears in the most genes' top-PR feature sets
    fg_expr = fg_cd4[:, expr_mask]
    feature_pr = compute_participation_ratio(fg_expr, axis=1)
    uncovered_genes = set(per_gene_feat_sets.keys())
    selected_gc = []
    while uncovered_genes:
        # Count how many uncovered genes each feature covers
        feat_counts = {}
        for gene in uncovered_genes:
            for f in per_gene_feat_sets[gene]:
                feat_counts[f] = feat_counts.get(f, 0) + 1
        if not feat_counts:
            break
        best_feat = max(feat_counts, key=feat_counts.get)
        covered = {g for g in uncovered_genes if best_feat in per_gene_feat_sets[g]}
        if not covered:
            break
        selected_gc.append((best_feat, len(covered), sorted(covered)))
        uncovered_genes -= covered

    print(f"\n  Gene-centric minimum cover (from {len(union_features)} features):")
    print(f"  Minimum features: {len(selected_gc)}")
    if uncovered_genes:
        print(f"  Uncovered DEGs ({len(uncovered_genes)}): {sorted(uncovered_genes)}")
    cumulative = 0
    for feat_id, n_new, genes in selected_gc:
        cumulative += n_new
        pr = feature_pr[feat_id]
        print(f"    Feature {feat_id} (PR={pr:.1f}): +{n_new} DEGs ({cumulative}/50 total) — {', '.join(genes)}")
    # Overlap with optimizer-selected features
    min_cover_ids = set(f for f, _, _ in selected_gc)
    optimizer_ids = set(steered_features)
    overlap = min_cover_ids & optimizer_ids
    only_cover = min_cover_ids - optimizer_ids
    only_optimizer = optimizer_ids - min_cover_ids
    print(f"\n  Overlap: min-cover ({len(min_cover_ids)}) vs optimizer ({len(optimizer_ids)}):")
    print(f"    In both:          {len(overlap)}  {sorted(overlap)}")
    print(f"    Min-cover only:   {len(only_cover)}  {sorted(only_cover)}")
    print(f"    Optimizer only:   {len(only_optimizer)} (ratio correction / other)")

    union_optimizer_overlap = union_features & optimizer_ids
    union_only = union_features - optimizer_ids
    optimizer_outside_union = optimizer_ids - union_features
    print(f"\n  Overlap: gene-centric union ({len(union_features)}) vs optimizer ({len(optimizer_ids)}):")
    print(f"    In both:          {len(union_optimizer_overlap)}")
    print(f"    Union only:       {len(union_only)}")
    print(f"    Optimizer only:   {len(optimizer_outside_union)} (outside gene-centric sets)")

    print(f"\n  Per-gene breakdown (sorted by |logFC|):")
    for gene, info in sorted(gene_feature_map.items(), key=lambda x: abs(x[1]['logfc']), reverse=True):
        top3 = ', '.join(f'{f}({a:.3f})' for f, a in info['top_features'][:3])
        logfc = abs(info['logfc'])
        n_feat = info['n_features']
        gpr = info['gene_pr']
        print(f"    {gene:12s} |logFC|={logfc:.3f}  n_features={n_feat:3d}  gene_PR={gpr:.1f}  top: {top3}")

    # --- Plot ---
    print("\n9. Generating plots...")
    plot_entanglement_sweep(sweep_results, args.output_dir)
    plot_de_fraction(sweep_results, args.output_dir)
    plot_deg_logfc_curve(deg_info, deg_cutoffs, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()

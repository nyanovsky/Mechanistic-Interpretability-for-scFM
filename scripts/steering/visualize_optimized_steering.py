"""
Visualize results from steering vector optimization.

Creates:
1. Before/after UMAP with steered cell type highlighted (separate UMAPs)
2. Before vs after distance scatter plot with y=x diagonal

Example:
    python visualize_optimized_steering.py \
        --alpha_file results/steering_optimization/CD4_T_cells_to_CD8_T_cells_alpha_vector.pt \
        --source_celltype "CD4 T cells" \
        --target_celltype "CD8 T cells"
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_sae
from utils.steering import SAESteeringModel, ActivationHook
from utils.data_utils import get_expressed_genes_mask, compute_de_genes

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts


def load_precomputed_embeddings(embeddings_path, processed_path):
    """Load pre-computed embeddings and align with cell type annotations."""
    embeddings_dict = torch.load(embeddings_path, map_location='cpu')

    if isinstance(embeddings_dict, dict):
        embeddings = embeddings_dict['embeddings']
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.numpy()
        cell_names = embeddings_dict['cell_names']
    else:
        embeddings = embeddings_dict
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.numpy()
        cell_names = None

    adata_processed = ad.read_h5ad(processed_path)

    if 'celltype' in adata_processed.obs.columns:
        cell_type_col = 'celltype'
    elif 'louvain' in adata_processed.obs.columns:
        cell_type_col = 'louvain'
    else:
        raise ValueError("No cell type column found")

    if cell_names is None:
        cell_names = list(adata_processed.obs_names)
        cell_types = adata_processed.obs[cell_type_col].values
    else:
        name_to_idx_embeddings = {name: i for i, name in enumerate(cell_names)}

        aligned_embeddings = []
        aligned_cell_types = []
        aligned_names = []

        for name in adata_processed.obs_names:
            if name in name_to_idx_embeddings:
                idx = name_to_idx_embeddings[name]
                aligned_embeddings.append(embeddings[idx])
                aligned_cell_types.append(adata_processed[name].obs[cell_type_col].values[0])
                aligned_names.append(name)

        embeddings = np.array(aligned_embeddings)
        cell_types = np.array(aligned_cell_types)
        cell_names = aligned_names

    return embeddings, cell_types, cell_names


def compute_steered_outputs(
    model,
    sae,
    alpha_vector,
    adata,
    attention_mask,
    layer_idx,
    device,
    batch_size=4
):
    """
    Compute steered embeddings and logits using the learned alpha vector.
    Returns (embeddings_array, logits_tensor).
    """
    # Create steering model with full alpha vector
    steering_model = SAESteeringModel(
        model=model,
        sae=sae,
        layer_idx=layer_idx,
        alpha_vector=alpha_vector
    )

    # Create attention mask tensor
    attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(device)
    attn_mask_bool = attention_mask.astype(bool)

    # Hook to capture embeddings
    embedding_hook = ActivationHook(model.bert.encoder.ln)

    embeddings_list = []
    logits_list = []

    try:
        with torch.no_grad():
            for i in tqdm(range(0, adata.n_obs, batch_size), desc="Computing steered outputs"):
                batch_adata = adata[i:i+batch_size]

                batch_counts = batch_adata.X
                if hasattr(batch_counts, 'toarray'):
                    batch_counts = batch_counts.toarray()

                batch_processed = preprocess_counts(batch_counts, device=device)

                # Prepare attention mask
                batch_attn_mask = attn_mask_tensor.repeat(batch_processed.shape[0], 1)
                depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=device)
                batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

                # Forward pass with steering
                output = steering_model(batch_processed, batch_attn_mask)

                # Get embeddings from hook
                last_hidden = embedding_hook.get_activations()
                last_hidden = last_hidden[:, :-2, :]
                last_hidden = last_hidden[:, attn_mask_bool, :]
                cell_embeddings = last_hidden.mean(dim=1).float()

                # Get logits
                logits = output.logits[:, :-2, :].squeeze(-1)
                logits_filtered = logits[:, attn_mask_bool].float()

                embeddings_list.append(cell_embeddings.cpu().numpy())
                logits_list.append(logits_filtered.cpu())

    finally:
        embedding_hook.remove()
        steering_model.remove_hook()

    return np.concatenate(embeddings_list, axis=0), torch.cat(logits_list, dim=0)


def plot_before_after_umap(
    pre_embeddings,
    post_embeddings,
    cell_types,
    source_celltype,
    target_celltype,
    output_dir
):
    """
    Create before/after UMAP visualization with independent UMAPs.
    """
    unique_types = np.unique(cell_types)
    colors = dict(zip(unique_types, sns.color_palette('tab10', n_colors=len(unique_types))))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Pre-steer UMAP
    print("  Computing pre-steer UMAP...")
    adata_pre = ad.AnnData(X=pre_embeddings)
    adata_pre.obs['cell_type'] = cell_types
    sc.pp.neighbors(adata_pre, use_rep='X', n_neighbors=15)
    sc.tl.umap(adata_pre)
    umap_pre = adata_pre.obsm['X_umap']

    for ct in unique_types:
        mask = cell_types == ct
        axes[0].scatter(umap_pre[mask, 0], umap_pre[mask, 1],
                       c=[colors[ct]], label=ct, alpha=0.4, s=5)

    axes[0].set_xlabel('UMAP 1', fontsize=8)
    axes[0].set_ylabel('UMAP 2', fontsize=8)
    axes[0].tick_params(axis='both', labelsize=7)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Post-steer UMAP
    print("  Computing post-steer UMAP...")
    adata_post = ad.AnnData(X=post_embeddings)
    adata_post.obs['cell_type'] = cell_types
    sc.pp.neighbors(adata_post, use_rep='X', n_neighbors=15)
    sc.tl.umap(adata_post)
    umap_post = adata_post.obsm['X_umap']

    for ct in unique_types:
        mask = cell_types == ct
        axes[1].scatter(umap_post[mask, 0], umap_post[mask, 1],
                       c=[colors[ct]], label=ct, alpha=0.4, s=5)

    axes[1].set_xlabel('UMAP 1', fontsize=8)
    axes[1].set_ylabel('UMAP 2', fontsize=8)
    axes[1].tick_params(axis='both', labelsize=7)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].legend(fontsize=6, markerscale=1.5, frameon=False,
                   bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'umap_before_after.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved UMAP to {output_path}")


def plot_distance_scatter(
    pre_embeddings,
    post_embeddings,
    cell_types,
    source_celltype,
    target_celltype,
    output_dir
):
    """
    Create scatter plot of before vs after distances to target centroid.
    """
    # Compute target centroid (from pre-steer embeddings)
    target_mask = cell_types == target_celltype
    target_centroid = pre_embeddings[target_mask].mean(axis=0)

    # Get source cells
    source_mask = cell_types == source_celltype

    # Compute distances for source cells
    pre_distances = np.linalg.norm(pre_embeddings[source_mask] - target_centroid, axis=1)
    post_distances = np.linalg.norm(post_embeddings[source_mask] - target_centroid, axis=1)

    # Count how many cells got closer
    n_closer = (post_distances < pre_distances).sum()
    n_total = len(pre_distances)
    pct_closer = n_closer / n_total * 100

    # Create scatter plot with color-coded points
    closer_mask = post_distances < pre_distances
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.scatter(pre_distances[closer_mask], post_distances[closer_mask],
               alpha=0.4, s=8, color='forestgreen', label=f'Closer ({n_closer})')
    ax.scatter(pre_distances[~closer_mask], post_distances[~closer_mask],
               alpha=0.4, s=8, color='firebrick', label=f'Farther ({n_total - n_closer})')

    # Add y=x diagonal
    max_dist = max(pre_distances.max(), post_distances.max())
    min_dist = min(pre_distances.min(), post_distances.min())
    ax.plot([min_dist, max_dist], [min_dist, max_dist], 'k--', linewidth=1, alpha=0.5)

    ax.set_xlabel(f'Pre-steer distance to {target_celltype}', fontsize=8)
    ax.set_ylabel(f'Post-steer distance to {target_celltype}', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.legend(fontsize=6, markerscale=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'distance_scatter.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved distance scatter to {output_path}")
    print(f"\n  {n_closer}/{n_total} ({pct_closer:.1f}%) cells moved closer to {target_celltype}")

    mean_pre = pre_distances.mean()
    mean_post = post_distances.mean()
    print(f"  Mean distance before: {mean_pre:.4f}")
    print(f"  Mean distance after: {mean_post:.4f}")
    print(f"  Change: {mean_post - mean_pre:.4f} ({(mean_post - mean_pre) / mean_pre * 100:.1f}%)")


def plot_nearest_target_distance(
    pre_embeddings,
    post_embeddings,
    cell_types,
    source_celltype,
    target_celltype,
    output_dir
):
    """
    Plot distribution of distance to nearest target cell for each source cell,
    before vs after steering.
    """
    from scipy.spatial.distance import cdist
    from scipy.stats import wilcoxon

    source_mask = cell_types == source_celltype
    target_mask = cell_types == target_celltype

    # Distances from each source cell to every target cell
    # Target embeddings don't change (only source cells are steered)
    target_embeddings = pre_embeddings[target_mask]

    pre_dists = cdist(pre_embeddings[source_mask], target_embeddings, metric='euclidean')
    pre_nearest = pre_dists.min(axis=1)

    post_dists = cdist(post_embeddings[source_mask], target_embeddings, metric='euclidean')
    post_nearest = post_dists.min(axis=1)

    # Paired Wilcoxon signed-rank test
    _, pval = wilcoxon(pre_nearest, post_nearest, alternative='greater')

    # Star notation
    if pval < 0.001:
        sig_text = '***'
    elif pval < 0.01:
        sig_text = '**'
    elif pval < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'

    # Plot side-by-side boxplots
    fig, ax = plt.subplots(figsize=(2.5, 3.5))

    bp = ax.boxplot(
        [pre_nearest, post_nearest],
        labels=['Pre-steering', 'Post-steering'],
        patch_artist=True,
        widths=0.5,
        showfliers=False
    )
    bp['boxes'][0].set_facecolor('#4878CF')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#E1812C')
    bp['boxes'][1].set_alpha(0.7)
    for box in bp['boxes']:
        box.set_edgecolor('black')
        box.set_linewidth(0.8)
    for median_line in bp['medians']:
        median_line.set_color('black')
        median_line.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(0.8)
    for cap in bp['caps']:
        cap.set_linewidth(0.8)

    # Significance bracket — position above the highest whisker
    whisker_tops = [w.get_ydata().max() for w in bp['whiskers']]
    y_bracket = max(whisker_tops) * 1.05
    h = y_bracket * 0.02
    ax.plot([1, 1, 2, 2], [y_bracket, y_bracket + h, y_bracket + h, y_bracket],
            color='black', linewidth=0.8)
    ax.text(1.5, y_bracket + h, sig_text, ha='center', va='bottom', fontsize=9)

    ax.set_ylabel(f'Distance to nearest\n{target_celltype} cell', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'nearest_target_distance.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved nearest target distance plot to {output_path}")
    print(f"  Mean distance to nearest {target_celltype}: {pre_nearest.mean():.3f} (pre) -> {post_nearest.mean():.3f} (post)")
    print(f"  Wilcoxon signed-rank test: p = {pval:.2e}")


def _plot_distance_boxplot(pre_dists, post_dists, ylabel, title, output_path):
    """Helper: styled boxplot of pre vs post distances with Wilcoxon test."""
    from scipy.stats import wilcoxon

    _, pval = wilcoxon(post_dists, pre_dists, alternative='less')

    if pval < 0.001:
        sig_text = '***'
    elif pval < 0.01:
        sig_text = '**'
    elif pval < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'

    fig, ax = plt.subplots(figsize=(2.5, 3.5))

    bp = ax.boxplot(
        [pre_dists, post_dists],
        labels=['Pre-steering', 'Post-steering'],
        patch_artist=True,
        widths=0.5,
        showfliers=False
    )
    bp['boxes'][0].set_facecolor('#4878CF')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#E1812C')
    bp['boxes'][1].set_alpha(0.7)
    for box in bp['boxes']:
        box.set_edgecolor('black')
        box.set_linewidth(0.8)
    for median_line in bp['medians']:
        median_line.set_color('black')
        median_line.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(0.8)
    for cap in bp['caps']:
        cap.set_linewidth(0.8)

    whisker_tops = [w.get_ydata().max() for w in bp['whiskers']]
    y_bracket = max(whisker_tops) * 1.01
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([1, 1, 2, 2], [y_bracket, y_bracket + h, y_bracket + h, y_bracket],
            color='black', linewidth=0.8)
    ax.text(1.5, y_bracket + h, sig_text, ha='center', va='bottom', fontsize=9)

    ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    delta = post_dists.mean() - pre_dists.mean()
    print(f"  Saved plot to {output_path}")
    print(f"  Mean distance: {pre_dists.mean():.4f} (pre) -> {post_dists.mean():.4f} (post), delta={delta:+.4f}")
    print(f"  Wilcoxon signed-rank test (post < pre): p = {pval:.2e}")


def plot_celltype_de_distance(
    pre_logits_source, post_logits_source, pre_logits_target,
    expr_mask, source_celltype, target_celltype, output_dir, n_de_genes=100
):
    """Euclidean distance to mean target profile on cell-type-DE genes, split up/down.

    DE genes selected by mean difference between source and target (pre-steer),
    filtered to expressed genes, split by direction.
    """
    # Filter to expressed genes
    pre_src = pre_logits_source[:, expr_mask]
    post_src = post_logits_source[:, expr_mask]
    pre_tgt = pre_logits_target[:, expr_mask]

    source_mean = pre_src.mean(dim=0)
    target_mean = pre_tgt.mean(dim=0)
    diff = target_mean - source_mean  # positive = higher in target

    for direction, label in [('up', 'Up in target'), ('down', 'Down in target')]:
        if direction == 'up':
            mask = diff > 0
            signed_diff = diff[mask]
            gene_idx_in_mask = signed_diff.argsort(descending=True)[:n_de_genes]
            # Map back to expressed-gene indices
            full_indices = torch.where(mask)[0]
            top_idx = full_indices[gene_idx_in_mask]
        else:
            mask = diff < 0
            signed_diff = diff[mask].abs()
            gene_idx_in_mask = signed_diff.argsort(descending=True)[:n_de_genes]
            full_indices = torch.where(mask)[0]
            top_idx = full_indices[gene_idx_in_mask]

        target_profile = target_mean[top_idx].numpy()
        pre_dists = np.linalg.norm(pre_src[:, top_idx].numpy() - target_profile, axis=1)
        post_dists = np.linalg.norm(post_src[:, top_idx].numpy() - target_profile, axis=1)

        output_path = os.path.join(output_dir, f'celltype_de_distance_{direction}.pdf')
        _plot_distance_boxplot(
            pre_dists, post_dists,
            ylabel=f'Euclidean distance to\nmean {target_celltype}',
            title=f'Cell-type DE: {label} (top {n_de_genes})',
            output_path=output_path
        )


def plot_steering_de_distance(
    pre_logits_source, post_logits_source, pre_logits_target,
    expr_mask, gene_names, source_celltype, target_celltype, output_dir, n_de_genes=100
):
    """Euclidean distance to mean target profile on steering-DE genes, split up/down.

    DE genes selected by paired t-test (pre vs post steering), filtered to expressed
    and significant genes, split by direction.
    """
    # Filter to expressed genes
    pre_src = pre_logits_source[:, expr_mask].numpy()
    post_src = post_logits_source[:, expr_mask].numpy()
    pre_tgt = pre_logits_target[:, expr_mask]
    names_expr = gene_names[expr_mask]

    # Paired DE: post-steer vs pre-steer source cells
    top_up_genes, top_down_genes, de_stats = compute_de_genes(
        post_src, pre_src, names_expr, top_n=n_de_genes
    )

    sig_count = de_stats['sig_mask'].sum()
    print(f"  Steering DE: {sig_count} significant genes")

    target_mean = pre_tgt.mean(dim=0).numpy()

    for genes, direction, label in [
        (top_up_genes, 'up', 'Upregulated'),
        (top_down_genes, 'down', 'Downregulated')
    ]:
        if not genes:
            print(f"  No {label.lower()} steering-DE genes found, skipping plot")
            continue

        # Map gene names to indices in the expressed-gene space
        name_to_idx = {name: i for i, name in enumerate(names_expr)}
        gene_idx = np.array([name_to_idx[g] for g in genes if g in name_to_idx])

        target_profile = target_mean[gene_idx]
        pre_dists = np.linalg.norm(pre_src[:, gene_idx] - target_profile, axis=1)
        post_dists = np.linalg.norm(post_src[:, gene_idx] - target_profile, axis=1)

        output_path = os.path.join(output_dir, f'steering_de_distance_{direction}.pdf')
        _plot_distance_boxplot(
            pre_dists, post_dists,
            ylabel=f'Euclidean distance to\nmean {target_celltype}',
            title=f'Steering DE: {label} (top {len(gene_idx)})',
            output_path=output_path
        )


def main():
    parser = argparse.ArgumentParser(description='Visualize optimized steering results')

    parser.add_argument('--alpha_file', type=str, required=True,
                       help='Path to alpha vector .pt file')
    parser.add_argument('--source_celltype', type=str, required=True,
                       help='Source cell type that was steered')
    parser.add_argument('--target_celltype', type=str, required=True,
                       help='Target cell type')

    parser.add_argument('--data_file', type=str, default='data/pbmc/pbmc3k_raw.h5ad')
    parser.add_argument('--processed_file', type=str, default='data/pbmc/pbmc3k_processed.h5ad')
    parser.add_argument('--embeddings_file', type=str, default='data/pbmc/aido_cell_pre_steer_embeddings.pt')
    parser.add_argument('--logits_file', type=str, default='/biodata/nyanovsky/datasets/pbmc3k/pbmc3k_logits.h5ad',
                       help='Pre-steer logits h5ad file (cells x genes)')
    parser.add_argument('--n_de_genes', type=int, default=100,
                       help='Number of top DE genes to use for expression similarity')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--sae_dir', type=str)

    parser.add_argument('--model_name', type=str, default='genbio-ai/AIDO.Cell-100M')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--save_embeddings', type=str, default=None,
                       help='Directory to save post-steer embeddings .pt file')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, '..', '..')

    if args.sae_dir is None:
        args.sae_dir = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}/sae_k_32_5120"

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.alpha_file)

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("VISUALIZE OPTIMIZED STEERING")
    print("="*70)

    # 1. Load alpha vector
    print("\n1. Loading alpha vector...")
    alpha_data = torch.load(args.alpha_file, map_location='cpu')
    alpha_vector = alpha_data['alpha_vector'].to(args.device)
    print(f"  Alpha vector shape: {alpha_vector.shape}")
    print(f"  Non-identity features: {(torch.abs(alpha_vector - 1.0) > 1e-6).sum().item()}")

    # 2. Load pre-computed embeddings
    print("\n2. Loading pre-computed embeddings...")
    embeddings_path = os.path.join(repo_root, args.embeddings_file)
    processed_path = os.path.join(repo_root, args.processed_file)

    pre_embeddings, cell_types, cell_names = load_precomputed_embeddings(embeddings_path, processed_path)
    print(f"  Loaded {len(pre_embeddings)} cells")

    source_mask = cell_types == args.source_celltype
    target_mask = cell_types == args.target_celltype
    raw_path = os.path.join(repo_root, args.data_file)

    # Check for cached post-steer results (must have both embeddings and logits)
    cache_path = os.path.join(args.save_embeddings, 'post_steer_embeddings.pt') if args.save_embeddings else None
    cached = None
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if 'post_logits_source' not in cached:
            print(f"\n  Cache at {cache_path} missing logits, will recompute...")
            cached = None

    if cached is not None:
        print("\n3-4. Loading cached post-steer results...")
        post_embeddings = cached['embeddings']
        if not isinstance(post_embeddings, np.ndarray):
            post_embeddings = post_embeddings.numpy()
        post_logits_source = cached['post_logits_source']
        print(f"  Loaded from {cache_path}")
    else:
        # 3. Load model and SAE
        print("\n3. Loading model and SAE...")
        config = CellFoundationConfig.from_pretrained(args.model_name)
        model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config)
        model = model.to(args.device)
        if args.device == 'cuda':
            model = model.to(torch.bfloat16)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        sae = load_sae(args.sae_dir, args.device)

        # 4. Compute steered outputs for source cells
        print("\n4. Computing steered outputs...")
        adata_raw = ad.read_h5ad(raw_path)
        adata_aligned, attention_mask = align_adata(adata_raw)

        common_cells = [name for name in cell_names if name in adata_aligned.obs_names]
        adata_aligned = adata_aligned[common_cells].copy()

        source_cell_names = [cell_names[i] for i in range(len(cell_names)) if source_mask[i]]
        adata_source = adata_aligned[source_cell_names].copy()

        steered_source_embeddings, post_logits_source = compute_steered_outputs(
            model=model,
            sae=sae,
            alpha_vector=alpha_vector,
            adata=adata_source,
            attention_mask=attention_mask,
            layer_idx=args.layer,
            device=args.device,
            batch_size=args.batch_size
        )

        post_embeddings = pre_embeddings.copy()
        post_embeddings[source_mask] = steered_source_embeddings

        # Save if requested
        if args.save_embeddings is not None:
            os.makedirs(args.save_embeddings, exist_ok=True)
            save_path = os.path.join(args.save_embeddings, 'post_steer_embeddings.pt')
            torch.save({
                'embeddings': post_embeddings,
                'post_logits_source': post_logits_source,
                'cell_types': cell_types,
                'cell_names': cell_names,
            }, save_path)
            print(f"  Saved post-steer results to {save_path}")

    # Load pre-steer logits and align to cell_names
    print("\n  Loading pre-steer logits...")
    adata_logits = ad.read_h5ad(args.logits_file)
    logits_cell_indices = [adata_logits.obs_names.get_loc(name) for name in cell_names]
    all_logits = torch.from_numpy(adata_logits.X[logits_cell_indices])
    pre_logits_source = all_logits[source_mask]
    pre_logits_target = all_logits[target_mask]

    # Expression mask and gene names for DE analysis
    expr_mask = get_expressed_genes_mask(raw_path)
    gene_names = np.array(adata_logits.var_names)

    # 5. Create visualizations
    print("\n5. Creating visualizations...")

    plot_before_after_umap(
        pre_embeddings,
        post_embeddings,
        cell_types,
        args.source_celltype,
        args.target_celltype,
        args.output_dir
    )

    plot_distance_scatter(
        pre_embeddings,
        post_embeddings,
        cell_types,
        args.source_celltype,
        args.target_celltype,
        args.output_dir
    )

    plot_celltype_de_distance(
        pre_logits_source, post_logits_source, pre_logits_target,
        expr_mask, args.source_celltype, args.target_celltype,
        args.output_dir, n_de_genes=args.n_de_genes
    )

    plot_steering_de_distance(
        pre_logits_source, post_logits_source, pre_logits_target,
        expr_mask, gene_names, args.source_celltype, args.target_celltype,
        args.output_dir, n_de_genes=args.n_de_genes
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

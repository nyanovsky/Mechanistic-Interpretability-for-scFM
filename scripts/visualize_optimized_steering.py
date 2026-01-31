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

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ModelGenerator/huggingface/aido.cell'))

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts

from steering_utils import load_sae, SAESteeringModel, ActivationHook


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


def compute_steered_embeddings(
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
    Compute steered embeddings using the learned alpha vector.
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

    try:
        with torch.no_grad():
            for i in tqdm(range(0, adata.n_obs, batch_size), desc="Computing steered embeddings"):
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
                _ = steering_model(batch_processed, batch_attn_mask)

                # Get embeddings from hook
                last_hidden = embedding_hook.get_activations()
                last_hidden = last_hidden[:, :-2, :]
                last_hidden = last_hidden[:, attn_mask_bool, :]
                cell_embeddings = last_hidden.mean(dim=1).float()

                embeddings_list.append(cell_embeddings.cpu().numpy())

    finally:
        embedding_hook.remove()
        steering_model.remove_hook()

    return np.concatenate(embeddings_list, axis=0)


def plot_before_after_umap(
    pre_embeddings,
    post_embeddings,
    cell_types,
    source_celltype,
    target_celltype,
    output_dir
):
    """
    Create before/after UMAP visualization with separate UMAPs.
    """
    unique_types = np.unique(cell_types)
    colors = dict(zip(unique_types, sns.color_palette('tab10', n_colors=len(unique_types))))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pre-steer UMAP
    print("  Computing pre-steer UMAP...")
    adata_pre = ad.AnnData(X=pre_embeddings)
    adata_pre.obs['cell_type'] = cell_types
    sc.pp.neighbors(adata_pre, use_rep='X', n_neighbors=15)
    sc.tl.umap(adata_pre)
    umap_pre = adata_pre.obsm['X_umap']

    for ct in unique_types:
        mask = cell_types == ct
        alpha = 0.8 if ct in [source_celltype, target_celltype] else 0.4
        size = 15 if ct in [source_celltype, target_celltype] else 8
        axes[0].scatter(umap_pre[mask, 0], umap_pre[mask, 1],
                       c=[colors[ct]], label=ct, alpha=alpha, s=size)

    axes[0].set_title('Pre-steering', fontsize=14)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    # Post-steer UMAP
    print("  Computing post-steer UMAP...")
    adata_post = ad.AnnData(X=post_embeddings)
    adata_post.obs['cell_type'] = cell_types
    sc.pp.neighbors(adata_post, use_rep='X', n_neighbors=15)
    sc.tl.umap(adata_post)
    umap_post = adata_post.obsm['X_umap']

    for ct in unique_types:
        mask = cell_types == ct
        alpha = 0.8 if ct in [source_celltype, target_celltype] else 0.4
        size = 15 if ct in [source_celltype, target_celltype] else 8
        axes[1].scatter(umap_post[mask, 0], umap_post[mask, 1],
                       c=[colors[ct]], label=ct, alpha=alpha, s=size)

    axes[1].set_title(f'Post-steering ({source_celltype} steered)', fontsize=14)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'umap_before_after.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(pre_distances, post_distances, alpha=0.5, s=20)

    # Add y=x diagonal
    max_dist = max(pre_distances.max(), post_distances.max())
    min_dist = min(pre_distances.min(), post_distances.min())
    ax.plot([min_dist, max_dist], [min_dist, max_dist], 'r--', linewidth=2, label='y=x (no change)')

    ax.set_xlabel(f'Pre-steer distance to {target_celltype} centroid', fontsize=12)
    ax.set_ylabel(f'Post-steer distance to {target_celltype} centroid', fontsize=12)
    ax.set_title(f'{source_celltype}: Before vs After Steering\n'
                f'{n_closer}/{n_total} ({pct_closer:.1f}%) cells moved closer to {target_celltype}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'distance_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved distance scatter to {output_path}")
    print(f"\n  {n_closer}/{n_total} ({pct_closer:.1f}%) cells moved closer to {target_celltype}")

    mean_pre = pre_distances.mean()
    mean_post = post_distances.mean()
    print(f"  Mean distance before: {mean_pre:.4f}")
    print(f"  Mean distance after: {mean_post:.4f}")
    print(f"  Change: {mean_post - mean_pre:.4f} ({(mean_post - mean_pre) / mean_pre * 100:.1f}%)")


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
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--sae_dir', type=str)

    parser.add_argument('--model_name', type=str, default='genbio-ai/AIDO.Cell-100M')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

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
    embeddings_path = os.path.join(script_dir, '..', args.embeddings_file)
    processed_path = os.path.join(script_dir, '..', args.processed_file)

    pre_embeddings, cell_types, cell_names = load_precomputed_embeddings(embeddings_path, processed_path)
    print(f"  Loaded {len(pre_embeddings)} cells")

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

    # 4. Compute steered embeddings for source cells
    print("\n4. Computing steered embeddings...")
    raw_path = os.path.join(script_dir, '..', args.data_file)
    adata_raw = ad.read_h5ad(raw_path)
    adata_aligned, attention_mask = align_adata(adata_raw)

    common_cells = [name for name in cell_names if name in adata_aligned.obs_names]
    adata_aligned = adata_aligned[common_cells].copy()

    # Create post-steer embeddings (copy pre-steer, replace source cells)
    post_embeddings = pre_embeddings.copy()

    source_mask = cell_types == args.source_celltype
    source_cell_names = [cell_names[i] for i in range(len(cell_names)) if source_mask[i]]

    adata_source = adata_aligned[source_cell_names].copy()

    steered_source_embeddings = compute_steered_embeddings(
        model=model,
        sae=sae,
        alpha_vector=alpha_vector,
        adata=adata_source,
        attention_mask=attention_mask,
        layer_idx=args.layer,
        device=args.device,
        batch_size=args.batch_size
    )

    post_embeddings[source_mask] = steered_source_embeddings

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

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

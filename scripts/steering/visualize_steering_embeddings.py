"""
Visualize cell embeddings from steering experiments.

Creates UMAPs of pre-steered and steered embeddings, colored by cell type,
and analyzes how cells move relative to fixed B-cell centroid.
"""
import os
import sys
import argparse
import torch
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats

# Paths
PROCESSED_DATA = "data/pbmc/pbmc3k_processed.h5ad"
PRE_STEERED_EMBEDDINGS = "data/pbmc/aido_cell_pre_steer_embeddings.pt"


def load_cell_types(processed_path):
    """Load cell type annotations from processed data."""
    adata = ad.read_h5ad(processed_path)

    # Check which column has cell types
    if 'celltype' in adata.obs.columns:
        cell_type_col = 'celltype'
    elif 'louvain' in adata.obs.columns:
        cell_type_col = 'louvain'
    else:
        raise ValueError("No cell type column found in processed data")

    return adata.obs[cell_type_col].values, adata.obs_names


def compute_centroid(embeddings, mask):
    """Compute centroid for a subset of cells."""
    return embeddings[mask].mean(axis=0)


def compute_distances_to_centroid(embeddings, centroid):
    """Compute Euclidean distance from each cell to a centroid."""
    distances = np.linalg.norm(embeddings - centroid[np.newaxis, :], axis=1)
    return distances


def compute_all_centroids(embeddings, cell_types):
    """Compute centroids for all cell types."""
    unique_types = np.unique(cell_types)
    centroids = {}

    for ctype in unique_types:
        mask = cell_types == ctype
        centroids[ctype] = embeddings[mask].mean(axis=0)

    return centroids


def compute_centroid_distances(centroids, reference_celltype):
    """
    Compute distance from each cell type centroid to the reference centroid.

    Returns:
        dict: {celltype: distance}
    """
    ref_centroid = centroids[reference_celltype]
    distances = {}

    for ctype, centroid in centroids.items():
        distances[ctype] = np.linalg.norm(centroid - ref_centroid)

    return distances


def plot_umap_scanpy(adata, basis_key, color_key, title, output_path, alpha=0.6):
    """
    Create UMAP visualization using scanpy.

    Args:
        adata: AnnData object with embeddings and UMAP coordinates
        basis_key: Key in adata.obsm for UMAP coordinates (e.g., 'X_umap_pre')
        color_key: Key in adata.obs for coloring (e.g., 'cell_type')
        title: Plot title
        output_path: Where to save figure
        alpha: Transparency for all points (default: 0.6)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sc.pl.embedding(
        adata,
        basis=basis_key,
        color=color_key,
        ax=ax,
        show=False,
        title=title,
        frameon=False,
        alpha=alpha
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved UMAP to {output_path}")


def plot_pairwise_distance_distributions(embeddings, cell_types, output_dir):
    """
    For each cell type, plot distance distributions from its cells to all other cell type centroids.
    This helps identify the closest cell type pairs.

    Args:
        embeddings: (n_cells, n_dims) cell embeddings
        cell_types: (n_cells,) cell type labels
        output_dir: Where to save figures
    """
    unique_types = np.unique(cell_types)

    # Compute all centroids
    centroids = compute_all_centroids(embeddings, cell_types)

    # Create subdirectory for pairwise plots
    pairwise_dir = os.path.join(output_dir, 'pairwise_distances')
    os.makedirs(pairwise_dir, exist_ok=True)

    print(f"\nGenerating pairwise distance plots for {len(unique_types)} cell types...")

    # For each source cell type
    for source_type in unique_types:
        source_mask = cell_types == source_type
        source_cells = embeddings[source_mask]

        # Compute distances from source cells to all target centroids
        target_types = [ct for ct in unique_types if ct != source_type]
        distances_to_targets = {}
        mean_distances = {}

        for target_type in target_types:
            target_centroid = centroids[target_type]
            distances = compute_distances_to_centroid(source_cells, target_centroid)
            distances_to_targets[target_type] = distances
            mean_distances[target_type] = distances.mean()

        # Sort targets by mean distance (closest first)
        sorted_targets = sorted(target_types, key=lambda ct: mean_distances[ct])

        # Create violin plot
        fig, ax = plt.subplots(figsize=(12, 6))

        data_for_violin = []
        labels = []
        for target_type in sorted_targets:
            data_for_violin.append(distances_to_targets[target_type])
            labels.append(target_type)

        parts = ax.violinplot(data_for_violin, positions=range(len(labels)),
                              showmeans=True, showmedians=True)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Distance to target centroid', fontsize=12)
        ax.set_title(f'{source_type} cells: distance to other cell type centroids (sorted)', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Sanitize filename
        safe_filename = source_type.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(pairwise_dir, f'distances_from_{safe_filename}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Print top 3 closest
        print(f"\n  {source_type} - closest cell types:")
        for i, target_type in enumerate(sorted_targets[:3]):
            mean_dist = mean_distances[target_type]
            std_dist = distances_to_targets[target_type].std()
            print(f"    {i+1}. {target_type}: {mean_dist:.4f} ± {std_dist:.4f}")

    print(f"\nSaved pairwise distance plots to {pairwise_dir}/")

    # Also create a summary: find the overall closest pair
    print("\n" + "="*70)
    print("CLOSEST CELL TYPE PAIRS (by mean distance)")
    print("="*70)

    pairs = []
    for source_type in unique_types:
        source_mask = cell_types == source_type
        source_cells = embeddings[source_mask]

        for target_type in unique_types:
            if source_type == target_type:
                continue

            target_centroid = centroids[target_type]
            distances = compute_distances_to_centroid(source_cells, target_centroid)
            mean_dist = distances.mean()

            # Store as sorted tuple to avoid duplicates (A->B and B->A)
            pair = tuple(sorted([source_type, target_type]))
            pairs.append((pair, mean_dist))

    # Remove duplicates and sort
    unique_pairs = {}
    for pair, dist in pairs:
        if pair not in unique_pairs:
            unique_pairs[pair] = []
        unique_pairs[pair].append(dist)

    # Average the two directions (A->B and B->A)
    avg_pairs = [(pair, np.mean(dists)) for pair, dists in unique_pairs.items()]
    avg_pairs.sort(key=lambda x: x[1])

    for i, (pair, avg_dist) in enumerate(avg_pairs[:5]):
        print(f"{i+1}. {pair[0]} <-> {pair[1]}: {avg_dist:.4f}")

    print("="*70)

    return avg_pairs[0] if avg_pairs else None


def plot_initial_distance_distribution(pre_distances, cell_types, output_dir, reference_celltype='B cells'):
    """
    Plot distribution of pre-steering distances to reference cell type centroid for each cell type.

    Args:
        pre_distances: (n_cells,) distances before steering
        cell_types: (n_cells,) cell type labels
        output_dir: Where to save figures
        reference_celltype: The reference cell type
    """
    unique_types = np.unique(cell_types)

    # Compute mean distances and sort cell types by proximity
    mean_distances = []
    for ctype in unique_types:
        mask = cell_types == ctype
        mean_distances.append(pre_distances[mask].mean())

    # Sort by mean distance (closest first)
    sorted_indices = np.argsort(mean_distances)
    sorted_types = unique_types[sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create violin plot for each cell type (sorted by proximity)
    data_for_violin = []
    labels = []
    for ctype in sorted_types:
        mask = cell_types == ctype
        data_for_violin.append(pre_distances[mask])
        labels.append(ctype)

    parts = ax.violinplot(data_for_violin, positions=range(len(labels)),
                          showmeans=True, showmedians=True)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(f'Distance to {reference_celltype} centroid', fontsize=12)
    ax.set_title(f'Pre-steering distance distributions (sorted by proximity)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'initial_distance_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved initial distance distribution to {output_path}")
    print(f"\nCell types sorted by mean distance to {reference_celltype} centroid:")
    for i, ctype in enumerate(sorted_types):
        mask = cell_types == ctype
        mean_dist = pre_distances[mask].mean()
        std_dist = pre_distances[mask].std()
        print(f"  {i+1}. {ctype}: {mean_dist:.4f} ± {std_dist:.4f}")


def plot_centroid_distances(
    pre_centroid_dists,
    post_centroid_dists_dict,
    output_dir,
    reference_celltype='B cells',
    centroid_mode='contemporary'
):
    """
    Plot centroid-to-centroid distances.

    Args:
        pre_centroid_dists: {celltype: distance} for pre-steered
        post_centroid_dists_dict: {alpha: {celltype: distance}} for each alpha
        output_dir: Where to save figures
        reference_celltype: The reference cell type
        centroid_mode: 'fixed' or 'contemporary'
    """
    alphas = sorted([k for k in post_centroid_dists_dict.keys() if isinstance(k, (int, float))])

    # Set titles based on mode
    if centroid_mode == 'fixed':
        centroid_desc = 'fixed pre-steered'
    else:
        centroid_desc = 'contemporary'

    # Get all cell types
    unique_types = [ct for ct in pre_centroid_dists.keys() if ct != reference_celltype]
    colors = sns.color_palette('tab10', n_colors=len(unique_types))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute centroid-to-centroid distances
    for i, ctype in enumerate(unique_types):
        distances = [pre_centroid_dists[ctype]]
        for alpha in alphas:
            distances.append(post_centroid_dists_dict[alpha][ctype])

        x_vals = ['Pre'] + [f'{alpha}' for alpha in alphas]
        axes[0].plot(x_vals, distances, 'o-', label=ctype, color=colors[i],
                    linewidth=2, markersize=6)

    axes[0].set_xlabel('Alpha', fontsize=12)
    axes[0].set_ylabel(f'Distance between centroids', fontsize=12)
    axes[0].set_title(f'Centroid-to-centroid distance to {reference_celltype} ({centroid_desc})', fontsize=13)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Relative change
    for i, ctype in enumerate(unique_types):
        pre_dist = pre_centroid_dists[ctype]

        relative_changes = []
        for alpha in alphas:
            post_dist = post_centroid_dists_dict[alpha][ctype]
            relative_changes.append((post_dist - pre_dist) / pre_dist * 100)

        axes[1].plot([f'{alpha}' for alpha in alphas], relative_changes, 'o-',
                    label=ctype, color=colors[i], linewidth=2, markersize=6)

    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Alpha', fontsize=12)
    axes[1].set_ylabel('% change in centroid distance', fontsize=12)
    axes[1].set_title(f'Relative change in centroid distance to {reference_celltype} ({centroid_desc})', fontsize=13)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'centroid_distance_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved centroid distance analysis to {output_path}")


def plot_distance_analysis(
    pre_distances,
    post_distances_dict,
    cell_types,
    output_dir,
    reference_celltype='B cells',
    centroid_mode='contemporary'
):
    """
    Plot distance analysis: how cells move relative to B-cell centroid.

    Args:
        pre_distances: (n_cells,) distances before steering
        post_distances_dict: {alpha: (n_cells,) distances} after steering
        cell_types: (n_cells,) cell type labels
        output_dir: Where to save figures
        reference_celltype: The cell type we're measuring distance to
        centroid_mode: 'fixed' or 'contemporary'
    """
    alphas = sorted([k for k in post_distances_dict.keys() if isinstance(k, (int, float))])

    # Set titles based on mode
    if centroid_mode == 'fixed':
        centroid_desc = 'fixed pre-steered'
    else:
        centroid_desc = 'contemporary'

    # Get all cell types (including reference for debugging)
    unique_types = list(np.unique(cell_types))
    colors = sns.color_palette('tab10', n_colors=len(unique_types))

    # Highlight reference cell type with thicker line/different marker
    reference_idx = unique_types.index(reference_celltype) if reference_celltype in unique_types else None

    # 1. Distance trajectories per cell type
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute distances
    for i, ctype in enumerate(unique_types):
        mask = cell_types == ctype

        mean_distances = [pre_distances[mask].mean()]
        for alpha in alphas:
            mean_distances.append(post_distances_dict[alpha][mask].mean())

        x_vals = ['Pre'] + [f'{alpha}' for alpha in alphas]

        # Highlight reference cell type
        if i == reference_idx:
            axes[0].plot(x_vals, mean_distances, 'o-', label=f'{ctype} (reference)',
                        color=colors[i], linewidth=3, markersize=8, linestyle='--', alpha=0.8)
        else:
            axes[0].plot(x_vals, mean_distances, 'o-', label=ctype,
                        color=colors[i], linewidth=2, markersize=6)

    axes[0].set_xlabel('Alpha', fontsize=12)
    axes[0].set_ylabel(f'Mean distance to {reference_celltype} centroid', fontsize=12)
    axes[0].set_title(f'Distance to {reference_celltype} centroid ({centroid_desc})', fontsize=13)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Relative change (normalized to pre-steered)
    for i, ctype in enumerate(unique_types):
        mask = cell_types == ctype
        pre_mean = pre_distances[mask].mean()

        relative_changes = []
        for alpha in alphas:
            post_mean = post_distances_dict[alpha][mask].mean()
            relative_changes.append((post_mean - pre_mean) / pre_mean * 100)

        # Highlight reference cell type
        if i == reference_idx:
            axes[1].plot([f'{alpha}' for alpha in alphas], relative_changes, 'o-',
                        label=f'{ctype} (reference)', color=colors[i],
                        linewidth=3, markersize=8, linestyle='--', alpha=0.8)
        else:
            axes[1].plot([f'{alpha}' for alpha in alphas], relative_changes, 'o-',
                        label=ctype, color=colors[i], linewidth=2, markersize=6)

    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Alpha', fontsize=12)
    axes[1].set_ylabel('% change in distance', fontsize=12)
    axes[1].set_title(f'Relative change in distance to {reference_celltype} ({centroid_desc})', fontsize=13)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'distance_analysis_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved distance analysis to {output_path}")

    # 2. Per-alpha detailed violin plots
    for alpha in alphas:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute distance change for each cell
        distance_changes = post_distances_dict[alpha] - pre_distances

        # Create violin plot for each cell type
        data_for_violin = []
        labels = []
        for ctype in unique_types:
            mask = cell_types == ctype
            data_for_violin.append(distance_changes[mask])
            labels.append(ctype)

        parts = ax.violinplot(data_for_violin, positions=range(len(labels)),
                              showmeans=True, showmedians=True)

        ax.axhline(0, color='red', linestyle='--', linewidth=1, label='No change')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(f'Distance change to {reference_celltype} centroid', fontsize=12)
        ax.set_title(f'Per-cell distance changes (Alpha={alpha})', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'distance_distribution_alpha{alpha}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved distance distribution for alpha={alpha}")


def main():
    parser = argparse.ArgumentParser(description='Visualize steering embeddings')
    parser.add_argument('--steering_file', required=True,
                       help='Path to steering results .pt file')
    parser.add_argument('--experiment_name', required=True,
                       help='Name of experiment (e.g., bcell_identity)')
    parser.add_argument('--reference_celltype', default='B cells',
                       help='Cell type to measure distances to (default: B cells)')
    parser.add_argument('--centroid_mode', default='contemporary', choices=['fixed', 'contemporary'],
                       help='Centroid mode: "fixed" (pre-steered centroid for all alphas) or '
                            '"contemporary" (compute centroid at each alpha, default: contemporary)')
    args = parser.parse_args()

    # Setup paths
    processed_path = PROCESSED_DATA
    pre_embed_path = PRE_STEERED_EMBEDDINGS

    output_dir = f"../../plots/sae/layer_12/steering_analysis/{args.experiment_name}/embeddings"
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("STEERING EMBEDDING VISUALIZATION")
    print("="*70)

    # 1. Load cell types
    print("\n1. Loading cell type annotations...")
    cell_types, cell_names = load_cell_types(processed_path)
    print(f"   Loaded {len(cell_types)} cells")
    print(f"   Cell types: {np.unique(cell_types)}")

    # 2. Load pre-steered embeddings
    print("\n2. Loading pre-steered embeddings...")
    pre_embeddings_dict = torch.load(pre_embed_path, map_location='cpu')

    # Handle both old and new format
    if isinstance(pre_embeddings_dict, dict):
        pre_embeddings = pre_embeddings_dict['embeddings']
        if not isinstance(pre_embeddings, np.ndarray):
            pre_embeddings = pre_embeddings.numpy()
        pre_cell_names = pre_embeddings_dict['cell_names']
    else:
        # Old format: just the tensor
        pre_embeddings = pre_embeddings_dict
        if not isinstance(pre_embeddings, np.ndarray):
            pre_embeddings = pre_embeddings.numpy()
        pre_cell_names = cell_names  # Assume same order

    print(f"   Shape: {pre_embeddings.shape}")

    # 3. Load steering results
    print("\n3. Loading steering results...")
    steering_results = torch.load(args.steering_file, map_location='cpu')
    inner_results = steering_results['results']
    alphas = sorted([k for k in inner_results.keys() if isinstance(k, (int, float))])
    print(f"   Alphas: {alphas}")

    # Get cell indices from steering results
    first_alpha = alphas[0]
    cell_indices_steered = inner_results[first_alpha]['cell_indices']
    if isinstance(cell_indices_steered, torch.Tensor):
        cell_indices_steered = cell_indices_steered.numpy()

    print(f"   Steered cells: {len(cell_indices_steered)}")

    # Load raw data to get cell names for steered cells
    print("\n3b. Loading raw data to map cell indices...")
    raw_data_path = "data/pbmc/pbmc3k_raw.h5ad"
    adata_raw = ad.read_h5ad(raw_data_path)
    steered_cell_names = adata_raw.obs_names[cell_indices_steered]
    print(f"   Raw data cells: {adata_raw.n_obs}")

    # Find overlap between steered cells and cells with annotations
    common_with_steered = list(set(steered_cell_names) & set(pre_cell_names))
    print(f"   Common cells (steered & annotated): {len(common_with_steered)}")

    # Create mapping for filtering
    steered_name_to_idx = {name: i for i, name in enumerate(steered_cell_names)}
    pre_name_to_idx_new = {name: i for i, name in enumerate(pre_cell_names)}

    # Indices in steered results that correspond to annotated cells
    steered_indices_to_keep = [steered_name_to_idx[name] for name in common_with_steered if name in steered_name_to_idx]
    pre_indices_to_keep = [pre_name_to_idx_new[name] for name in common_with_steered if name in pre_name_to_idx_new]

    # Filter pre-steered embeddings to common cells
    pre_embeddings = pre_embeddings[pre_indices_to_keep]
    pre_cell_names = [common_with_steered[i] for i in range(len(common_with_steered))]

    # Now align cell types with filtered embeddings
    name_to_idx = {name: i for i, name in enumerate(cell_names)}
    cell_types_aligned = []
    for name in pre_cell_names:
        if name in name_to_idx:
            cell_types_aligned.append(cell_types[name_to_idx[name]])
        else:
            cell_types_aligned.append('Unknown')
    cell_types_aligned = np.array(cell_types_aligned)

    print(f"   Filtered embeddings shape: {pre_embeddings.shape}")
    print(f"   Filtered cell types: {len(cell_types_aligned)}")
    print(f"   Cell type distribution: {dict(zip(*np.unique(cell_types_aligned, return_counts=True)))}")

    # 4. Compute pre-steered reference cell type centroid
    print(f"\n4. Computing pre-steered {args.reference_celltype} centroid...")
    ref_mask = cell_types_aligned == args.reference_celltype

    if ref_mask.sum() == 0:
        print(f"   ERROR: '{args.reference_celltype}' not found in cell types!")
        print(f"   Available: {np.unique(cell_types_aligned)}")
        sys.exit(1)

    ref_centroid_pre = compute_centroid(pre_embeddings, ref_mask)
    print(f"   Centroid shape: {ref_centroid_pre.shape}")
    print(f"   {args.reference_celltype} cells: {ref_mask.sum()}")

    # 5. Compute distances
    if args.centroid_mode == 'fixed':
        print(f"\n5. Computing distances to FIXED pre-steered {args.reference_celltype} centroid...")
        pre_distances = compute_distances_to_centroid(pre_embeddings, ref_centroid_pre)

        post_distances_dict = {}
        steered_embeddings_dict = {}

        for alpha in alphas:
            steered_emb_full = inner_results[alpha]['embeddings']
            if not isinstance(steered_emb_full, np.ndarray):
                steered_emb_full = steered_emb_full.float().numpy()

            # Filter to only cells with annotations
            steered_emb = steered_emb_full[steered_indices_to_keep]
            steered_embeddings_dict[alpha] = steered_emb

            # Measure distance to FIXED pre-steered centroid
            distances = compute_distances_to_centroid(steered_emb, ref_centroid_pre)
            post_distances_dict[alpha] = distances

            print(f"   Alpha {alpha}: mean distance = {distances.mean():.4f} (pre: {pre_distances.mean():.4f})")

    else:  # contemporary mode
        print(f"\n5. Computing distances to CONTEMPORARY {args.reference_celltype} centroid...")

        # Pre-steered: distance to pre-steered centroid
        pre_distances = compute_distances_to_centroid(pre_embeddings, ref_centroid_pre)
        print(f"   Pre-steered: mean distance = {pre_distances.mean():.4f}")

        post_distances_dict = {}
        steered_embeddings_dict = {}

        for alpha in alphas:
            steered_emb_full = inner_results[alpha]['embeddings']
            if not isinstance(steered_emb_full, np.ndarray):
                steered_emb_full = steered_emb_full.float().numpy()

            # Filter to only cells with annotations
            steered_emb = steered_emb_full[steered_indices_to_keep]
            steered_embeddings_dict[alpha] = steered_emb

            # Compute contemporary centroid for this alpha
            ref_mask_alpha = cell_types_aligned == args.reference_celltype
            ref_centroid_alpha = compute_centroid(steered_emb, ref_mask_alpha)

            # Measure distance to contemporary centroid at this alpha
            distances = compute_distances_to_centroid(steered_emb, ref_centroid_alpha)
            post_distances_dict[alpha] = distances

            print(f"   Alpha {alpha}: mean distance = {distances.mean():.4f}")

    # 6. Create AnnData for UMAP visualization
    print("\n6. Creating UMAP visualizations with scanpy...")

    # Create AnnData with pre-steered embeddings
    adata = ad.AnnData(X=pre_embeddings)
    adata.obs['cell_type'] = cell_types_aligned
    adata.obsm['X_embedding_pre'] = pre_embeddings

    # Compute neighbors and UMAP for pre-steered
    print("   Computing pre-steered UMAP...")
    sc.pp.neighbors(adata, use_rep='X_embedding_pre', key_added='pre_steer', n_neighbors=15)
    sc.tl.umap(adata, neighbors_key='pre_steer')
    adata.obsm['X_umap_pre'] = adata.obsm['X_umap'].copy()

    # Plot pre-steered UMAP
    umap_path = os.path.join(output_dir, 'umap_pre_steered.png')
    plot_umap_scanpy(adata, 'X_umap_pre', 'cell_type', 'Pre-steered embeddings', umap_path)

    # Compute UMAP for each steered condition
    for alpha in alphas:
        print(f"   Computing UMAP for alpha={alpha}...")
        steered_emb = steered_embeddings_dict[alpha]

        # Add to adata
        adata.obsm[f'X_embedding_alpha{alpha}'] = steered_emb

        # Compute neighbors and UMAP
        sc.pp.neighbors(adata, use_rep=f'X_embedding_alpha{alpha}',
                       key_added=f'alpha{alpha}', n_neighbors=15)
        sc.tl.umap(adata, neighbors_key=f'alpha{alpha}')
        adata.obsm[f'X_umap_alpha{alpha}'] = adata.obsm['X_umap'].copy()

        # Plot
        umap_path = os.path.join(output_dir, f'umap_alpha{alpha}.png')
        plot_umap_scanpy(adata, f'X_umap_alpha{alpha}', 'cell_type',
                        f'Steered embeddings (Alpha={alpha})', umap_path)

    # 7. Compute centroid-to-centroid distances
    print("\n7. Computing centroid-to-centroid distances...")

    # Compute all centroids for pre-steered
    pre_centroids = compute_all_centroids(pre_embeddings, cell_types_aligned)
    pre_centroid_dists = compute_centroid_distances(pre_centroids, args.reference_celltype)

    # Compute for each alpha
    post_centroid_dists_dict = {}
    for alpha in alphas:
        steered_emb = steered_embeddings_dict[alpha]
        alpha_centroids = compute_all_centroids(steered_emb, cell_types_aligned)
        post_centroid_dists_dict[alpha] = compute_centroid_distances(alpha_centroids, args.reference_celltype)

    # 8. Distance analysis plots
    print("\n8. Creating distance analysis plots...")

    # Plot pairwise distance distributions for all cell types
    print("\n8a. Analyzing all pairwise distances...")
    closest_pair = plot_pairwise_distance_distributions(
        pre_embeddings,
        cell_types_aligned,
        output_dir
    )

    if closest_pair:
        pair, dist = closest_pair
        print(f"\nRecommendation: For optimization experiment, steer {pair[0]} → {pair[1]} (distance: {dist:.4f})")

    # Plot initial distance distribution to identify closest cell types (to reference)
    print(f"\n8b. Analyzing distances to {args.reference_celltype}...")
    plot_initial_distance_distribution(
        pre_distances,
        cell_types_aligned,
        output_dir,
        args.reference_celltype
    )

    # Plot centroid-to-centroid distances
    plot_centroid_distances(
        pre_centroid_dists,
        post_centroid_dists_dict,
        output_dir,
        args.reference_celltype,
        args.centroid_mode
    )

    # Plot per-cell distances (original)
    plot_distance_analysis(
        pre_distances,
        post_distances_dict,
        cell_types_aligned,
        output_dir,
        args.reference_celltype,
        args.centroid_mode
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

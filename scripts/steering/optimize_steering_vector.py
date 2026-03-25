"""
Optimize a sparse steering vector to move one cell type toward another.

Uses gradient descent with L1 regularization to find a sparse set of SAE features
that, when steered, move source cell type toward target cell type.

Supports two objectives:
- latent: minimize L2 distance in cell embedding space (original)
- expression: minimize weighted MSE in gene expression (logit) space

Example:
    python optimize_steering_vector.py \
        --source_celltype "CD4 T cells" \
        --target_celltype "CD8 T cells" \
        --data_file data/pbmc/pbmc3k_raw.h5ad \
        --objective expression \
        --logits_file data/pbmc/pbmc3k_logits.h5ad \
        --gene_weight_mode effect_size
"""

import os
import sys
import argparse
import torch
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_sae, get_annotated_features
from utils.steering import SAESteeringModel

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts


def resolve_path(path, script_dir):
    """Resolve path: absolute paths pass through, relative paths resolve from project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(script_dir, '..', '..', path)


def load_precomputed_embeddings(embeddings_path, cell_names):
    """
    Load pre-computed embeddings aligned to cell_names ordering.

    Returns:
        embeddings: (n_cells, hidden_dim) numpy array
    """
    embeddings_dict = torch.load(embeddings_path, map_location='cpu')

    if isinstance(embeddings_dict, dict):
        embeddings = embeddings_dict['embeddings']
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.numpy()
        emb_cell_names = embeddings_dict['cell_names']
    else:
        embeddings = embeddings_dict
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.numpy()
        return embeddings

    # Align to cell_names ordering
    name_to_idx = {name: i for i, name in enumerate(emb_cell_names)}
    aligned_embeddings = np.array([embeddings[name_to_idx[name]] for name in cell_names
                                    if name in name_to_idx])

    return aligned_embeddings


def main():
    parser = argparse.ArgumentParser(description='Optimize steering vector')

    # Cell types
    parser.add_argument('--source_celltype', type=str, required=True,
                       help='Source cell type to steer')
    parser.add_argument('--target_celltype', type=str, required=True,
                       help='Target cell type to steer toward')

    # Objective
    parser.add_argument('--objective', type=str, default='latent',
                       choices=['latent', 'expression'],
                       help='Optimization objective: latent (embedding L2) or expression (logit MSE)')
    parser.add_argument('--logits_file', type=str, default=None,
                       help='Path to pre-computed logits .h5ad (required for expression objective)')
    parser.add_argument('--gene_weight_mode', type=str, default='uniform',
                       choices=['uniform', 'effect_size'],
                       help='Gene weighting for expression objective')

    # Data paths
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to raw .h5ad file')
    parser.add_argument('--processed_file', type=str,
                       default='data/pbmc/pbmc3k_processed.h5ad',
                       help='Path to processed .h5ad with cell type annotations')
    parser.add_argument('--embeddings_file', type=str,
                       default='data/pbmc/aido_cell_pre_steer_embeddings.pt',
                       help='Path to pre-computed embeddings')
    parser.add_argument('--layer', type=int, default=12,
                       help='Layer to steer (0-indexed)')
    parser.add_argument('--sae_dir', type=str,
                       help='Path to SAE directory (auto-detect if not specified)')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=0.001,
                       help='L1 regularization strength')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of optimization epochs')
    parser.add_argument('--threshold', type=float, default=1e-4,
                       help='Threshold for zeroing alphas to 1.0')

    # Model parameters
    parser.add_argument('--model_name', type=str,
                       default='genbio-ai/AIDO.Cell-100M',
                       help='AIDO.Cell model name')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')

    # Output
    parser.add_argument('--output_dir', type=str,
                       default='results/steering_optimization',
                       help='Output directory')

    args = parser.parse_args()

    if args.objective == 'expression' and args.logits_file is None:
        parser.error("--logits_file is required when --objective is 'expression'")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.sae_dir is None:
        BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
        args.sae_dir = f"{BASE_DIR}/sae_k_32_5120"

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("STEERING VECTOR OPTIMIZATION")
    print("="*70)
    print(f"Source: {args.source_celltype}")
    print(f"Target: {args.target_celltype}")
    print(f"Objective: {args.objective}")
    if args.objective == 'expression':
        print(f"Gene weighting: {args.gene_weight_mode}")
    print(f"Layer: {args.layer}")
    print(f"Learning rate: {args.lr}")
    print(f"L1 lambda: {args.lambda_l1}")
    print(f"Epochs: {args.num_epochs}")
    print("="*70)

    # 1. Load cell type annotations
    print("\n1. Loading cell type annotations...")
    processed_path = resolve_path(args.processed_file, script_dir)
    adata_processed = ad.read_h5ad(processed_path)

    if 'celltype' in adata_processed.obs.columns:
        cell_type_col = 'celltype'
    elif 'louvain' in adata_processed.obs.columns:
        cell_type_col = 'louvain'
    else:
        raise ValueError("No cell type column found in processed file")

    cell_types = adata_processed.obs[cell_type_col].values
    cell_names = list(adata_processed.obs_names)

    source_mask = cell_types == args.source_celltype
    target_mask = cell_types == args.target_celltype

    if source_mask.sum() == 0:
        raise ValueError(f"No cells found for '{args.source_celltype}'")
    if target_mask.sum() == 0:
        raise ValueError(f"No cells found for '{args.target_celltype}'")

    print(f"  Loaded {len(cell_names)} cells")
    print(f"  Cell types: {np.unique(cell_types)}")
    print(f"  Source cells: {source_mask.sum()}")
    print(f"  Target cells: {target_mask.sum()}")

    del adata_processed

    # 2. Prepare objective targets
    print("\n2. Preparing optimization targets...")

    target_centroid = None
    target_expression = None
    gene_weights = None

    if args.objective == 'latent':
        embeddings_path = resolve_path(args.embeddings_file, script_dir)
        embeddings = load_precomputed_embeddings(embeddings_path, cell_names)
        target_centroid = torch.from_numpy(embeddings[target_mask].mean(axis=0)).to(args.device)
        print(f"  Target centroid shape: {target_centroid.shape}")
        del embeddings

    elif args.objective == 'expression':
        logits_path = resolve_path(args.logits_file, script_dir)
        adata_logits = ad.read_h5ad(logits_path)

        # Align to cell_names ordering
        logits_cell_indices = [adata_logits.obs_names.get_loc(name) for name in cell_names]
        all_logits = torch.from_numpy(adata_logits.X[logits_cell_indices]).float()

        target_expression = all_logits[target_mask].mean(dim=0).to(args.device)
        print(f"  Target expression shape: {target_expression.shape}")

        if args.gene_weight_mode == 'effect_size':
            source_expression = all_logits[source_mask].mean(dim=0)
            gene_weights = (target_expression.cpu() - source_expression).abs()
            gene_weights = gene_weights.to(args.device)
            print(f"  Gene weights: effect_size mode, top weight={gene_weights.max():.6f}")
        else:
            print(f"  Gene weights: uniform")

        del adata_logits, all_logits

    # 3. Load model
    print("\n3. Loading AIDO.Cell model...")
    config = CellFoundationConfig.from_pretrained(args.model_name)
    model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config)
    model = model.to(args.device)

    if args.device == "cuda":
        model = model.to(torch.bfloat16)

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print("✓ Model loaded")

    # 4. Load SAE
    print("\n4. Loading SAE...")
    sae = load_sae(args.sae_dir, args.device)

    for param in sae.parameters():
        param.requires_grad = False
    sae.eval()

    print("✓ SAE loaded")

    # 5. Load data for source cells
    print("\n5. Loading source cell data...")
    raw_path = resolve_path(args.data_file, script_dir)

    adata_raw = ad.read_h5ad(raw_path)

    # Align data
    adata_aligned, attention_mask = align_adata(adata_raw)

    # Filter to cells that have embeddings
    common_cells = [name for name in cell_names if name in adata_aligned.obs_names]
    adata_aligned = adata_aligned[common_cells].copy()

    # Get source cells
    source_cell_names = [cell_names[i] for i in range(len(cell_names)) if source_mask[i]]
    adata_source = adata_aligned[source_cell_names].copy()

    print(f"  Source cells for optimization: {adata_source.n_obs}")

    # Calculate number of batches
    n_batches_per_epoch = (adata_source.n_obs + args.batch_size - 1) // args.batch_size
    print(f"  Batches per epoch: {n_batches_per_epoch}")

    # 6. Load annotated features
    print("\n6. Loading annotated features...")
    n_features = sae.latent_dim
    annotated_mask = get_annotated_features(args.sae_dir, n_features)
    annotated_mask_tensor = torch.from_numpy(annotated_mask).to(args.device)

    # 7. Initialize learnable alpha vector and steering model
    print("\n7. Initializing learnable alpha vector...")

    # Base alpha vector (fixed at 1.0, no gradients)
    alpha_vector_base = torch.ones(n_features, device=args.device, requires_grad=False)

    # Learnable parameters only for annotated features
    n_learnable = annotated_mask.sum()
    alpha_learnable = torch.ones(n_learnable, device=args.device, requires_grad=True)

    print(f"  Total features: {n_features}")
    print(f"  Learnable features: {n_learnable}")
    print(f"  Fixed features: {n_features - n_learnable}")

    # Create initial full alpha vector
    alpha_full_init = alpha_vector_base.clone()
    alpha_full_init[annotated_mask_tensor] = alpha_learnable

    # Convert attention mask to boolean and tensor
    attn_mask_bool = attention_mask.astype(bool)
    attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(args.device)

    steering_model = SAESteeringModel(
        model=model,
        sae=sae,
        layer_idx=args.layer,
        alpha_vector=alpha_full_init,
    )

    # 8. Optimize
    print("\n8. Running optimization...")
    optimizer = torch.optim.Adam([alpha_learnable], lr=args.lr)

    losses = []
    objective_losses = []
    sparsity_losses = []
    non_zero_counts = []

    for epoch in range(args.num_epochs):
        epoch_objective_loss = 0.0
        epoch_sparsity_loss = 0.0
        n_batches = 0

        # Progress bar for batches
        pbar = tqdm(range(0, adata_source.n_obs, args.batch_size),
                   desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)

        # Process source cells in batches
        for i in pbar:
            batch_adata = adata_source[i:i+args.batch_size]

            # Get counts
            batch_counts = batch_adata.X
            if hasattr(batch_counts, 'toarray'):
                batch_counts = batch_counts.toarray()

            # Preprocess
            batch_processed = preprocess_counts(batch_counts, device=args.device)

            # Prepare attention mask (including depth tokens)
            batch_attn_mask = attn_mask_tensor.repeat(batch_processed.shape[0], 1)
            depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=args.device)
            batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

            optimizer.zero_grad()

            # Construct differentiable alpha vector and update model
            alpha_full = alpha_vector_base.clone()
            alpha_full[annotated_mask_tensor] = alpha_learnable
            steering_model.update_alpha_vector(alpha_full)

            # Forward pass
            output = steering_model(
                batch_processed, batch_attn_mask,
                output_hidden_states=(args.objective == 'latent')
            )

            # Compute objective loss
            if args.objective == 'latent':
                last_hidden = output.hidden_states[-1][:, :-2, :]
                cell_embeddings = last_hidden[:, attn_mask_bool, :].mean(dim=1)
                distances = torch.norm(cell_embeddings - target_centroid, dim=1)
                objective_loss = distances.mean()

            elif args.objective == 'expression':
                logits = output.logits[:, :-2, :].squeeze(-1)
                logits_filtered = logits[:, attn_mask_bool]
                diff = logits_filtered - target_expression
                if gene_weights is not None:
                    objective_loss = (diff ** 2 * gene_weights).sum(dim=1).mean()
                else:
                    objective_loss = (diff ** 2).mean()

            # Sparsity loss: L1 on (alpha - 1) to encourage alphas near 1 (no change)
            sparsity_loss = args.lambda_l1 * torch.abs(alpha_learnable - 1.0).sum()

            # Total loss
            loss = objective_loss + sparsity_loss

            # Backward
            loss.backward()
            optimizer.step()

            epoch_objective_loss += objective_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'obj': f'{objective_loss.item():.4f}'
            })

        # Average losses
        epoch_objective_loss /= n_batches
        epoch_sparsity_loss /= n_batches
        epoch_loss = epoch_objective_loss + epoch_sparsity_loss

        losses.append(epoch_loss)
        objective_losses.append(epoch_objective_loss)
        sparsity_losses.append(epoch_sparsity_loss)

        # Hard threshold: set alphas close to 1.0 to exactly 1.0 (no steering)
        with torch.no_grad():
            mask = torch.abs(alpha_learnable - 1.0) < args.threshold
            alpha_learnable[mask] = 1.0

        non_zero_count = (torch.abs(alpha_learnable - 1.0) > 1e-6).sum().item()
        non_zero_counts.append(non_zero_count)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.num_epochs}: "
                  f"loss={epoch_loss:.4f}, "
                  f"obj={epoch_objective_loss:.4f}, "
                  f"l1={epoch_sparsity_loss:.4f}, "
                  f"non_identity={non_zero_count}")

    print("\n✓ Optimization complete")

    # 9. Analyze results
    print("\n9. Analyzing learned alpha vector...")

    # Reconstruct full alpha vector
    alpha_vector_full_final = alpha_vector_base.clone()
    alpha_vector_full_final[annotated_mask_tensor] = alpha_learnable

    alphas_np = alpha_vector_full_final.detach().cpu().numpy()
    non_identity_mask = np.abs(alphas_np - 1.0) > 1e-6
    non_identity_alphas = alphas_np[non_identity_mask]
    non_identity_features = np.where(non_identity_mask)[0]

    print(f"  Non-identity features (alpha != 1): {len(non_identity_features)} / {n_features}")
    print(f"  Alpha range: [{alphas_np.min():.4f}, {alphas_np.max():.4f}]")

    if len(non_identity_features) > 0:
        # Sort by distance from 1.0
        sorted_indices = np.argsort(np.abs(non_identity_alphas - 1.0))[::-1]

        print(f"\n  Top 20 features by |alpha - 1|:")
        for i, idx in enumerate(sorted_indices[:20]):
            feat_id = non_identity_features[idx]
            alpha = non_identity_alphas[idx]
            print(f"    {i+1}. Feature {feat_id}: alpha={alpha:.4f}")

    # 10. Save results
    print("\n10. Saving results...")

    # Save alpha vector
    output_file = os.path.join(args.output_dir,
                               f"{args.source_celltype.replace(' ', '_')}_to_"
                               f"{args.target_celltype.replace(' ', '_')}_alpha_vector.pt")

    torch.save({
        'alpha_vector': alpha_vector_full_final.detach().cpu(),
        'alpha_learnable': alpha_learnable.detach().cpu(),
        'annotated_mask': annotated_mask,
        'non_identity_features': non_identity_features,
        'non_identity_alphas': non_identity_alphas,
        'losses': losses,
        'objective_losses': objective_losses,
        'sparsity_losses': sparsity_losses,
        'non_zero_counts': non_zero_counts,
        'args': vars(args)
    }, output_file)

    print(f"  Saved alpha vector to {output_file}")

    # Plot loss curves
    obj_label = 'Distance Loss (to target centroid)' if args.objective == 'latent' \
        else 'Expression Loss (weighted MSE)'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(objective_losses)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Objective Loss')
    axes[0, 1].set_title(obj_label)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(sparsity_losses)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sparsity Loss (L1)')
    axes[1, 0].set_title('Sparsity Loss')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(non_zero_counts)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Non-identity Features')
    axes[1, 1].set_title('Number of Features with alpha != 1')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    loss_plot_path = os.path.join(args.output_dir, 'optimization_curves.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved loss curves to {loss_plot_path}")

    # Clean up
    steering_model.remove_hook()

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Objective: {args.objective}")
    print(f"Non-identity features: {len(non_identity_features)}")
    print(f"Final objective loss: {objective_losses[-1]:.4f}")


if __name__ == "__main__":
    main()

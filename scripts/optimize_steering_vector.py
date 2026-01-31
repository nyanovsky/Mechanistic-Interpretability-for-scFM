"""
Optimize a sparse steering vector to move one cell type toward another.

Uses gradient descent with L1 regularization to find a sparse set of SAE features
that, when steered, move source cell type embeddings toward target cell type centroid.

Example:
    python optimize_steering_vector.py \
        --source_celltype "CD4 T cells" \
        --target_celltype "CD8 T cells" \
        --data_file data/pbmc/pbmc3k_raw.h5ad \
        --layer 12
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ModelGenerator/huggingface/aido.cell'))

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts

from steering_utils import TopKSAE, load_sae


class OptimizableSteeringModel(nn.Module):
    """
    Steering model with learnable per-feature alpha vector.

    Applies SAE-based steering with a learnable alpha for each feature.
    """

    def __init__(
        self,
        model,
        sae,
        alpha_vector_full,
        alpha_learnable,
        annotated_mask,
        attention_mask,
        layer_idx: int,
        device: str
    ):
        super().__init__()
        self.model = model
        self.sae = sae
        self.alpha_vector_full = alpha_vector_full  # Full vector (fixed at 1.0)
        self.alpha_learnable = alpha_learnable  # Learnable subset
        self.annotated_mask = annotated_mask  # Which features are learnable
        self.attention_mask = attention_mask  # (n_genes,) boolean mask
        self.layer_idx = layer_idx
        self.device = device
        self.hook_handle = None

        # Register hook
        self._register_steering_hook()

    def _register_steering_hook(self):
        """Register forward hook for steering."""
        def steering_hook(module, input, output):
            # output is a tuple (hidden_states,) for BERT layers
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Apply steering
            steered_states = self._apply_sae_steering(hidden_states)

            # Return as tuple if original was tuple
            if isinstance(output, tuple):
                return (steered_states,) + output[1:]
            return steered_states

        target_layer = self.model.bert.encoder.layer[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(steering_hook)

    def _apply_sae_steering(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply SAE-based steering with learnable alpha vector.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            steered_states: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        dtype = hidden_states.dtype

        # Flatten batch and sequence dimensions for SAE
        flat_states = hidden_states.reshape(-1, hidden_size)

        # Encode with SAE
        features = self.sae.encode(flat_states.float())

        # Get original reconstruction and error
        decoder_weights = self.sae.decoder.weight.T
        x_reconstructed = features @ decoder_weights
        error = flat_states.float() - x_reconstructed

        # Construct full alpha vector from learnable subset
        alpha_full = self.alpha_vector_full.clone()
        alpha_full[self.annotated_mask] = self.alpha_learnable

        # STEER: Scale features by alpha vector
        # Broadcasting: features (batch*seq, n_features) * alpha_full (n_features,)
        features_steered = features * alpha_full

        # Reconstruct from steered features
        x_reconstructed_steered = features_steered @ decoder_weights

        # Add back error term
        x_steered = x_reconstructed_steered + error

        # Reshape back
        steered_states = x_steered.reshape(batch_size, seq_len, hidden_size).to(dtype)

        return steered_states

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass with steering applied.

        Returns cell embeddings computed with proper attention mask filtering.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract final hidden states
        last_hidden = outputs.hidden_states[-1]

        # Remove depth tokens
        last_hidden = last_hidden[:, :-2, :]

        # Filter to valid genes using attention mask
        last_hidden_filtered = last_hidden[:, self.attention_mask, :]

        # Average over valid genes
        cell_embeddings = last_hidden_filtered.mean(dim=1)

        return cell_embeddings

    def remove_hooks(self):
        """Remove steering hooks."""
        if self.hook_handle is not None:
            self.hook_handle.remove()


def load_precomputed_embeddings(embeddings_path, processed_path):
    """
    Load pre-computed embeddings and align with cell type annotations.

    Returns:
        embeddings: (n_cells, hidden_dim) numpy array
        cell_types: (n_cells,) cell type labels
        cell_names: List of cell names
    """
    # Load embeddings
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

    # Load cell types
    adata_processed = ad.read_h5ad(processed_path)

    if 'celltype' in adata_processed.obs.columns:
        cell_type_col = 'celltype'
    elif 'louvain' in adata_processed.obs.columns:
        cell_type_col = 'louvain'
    else:
        raise ValueError("No cell type column found")

    # Align
    if cell_names is None:
        cell_names = adata_processed.obs_names
        cell_types = adata_processed.obs[cell_type_col].values
    else:
        # Filter to common cells and align order
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


def plot_alpha_distribution(alpha_vector, output_path):
    """Plot distribution of learned alpha values."""
    alphas = alpha_vector.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Histogram of all alphas
    axes[0].hist(alphas, bins=50, edgecolor='black')
    axes[0].set_xlabel('Alpha value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of all alpha values')
    axes[0].axvline(1.0, color='red', linestyle='--', linewidth=1, label='Identity (alpha=1)')
    axes[0].legend()

    # Plot 2: Histogram of non-identity alphas
    non_identity_alphas = alphas[np.abs(alphas - 1.0) > 1e-6]
    if len(non_identity_alphas) > 0:
        axes[1].hist(non_identity_alphas, bins=50, edgecolor='black')
        axes[1].set_xlabel('Alpha value')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Non-identity alphas (n={len(non_identity_alphas)})')
        axes[1].axvline(1.0, color='red', linestyle='--', linewidth=1)
    else:
        axes[1].text(0.5, 0.5, 'No non-identity alphas', ha='center', va='center')
        axes[1].set_title('Non-identity alphas')

    # Plot 3: Top-K largest |alpha - 1| values
    top_k = 50
    deviations = np.abs(alphas - 1.0)
    top_indices = np.argsort(deviations)[-top_k:][::-1]
    top_alphas = alphas[top_indices]

    colors = ['red' if a < 1.0 else 'blue' for a in top_alphas]
    axes[2].barh(range(len(top_alphas)), top_alphas, color=colors)
    axes[2].axvline(1.0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Alpha value')
    axes[2].set_ylabel(f'Top {top_k} features')
    axes[2].set_title(f'Top {top_k} features by |alpha - 1|')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved alpha distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize steering vector')

    # Cell types
    parser.add_argument('--source_celltype', type=str, required=True,
                       help='Source cell type to steer')
    parser.add_argument('--target_celltype', type=str, required=True,
                       help='Target cell type to steer toward')

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
    print(f"Layer: {args.layer}")
    print(f"Learning rate: {args.lr}")
    print(f"L1 lambda: {args.lambda_l1}")
    print(f"Epochs: {args.num_epochs}")
    print("="*70)

    # 1. Load pre-computed embeddings and compute target centroid
    print("\n1. Loading pre-computed embeddings...")
    embeddings_path = os.path.join(script_dir, '..', args.embeddings_file)
    processed_path = os.path.join(script_dir, '..', args.processed_file)

    embeddings, cell_types, cell_names = load_precomputed_embeddings(embeddings_path, processed_path)

    print(f"  Loaded {len(embeddings)} cells")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Cell types: {np.unique(cell_types)}")

    # Identify source and target
    source_mask = cell_types == args.source_celltype
    target_mask = cell_types == args.target_celltype

    if source_mask.sum() == 0:
        raise ValueError(f"No cells found for '{args.source_celltype}'")
    if target_mask.sum() == 0:
        raise ValueError(f"No cells found for '{args.target_celltype}'")

    print(f"  Source cells: {source_mask.sum()}")
    print(f"  Target cells: {target_mask.sum()}")

    # Compute target centroid from pre-computed embeddings
    target_embeddings = embeddings[target_mask]
    target_centroid = torch.from_numpy(target_embeddings.mean(axis=0)).to(args.device)

    print(f"  Target centroid shape: {target_centroid.shape}")

    # 2. Load model
    print("\n2. Loading AIDO.Cell model...")
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

    # 3. Load SAE
    print("\n3. Loading SAE...")
    sae = load_sae(args.sae_dir, args.device)
    print("✓ SAE loaded")

    # 4. Load data for source cells
    print("\n4. Loading source cell data...")
    raw_path = os.path.join(script_dir, '..', args.data_file)

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

    # 5. Load annotated features
    print("\n5. Loading annotated features...")

    annotated_file = os.path.join(script_dir, '..', 'plots', 'sae', f'layer_{args.layer}', 'annotated_features.csv')
    if not os.path.exists(annotated_file):
        raise FileNotFoundError(f"Annotated features file not found at {annotated_file}")

    annotated_df = pd.read_csv(annotated_file)

    # Get mask for annotated features
    annotated_mask = annotated_df['Annotated'].values.astype(bool)

    print(f"  Total features: {len(annotated_mask)}")
    print(f"  Annotated features: {annotated_mask.sum()}")

    # Convert to tensor
    annotated_mask_tensor = torch.from_numpy(annotated_mask).to(args.device)

    # 6. Initialize learnable alpha vector
    print("\n6. Initializing learnable alpha vector...")
    n_features = sae.latent_dim

    # Create full alpha vector (fixed at 1.0, no gradients)
    alpha_vector_full = torch.ones(n_features, device=args.device, requires_grad=False)

    # Create learnable parameters only for annotated features
    n_learnable = annotated_mask.sum()
    alpha_learnable = torch.ones(n_learnable, device=args.device, requires_grad=True)

    print(f"  Total features: {n_features}")
    print(f"  Learnable features: {n_learnable}")
    print(f"  Fixed features: {n_features - n_learnable}")

    # 7. Create steering model
    print("\n7. Creating optimizable steering model...")

    # Convert attention mask to boolean
    attn_mask_bool = attention_mask.astype(bool)

    # Create attention mask tensor for batching
    attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(args.device)

    steering_model = OptimizableSteeringModel(
        model=model,
        sae=sae,
        alpha_vector_full=alpha_vector_full,
        alpha_learnable=alpha_learnable,
        annotated_mask=annotated_mask_tensor,
        attention_mask=attn_mask_bool,
        layer_idx=args.layer,
        device=args.device
    )

    # 8. Optimize
    print("\n8. Running optimization...")
    optimizer = torch.optim.Adam([alpha_learnable], lr=args.lr)

    losses = []
    distance_losses = []
    sparsity_losses = []
    non_zero_counts = []

    for epoch in range(args.num_epochs):
        epoch_distance_loss = 0.0
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

            # Forward pass with steering
            steered_embeddings = steering_model(batch_processed, batch_attn_mask)

            # Distance loss: mean per-cell distance to target centroid
            distances = torch.norm(steered_embeddings - target_centroid, dim=1)
            distance_loss = distances.mean()

            # Sparsity loss: L1 on (alpha - 1) to encourage alphas near 1 (no change)
            sparsity_loss = args.lambda_l1 * torch.abs(alpha_learnable - 1.0).sum()

            # Total loss
            loss = distance_loss + sparsity_loss

            # Backward
            loss.backward()
            optimizer.step()

            epoch_distance_loss += distance_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dist': f'{distance_loss.item():.4f}'
            })

        # Average losses
        epoch_distance_loss /= n_batches
        epoch_sparsity_loss /= n_batches
        epoch_loss = epoch_distance_loss + epoch_sparsity_loss

        losses.append(epoch_loss)
        distance_losses.append(epoch_distance_loss)
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
                  f"dist={epoch_distance_loss:.4f}, "
                  f"l1={epoch_sparsity_loss:.4f}, "
                  f"non_identity={non_zero_count}")

    print("\n✓ Optimization complete")

    # 9. Analyze results
    print("\n9. Analyzing learned alpha vector...")

    # Reconstruct full alpha vector
    alpha_vector_full_final = alpha_vector_full.clone()
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
        'distance_losses': distance_losses,
        'sparsity_losses': sparsity_losses,
        'non_zero_counts': non_zero_counts,
        'args': vars(args)
    }, output_file)

    print(f"  Saved alpha vector to {output_file}")

    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(distance_losses)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Distance Loss')
    axes[0, 1].set_title('Distance Loss (to target centroid)')
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

    # Plot alpha distribution
    alpha_plot_path = os.path.join(args.output_dir, 'alpha_distribution.png')
    plot_alpha_distribution(alpha_vector_full_final, alpha_plot_path)

    # Clean up
    steering_model.remove_hooks()

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Non-identity features: {len(non_identity_features)}")
    print(f"Final distance loss: {distance_losses[-1]:.4f}")


if __name__ == "__main__":
    main()

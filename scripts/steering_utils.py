"""
Utility classes and functions for SAE-based steering experiments on AIDO.Cell.

This module provides the core infrastructure for steering experiments:
- ActivationHook: Generic hook for capturing layer activations
- SAESteeringModel: Model wrapper that applies SAE-based steering via hooks
- SteeringExperiment: High-level interface for running experiments
- Helper functions for loading statistics and data
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass

# Add ModelGenerator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ModelGenerator/huggingface/aido.cell'))

from aido_cell.utils import preprocess_counts


@dataclass
class SteeringConfig:
    """Configuration for a steering experiment."""
    name: str
    steering_features: List[int]
    alphas: List[float]
    layer_idx: int = 12
    reference: str = 'max'  # 'max', 'mean', or 'median'
    description: str = ""


class ActivationHook:
    """
    Generic hook for capturing layer activations.

    Can be used to capture activations from any layer for analysis.
    """

    def __init__(self, module: nn.Module):
        """
        Args:
            module: Module to hook into
        """
        self.activations = None
        self.handle = module.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, input, output):
        """Capture output activations."""
        # Store first element if output is tuple (common pattern)
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
        return output

    def get_activations(self) -> torch.Tensor:
        """Get captured activations."""
        return self.activations

    def remove(self):
        """Remove the hook."""
        self.handle.remove()


class SAESteeringModel(nn.Module):
    """
    Wrapper around CellFoundationForMaskedLM with SAE-based steering.

    This model hooks into a specified layer and applies SAE-based steering
    by SETTING feature activations to alpha * reference_value.
    """

    def __init__(
        self,
        model: nn.Module,
        sae: nn.Module,
        feature_stats: Dict[str, torch.Tensor],
        layer_idx: int,
        steering_features: List[int],
        alpha: float,
        reference: str = 'max'
    ):
        """
        Args:
            model: AIDO.Cell model
            sae: Trained SAE
            feature_stats: Feature activation statistics dict
            layer_idx: Layer to steer (0-indexed)
            steering_features: List of feature IDs to steer
            alpha: Steering strength (Anthropic uses [-10, 10])
            reference: Reference statistic ('max', 'mean', 'median')
        """
        super().__init__()
        self.model = model
        self.sae = sae
        self.feature_stats = feature_stats
        self.layer_idx = layer_idx
        self.steering_features = steering_features
        self.alpha = alpha
        self.reference = reference

        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook on the specified layer."""
        def steering_hook(module, input, output):
            """
            Hook that captures activations, applies SAE steering, and returns steered activations.

            Anthropic's approach:
            1. Encode activations with SAE to get features
            2. SET steering features to alpha * reference_value (REPLACE, not add)
            3. Reconstruct from modified features
            4. Add back the SAE reconstruction error

            Args:
                output: Tuple of (hidden_states, ...) from the layer

            Returns:
                Tuple with steered hidden_states as first element
            """
            hidden_states = output[0]  # Shape: (batch, seq_len, hidden_size)

            # Apply SAE steering
            steered_states = self._apply_sae_steering(hidden_states)

            # Return as tuple with same structure
            return (steered_states,) + output[1:]

        # Access the layer: model.bert.encoder.layer[layer_idx]
        target_layer = self.model.bert.encoder.layer[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(steering_hook)

    def _apply_sae_steering(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply SAE-based steering to hidden states.

        Method (Anthropic's approach):
        1. Encode: features = SAE.encode(x)
        2. Reconstruct: x_recon = features @ decoder
        3. Compute error: error = x - x_recon
        4. Steer: features_steered[i] = alpha * reference[i] for i in steering_features
        5. Reconstruct steered: x_recon_steered = features_steered @ decoder
        6. Return: x_steered = x_recon_steered + error

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            steered_states: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Flatten batch and sequence dimensions for SAE
        flat_states = hidden_states.reshape(-1, hidden_size)  # (batch*seq_len, hidden_size)

        with torch.no_grad():
            # Step 1: Encode with SAE
            features = self.sae.encode(flat_states.float())  # (batch*seq_len, n_features)

            # Step 2: Get original reconstruction and error
            # Use decoder weight matrix transposed: (latent_dim, input_dim)
            decoder_weights = self.sae.decoder.weight.T
            x_reconstructed = features @ decoder_weights
            error = flat_states.float() - x_reconstructed

            # Step 3: STEER - SET feature activations (REPLACE, not add)
            features_steered = features.clone()
            reference_values = self.feature_stats[self.reference].to(device)

            for feat_id in self.steering_features:
                # SET (=) feature to alpha * reference_value
                # alpha=0 means ABLATE (set to 0)
                # alpha=1 means set to reference value (e.g., max)
                # alpha=5 means set to 5x reference value
                features_steered[:, feat_id] = self.alpha * reference_values[feat_id]

            # Step 4: Reconstruct from steered features
            x_reconstructed_steered = features_steered @ decoder_weights

            # Step 5: Add back error term (Anthropic's approach)
            # This preserves information the SAE couldn't capture
            x_steered = x_reconstructed_steered + error

        # Reshape back to (batch, seq_len, hidden_size)
        steered_states = x_steered.reshape(batch_size, seq_len, hidden_size).to(dtype)

        return steered_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        Forward pass with steering applied.

        Args:
            input_ids: Preprocessed gene counts
            attention_mask: Attention mask (including depth tokens)

        Returns:
            Model output with logits
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

    def remove_hook(self):
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class SteeringExperiment:
    """Main class to run steering experiments."""

    def __init__(
        self,
        model: nn.Module,
        sae: nn.Module,
        feature_stats: Dict[str, torch.Tensor],
        attention_mask: np.ndarray,
        layer_idx: int = 12,
        device: str = 'cuda',
        batch_size: int = 4
    ):
        """
        Args:
            model: AIDO.Cell model
            sae: Trained SAE
            feature_stats: Feature activation statistics
            attention_mask: Gene attention mask
            layer_idx: Layer to steer (0-indexed)
            device: Device for computation
            batch_size: Batch size for processing
        """
        self.model = model
        self.sae = sae
        self.feature_stats = feature_stats
        self.attention_mask = attention_mask
        self.layer_idx = layer_idx
        self.device = device
        self.batch_size = batch_size

        # Create attention mask tensor
        self.attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(device)

    def run_baseline(
        self,
        adata,
        cell_indices: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run baseline (NO steering) forward pass.

        This is a completely separate run without any steering hooks.
        This is different from alpha=0, which is ABLATION.

        Args:
            adata: AnnData object with gene counts
            cell_indices: Optional indices of cells to process (default: all)

        Returns:
            results: Dict with outputs and embeddings
        """
        if cell_indices is None:
            cell_indices = np.arange(adata.n_obs)

        print("Running BASELINE (no steering)...")

        # Hook the final layer norm to capture embeddings
        embedding_hook = ActivationHook(self.model.bert.encoder.ln)

        outputs_list = []
        embeddings_list = []

        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(cell_indices), self.batch_size), desc="Baseline"):
                    batch_idx = cell_indices[i:i+self.batch_size]
                    batch_counts = adata[batch_idx].X
                    if hasattr(batch_counts, 'toarray'):
                        batch_counts = batch_counts.toarray()

                    # Preprocess
                    batch_processed = preprocess_counts(batch_counts, device=self.device)

                    # Prepare attention mask
                    batch_attn_mask = self.attn_mask_tensor.repeat(batch_processed.shape[0], 1)
                    depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=self.device)
                    batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

                    # Single forward pass
                    output = self.model(
                        input_ids=batch_processed,
                        attention_mask=batch_attn_mask,
                        return_dict=True
                    )

                    # Get captured embeddings from hook
                    last_hidden = embedding_hook.get_activations()

                    # Extract cell embeddings (remove depth tokens, mask genes, average)
                    last_hidden = last_hidden[:, :-2, :]
                    last_hidden = last_hidden[:, self.attention_mask.astype(bool), :]
                    cell_embeddings = last_hidden.mean(dim=1)

                    outputs_list.append(output.logits.cpu())
                    embeddings_list.append(cell_embeddings.cpu())
        finally:
            embedding_hook.remove()

        return {
            'logits': torch.cat(outputs_list, dim=0),
            'embeddings': torch.cat(embeddings_list, dim=0),
            'cell_indices': cell_indices,
            'alpha': None,  # Baseline has no alpha
            'is_baseline': True
        }

    def run_steered(
        self,
        adata,
        steering_features: List[int],
        alpha: float,
        reference: str = 'max',
        cell_indices: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run steered forward pass with a SINGLE forward pass per batch.

        Note: alpha=0 means ABLATION (features set to 0), not baseline.
        For true baseline, use run_baseline().

        Args:
            adata: AnnData object with gene counts
            steering_features: List of feature IDs to steer
            alpha: Steering strength
                   - alpha=0:   Ablation (features turned off)
                   - alpha=0.5: Features at 0.5x reference
                   - alpha=1:   Features at reference (e.g., max)
                   - alpha=5:   Features at 5x reference
            reference: Reference statistic ('max', 'mean', 'median')
            cell_indices: Optional indices of cells to process

        Returns:
            results: Dict with outputs and metadata
        """
        if cell_indices is None:
            cell_indices = np.arange(adata.n_obs)

        if alpha == 0:
            print(f"Running ABLATION (alpha=0, features turned OFF)...")
        else:
            print(f"Running STEERED pass (alpha={alpha})...")

        # Create steerable model
        steering_model = SAESteeringModel(
            model=self.model,
            sae=self.sae,
            feature_stats=self.feature_stats,
            layer_idx=self.layer_idx,
            steering_features=steering_features,
            alpha=alpha,
            reference=reference
        )

        # Hook the final layer norm to capture embeddings
        embedding_hook = ActivationHook(self.model.bert.encoder.ln)

        outputs_list = []
        embeddings_list = []

        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(cell_indices), self.batch_size), desc=f"Alpha={alpha}"):
                    batch_idx = cell_indices[i:i+self.batch_size]
                    batch_counts = adata[batch_idx].X
                    if hasattr(batch_counts, 'toarray'):
                        batch_counts = batch_counts.toarray()

                    # Preprocess
                    batch_processed = preprocess_counts(batch_counts, device=self.device)

                    # Prepare attention mask
                    batch_attn_mask = self.attn_mask_tensor.repeat(batch_processed.shape[0], 1)
                    depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=self.device)
                    batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

                    # SINGLE forward pass with steering
                    output = steering_model(
                        input_ids=batch_processed,
                        attention_mask=batch_attn_mask
                    )

                    # Get captured embeddings from hook
                    last_hidden = embedding_hook.get_activations()

                    # Extract cell embeddings
                    last_hidden = last_hidden[:, :-2, :]
                    last_hidden = last_hidden[:, self.attention_mask.astype(bool), :]
                    cell_embeddings = last_hidden.mean(dim=1)

                    outputs_list.append(output.logits.cpu())
                    embeddings_list.append(cell_embeddings.cpu())
        finally:
            embedding_hook.remove()
            steering_model.remove_hook()

        return {
            'logits': torch.cat(outputs_list, dim=0),
            'embeddings': torch.cat(embeddings_list, dim=0),
            'cell_indices': cell_indices,
            'alpha': alpha,
            'steering_features': steering_features,
            'reference': reference,
            'is_baseline': False
        }

    def run_experiment(
        self,
        config: SteeringConfig,
        adata,
        cell_indices: Optional[np.ndarray] = None,
        include_baseline: bool = True
    ) -> Dict:
        """
        Run complete steering experiment across multiple alphas.

        Args:
            config: SteeringConfig with experiment parameters
            adata: AnnData object with gene counts
            cell_indices: Optional cell indices to process
            include_baseline: Whether to include baseline (no steering) run

        Returns:
            results: Dict with keys:
                     - 'baseline': baseline results (if include_baseline=True)
                     - alpha values: results for each alpha
        """
        print("="*70)
        print(f"STEERING EXPERIMENT: {config.name}")
        print("="*70)
        if config.description:
            print(f"Description: {config.description}")
        print(f"Steering {len(config.steering_features)} features: {config.steering_features}")
        print(f"Alpha values: {config.alphas}")
        print(f"Reference: {config.reference}")
        print(f"Layer: {config.layer_idx} (0-indexed)")
        print("="*70)

        # Print feature reference values
        print("\nFeature reference values:")
        for feat_id in config.steering_features[:5]:
            max_val = self.feature_stats['max'][feat_id].item()
            mean_val = self.feature_stats['mean'][feat_id].item()
            print(f"  Feature {feat_id}: max={max_val:.4f}, mean={mean_val:.4f}, ratio={mean_val/max_val:.3f}")
        if len(config.steering_features) > 5:
            print(f"  ... and {len(config.steering_features) - 5} more features")

        print("\nInterpretation:")
        print("  - Baseline: No steering applied")
        print("  - alpha=0: Ablation (features turned OFF)")
        print("  - alpha=1: Features at reference level (e.g., max)")
        print("  - alpha=5: Features at 5x reference")
        print()

        results = {}

        # Baseline (no steering)
        if include_baseline:
            results['baseline'] = self.run_baseline(adata, cell_indices)

        # Steered runs (including alpha=0 which is ablation)
        for alpha in config.alphas:
            results[alpha] = self.run_steered(
                adata,
                config.steering_features,
                alpha,
                config.reference,
                cell_indices
            )

        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        print(f"Results keys: {list(results.keys())}")

        return results


def load_feature_statistics(interpretation_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load feature activation statistics from cell-feature matrix.

    Args:
        interpretation_dir: Path to SAE interpretation directory

    Returns:
        Dict with 'max', 'mean', 'median', 'std', 'p90' tensors of shape (n_features,)
    """
    cell_feature_path = os.path.join(interpretation_dir, "cell_feature_matrix.npy")

    if not os.path.exists(cell_feature_path):
        raise FileNotFoundError(f"Cell-feature matrix not found at {cell_feature_path}")

    cell_feature_matrix = np.load(cell_feature_path)
    print(f"Loaded cell-feature matrix with shape: {cell_feature_matrix.shape}")

    # Handle shape: want (n_features, n_cells)
    if cell_feature_matrix.shape[0] != 5120:
        print("Detected shape (n_cells, n_features), transposing...")
        cell_feature_matrix = cell_feature_matrix.T

    n_features, n_cells = cell_feature_matrix.shape
    print(f"Working with {n_features} features across {n_cells} cells")

    # Compute statistics across cells (axis=1)
    stats = {
        'mean': torch.tensor(np.mean(cell_feature_matrix, axis=1), dtype=torch.float32),
        'median': torch.tensor(np.median(cell_feature_matrix, axis=1), dtype=torch.float32),
        'max': torch.tensor(np.max(cell_feature_matrix, axis=1), dtype=torch.float32),
        'std': torch.tensor(np.std(cell_feature_matrix, axis=1), dtype=torch.float32),
        'p90': torch.tensor(np.percentile(cell_feature_matrix, 90, axis=1), dtype=torch.float32),
    }

    print("\nFeature activation statistics:")
    print(f"  Max - range: [{stats['max'].min():.4f}, {stats['max'].max():.4f}]")
    print(f"  Mean - range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
    print(f"  Mean/Max ratio: {(stats['mean'] / (stats['max'] + 1e-8)).mean():.3f}")

    return stats

"""Steering infrastructure: hooks, steerable model wrapper, and experiment runner."""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict, Optional
from dataclasses import dataclass

from aido_cell.utils import preprocess_counts


@dataclass
class SteeringConfig:
    """Configuration for a steering experiment."""
    name: str
    steering_features: List[int]
    alphas: List[float]
    layer_idx: int = 12
    description: str = ""


class ActivationHook:
    """Generic hook for capturing layer activations."""

    def __init__(self, module: nn.Module):
        self.activations = None
        self.handle = module.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, input, output):
        """Capture output activations."""
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
        return output

    def get_activations(self) -> torch.Tensor:
        return self.activations

    def remove(self):
        self.handle.remove()


class SAESteeringModel(nn.Module):
    """Wrapper around CellFoundationForMaskedLM with SAE-based steering.

    Supports three modes:
    1. Full alpha vector (multiplicative): alpha_vector (shape [n_features]) scales
       all features via the SAE encode/decode pipeline.
    2. Sparse multiplicative: steering_features + alpha scales specific features via
       the encode/decode pipeline.
    3. Sparse additive: steering_features + alpha adds alpha * decoder_row[:, feat_id]
       directly to the masked hidden states. No encode/decode (algebraically
       equivalent to encode-add-decode + error term; see docstring of
       _apply_sae_steering).
    """

    def __init__(
        self,
        model: nn.Module,
        sae: nn.Module,
        layer_idx: int,
        alpha_vector: Optional[torch.Tensor] = None,
        steering_features: Optional[List[int]] = None,
        alpha: Optional[float] = None,
        token_mask: Optional[torch.Tensor] = None,
        steering_mode: str = 'multiplicative',
    ):
        super().__init__()
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        self.token_mask = token_mask  # Shape: (seq_len,) bool, True = steer this position
        if steering_mode not in ('multiplicative', 'additive'):
            raise ValueError(f"steering_mode must be 'multiplicative' or 'additive', got {steering_mode}")
        self.steering_mode = steering_mode

        # Determine steering mode
        if alpha_vector is not None:
            if steering_mode != 'multiplicative':
                raise ValueError("alpha_vector is only supported with steering_mode='multiplicative'")
            self.mode = 'full_vector'
            self.alpha_vector = alpha_vector
        elif steering_features is not None and alpha is not None:
            self.mode = 'sparse'
            self.steering_features = steering_features
            self.alpha = alpha
        else:
            raise ValueError("Must provide either alpha_vector OR (steering_features + alpha)")

        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook on the specified layer."""
        def steering_hook(module, input, output):
            hidden_states = output[0]  # Shape: (batch, seq_len, hidden_size)
            steered_states = self._apply_sae_steering(hidden_states)
            return (steered_states,) + output[1:]

        target_layer = self.model.bert.encoder.layer[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(steering_hook)

    def _apply_sae_steering(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply SAE-based steering to hidden states.

        Multiplicative mode:
        1. Encode: features = SAE.encode(x)
        2. Reconstruct: x_recon = features @ decoder
        3. error = x - x_recon
        4. features_steered *= alpha_vector (or scale specific features)
        5. x_recon_steered = features_steered @ decoder
        6. Return: x_recon_steered + error

        Additive mode (sparse only):
        - h_steered[mask] = h[mask] + alpha * decoder_row[:, feat_id] (summed over
          listed features). No encode/decode: algebraically equivalent to
          encode -> features += alpha * e_{feat_id} -> decode -> + error.

        When token_mask is set, only masked positions are steered; non-masked
        positions pass through with original hidden states.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        dtype = hidden_states.dtype

        flat_states = hidden_states.reshape(-1, hidden_size)

        # Build mask: all-True if no token_mask provided, otherwise expand for batch
        if self.token_mask is not None:
            mask = self.token_mask.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        else:
            mask = torch.ones(batch_size * seq_len, dtype=torch.bool, device=flat_states.device)

        masked_states = flat_states[mask]

        if self.steering_mode == 'additive':
            # Direct hidden-state edit; skip encode/decode entirely.
            decoder_weights = self.sae.decoder.weight  # (hidden_size, n_features)
            with torch.no_grad():
                delta = torch.zeros_like(masked_states, dtype=torch.float32)
                for feat_id in self.steering_features:
                    delta = delta + self.alpha * decoder_weights[:, feat_id].float()
            x_steered_masked = (masked_states.float() + delta).to(dtype)
        else:
            with torch.no_grad():
                features = self.sae.encode(masked_states.float())
                x_reconstructed, _ = self.sae(masked_states.float())
                error = masked_states.float() - x_reconstructed

            if self.mode == 'full_vector':
                features_steered = features * self.alpha_vector
            else:
                features_steered = features.clone()
                for feat_id in self.steering_features:
                    features_steered[:, feat_id] *= self.alpha

            decoder_weights = self.sae.decoder.weight.T
            x_reconstructed_steered = features_steered @ decoder_weights
            if self.sae.decoder.bias is not None:
                x_reconstructed_steered = x_reconstructed_steered + self.sae.decoder.bias
            if self.sae.use_b_pre:
                x_reconstructed_steered = x_reconstructed_steered + self.sae.b_pre

            x_steered_masked = (x_reconstructed_steered + error).to(dtype)

        output = flat_states.clone()
        output[mask] = x_steered_masked

        return output.reshape(batch_size, seq_len, hidden_size)

    def set_token_mask(self, token_mask: Optional[torch.Tensor]):
        """Update the token mask without recreating the model."""
        self.token_mask = token_mask

    def update_alpha_vector(self, alpha_vector: torch.Tensor):
        """Update the alpha vector (for optimization)."""
        if self.mode != 'full_vector':
            raise ValueError("Can only update alpha_vector in full_vector mode")
        self.alpha_vector = alpha_vector

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Forward pass with steering applied."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
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
        attention_mask: np.ndarray,
        layer_idx: int = 12,
        device: str = 'cuda',
        batch_size: int = 4
    ):
        self.model = model
        self.sae = sae
        self.attention_mask = attention_mask
        self.layer_idx = layer_idx
        self.device = device
        self.batch_size = batch_size

        self.attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(device)

    def run_steered(self, adata, steering_features, alpha, cell_indices=None,
                     token_mask=None, steering_mode='multiplicative'):
        """Run steered forward pass with a SINGLE forward pass per batch.

        Notes on semantics of alpha=0:
          - multiplicative: features set to 0 (ABLATION).
          - additive: no-op (h unchanged); useful as a baseline control.
        """
        if cell_indices is None:
            cell_indices = np.arange(adata.n_obs)

        if steering_mode == 'multiplicative' and alpha == 0:
            print(f"Running ABLATION (alpha=0, features turned OFF)...")
        else:
            print(f"Running STEERED pass ({steering_mode}, alpha={alpha})...")

        steering_model = SAESteeringModel(
            model=self.model,
            sae=self.sae,
            layer_idx=self.layer_idx,
            steering_features=steering_features,
            alpha=alpha,
            token_mask=token_mask,
            steering_mode=steering_mode,
        )

        embedding_hook = ActivationHook(self.model.bert.encoder.ln)

        outputs_list = []
        embeddings_list = []

        try:
            with torch.no_grad():
                pbar = tqdm(range(0, len(cell_indices), self.batch_size), desc=f"Alpha={alpha}")
                for i in pbar:
                    batch_idx = cell_indices[i:i+self.batch_size]
                    batch_counts = adata[batch_idx].X
                    if hasattr(batch_counts, 'toarray'):
                        batch_counts = batch_counts.toarray()

                    batch_processed = preprocess_counts(batch_counts, device=self.device)

                    seq_len = batch_processed.shape[1]
                    pbar.set_postfix({"seq_len": seq_len})

                    batch_attn_mask = self.attn_mask_tensor.repeat(batch_processed.shape[0], 1)
                    depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=self.device)
                    batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

                    output = steering_model(
                        input_ids=batch_processed,
                        attention_mask=batch_attn_mask
                    )

                    last_hidden = embedding_hook.get_activations()

                    last_hidden = last_hidden[:, :-2, :]
                    last_hidden = last_hidden[:, self.attention_mask.astype(bool), :]
                    cell_embeddings = last_hidden.mean(dim=1)

                    logits = output.logits[:, :-2, :].squeeze(-1)
                    logits_filtered = logits[:, self.attention_mask.astype(bool)]

                    outputs_list.append(logits_filtered.cpu())
                    embeddings_list.append(cell_embeddings.cpu())

                    del output, last_hidden, batch_processed, batch_attn_mask
        finally:
            embedding_hook.remove()
            steering_model.remove_hook()
            torch.cuda.empty_cache()

        return {
            'logits': torch.cat(outputs_list, dim=0),
            'embeddings': torch.cat(embeddings_list, dim=0),
            'cell_indices': cell_indices,
            'alpha': alpha,
            'steering_features': steering_features,
            'steering_mode': steering_mode,
        }

    def run_experiment(self, config: SteeringConfig, adata, cell_indices=None):
        """Run complete steering experiment across multiple alphas."""
        print("="*70)
        print(f"STEERING EXPERIMENT: {config.name}")
        print("="*70)
        if config.description:
            print(f"Description: {config.description}")
        print(f"Steering {len(config.steering_features)} features: {config.steering_features}")
        print(f"Alpha values: {config.alphas}")
        print(f"Layer: {config.layer_idx} (0-indexed)")
        print("="*70)

        results = {}

        for alpha in config.alphas:
            results[alpha] = self.run_steered(
                adata,
                config.steering_features,
                alpha,
                cell_indices
            )

        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        print(f"Results keys: {list(results.keys())}")

        return results

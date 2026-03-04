"""Top-K Sparse Autoencoder model definition and loading utilities."""

import os
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder with optional pre-encoder centering bias."""

    def __init__(self, input_dim=640, expansion=4, k=32, use_b_pre=False):
        super().__init__()
        latent_dim = input_dim * expansion
        self.use_b_pre = use_b_pre
        if use_b_pre:
            self.b_pre = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        self.k = k
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def encode(self, x):
        """Encode input to sparse latent representation."""
        if self.use_b_pre:
            x = x - self.b_pre
        latents = self.encoder(x)
        topk_vals, topk_idx = latents.topk(self.k, dim=-1)
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(-1, topk_idx, topk_vals)
        return sparse_latents

    def forward(self, x):
        """Forward pass: encode to sparse latents, decode back."""
        sparse_latents = self.encode(x)
        recon = self.decoder(sparse_latents)
        if self.use_b_pre:
            recon = recon + self.b_pre
        return recon, sparse_latents


# Backward compatibility alias
TopKSAEWithBPre = lambda input_dim=640, expansion=4, k=32: TopKSAE(input_dim, expansion, k, use_b_pre=True)


def load_sae(sae_dir: str, device: str = 'cuda') -> TopKSAE:
    """Load trained SAE model from checkpoint.

    Auto-detects use_b_pre from checkpoint metadata or state_dict keys.

    Args:
        sae_dir: Path to SAE directory containing topk_sae.pt
        device: Device to load model to

    Returns:
        Loaded TopKSAE model
    """
    checkpoint_path = os.path.join(sae_dir, 'topk_sae.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAE checkpoint not found at {checkpoint_path}")

    print(f"Loading SAE from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = checkpoint['input_dim']
    expansion = checkpoint['expansion']
    k = checkpoint['k']

    # Auto-detect b_pre: check metadata first, then state_dict keys
    if 'use_b_pre' in checkpoint:
        use_b_pre = checkpoint['use_b_pre']
    else:
        use_b_pre = 'b_pre' in checkpoint['model_state_dict']

    print(f"  Input dim: {input_dim}, Expansion: {expansion}x, Top-K: {k}, b_pre: {use_b_pre}")

    sae = TopKSAE(input_dim=input_dim, expansion=expansion, k=k, use_b_pre=use_b_pre)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae = sae.to(device)
    sae.eval()

    return sae

"""Analyze dead features in trained SAEs.

A "dead" feature is one that never activates across the dataset.
For TopK SAEs, this means features that never win a top-k slot.
"""

import os
import sys
import argparse
import torch
import h5py
import numpy as np
from tqdm import tqdm

from steering_utils import TopKSAE

BASE_DIR = "/biodata/nyanovsky/datasets/pbmc3k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def analyze_layer(layer, expansion=8, k=32):
    """Analyze dead features for a single layer's SAE."""
    input_dim = 640
    latent_dim = input_dim * expansion

    sae_dir = os.path.join(BASE_DIR, f"layer_{layer}", f"sae_k_{k}_{latent_dim}")
    sae_path = os.path.join(sae_dir, "topk_sae.pt")
    activations_path = os.path.join(BASE_DIR, f"layer_{layer}", f"layer{layer}_activations.h5")

    if not os.path.exists(sae_path):
        print(f"  SAE not found at {sae_path}")
        return None
    if not os.path.exists(activations_path):
        print(f"  Activations not found at {activations_path}")
        return None

    # Load SAE
    checkpoint = torch.load(sae_path, map_location=DEVICE)
    sae = TopKSAE(
        input_dim=checkpoint['input_dim'],
        expansion=checkpoint['expansion'],
        k=checkpoint['k']
    ).to(DEVICE)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()

    # Track feature activation counts
    n_features = sae.latent_dim
    feature_activation_count = np.zeros(n_features, dtype=np.int64)
    feature_activation_sum = np.zeros(n_features, dtype=np.float64)
    total_samples = 0

    with h5py.File(activations_path, 'r') as f:
        n_cells, n_genes, hidden_dim = f['activations'].shape

        for cell_idx in tqdm(range(n_cells), desc=f"Layer {layer}"):
            cell_activations = torch.from_numpy(
                f['activations'][cell_idx, :, :].astype(np.float32)
            ).to(DEVICE)

            with torch.no_grad():
                sparse_latents = sae.encode(cell_activations)

            # Count activations per feature (across all genes in this cell)
            activated = (sparse_latents != 0).sum(dim=0).cpu().numpy()
            feature_activation_count += activated

            # Sum activation magnitudes
            feature_activation_sum += sparse_latents.abs().sum(dim=0).cpu().numpy()

            total_samples += n_genes

    # Compute statistics
    dead_mask = feature_activation_count == 0
    n_dead = dead_mask.sum()
    pct_dead = 100 * n_dead / n_features

    # Features that activate but rarely (< 0.1% of samples)
    rare_threshold = 0.001 * total_samples
    rare_mask = (feature_activation_count > 0) & (feature_activation_count < rare_threshold)
    n_rare = rare_mask.sum()

    # Mean activation frequency for non-dead features
    alive_mask = ~dead_mask
    if alive_mask.sum() > 0:
        mean_activation_freq = feature_activation_count[alive_mask].mean() / total_samples
    else:
        mean_activation_freq = 0

    results = {
        'layer': layer,
        'n_features': n_features,
        'n_dead': int(n_dead),
        'pct_dead': pct_dead,
        'n_rare': int(n_rare),
        'total_samples': total_samples,
        'feature_activation_count': feature_activation_count,
        'feature_activation_sum': feature_activation_sum,
        'mean_activation_freq': mean_activation_freq,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze dead features in SAEs")
    parser.add_argument("--layers", type=int, nargs='+', default=[9, 12, 15],
                        help="Layers to analyze")
    parser.add_argument("--expansion", type=int, default=8, help="SAE expansion factor")
    parser.add_argument("--k", type=int, default=32, help="Top-K sparsity")
    args = parser.parse_args()

    print("="*60)
    print("SAE Dead Feature Analysis")
    print("="*60)
    print(f"Layers: {args.layers}")
    print(f"Expansion: {args.expansion}x, K={args.k}")
    print(f"Device: {DEVICE}")
    print("="*60)

    all_results = []

    for layer in args.layers:
        print(f"\nAnalyzing Layer {layer}...")
        results = analyze_layer(layer, args.expansion, args.k)
        if results:
            all_results.append(results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Layer':<8} {'Dead':<10} {'Dead %':<10} {'Rare (<0.1%)':<15} {'Mean Freq':<12}")
    print("-"*60)

    for r in all_results:
        print(f"{r['layer']:<8} {r['n_dead']:<10} {r['pct_dead']:<10.2f} {r['n_rare']:<15} {r['mean_activation_freq']:<12.4f}")

    # Save detailed results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    for r in all_results:
        layer = r['layer']
        np.savez(
            os.path.join(output_dir, f'../data/pbmc/dead_features_layer_{layer}.npz'),
            feature_activation_count=r['feature_activation_count'],
            feature_activation_sum=r['feature_activation_sum'],
            n_dead=r['n_dead'],
            n_rare=r['n_rare'],
            total_samples=r['total_samples']
        )

    print(f"\nDetailed results saved to data/pbmc/dead_features_layer_*.npz")


if __name__ == "__main__":
    main()

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze scale/sparsity of SAE features using Participation Ratio')
    parser.add_argument('--layer', type=int, default=12, help='Layer to analyze')
    parser.add_argument('--expansion', type=int, default=8, help='Expansion factor')
    parser.add_argument('--k', type=int, default=32, help='Top-K sparsity')
    parser.add_argument('--online', action='store_true',
                        help='Use online SAE directory (sae_k_{k}_{latent_dim}_online)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Construct path
    latent_dim = 640 * args.expansion
    sae_suffix = "_online" if args.online else ""
    base_dir = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}/sae_k_{args.k}_{latent_dim}{sae_suffix}"
    interp_dir = f"{base_dir}/interpretations_filter_zero_expressed"
    matrix_path = os.path.join(interp_dir, 'feature_gene_matrix.npy')
    plot_dir = os.path.join(os.path.dirname(__file__), f"../plots/sae/layer_{args.layer}")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Analyzing Feature Scale for Layer {args.layer}...")
    
    if not os.path.exists(matrix_path):
        print(f"Error: Matrix not found at {matrix_path}")
        return

    # 1. Load Data
    # Matrix shape: [n_features, n_genes]
    print("Loading feature-gene matrix...")
    fg_matrix = np.load(matrix_path)
    n_features, n_genes = fg_matrix.shape
    print(f"Shape: {n_features} features x {n_genes} genes")

    # 2. Compute Participation Ratio (PR)
    print("Computing Participation Ratio...")
    
    # Square the activations to get "energy"
    energy = fg_matrix ** 2
    
    # Normalize to get probability distribution P_i for each feature
    # Sum over genes (axis 1)
    energy_sum = energy.sum(axis=1, keepdims=True)
    
    # Avoid division by zero for dead features
    energy_sum[energy_sum == 0] = 1.0 
    
    probs = energy / energy_sum
    
    # PR = 1 / Sum(P_i^2)
    # Sum of squares of probabilities (Inverse Simpson Index)
    ipr = (probs ** 2).sum(axis=1)
    
    # Handle dead features (ipr will be 0 if energy was 0)
    pr = np.zeros_like(ipr)
    mask = ipr > 0
    pr[mask] = 1.0 / ipr[mask]
    
    # 3. Statistics
    print("\nFeature Scale Statistics (Effective Gene Count):")
    print(f"Min PR: {pr.min():.2f}")
    print(f"Max PR: {pr.max():.2f}")
    print(f"Mean PR: {pr.mean():.2f}")
    print(f"Median PR: {np.median(pr):.2f}")
    
    # 4. Visualization
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(pr, bins=100, log_scale=(True, False)) # Log x-axis helps seeing the range
    plt.xlabel('Effective Gene Count (PR)')
    plt.ylabel('Count')
    plt.title('Distribution of Feature Scales')
    plt.axvline(np.median(pr), color='r', linestyle='--', label=f'Median: {np.median(pr):.1f}')
    plt.legend()
    
    # Cumulative Distribution
    plt.subplot(1, 2, 2)
    sns.ecdfplot(pr)
    plt.xlabel('Effective Gene Count (PR)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Feature Scales')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'feature_scale_distribution.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    
    # 5. Examples
    print("\nExamples of Specific Features (Low PR):")
    indices = np.argsort(pr)
    
    # Filter out dead features (PR=0)
    valid_indices = [i for i in indices if pr[i] >= 1.0]
    
    for i in valid_indices[:10]:
        print(f"Feature {i:<4} | PR: {pr[i]:.2f}")
        
    print("\nExamples of Pathway Features (PR ~ 20-50):")
    # Find features closest to PR=30
    mid_indices = [i for i in valid_indices if 20 < pr[i] < 50]
    for i in mid_indices[:10]:
        print(f"Feature {i:<4} | PR: {pr[i]:.2f}")

    print("\nExamples of Global Features (High PR):")
    for i in valid_indices[-10:]:
        print(f"Feature {i:<4} | PR: {pr[i]:.2f}")

    # Save PRs to file for later use
    np.save(os.path.join(interp_dir, 'feature_participation_ratios.npy'), pr)
    print(f"\nPR values saved to {os.path.join(interp_dir, 'feature_participation_ratios.npy')}")

if __name__ == "__main__":
    main()

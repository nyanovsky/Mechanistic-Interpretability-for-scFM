import os
import sys
import argparse
import re
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import load_gene_names

def parse_args():
    parser = argparse.ArgumentParser(description='Detailed analysis of feature scale vs activation magnitude (gene-wise)')
    parser.add_argument('--layer', type=int, default=12, help='Layer to analyze')
    parser.add_argument('--expansion', type=int, default=8, help='Expansion factor')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling method used in SAE for fg matrix')
    parser.add_argument('--online', action='store_true',
                        help='Use online SAE directory (sae_k_{k}_{latent_dim}_online)')
    parser.add_argument('--dataset', default='pbmc3k', help='Dataset name for raw data path')
    return parser.parse_args()


def load_decoder_weights(sae_dir):
    """Load decoder weights and return L2 norms per feature."""
    decoder_path = os.path.join(sae_dir, 'sae_decoder.pt')
    if os.path.exists(decoder_path):
        decoder_state = torch.load(decoder_path, map_location='cpu')
        W = decoder_state['weight']
    else:
        ckpt_path = os.path.join(sae_dir, 'topk_sae.pt')
        if not os.path.exists(ckpt_path):
            return None
        ckpt = torch.load(ckpt_path, map_location='cpu')
        W = ckpt['model_state_dict']['decoder.weight']
    
    if W.shape[0] == 640:
        norms = torch.norm(W, p=2, dim=0) 
    else:
        norms = torch.norm(W, p=2, dim=1) 
        
    return norms.detach().numpy()

def main():
    args = parse_args()
    
    latent_dim = 640 * args.expansion
    SAE_SUFFIX = "_online" if args.online else ""
    base_dir = f"/biodata/nyanovsky/datasets/{args.dataset}/layer_{args.layer}/sae_k_32_{latent_dim}{SAE_SUFFIX}"
    pooling = "custom_pooling" if args.pooling == "custom" else "filter_zero_expressed"
    interp_dir = f"{base_dir}/interpretations_{pooling}"
    raw_data_path = os.path.join(os.path.dirname(__file__), "../../data/pbmc/pbmc3k_raw.h5ad")
    
    print(f"Analyzing Feature Scale & Peak Gene Activation for Layer {args.layer}...")

    # 1. Load Matrix
    fg_path = os.path.join(interp_dir, 'feature_gene_matrix.npy') 
    if not os.path.exists(fg_path):
        print("Error: Missing feature-gene matrix. Run interpret_sae.py first.")
        return

    print("Loading matrix...")
    fg_matrix = np.load(fg_path) 
    n_features, n_genes = fg_matrix.shape
    
    # 2. Compute Metrics
    print("Computing metrics...")
    energy = fg_matrix ** 2
    energy_sum = energy.sum(axis=1, keepdims=True)
    energy_sum[energy_sum == 0] = 1.0
    probs = energy / energy_sum
    ipr = (probs ** 2).sum(axis=1)
    pr = np.zeros_like(ipr)
    mask = ipr > 0
    pr[mask] = 1.0 / ipr[mask]
    
    max_gene_activation = fg_matrix.max(axis=1)
    
    # 3. Load Decoder Norms
    print("Loading decoder norms...")
    norms = load_decoder_weights(base_dir)
    
    # 4. Check Annotations (both presence and significance)
    print("Checking for annotation folders and significant enrichment...")
    annotated_dirs = glob(os.path.join(interp_dir, 'feature_*_enrichr'))
    annotated_ids = set()
    significant_ids = set()

    for d in annotated_dirs:
        match = re.search(r'feature_(\d+)_enrichr', d)
        if not match:
            continue
        feat_id = int(match.group(1))
        annotated_ids.add(feat_id)

        # Check if any GO enrichment file has significant results (p < 0.05)
        txt_files = glob(os.path.join(d, "*.txt"))
        has_significant = False
        for txt in txt_files:
            try:
                df_go = pd.read_csv(txt, sep='\t')
                if 'Adjusted P-value' in df_go.columns and (df_go['Adjusted P-value'] < 0.05).any():
                    has_significant = True
                    break
            except:
                continue

        if has_significant:
            significant_ids.add(feat_id)

    print(f"Found {len(annotated_ids)} annotated features, {len(significant_ids)} with significant enrichment")
    
    # 5. Build DataFrame
    data = {
        'FeatureID': range(n_features),
        'ParticipationRatio': pr,
        'MaxGeneActivation': max_gene_activation,
        'IsAnnotated': [i in annotated_ids for i in range(n_features)],
        'Significant': [i in significant_ids for i in range(n_features)]
    }
    if norms is not None:
        data['DecoderNorm'] = norms

    df = pd.DataFrame(data)

    # Create AnnotationStatus column for plotting
    def get_status(row):
        if row['Significant']:
            return 'Significant'
        elif row['IsAnnotated']:
            return 'Annotated (not sig.)'
        else:
            return 'Not annotated'
    df['AnnotationStatus'] = df.apply(get_status, axis=1)
    df = df[df['ParticipationRatio'] >= 1.0].copy()
    
    # Bins
    bins = [0, 5, 20, 100, 500, 10000]
    labels = ['Specific (<5)', 'Small (5-20)', 'Pathway (20-100)', 'Broad (100-500)', 'Global (>500)']
    df['ScaleBin'] = pd.cut(df['ParticipationRatio'], bins=bins, labels=labels)
    
    # Save CSV
    csv_path = os.path.join(interp_dir, 'feature_scale_analysis_detailed.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved analysis to {csv_path}")
    
    # 6. Plotting
    plot_dir = os.path.join(os.path.dirname(__file__), f"../../plots/sae/layer_{args.layer}/{pooling}")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: PR vs Max Activation
    plt.figure(figsize=(10, 6))
    # Define custom color palette: red (not annotated), orange (annotated but not sig), green (significant)
    palette = {'Not annotated': '#d62728', 'Annotated (not sig.)': '#ff7f0e', 'Significant': '#2ca02c'}
    sns.scatterplot(data=df, x='ParticipationRatio', y='MaxGeneActivation',
                    hue='AnnotationStatus', hue_order=['Not annotated', 'Annotated (not sig.)', 'Significant'],
                    palette=palette, alpha=0.6, s=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Effective Gene Count (PR)')
    plt.ylabel('Peak Gene Activation')
    plt.title('Feature Scale vs Peak Gene Activation')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Annotation Status', loc='best')
    plt.savefig(os.path.join(plot_dir, 'feature_scale_vs_peak_activation.png'), dpi=150)
    
    # Plot 2: Representative Bars
    print("Generating representative plots...")
    gene_names = load_gene_names(raw_data_path)
    reps = []
    for label in labels:
        subset = df[df['ScaleBin'] == label]
        if not subset.empty:
            median_pr = subset['ParticipationRatio'].median()
            closest_idx = (subset['ParticipationRatio'] - median_pr).abs().idxmin()
            feat_id = int(subset.loc[closest_idx, 'FeatureID'])
            reps.append((label, feat_id))
    
    fig, axes = plt.subplots(len(reps), 1, figsize=(10, 4 * len(reps)))
    if len(reps) == 1: axes = [axes]
    
    for i, (label, feat_id) in enumerate(reps):
        ax = axes[i]
        activations = fg_matrix[feat_id]
        top_indices = np.argsort(activations)[::-1][:20]
        top_genes = gene_names[top_indices]
        top_values = activations[top_indices]
        
        ax.bar(range(20), top_values, color='steelblue')
        ax.set_xticks(range(20))
        ax.set_xticklabels(top_genes, rotation=45, ha='right')
        ax.set_title(f"{label}: Feature {feat_id} (PR={df.loc[df['FeatureID']==feat_id, 'ParticipationRatio'].values[0]:.1f})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_scale_representatives.png'), dpi=150)
    
    # Plot 3: PR vs Decoder Norm (NEW)
    if norms is not None:
        plt.figure(figsize=(10, 6))
        palette = {'Not annotated': '#d62728', 'Annotated (not sig.)': '#ff7f0e', 'Significant': '#2ca02c'}
        sns.scatterplot(data=df, x='ParticipationRatio', y='DecoderNorm',
                        hue='AnnotationStatus', hue_order=['Not annotated', 'Annotated (not sig.)', 'Significant'],
                        palette=palette, alpha=0.6, s=15)
        plt.xscale('log')
        plt.xlabel('Effective Gene Count (PR)')
        plt.ylabel('Decoder Vector L2 Norm')
        plt.title('Participation Ratio vs Decoder Norm')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Annotation Status', loc='best')
        plt.savefig(os.path.join(plot_dir, 'feature_scale_vs_norm.png'), dpi=150)
        print(f"Norm plot saved to {plot_dir}/feature_scale_vs_norm.png")
        
        print("\nCorrelations:")
        print(df[['ParticipationRatio', 'DecoderNorm', 'MaxGeneActivation']].corr())

if __name__ == "__main__":
    main()

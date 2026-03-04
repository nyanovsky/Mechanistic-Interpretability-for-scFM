import os
import argparse
import pandas as pd
import glob
import numpy as np

# CLI arguments
parser = argparse.ArgumentParser(description='Summarize significant SAE features')
parser.add_argument('interp_dir', type=str, help='Path to SAE interpretation directory')
parser.add_argument('--layer', type=int, default=12,
                    help='AIDO.Cell layer (e.g., 9, 12, 15)')
parser.add_argument('--expansion', type=int, default=8,
                    help='SAE expansion factor (4 or 8)')
parser.add_argument('--k', type=int, default=32,
                    help='Top-K sparsity')
parser.add_argument('--online', action='store_true',
                        help='Use online SAE directory (sae_k_{k}_{latent_dim}_online)')
parser.add_argument('--pooling', type=str, default='mean')
args = parser.parse_args()

# Build paths from arguments
INPUT_DIM = 640
latent_dim = INPUT_DIM * args.expansion

base_dir = args.interp_dir

print(f"Analyzing: Layer {args.layer}, {args.expansion}x expansion (K={args.k})")
print(f"Directory: {base_dir}\n")

feature_dirs = glob.glob(os.path.join(base_dir, "feature_*_enrichr"))
corr_file = os.path.join(base_dir, "feature_celltype_correlations.csv")

# Load correlations if available
try:
    corr_df = pd.read_csv(corr_file, index_col=0)
except:
    corr_df = None

print(f"{ 'Feature':<10} | { 'Top Cell Type (r)':<25} | { 'Top Significant GO Terms'}")
print("-" * 120)

significant_count = 0

for fdir in feature_dirs:
    feat_id = fdir.split('_')[-2]
    
    # Collect all significant terms from all files in this folder
    all_sig_terms = []
    txt_files = glob.glob(os.path.join(fdir, "*.txt"))
    
    for txt in txt_files:
        try:
            df = pd.read_csv(txt, sep='\t')
            if 'Adjusted P-value' in df.columns:
                sig_df = df[df['Adjusted P-value'] < 0.05]
                for _, row in sig_df.iterrows():
                    all_sig_terms.append((row['Term'], row['Adjusted P-value'], row['Gene_set']))
        except:
            continue
    
    if not all_sig_terms:
        continue
        
    significant_count += 1
    
    # Sort by p-value (ascending)
    all_sig_terms.sort(key=lambda x: x[1])
    
    # Get top cell type correlation
    cell_info = "N/A"
    if corr_df is not None:
        try:
            # Try string first
            if feat_id in corr_df.index:
                row = corr_df.loc[feat_id]
            # Try integer
            elif int(feat_id) in corr_df.index:
                row = corr_df.loc[int(feat_id)]
            else:
                row = None
        except:
            row = None
            
        if row is not None:
            top_type = row.abs().idxmax()
            top_val = row[top_type]
            cell_info = f"{top_type} ({top_val:.2f})"

    # Format output
    top_terms = all_sig_terms[:3]
    terms_str = " || ".join([f"{t[0]} ({t[2]})" for t in top_terms])
    
    print(f"{feat_id:<10} | {cell_info:<25} | {terms_str}")

print("-" * 120)
print(f"Total significant features found: {significant_count}")

print("-" * 120)
print(f"Top feature by cell type correlation:")
if corr_df is not None:
    for cell_type in corr_df.columns:
        top_feat = corr_df[cell_type].abs().idxmax()
        corr_val = corr_df.loc[top_feat, cell_type]
        print(f"  {cell_type}: Feature {top_feat} (r={corr_val:.3f})")

print("-" * 120)
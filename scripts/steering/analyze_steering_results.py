"""
Analyze results from SAE steering experiments.

1. Calculates significantly DE genes for ALL alphas using Paired T-tests.
   - Comparison 1: Steered (Alpha=X) vs Baseline (No Steering)
   - Comparison 2: Steered (Alpha=X) vs Ablated (Alpha=0)
2. Filters for expressed genes to remove model artifacts.
3. Runs GO enrichment for top verified DE genes.
4. Plots trajectories of top DE genes across conditions.
"""

import os
import sys
import argparse
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import get_expressed_genes_mask, compute_de_genes
from utils.go_utils import run_go_enrichment

from aido_cell.utils import align_adata

# --- Constants ---
RAW_DATA_FILE = "../../data/pbmc/pbmc3k_raw.h5ad"
PROCESSED_DATA_FILE = "../../data/pbmc/pbmc3k_processed.h5ad"


# --- Processing and Analysis Functions ---

def process_steered_logits(logits_tensor, attention_mask):
    """
    Convert steering logits to numpy array.

    Note: As of the latest version, steering_utils.py saves already-processed logits
    (depth tokens removed, filtered by attention mask), so no processing needed here.
    """
    # Logits are already processed, just convert to numpy
    return logits_tensor.float().cpu().numpy()

def analyze_de_paired(logits_cond, logits_ref, gene_names, expr_mask, label, output_dir, top_n=100, background=None):
    """Identify robustly DE genes using a Paired T-Test, sorted by Effect Size."""
    print(f"\n--- Analyzing: {label} ---")

    # Subset to expressed genes
    cond_expr = logits_cond[:, expr_mask]
    ref_expr = logits_ref[:, expr_mask]
    names_expr = gene_names[expr_mask]

    top_up_genes, top_down_genes, de_stats = compute_de_genes(cond_expr, ref_expr, names_expr, top_n=top_n)

    sig_count = de_stats['sig_mask'].sum()
    if sig_count > 0:
        mean_diff = de_stats['mean_diff']
        sig_sorted = de_stats['sig_sorted_indices']
        median_abs_logfc = np.median(np.abs(mean_diff[de_stats['sig_mask']]))
        print(f"  Significant genes: {sig_count}/{len(names_expr)} ({100*sig_count/len(names_expr):.1f}%)")
        print(f"  Effect sizes (logFC): max_up={mean_diff[sig_sorted[0]]:.4f}, max_down={mean_diff[sig_sorted[-1]]:.4f}, median_abs={median_abs_logfc:.4f}")
    else:
        print(f"  No significant genes found")

    print(f"  Top Up (by Magnitude): {top_up_genes[:10]}")
    print(f"  Top Down (by Magnitude): {top_down_genes[:10]}")

    # Run GO
    if top_up_genes:
        run_go_enrichment(top_up_genes, output_dir, background=background, identifier=f"Up_{label}")
    if top_down_genes:
        run_go_enrichment(top_down_genes, output_dir, background=background, identifier=f"Down_{label}")

    return top_up_genes, top_down_genes

def plot_trajectories_side_by_side(means_dict, gene_names, top_up, top_down, label, output_dir):
    """
    Plot top up/down gene trajectories side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    alphas = sorted([k for k in means_dict.keys() if k != 'baseline'])
    conditions = ['baseline'] + alphas
    x_points = range(len(conditions))
    x_labels = ['Base'] + [str(a) for a in alphas]
    
    # Plot Up
    if top_up:
        for gene in top_up[:10]:
            if gene in gene_names:
                idx = np.where(gene_names == gene)[0][0]
                y_vals = [means_dict[c][idx] for c in conditions]
                axes[0].plot(x_points, y_vals, marker='o', label=gene)
        axes[0].set_title(f"Top 10 Upregulated ({label})")
        axes[0].set_xticks(x_points)
        axes[0].set_xticklabels(x_labels)
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[0].grid(True, alpha=0.3)

    # Plot Down
    if top_down:
        for gene in top_down[:10]:
            if gene in gene_names:
                idx = np.where(gene_names == gene)[0][0]
                y_vals = [means_dict[c][idx] for c in conditions]
                axes[1].plot(x_points, y_vals, marker='o', label=gene)
        axes[1].set_title(f"Top 10 Downregulated ({label})")
        axes[1].set_xticks(x_points)
        axes[1].set_xticklabels(x_labels)
        axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"combined_trajectories_{label}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Saved combined plot to {output_dir}/{filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steering_file', required=True)
    parser.add_argument('--baseline_file', required=True)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--background', action='store_true',
                        help='Use expressed genes as background for GO enrichment (gp.enrich instead of gp.enrichr)')
    args = parser.parse_args()

    PLOT_DIR = f"../../plots/sae/layer_12/steering_analysis/{args.experiment_name}"
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Load data and filter cells
    print("Loading data and metadata...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, RAW_DATA_FILE)
    processed_path = os.path.join(script_dir, PROCESSED_DATA_FILE)

    adata_processed = ad.read_h5ad(processed_path)
    adata_raw = ad.read_h5ad(raw_path)

    common_cells = adata_raw.obs_names.intersection(adata_processed.obs_names)
    print(f"Filtered to {len(common_cells)} common cells.")

    adata_aligned, attention_mask = align_adata(adata_raw)
    gene_names = adata_aligned.var_names[attention_mask.astype(bool)].to_numpy()
    expr_mask = get_expressed_genes_mask(raw_path)

    # Set up background for GO enrichment if requested
    go_background = gene_names[expr_mask].tolist() if args.background else None
    if args.background:
        print(f"Using {len(go_background)} expressed genes as GO enrichment background")

    # 3. Load baseline
    print("Loading baseline logits...")
    baseline_adata = ad.read_h5ad(args.baseline_file)
    baseline_adata = baseline_adata[common_cells].copy()
    baseline_logits = baseline_adata.X
    
    # 4. Load steering results and align
    print("Loading steering results...")
    steering_payload = torch.load(args.steering_file, map_location='cpu')
    inner_results = steering_payload['results']
    alphas = sorted([k for k in inner_results.keys() if isinstance(k, (int, float))])
    
    full_data = {'baseline': baseline_logits}
    means_dict = {'baseline': baseline_logits.mean(axis=0)}
    
    # Pre-map cell names for alignment
    name_to_idx_common = {name: i for i, name in enumerate(common_cells)}

    for alpha in alphas:
        entry = inner_results[alpha]
        logits_tensor = entry['logits']
        cell_indices = entry['cell_indices']
        
        # Get names from full raw data
        cell_names_steered = adata_raw.obs_names[cell_indices]
        
        # Intersection and alignment
        steered_mask = []
        common_indices_for_steered = []
        for i, name in enumerate(cell_names_steered):
            if name in name_to_idx_common:
                steered_mask.append(i)
                common_indices_for_steered.append(name_to_idx_common[name])
        
        # Filter and process
        filtered_tensor = logits_tensor[steered_mask]
        processed_npy = process_steered_logits(filtered_tensor, attention_mask)
        
        # Reorder to common_cells order
        reordered = np.zeros((len(common_cells), len(gene_names)), dtype=np.float32)
        reordered[common_indices_for_steered] = processed_npy

        n_zero_cells = (reordered.sum(axis=1) == 0).sum()                                                                                            
        print(f"  DEBUG: {n_zero_cells}/{len(common_cells)} cells are all-zeros")                                                                    
        print(f"  DEBUG: len(common_cells)={len(common_cells)}, len(steered_mask)={len(steered_mask)}")
                
        full_data[alpha] = reordered
        means_dict[alpha] = reordered.mean(axis=0)
        print(f"  Alpha {alpha}: Aligned {len(steered_mask)} cells.")

    # 5. DE Analysis and Plotting for ALL comparisons
    for alpha in alphas:
        # A) Steered vs Baseline
        label = f"Alpha{alpha}_vs_Baseline"
        up, down = analyze_de_paired(full_data[alpha], full_data['baseline'], gene_names, expr_mask, label, PLOT_DIR, background=go_background)
        # Plot immediately for this comparison
        plot_trajectories_side_by_side(means_dict, gene_names, up, down, label, PLOT_DIR)

        # B) Steered vs Ablated (alpha=0)
        if alpha != 0 and 0 in full_data:
            label_abl = f"Alpha{alpha}_vs_Ablated"
            up_abl, down_abl = analyze_de_paired(full_data[alpha], full_data[0], gene_names, expr_mask, label_abl, PLOT_DIR, background=go_background)
            plot_trajectories_side_by_side(means_dict, gene_names, up_abl, down_abl, label_abl, PLOT_DIR)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
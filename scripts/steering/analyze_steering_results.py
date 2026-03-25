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

def analyze_de_paired(logits_cond, logits_ref, gene_names, expr_mask, label, go_output_dir, top_n=100, background=None):
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

    # Run GO enrichment (saved to go_output_dir, separate from plots)
    if top_up_genes:
        run_go_enrichment(top_up_genes, go_output_dir, background=background, identifier=f"Up_{label}")
    if top_down_genes:
        run_go_enrichment(top_down_genes, go_output_dir, background=background, identifier=f"Down_{label}")

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
    parser.add_argument('--plot_dir', type=str, default=None,
                        help='Directory for plots and reports (default: plots/sae/pbmc/layer_12/steering_analysis/<experiment_name>)')
    parser.add_argument('--background', default=True, action='store_true',
                        help='Use expressed genes as background for GO enrichment (gp.enrich instead of gp.enrichr)')
    parser.add_argument('--celltypes', type=str, nargs='+', default=None,
                        help='Analyze specific cell types separately (default: all cells together)')
    parser.add_argument('--processed_file', type=str,
                        default='data/pbmc/pbmc3k_processed.h5ad',
                        help='Processed .h5ad with celltype annotations (used with --celltypes)')
    args = parser.parse_args()

    PLOT_DIR = args.plot_dir

    # GO enrichment outputs go next to the steering file, not in plots
    steering_dir = os.path.dirname(os.path.abspath(args.steering_file))
    GO_OUTPUT_DIR = os.path.join(steering_dir, "DE_go")

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(GO_OUTPUT_DIR, exist_ok=True)
    
    # 1. Load data and filter cells
    print("Loading data and metadata...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, RAW_DATA_FILE)
    processed_path = os.path.join(script_dir, PROCESSED_DATA_FILE)

    adata_processed = ad.read_h5ad(processed_path)
    adata_raw = ad.read_h5ad(raw_path)

    common_cells = adata_raw.obs_names.intersection(adata_processed.obs_names)
    print(f"Filtered to {len(common_cells)} common cells.")

    # Build celltype masks if requested
    celltype_masks = {}
    if args.celltypes is not None:
        if 'celltype' in adata_processed.obs.columns:
            cell_type_col = 'celltype'
        elif 'louvain' in adata_processed.obs.columns:
            cell_type_col = 'louvain'
        else:
            raise ValueError("No celltype column found in processed file")
        ct_labels = adata_processed.obs.loc[common_cells, cell_type_col]
        for ct in args.celltypes:
            mask = (ct_labels == ct).values
            celltype_masks[ct] = mask
            print(f"  {ct}: {mask.sum()} cells")

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
    # If celltypes specified, run separately per celltype; otherwise run on all cells
    analysis_groups = celltype_masks if celltype_masks else {'all': None}

    for group_name, group_mask in analysis_groups.items():
        if group_mask is not None:
            print(f"\n{'='*70}")
            print(f"ANALYZING: {group_name}")
            print(f"{'='*70}")
            group_plot_dir = os.path.join(PLOT_DIR, group_name.replace(' ', '_'))
            group_go_dir = os.path.join(GO_OUTPUT_DIR, group_name.replace(' ', '_'))
            os.makedirs(group_plot_dir, exist_ok=True)
            os.makedirs(group_go_dir, exist_ok=True)
            group_data = {k: v[group_mask] for k, v in full_data.items()}
            group_means = {k: v[group_mask].mean(axis=0) for k, v in full_data.items()}
        else:
            group_plot_dir = PLOT_DIR
            group_go_dir = GO_OUTPUT_DIR
            group_data = full_data
            group_means = means_dict

        for alpha in alphas:
            # A) Steered vs Baseline
            label = f"Alpha{alpha}_vs_Baseline"
            up, down = analyze_de_paired(group_data[alpha], group_data['baseline'], gene_names, expr_mask, label, group_go_dir, background=go_background)
            plot_trajectories_side_by_side(group_means, gene_names, up, down, label, group_plot_dir)

            # B) Steered vs Ablated (alpha=0)
            if alpha != 0 and 0 in group_data:
                label_abl = f"Alpha{alpha}_vs_Ablated"
                up_abl, down_abl = analyze_de_paired(group_data[alpha], group_data[0], gene_names, expr_mask, label_abl, group_go_dir, background=go_background)
                plot_trajectories_side_by_side(group_means, gene_names, up_abl, down_abl, label_abl, group_plot_dir)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
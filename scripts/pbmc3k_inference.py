#!/usr/bin/env python3
"""
PBMC3K Inference with AIDO.Cell

Runs gene expression reconstruction on PBMC3K data using the pretrained
AIDO.Cell model with decoder head.

Output:
- Reconstructed expression stored in adata.layers['reconstructed']
- Input/output expression scatter plot
"""

import os
import sys
import torch
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

# Add the huggingface/aido.cell directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ModelGenerator/huggingface/aido.cell'))

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "genbio-ai/AIDO.Cell-100M"
INPUT_RAW_FILE = "../data/pbmc/pbmc3k_raw.h5ad"
INPUT_PROCESSED_FILE = "../data/pbmc/pbmc3k_processed.h5ad"
PLOT_FILE = "../plots/input_vs_output_expression.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Small batch size to prevent OOM

# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_RAW_FILE_PATH = os.path.join(script_dir, INPUT_RAW_FILE)
    INPUT_PROCESSED_FILE_PATH = os.path.join(script_dir, INPUT_PROCESSED_FILE)
    PLOT_FILE_PATH = os.path.join(script_dir, PLOT_FILE)

    print("\n" + "="*60)
    print("AIDO.Cell Gene Expression Reconstruction - PBMC3K")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*60 + "\n")

    # Load input data
    print("Loading PBMC3K data...")
    adata_raw = ad.read_h5ad(INPUT_RAW_FILE_PATH)
    adata_processed = ad.read_h5ad(INPUT_PROCESSED_FILE_PATH)

    common_cells = adata_raw.obs_names.intersection(adata_processed.obs_names)

    # Filter raw data to common cells and get cell types for those cells
    adata_raw = adata_raw[common_cells].copy()

    print(f'Loaded {adata_raw.n_obs} cells and {adata_raw.n_vars} genes from raw data\n')

    # Align data to AIDO.Cell gene set
    adata_aligned, attention_mask = align_adata(adata_raw)

    # Load model with decoder head
    print(f"\n{'='*60}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"{'='*60}\n")

    config = CellFoundationConfig.from_pretrained(MODEL_NAME)
    model = CellFoundationForMaskedLM.from_pretrained(MODEL_NAME, config=config)
    model = model.to(DEVICE)

    if DEVICE == "cuda":
        model = model.to(torch.bfloat16)
        print("Model converted to bfloat16 for Flash Attention")

    model.eval()
    print("Model loaded successfully\n")

    # Perform gene expression reconstruction
    print(f"\n{'='*60}")
    print("Performing gene expression reconstruction")
    print(f"Number of cells: {adata_aligned.n_obs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    n_cells = adata_aligned.n_obs
    all_reconstructions = []

    # Convert attention mask to torch tensor
    attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(DEVICE)

    # Process in batches
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_cells, BATCH_SIZE), desc="Reconstructing"):
            end_idx = min(start_idx + BATCH_SIZE, n_cells)

            # Get batch of raw counts
            batch_counts = adata_aligned.X[start_idx:end_idx]
            if hasattr(batch_counts, 'toarray'):
                batch_counts = batch_counts.toarray()

            # Preprocess counts (normalize to log1p(CPM), add depth tokens)
            batch_processed = preprocess_counts(batch_counts, device=DEVICE)

            # Expand attention mask to match batch size
            batch_attn_mask = attn_mask_tensor.repeat(batch_processed.shape[0], 1)

            # Add 2 positions to attention mask for depth tokens
            depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=DEVICE)
            batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

            # Forward pass through model
            outputs = model(
                input_ids=batch_processed,
                attention_mask=batch_attn_mask,
                return_dict=True
            )

            # Extract gene predictions from logits
            logits = outputs.logits  # [batch, 19266, 1]
            gene_logits = logits[:, :-2, :]  # Remove depth token predictions
            gene_logits = gene_logits[:, attention_mask.astype(bool), :]  # Apply attention mask
            gene_predictions = gene_logits.squeeze(-1)  # [batch, 19264]

            all_reconstructions.append(gene_predictions.float().cpu().numpy())

    # Concatenate all batches
    reconstructed = np.vstack(all_reconstructions)
    print(f"\nReconstructed expression shape: {reconstructed.shape}")
    print("Values are in log1p(CPM) space\n")

    # Get original data in log1p(CPM) space for comparison
    original_data = adata_aligned.X
    if hasattr(original_data, 'toarray'):
        original_data = original_data.toarray()

    original_data_processed = preprocess_counts(original_data, device='cpu').numpy()[:, :-2]  # Remove depth tokens
    original_data_processed = original_data_processed[:, attention_mask.astype(bool)]

    # Calculate MSE
    mse = np.mean((original_data_processed - reconstructed) ** 2)
    print(f"Mean Squared Error (log1p(CPM) space): {mse:.4f}\n")

    # Create scatter plot
    print("Creating input vs output expression scatter plot...")
    create_scatter_plot(original_data_processed, reconstructed, PLOT_FILE_PATH, mse)
    print(f"Plot saved to: {PLOT_FILE_PATH}\n")


def create_scatter_plot(original, reconstructed, save_path, mse):
    """Create a scatter plot of input vs output expression values."""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Flatten arrays and subsample for visualization
    x_flat = original.flatten()
    y_flat = reconstructed.flatten()

    # Subsample points (max 200k for performance)
    n_points = min(200000, len(x_flat))
    idx = np.random.choice(len(x_flat), n_points, replace=False)
    x_sample = x_flat[idx]
    y_sample = y_flat[idx]

    # Plot scatter
    ax.scatter(x_sample, y_sample, alpha=0.05, s=1, c='blue')

    # Add diagonal line
    max_val = max(x_sample.max(), y_sample.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')

    # Calculate overall correlation
    mask = (x_flat > 0) | (y_flat > 0)
    corr, _ = stats.pearsonr(x_flat[mask], y_flat[mask])

    ax.set_xlabel('Input Expression (log1p(CPM))', fontsize=14)
    ax.set_ylabel('Reconstructed Expression (log1p(CPM))', fontsize=14)
    ax.set_title(f'AIDO.Cell Reconstruction: Input vs Output\n'
                 f'MSE = {mse:.4f}, Pearson r = {corr:.4f}', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

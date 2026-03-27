import os
import sys
import torch
import torch.nn as nn
import anndata as ad
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils  # noqa: F401 – triggers AIDO.Cell path setup

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts


MODEL_NAME = "genbio-ai/AIDO.Cell-100M"
RAW_DATA_FILE = "data/pbmc/pbmc3k_raw.h5ad"
PROCESSED_DATA_FILE = "data/pbmc/pbmc3k_processed.h5ad"
OUTPUT_DIR = "/biodata/nyanovsky/datasets/pbmc3k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6  # Small due to memory constraints

def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, RAW_DATA_FILE)
    processed_data_path = os.path.join(script_dir, PROCESSED_DATA_FILE)
    output_path = os.path.join(OUTPUT_DIR, "pbmc3k_logits.h5ad")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    adata_processed = ad.read_h5ad(processed_data_path)
    adata_raw = ad.read_h5ad(raw_data_path)

    # Get common cells
    common_cells = adata_raw.obs_names.intersection(adata_processed.obs_names)
    adata_raw = adata_raw[common_cells].copy()

    # Align to AIDO.Cell vocab
    adata_aligned, attention_mask = align_adata(adata_raw)
    n_cells = adata_aligned.n_obs
    n_valid_genes = int(attention_mask.sum())
    valid_mask = attention_mask.astype(bool)

    print(f"Aligned data shape: {adata_aligned.shape}")
    print(f"Valid genes (in vocab): {n_valid_genes}")
    print(f"Total cells: {n_cells}")

    print(f"\nLoading model: {MODEL_NAME}")
    config = CellFoundationConfig.from_pretrained(MODEL_NAME)
    model = CellFoundationForMaskedLM.from_pretrained(MODEL_NAME, config=config)
    model = model.to(DEVICE)

    if DEVICE == "cuda":
        model = model.to(torch.bfloat16)

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    print("Model loaded and frozen")

    attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(DEVICE)
    torch_valid_mask = torch.from_numpy(valid_mask).to(DEVICE)

    gene_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, n_cells, BATCH_SIZE), desc="Processing cells"):

            batch_counts = adata_aligned[i:i+BATCH_SIZE].X.toarray()

            batch_processed = preprocess_counts(batch_counts, device=DEVICE)
            batch_attn_mask = attn_mask_tensor.repeat(batch_processed.shape[0], 1)
            depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=DEVICE)
            batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

            outputs = model(input_ids=batch_processed, attention_mask=batch_attn_mask, return_dict=True)
            
            # Extract logits: [batch, 19266, 1] -> [batch, 19264] (remove depth tokens)
            logits = outputs.logits[:, :-2, :].squeeze(-1).float()
            
            # Filter to valid genes only: [batch, n_valid_genes]
            gene_predictions = logits[:, torch_valid_mask]

            gene_preds.append(gene_predictions.cpu().numpy())

    gene_preds = np.concatenate(gene_preds, axis=0)
    print(f"Shape of gene_preds: {gene_preds.shape}")
    
    # Save to h5ad
    adata_out = ad.AnnData(X=gene_preds)
    adata_out.obs_names = adata_aligned.obs_names
    adata_out.var_names = adata_aligned.var_names[valid_mask]
    adata_out.write_h5ad(output_path)
    print(f"Saved logits to {output_path}")

if __name__ == "__main__":
    main()

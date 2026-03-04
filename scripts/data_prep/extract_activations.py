"""Extract layer 12 activations from AIDO.Cell for SAE training.

Saves per-gene embeddings to HDF5 in chunked format for memory-efficient streaming.
"""

import os
import sys
import torch
import h5py
import numpy as np
import anndata as ad
from tqdm import tqdm

import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils  # noqa: F401 – triggers AIDO.Cell path setup

from modelgenerator.tasks import Embed
from aido_cell.utils import align_adata

# Configuration
RAW_DATA_FILE = "../../data/pbmc/pbmc3k_raw.h5ad"
PROCESSED_DATA_FILE = "../../data/pbmc/pbmc3k_processed.h5ad"
BASE_OUTPUT_DIR = "/biodata/nyanovsky/datasets/pbmc3k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Small due to memory constraints

print(f"Device: {DEVICE}")


def main():
    parser = argparse.ArgumentParser(description="Extract layer activations from AIDO.Cell")
    parser.add_argument("--layer", type=int, default=12, help="Layer to extract (0-15)")
    args = parser.parse_args()
    
    TARGET_LAYER = args.layer
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"layer_{TARGET_LAYER}")
    OUTPUT_FILE = f"layer{TARGET_LAYER}_activations.h5"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, RAW_DATA_FILE)
    processed_data_path = os.path.join(script_dir, PROCESSED_DATA_FILE)

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    print(f"Target Layer: {TARGET_LAYER}")
    print(f"Output Path: {output_path}")

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

    print(f"Aligned data shape: {adata_aligned.shape}")
    print(f"Valid genes (in vocab): {n_valid_genes}")
    print(f"Total cells: {n_cells}")

    # Load model
    print("Loading AIDO.Cell model...")
    model = Embed.from_config({
        "model.backbone": "aido_cell_100m",
        "model.batch_size": BATCH_SIZE
    }).to(DEVICE).to(torch.bfloat16)

    model.configure_model()
    model.eval()

    # Get hidden size from a test forward pass
    with torch.no_grad():
        test_batch = torch.from_numpy(adata_aligned.X[0:1].toarray()).to(torch.bfloat16).to(DEVICE)
        test_transformed = model.transform({'sequences': test_batch})
        test_output = model(test_transformed).last_hidden_state
        hidden_dim = test_output.shape[-1]

    print(f"Hidden dimension: {hidden_dim}")

    # Setup hook to capture layer 12 activations
    layer_activations = {}

    def capture_layer_hook(module, input, output):
        # Output from transformer layer is typically (hidden_states, ...)
        if isinstance(output, tuple):
            layer_activations['hidden'] = output[0]
        else:
            layer_activations['hidden'] = output

    # Access the encoder layers - need to find the correct path
    # AIDO.Cell uses model.backbone.encoder.layers[TARGET_LAYER]
    encoder = model.backbone.encoder.encoder
    target_layer = encoder.layer[TARGET_LAYER]
    hook_handle = target_layer.register_forward_hook(capture_layer_hook)

    print(f"Registered hook on encoder layer {TARGET_LAYER}")

    # Create HDF5 file with chunked storage
    print(f"Creating HDF5 file at {output_path}...")

    # Estimate chunk size for efficient row access
    # Chunk by cells for efficient batched reading during SAE training
    chunk_cells = min(50, n_cells)  # 100 cells per chunk

    with h5py.File(output_path, 'w') as f:
        # Create dataset with chunking
        activations_ds = f.create_dataset(
            'activations',
            shape=(n_cells, n_valid_genes, hidden_dim),
            dtype='float32',
            chunks=(chunk_cells, n_valid_genes, hidden_dim)
            # No compression - faster writes, ~130GB uncompressed
        )

        # Store metadata
        f.attrs['n_cells'] = n_cells
        f.attrs['n_valid_genes'] = n_valid_genes
        f.attrs['hidden_dim'] = hidden_dim
        f.attrs['target_layer'] = TARGET_LAYER
        f.attrs['model'] = 'aido_cell_100m'

        # Store attention mask for reference
        f.create_dataset('attention_mask', data=attention_mask)

        # Store cell indices/names
        f.create_dataset('cell_names', data=np.array(adata_aligned.obs_names, dtype='S'))

        # Extract activations batch by batch
        print("Extracting layer 12 activations...")
        attn_mask_bool = attention_mask.astype(bool)
        n_batches = (n_cells + BATCH_SIZE - 1) // BATCH_SIZE

        with torch.no_grad():
            pbar = tqdm(range(0, n_cells, BATCH_SIZE),
                       total=n_batches,
                       desc="Extracting",
                       unit="batch")
            for batch_start in pbar:
                pbar.set_postfix(cells=f"{min(batch_start + BATCH_SIZE, n_cells)}/{n_cells}")
                batch_end = min(batch_start + BATCH_SIZE, n_cells)
                batch_counts = adata_aligned.X[batch_start:batch_end].toarray()

                batch_tensor = torch.from_numpy(batch_counts).to(torch.bfloat16).to(DEVICE)
                batch_transformed = model.transform({'sequences': batch_tensor})

                # Forward pass triggers the hook
                _ = model(batch_transformed)

                # Get captured activations and filter to valid genes
                hidden = layer_activations['hidden']
                # Remove depth tokens (last 2 positions) and filter by attention mask
                hidden = hidden[:, :-2, :]
                hidden = hidden[:, attn_mask_bool, :]

                # Convert to float32 and save
                hidden_np = hidden.float().cpu().numpy()
                activations_ds[batch_start:batch_end] = hidden_np

    # Cleanup
    hook_handle.remove()

    print(f"\nActivations saved to {output_path}")
    print(f"Shape: ({n_cells}, {n_valid_genes}, {hidden_dim})")

    # Verify file
    with h5py.File(output_path, 'r') as f:
        print(f"File size: {os.path.getsize(output_path) / 1e9:.2f} GB")
        print(f"Dataset shape: {f['activations'].shape}")
        print(f"Chunks: {f['activations'].chunks}")


if __name__ == "__main__":
    main()

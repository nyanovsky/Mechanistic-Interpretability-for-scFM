"""Compute feature-gene activation matrices from 100k CELLxGENE cells.

Loads AIDO.Cell + trained SAE, processes cells in batches, and accumulates:
- Mean-pooled feature-gene matrix (for PR computation, GO enrichment)
- Max-pooled feature-gene matrix (for peak activation analysis)

Only processes expressed genes (filters out zero-expression genes like olfactory receptors).

Usage:
    python scripts/compute_feature_matrices_online.py --layer 12
"""

import argparse
import os
import sys
import gc
import torch
import numpy as np
import anndata as ad
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import TopKSAE
from utils.data_utils import get_expressed_genes_mask

from modelgenerator.tasks import Embed
from aido_cell.utils import align_adata

# Speed optimizations
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Compute feature-gene activation matrices online")
    parser.add_argument("--data-file", type=str, required=True, help="Path to h5ad data file")
    parser.add_argument("--sae-file", type=str, required=True, help="Path to SAE checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output matrices")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--aido_batch", type=int, default=8, help="Cells per AIDO forward pass")
    parser.add_argument("--chunk_size", type=int, default=200, help="Cells per chunk")
    parser.add_argument("--min_mean_expr", type=float, default=0.01)
    parser.add_argument("--min_pct_cells", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start_chunk", type=int, default=0, help="Start from this chunk index")
    parser.add_argument("--end_chunk", type=int, default=None, help="Stop after this chunk index (exclusive)")
    parser.add_argument("--compute_cell_matrix", action="store_true", help="Also compute feature-cell matrices (~2GB each)")
    args = parser.parse_args()

    DATA_FILE = args.data_file
    SAE_FILE = args.sae_file
    OUTPUT_DIR = args.output_dir

    print(f"=== Computing Feature-Gene Matrices (Layer {args.layer}) ===")
    print(f"Device: {DEVICE}")

    # 1. Load data
    print("\n[1/5] Loading 100k subset...")
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        return

    adata = ad.read_h5ad(DATA_FILE)
    if not adata.obs_names.is_unique:
        print("  Warning: Duplicate cell names. Making unique...")
        adata.obs_names_make_unique()

    print(f"  Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    print("  Aligning to AIDO gene set...")
    adata_aligned, attn_mask = align_adata(adata)
    attn_mask_bool = attn_mask.astype(bool)
    n_genes_aido = attn_mask_bool.sum()
    print(f"  After AIDO alignment: {n_genes_aido} genes")

    # 2. Compute expressed genes mask
    print("\n[2/5] Computing expressed genes mask...")
    expressed_mask = get_expressed_genes_mask(
        adata_aligned, attn_mask_bool,
        min_mean_expr=args.min_mean_expr,
        min_pct_cells=args.min_pct_cells
    )
    n_expressed = expressed_mask.sum()
    print(f"  Expressed genes: {n_expressed} / {n_genes_aido} ({100*n_expressed/n_genes_aido:.1f}%)")

    # Get gene names for expressed genes
    all_gene_names = adata_aligned.var_names[attn_mask_bool].tolist()
    expressed_gene_names = [all_gene_names[i] for i in range(len(all_gene_names)) if expressed_mask[i]]

    # Create combined mask: AIDO valid genes AND expressed
    # We'll use this to index into the hidden states after AIDO forward pass
    expressed_indices = np.where(expressed_mask)[0]  # Indices within AIDO-valid genes

    # 3. Load models
    print("\n[3/5] Loading models...")

    # AIDO.Cell (load to CPU first, will move to GPU per chunk)
    print("  Loading AIDO.Cell...")
    aido = Embed.from_config({
        "model.backbone": "aido_cell_100m",
        "model.batch_size": args.aido_batch
    }).to(torch.bfloat16)
    aido.eval()

    # SAE
    print(f"  Loading SAE from {SAE_FILE}...")
    checkpoint = torch.load(SAE_FILE, map_location=DEVICE)

    # Handle checkpoint format
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        input_dim = checkpoint.get('input_dim', 640)
        expansion = checkpoint.get('expansion', args.expansion)
        k = checkpoint.get('k', args.k)
    else:
        state_dict = checkpoint
        input_dim = 640
        expansion = args.expansion
        k = args.k

    sae = TopKSAE(input_dim=input_dim, expansion=expansion, k=k).to(DEVICE)
    sae.load_state_dict(state_dict)
    sae.eval()

    n_features = sae.latent_dim
    print(f"  SAE: {input_dim} -> {n_features} (K={k})")

    # 4. Setup hook for layer activations
    print(f"\n[4/5] Setting up layer {args.layer} hook...")
    layer_outputs = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            layer_outputs['hidden'] = output[0]
        else:
            layer_outputs['hidden'] = output

    encoder = aido.backbone.encoder.encoder
    handle = encoder.layer[args.layer].register_forward_hook(hook)

    # 5. Process cells in chunks: AIDO extraction -> CPU storage -> SAE encoding
    # This allows AIDO to be offloaded between chunks, freeing GPU memory
    print(f"\n[5/5] Processing {adata.n_obs} cells (only {n_expressed} expressed genes)...")

    AIDO_BATCH_SIZE = args.aido_batch
    CHUNK_SIZE = args.chunk_size
    CHECKPOINT_EVERY = 50  # Save checkpoint every N chunks

    print(f"  AIDO batch size: {AIDO_BATCH_SIZE}")
    print(f"  Chunk size: {CHUNK_SIZE}")

    # Initialize or load accumulators
    start_chunk = args.start_chunk
    if args.resume and os.path.exists(args.resume):
        print(f"  Resuming from checkpoint: {args.resume}")
        ckpt = np.load(args.resume)
        feature_gene_sum = ckpt['feature_gene_sum']
        feature_gene_max = ckpt['feature_gene_max']
        n_cells_processed = int(ckpt['n_cells_processed'])
        start_chunk = max(start_chunk, int(ckpt['next_chunk']))
        print(f"  Resuming from chunk {start_chunk}, {n_cells_processed} cells processed")
    else:
        feature_gene_sum = np.zeros((n_features, n_expressed), dtype=np.float64)
        feature_gene_max = np.full((n_features, n_expressed), -np.inf, dtype=np.float64)
        n_cells_processed = 0

    # Feature-cell matrices (optional, ~2GB each)
    feature_cell_mean = None
    feature_cell_max = None
    if args.compute_cell_matrix:
        # Check if loaded from checkpoint
        if args.resume and os.path.exists(args.resume):
            ckpt = np.load(args.resume)
            if 'feature_cell_mean' in ckpt:
                feature_cell_mean = ckpt['feature_cell_mean']
                feature_cell_max = ckpt['feature_cell_max']
                print(f"  Loaded feature-cell matrices from checkpoint")
        if feature_cell_mean is None:
            print(f"  Allocating feature-cell matrices: {n_features} x {adata.n_obs}")
            feature_cell_mean = np.zeros((n_features, adata.n_obs), dtype=np.float32)
            feature_cell_max = np.zeros((n_features, adata.n_obs), dtype=np.float32)

    # Convert sparse to dense once
    if hasattr(adata_aligned.X, 'toarray'):
        print("  Converting sparse matrix to dense...")
        X_dense = adata_aligned.X.toarray().astype(np.float32)
    else:
        X_dense = adata_aligned.X.astype(np.float32)

    # Pre-compute final gene indices
    attn_valid_indices = np.where(attn_mask_bool)[0]
    final_gene_indices = attn_valid_indices[expressed_indices]

    n_chunks = (adata.n_obs + CHUNK_SIZE - 1) // CHUNK_SIZE
    end_chunk = args.end_chunk if args.end_chunk is not None else n_chunks
    end_chunk = min(end_chunk, n_chunks)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoint.npz')

    print(f"  Processing chunks {start_chunk} to {end_chunk} (of {n_chunks} total)")

    try:
        for chunk_idx in tqdm(range(start_chunk, end_chunk), desc="Chunks", initial=start_chunk, total=end_chunk):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, adata.n_obs)
            chunk_size = chunk_end - chunk_start

            # === PHASE 1: Extract AIDO activations for this chunk ===
            # AIDO on GPU, extract hidden states, store on CPU
            aido = aido.to(DEVICE)
            chunk_hidden_list = []

            with torch.no_grad():
                for batch_start in range(chunk_start, chunk_end, AIDO_BATCH_SIZE):
                    batch_end = min(batch_start + AIDO_BATCH_SIZE, chunk_end)
                    batch_counts = X_dense[batch_start:batch_end]

                    # Forward through AIDO
                    batch_tensor = torch.from_numpy(batch_counts).to(torch.bfloat16).to(DEVICE)
                    batch_transformed = aido.transform({'sequences': batch_tensor})
                    _ = aido(batch_transformed)

                    # Get layer activations and filter to expressed genes
                    hidden = layer_outputs['hidden']
                    hidden = hidden[:, :-2, :]  # Remove special tokens
                    hidden = hidden[:, final_gene_indices, :]  # [batch, n_expressed, hidden_dim]

                    # Store on CPU as float32
                    chunk_hidden_list.append(hidden.float().cpu())

            # Concatenate chunk activations on CPU
            chunk_hidden = torch.cat(chunk_hidden_list, dim=0)  # [chunk_size, n_expressed, hidden_dim]
            del chunk_hidden_list

            # === PHASE 2: Offload AIDO, process with SAE ===
            aido = aido.cpu()
            free_gpu()

            # Process SAE on GPU in batches
            SAE_BATCH_SIZE = 32  # Can be larger now that AIDO is offloaded
            chunk_cell_offset = chunk_start  # Global cell index for this chunk

            with torch.no_grad():
                for sae_batch_start in range(0, chunk_size, SAE_BATCH_SIZE):
                    sae_batch_end = min(sae_batch_start + SAE_BATCH_SIZE, chunk_size)
                    sae_batch_size = sae_batch_end - sae_batch_start

                    # Move batch to GPU
                    batch_hidden = chunk_hidden[sae_batch_start:sae_batch_end].to(DEVICE)

                    # Encode through SAE
                    hidden_flat = batch_hidden.reshape(-1, batch_hidden.shape[-1])
                    sparse_latents = sae.encode(hidden_flat)  # [batch * n_expressed, n_features]

                    # Reshape and permute for feature-gene: [batch, n_features, n_expressed]
                    sparse_latents = sparse_latents.reshape(sae_batch_size, n_expressed, n_features)
                    sparse_latents_t = sparse_latents.permute(0, 2, 1)

                    # Accumulate feature-gene (move to CPU immediately)
                    batch_sum = sparse_latents_t.sum(dim=0).cpu().numpy()
                    batch_max = sparse_latents_t.max(dim=0).values.cpu().numpy()

                    feature_gene_sum += batch_sum
                    feature_gene_max = np.maximum(feature_gene_max, batch_max)

                    # Accumulate feature-cell if requested
                    if args.compute_cell_matrix:
                        # Mean/max over genes (dim=2) -> [batch, n_features]
                        cell_mean = sparse_latents_t.mean(dim=2).cpu().numpy().T  # [n_features, batch]
                        cell_max = sparse_latents_t.max(dim=2).values.cpu().numpy().T

                        global_start = chunk_cell_offset + sae_batch_start
                        global_end = global_start + sae_batch_size
                        feature_cell_mean[:, global_start:global_end] = cell_mean
                        feature_cell_max[:, global_start:global_end] = cell_max

                    n_cells_processed += sae_batch_size

                    # Clean up
                    del batch_hidden, hidden_flat, sparse_latents, sparse_latents_t

            # Clean up chunk
            del chunk_hidden
            free_gpu()

            # Save checkpoint periodically
            if (chunk_idx + 1) % CHECKPOINT_EVERY == 0:
                print(f"\n  Saving checkpoint at chunk {chunk_idx + 1}/{n_chunks}...")
                ckpt_data = {
                    'feature_gene_sum': feature_gene_sum,
                    'feature_gene_max': feature_gene_max,
                    'n_cells_processed': n_cells_processed,
                    'next_chunk': chunk_idx + 1
                }
                if args.compute_cell_matrix:
                    ckpt_data['feature_cell_mean'] = feature_cell_mean
                    ckpt_data['feature_cell_max'] = feature_cell_max
                np.savez(checkpoint_path, **ckpt_data)

    finally:
        handle.remove()
        if 'aido' in dir():
            del aido
        del layer_outputs
        free_gpu()

    print(f"\n  Processed {n_cells_processed} cells")

    # Compute final matrices (already numpy)
    print("\nComputing final matrices...")
    feature_gene_mean = feature_gene_sum / n_cells_processed

    # Replace -inf with 0 in max matrix (for genes that were never activated)
    feature_gene_max[feature_gene_max == -np.inf] = 0

    print(f"  Mean matrix shape: {feature_gene_mean.shape}")
    print(f"  Max matrix shape: {feature_gene_max.shape}")
    print(f"  Mean matrix range: [{feature_gene_mean.min():.4f}, {feature_gene_mean.max():.4f}]")
    print(f"  Max matrix range: [{feature_gene_max.min():.4f}, {feature_gene_max.max():.4f}]")

    # Save
    mean_path = os.path.join(OUTPUT_DIR, 'feature_gene_matrix_mean.npy')
    max_path = os.path.join(OUTPUT_DIR, 'feature_gene_matrix_max.npy')

    np.save(mean_path, feature_gene_mean.astype(np.float32))
    np.save(max_path, feature_gene_max.astype(np.float32))

    # Save gene names for reference
    gene_names_path = os.path.join(OUTPUT_DIR, 'gene_names.txt')
    with open(gene_names_path, 'w') as f:
        f.write('\n'.join(expressed_gene_names))

    # Save expressed mask for reference (in case needed later)
    mask_path = os.path.join(OUTPUT_DIR, 'expressed_genes_mask.npy')
    np.save(mask_path, expressed_mask)

    # Save feature-cell matrices if computed
    if args.compute_cell_matrix:
        np.save(os.path.join(OUTPUT_DIR, 'feature_cell_matrix_mean.npy'), feature_cell_mean)
        np.save(os.path.join(OUTPUT_DIR, 'feature_cell_matrix_max.npy'), feature_cell_max)

    print(f"\nSaved to {OUTPUT_DIR}:")
    print(f"  - feature_gene_matrix_mean.npy ({n_features} x {n_expressed})")
    print(f"  - feature_gene_matrix_max.npy ({n_features} x {n_expressed})")
    print(f"  - gene_names.txt ({len(expressed_gene_names)} genes)")
    print(f"  - expressed_genes_mask.npy")
    if args.compute_cell_matrix:
        print(f"  - feature_cell_matrix_mean.npy ({n_features} x {adata.n_obs})")
        print(f"  - feature_cell_matrix_max.npy ({n_features} x {adata.n_obs})")


if __name__ == "__main__":
    main()

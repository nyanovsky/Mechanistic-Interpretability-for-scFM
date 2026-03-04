"""Train SAE on 100k CELLxGENE Subset (Online & RAM Optimized).

Supports resuming and chunk-based execution for long jobs.

Usage:
    # First run (0 to 42 chunks)
    python scripts/train_sae_online_100k.py --layer 12 --end_chunk 42 --save_name part1

    # Second run (42 to 84 chunks)
    python scripts/train_sae_online_100k.py --layer 12 --resume_from .../sae_part1.pt --start_chunk 42 --end_chunk 84 --save_name part2

    # Final run (84 to end)
    python scripts/train_sae_online_100k.py --layer 12 --resume_from .../sae_part2.pt --start_chunk 84
"""

import argparse
import os
import sys
import gc
import torch
import torch.nn.functional as F
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import TopKSAE
# aido_cell imports (path set up by utils/__init__.py)
from modelgenerator.tasks import Embed
from aido_cell.utils import align_adata

# Speed optimizations
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

# Configuration
DATA_FILE = "/biodata/nyanovsky/datasets/cell_x_gene/bone_marrow/bm_100k_subset.h5ad"
OUTPUT_BASE_DIR = "/biodata/nyanovsky/datasets/cell_x_gene/bone_marrow"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
AIDO_EXTRACT_BATCH = 8
CELLS_PER_BATCH = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def extract_chunk_activations(aido_model, adata, mask, indices, layer_idx, device):
    """Move AIDO to GPU -> Extract -> Move to CPU."""
    aido_model = aido_model.to(device)
    activations = []
    attn_mask_bool = mask.astype(bool)
    
    # Hook
    layer_outputs = {}
    def hook(module, input, output):
        if isinstance(output, tuple):
            layer_outputs['hidden'] = output[0]
        else:
            layer_outputs['hidden'] = output
            
    encoder = aido_model.backbone.encoder.encoder
    handle = encoder.layer[layer_idx].register_forward_hook(hook)
    
    try:
        with torch.no_grad():
            for i in range(0, len(indices), AIDO_EXTRACT_BATCH):
                batch_idx = indices[i : i + AIDO_EXTRACT_BATCH]
                
                # Handle sparse/dense
                if hasattr(adata.X, 'toarray'):
                    batch_counts = adata.X[batch_idx].toarray()
                else:
                    batch_counts = adata.X[batch_idx]
                    
                batch_tensor = torch.from_numpy(batch_counts).to(torch.bfloat16).to(device)
                
                batch_transformed = aido_model.transform({'sequences': batch_tensor})
                _ = aido_model(batch_transformed)
                
                hidden = layer_outputs['hidden']
                hidden = hidden[:, :-2, :] 
                hidden = hidden[:, attn_mask_bool, :] 
                
                activations.append(hidden.float().cpu())
                
    finally:
        handle.remove()
        aido_model = aido_model.cpu()
        del layer_outputs
        free_gpu()
        
    if len(activations) > 0:
        return torch.cat(activations, dim=0)
    else:
        return torch.empty(0)

def train_chunk(sae, optimizer, activations, device, epochs=1, scheduler=None):
    """Move SAE to GPU once per chunk."""
    sae = sae.to(device)
    sae.train()
    n_cells = activations.shape[0]
    total_loss = 0
    batches_per_epoch = (n_cells + CELLS_PER_BATCH - 1) // CELLS_PER_BATCH
    
    for epoch in range(epochs):
        perm = torch.randperm(n_cells)
        epoch_loss = 0
        for i in range(0, n_cells, CELLS_PER_BATCH):
            batch_idx = perm[i : i + CELLS_PER_BATCH]
            batch_data = activations[batch_idx]
            batch_flat = batch_data.view(-1, batch_data.shape[-1]).to(device, non_blocking=True)
            recon, _ = sae(batch_flat)
            loss = F.mse_loss(recon, batch_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        total_loss += (epoch_loss / batches_per_epoch)
        if scheduler is not None:
            scheduler.step()
            
    sae = sae.cpu()
    free_gpu()
    return total_loss / epochs if epochs > 0 else 0

def evaluate(sae, val_activations, device):
    sae = sae.to(device)
    sae.eval()
    total_loss = 0
    batches = 0
    n_cells = val_activations.shape[0]
    with torch.no_grad():
        for i in range(0, n_cells, CELLS_PER_BATCH):
            batch_data = val_activations[i : i + CELLS_PER_BATCH]
            batch_flat = batch_data.view(-1, batch_data.shape[-1]).to(device, non_blocking=True)
            recon, _ = sae(batch_flat)
            loss = F.mse_loss(recon, batch_flat)
            total_loss += loss.item()
            batches += 1
    sae = sae.cpu()
    free_gpu()
    return total_loss / batches if batches > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=400)
    # Resume / Chunking args
    parser.add_argument("--resume_from", type=str, default=None, help="Path to .pt checkpoint to resume from")
    parser.add_argument("--start_chunk", type=int, default=0, help="Start processing from this chunk index")
    parser.add_argument("--end_chunk", type=int, default=None, help="Stop after this chunk index")
    parser.add_argument("--save_name", type=str, default="final", help="Suffix for saved model file")
    args = parser.parse_args()
    
    print(f"=== 100k Online SAE Training (Layer {args.layer}) ===")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Run `python scripts/create_100k_subset.py` first!")
        return
        
    print("Loading 100k Subset...")
    adata = ad.read_h5ad(DATA_FILE)

    # Fix duplicate cell names if present
    if not adata.obs_names.is_unique:
        print("  Warning: Duplicate cell names detected. Making unique...")
        adata.obs_names_make_unique()

    print("Aligning to AIDO...")
    adata, mask = align_adata(adata)
    
    # 2. Init Models
    print("Initializing models...")
    aido = Embed.from_config({
        "model.backbone": "aido_cell_100m",
        "model.batch_size": AIDO_EXTRACT_BATCH
    }).to(torch.bfloat16)
    aido.eval()
    
    sae = TopKSAE(input_dim=640, expansion=args.expansion, k=args.k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    
    # 3. Resume Checkpoint?
    if args.resume_from:
        print(f"Resuming from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location='cpu')
        sae.load_state_dict(ckpt['model_state_dict'])
        # Optional: Load optimizer state if strictly resuming
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # But for 'staged' training where we might change LR schedules, resetting opt is sometimes safer.
        # Let's keep it simple and just load weights.
        print("Weights loaded.")

    # 4. Split & Shuffle
    all_indices = np.arange(adata.n_obs)
    train_idx, val_idx = train_test_split(all_indices, test_size=0.05, random_state=42)
    
    # Fixed Seed Shuffle for Train Indices to ensure consistency across runs!
    # This is CRITICAL for resuming. Chunk 0 must be the same Chunk 0 every time.
    rng = np.random.default_rng(seed=42)
    rng.shuffle(train_idx)
    
    # Subsample Val
    if len(val_idx) > 200:
        rng_val = np.random.default_rng(seed=42)
        val_idx = rng_val.choice(val_idx, 200, replace=False)
        
    print("Extracting Validation Set...")
    val_acts = extract_chunk_activations(aido, adata, mask, val_idx, args.layer, DEVICE)
    
    # 5. Determine Chunks
    n_chunks = (len(train_idx) + args.chunk_size - 1) // args.chunk_size
    start_chunk = args.start_chunk
    end_chunk = args.end_chunk if args.end_chunk is not None else n_chunks
    
    print(f"Total Chunks: {n_chunks}")
    print(f"Processing range: {start_chunk} -> {end_chunk}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_chunks * NUM_EPOCHS)
    # Fast-forward scheduler if resuming?
    if start_chunk > 0:
        for _ in range(start_chunk * NUM_EPOCHS):
            scheduler.step()
    
    # 6. Train Loop
    train_losses = []
    val_losses = []
    
    # Iterate specifically over the requested chunk range
    chunk_indices = range(start_chunk, end_chunk)
    pbar = tqdm(chunk_indices, desc="Chunks")
    
    for i in pbar:
        chunk_start = i * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(train_idx))
        c_idx = train_idx[chunk_start:chunk_end]
        
        # A. Extract
        c_acts = extract_chunk_activations(aido, adata, mask, c_idx, args.layer, DEVICE)
        
        # B. Train
        c_loss = train_chunk(sae, optimizer, c_acts, DEVICE, epochs=NUM_EPOCHS, scheduler=scheduler)
        
        train_losses.append(c_loss)
        del c_acts
        
        # C. Evaluate
        v_loss = evaluate(sae, val_acts, DEVICE)
        val_losses.append(v_loss)
        
        pbar.set_postfix(train=f"{c_loss:.5f}", val=f"{v_loss:.5f}")

    # 7. Save
    out_dir = os.path.join(OUTPUT_BASE_DIR, f"layer_{args.layer}", f"sae_k_{args.k}_online_100k")
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, f"sae_{args.save_name}.pt")
    torch.save({
        'model_state_dict': sae.state_dict(),
        'input_dim': 640,
        'expansion': args.expansion,
        'k': args.k,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, save_path)
    
    # Also save decoder weights separately
    torch.save(sae.decoder.state_dict(), os.path.join(out_dir, f"decoder_{args.save_name}.pt"))
    
    print(f"\nRun complete. Saved to {save_path}")

if __name__ == "__main__":
    main()
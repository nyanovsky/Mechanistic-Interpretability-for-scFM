"""Online SAE Training (Minimal & Speed Optimized)

Strategy:
- Iterate data in chunks (Single Pass).
- SWAP models to GPU:
  - Extraction: AIDO on GPU (Batch Size 32) -> Fast Extraction.
  - Training: SAE on GPU -> Fast Training.
- Keeps VRAM ~12GB.
- fast matrix multiplication enabled.

Usage:
    python train_sae_online_minimal.py --layer 12
"""

import argparse
import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add AIDO.Cell to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ModelGenerator/huggingface/aido.cell'))
from modelgenerator.tasks import Embed
from aido_cell.utils import align_adata

# Speed optimizations for RTX 3090
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

# Configuration
PBMC_RAW_FILE = "../data/pbmc/pbmc3k_raw.h5ad"
OUTPUT_BASE_DIR = "/biodata/nyanovsky/datasets/pbmc3k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
CELLS_PER_BATCH = 8       # SAE Training Batch (Gradient updates)
AIDO_EXTRACT_BATCH = 8   # AIDO Inference Batch (Maximized for speed)
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10           # Local epochs per chunk

# ===== UTILS =====

def free_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

# ===== MODELS =====

class TopKSAE(nn.Module):
    def __init__(self, input_dim=640, expansion=4, k=32):
        super().__init__()
        latent_dim = input_dim * expansion
        self.b_pre = nn.Parameter(torch.zeros(input_dim))  # Learned pre-bias for centering
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        self.k = k

    def forward(self, x):
        # Center activations before encoding
        x_centered = x - self.b_pre
        # Encode with TopK sparsity
        latents = self.encoder(x_centered)
        topk_vals, topk_idx = latents.topk(self.k, dim=-1)
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(-1, topk_idx, topk_vals)
        # Decode and add back baseline
        recon = self.decoder(sparse_latents) + self.b_pre
        return recon, sparse_latents

# ===== PROCESSING =====

def extract_chunk_activations(aido_model, adata, mask, indices, layer_idx, device):
    """
    1. Move AIDO to GPU.
    2. Extract activations to CPU Tensor using LARGE batches.
    3. Move AIDO to CPU.
    """
    # 1. Setup
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
    
    # 2. Extract
    try:
        with torch.no_grad():
            # Use larger batch size for inference
            for i in range(0, len(indices), AIDO_EXTRACT_BATCH):
                batch_idx = indices[i : i + AIDO_EXTRACT_BATCH]
                
                # Prepare input
                batch_counts = adata.X[batch_idx].toarray()
                batch_tensor = torch.from_numpy(batch_counts).to(torch.bfloat16).to(device)
                
                # Forward
                batch_transformed = aido_model.transform({'sequences': batch_tensor})
                _ = aido_model(batch_transformed)
                
                # Capture & Filter
                hidden = layer_outputs['hidden']
                hidden = hidden[:, :-2, :] # Remove special tokens
                hidden = hidden[:, attn_mask_bool, :] # Filter genes
                
                # Move to CPU immediately
                activations.append(hidden.float().cpu())
                
    finally:
        handle.remove()
        # 3. Cleanup
        aido_model = aido_model.cpu()
        del layer_outputs
        free_gpu()
        
    return torch.cat(activations, dim=0) # [N_Cells, N_Genes, Dim]


def train_chunk(sae, optimizer, activations, device, epochs=1, scheduler=None):
    """
    1. Move SAE to GPU.
    2. Train on CPU activations (moving batches to GPU).
    3. Move SAE to CPU.
    """
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
            
            # Load batch (CPU) -> GPU
            batch_data = activations[batch_idx] # [B, Genes, Dim]
            # Flatten: [B * Genes, Dim]
            batch_flat = batch_data.view(-1, batch_data.shape[-1]).to(device, non_blocking=True)
            
            # Step
            recon, _ = sae(batch_flat)
            loss = F.mse_loss(recon, batch_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        total_loss += (epoch_loss / batches_per_epoch)
        if scheduler is not None:
            scheduler.step()
            
    # Cleanup
    sae = sae.cpu()
    free_gpu()
    
    return total_loss / epochs if epochs > 0 else 0

def evaluate(sae, val_activations, device):
    """Eval on CPU activations (moving SAE to GPU temporarily)."""
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


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=200)
    args = parser.parse_args()
    
    print(f"=== Online SAE Training (Speed Optimized) - Layer {args.layer} ===")
    print(f"Device: {DEVICE}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Extraction Batch: {AIDO_EXTRACT_BATCH}")
    
    # 1. Load Data
    print("Loading Data...")
    raw_path = os.path.join(os.path.dirname(__file__), PBMC_RAW_FILE)
    adata = ad.read_h5ad(raw_path)
    adata, mask = align_adata(adata)
    print(f"Cells: {adata.n_obs}, Genes: {int(mask.sum())}")
    
    # 2. Load AIDO (CPU)
    print("Loading AIDO (CPU)...")
    aido = Embed.from_config({
        "model.backbone": "aido_cell_100m",
        "model.batch_size": AIDO_EXTRACT_BATCH
    }).to(torch.bfloat16)
    aido.eval()
    
    # 3. Init SAE (CPU)
    sae = TopKSAE(input_dim=640, expansion=args.expansion, k=args.k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    
    # 4. Split
    all_indices = np.arange(adata.n_obs)
    train_idx, val_idx = train_test_split(all_indices, test_size=0.1, random_state=42)
    
    # Subsample Val for RAM safety
    if len(val_idx) > 200:
        np.random.seed(42)
        val_idx = np.random.choice(val_idx, 200, replace=False)
    
    # 5. Pre-extract Validation (AIDO -> GPU -> CPU)
    print("Extracting Validation Set...")
    val_acts = extract_chunk_activations(aido, adata, mask, val_idx, args.layer, DEVICE)
    
    # 6. Training Loop
    print("\nStarting Training...")
    train_losses = []
    val_losses = []
    
    n_chunks = (len(train_idx) + args.chunk_size - 1) // args.chunk_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_chunks * NUM_EPOCHS)
    
    np.random.shuffle(train_idx)
    
    pbar = tqdm(range(0, len(train_idx), args.chunk_size), desc="Chunks")
    for chunk_start in pbar:
        chunk_end = min(chunk_start + args.chunk_size, len(train_idx))
        c_idx = train_idx[chunk_start:chunk_end]
        
        # A. Extract (Fast with large batch)
        c_acts = extract_chunk_activations(aido, adata, mask, c_idx, args.layer, DEVICE)
        
        # B. Train (Optimized: SAE moves to GPU once per chunk)
        c_loss = train_chunk(sae, optimizer, c_acts, DEVICE, epochs=NUM_EPOCHS, scheduler=scheduler)
        
        train_losses.append(c_loss)
        del c_acts 
        
        # C. Evaluate
        v_loss = evaluate(sae, val_acts, DEVICE)
        val_losses.append(v_loss)
        
        pbar.set_postfix(train=f"{c_loss:.5f}", val=f"{v_loss:.5f}")

    # 7. Save
    print("\nSaving...")
    out_dir = os.path.join(OUTPUT_BASE_DIR, f"layer_{args.layer}", f"sae_k_{args.k}_online_minimal")
    os.makedirs(out_dir, exist_ok=True)
    
    torch.save(sae.state_dict(), os.path.join(out_dir, "sae.pt"))
    torch.save(sae.decoder.state_dict(), os.path.join(out_dir, "decoder.pt"))
    
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training.png"))
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()

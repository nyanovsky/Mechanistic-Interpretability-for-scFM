"""Create a single H5AD subset containing only the sampled 100k cells.

Reads the sampling indices and the 19 raw chunk files, extracts the cells,
and saves them to a single file. This simplifies downstream training.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from glob import glob
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = "/biodata/nyanovsky/datasets/cell_x_gene/bone_marrow/raw/"
INDICES_DIR = os.path.join(os.path.dirname(__file__), "../plots/cellxgene_analysis")
OUTPUT_FILE = "/biodata/nyanovsky/datasets/cell_x_gene/bone_marrow/pbmc_100k_subset.h5ad"

# Strategy to load (must match what you want to train on)
STRATEGY = "k562_hybrid" 

def main():
    print("="*60)
    print(f"Creating 100k Subset H5AD ({STRATEGY})")
    print("="*60)
    
    # 1. Load Indices
    indices_path = os.path.join(INDICES_DIR, f"sampling_indices_{STRATEGY}.pkl")
    print(f"Loading indices from: {indices_path}")
    with open(indices_path, 'rb') as f:
        sampling_indices = pickle.load(f)
        
    total_cells = sum(len(x) for x in sampling_indices.values())
    print(f"Total cells to extract: {total_cells:,}")
    
    # 2. Find Files
    h5ad_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*.h5ad")))
    print(f"Found {len(h5ad_files)} raw h5ad files")
    
    adatas = []
    
    # 3. Iterate and Extract
    print("\nExtracting cells...")
    for chunk_id in sorted(sampling_indices.keys()):
        local_indices = sampling_indices[chunk_id]
        if not local_indices:
            continue
            
        file_path = h5ad_files[chunk_id]
        print(f"  Processing Chunk {chunk_id}: {len(local_indices):,} cells from {os.path.basename(file_path)}")
        
        # Load backing mode to save RAM
        adata_chunk = sc.read_h5ad(file_path) # Load fully into RAM (50k cells is small enough)
        
        # Subset
        adata_subset = adata_chunk[local_indices].copy()
        
        # Add metadata tracking
        adata_subset.obs['original_chunk_id'] = chunk_id
        adata_subset.obs['original_local_idx'] = local_indices
        
        adatas.append(adata_subset)
        
        # Explicit GC
        del adata_chunk
    
    # 4. Concatenate
    print("\nConcatenating...")
    # join='outer' to keep all genes (though they should be identical)
    adata_full = ad.concat(adatas, join='outer', merge='same')
    
    print(f"Final shape: {adata_full.shape}")
    
    # 5. Save
    print(f"Saving to {OUTPUT_FILE}...")
    # Ensure dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    adata_full.write_h5ad(OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    main()

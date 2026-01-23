"""Create sampling indices for 100k cell subset from CELLxGENE bone marrow.

Uses the K562-informed sampling strategy to generate cell indices for each chunk
that should be included in training.

Output: pickle file with dict mapping chunk_id -> list of cell indices
"""

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
from tqdm import tqdm
from collections import defaultdict

# Configuration
DATA_DIR = "/biodata/nyanovsky/datasets/cell_x_gene/bone_marrow/raw/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../plots/cellxgene_analysis")
SAMPLING_STRATEGY_FILE = os.path.join(OUTPUT_DIR, "k562_informed_sampling_strategies.csv")
METADATA_FILE = os.path.join(OUTPUT_DIR, "cellxgene_metadata.csv")

# Sampling parameters
STRATEGY = "k562_hybrid"  # Options: 'k562_weighted', 'k562_weighted_sqrt', 'k562_hybrid', 'sqrt_sample', 'proportional_sample'
TARGET_CELLS = 100000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print("="*80)
print("Creating Sampling Indices for CELLxGENE Bone Marrow")
print("="*80)
print(f"Strategy: {STRATEGY}")
print(f"Target cells: {TARGET_CELLS:,}")
print(f"Random seed: {RANDOM_SEED}")
print("="*80)

# Load sampling strategy
print("\nLoading sampling strategy...")
df_strategy = pd.read_csv(SAMPLING_STRATEGY_FILE)
print(f"Loaded {len(df_strategy)} cell types")

# Create dict: cell_type -> n_samples
sampling_dict = dict(zip(df_strategy['cell_type'], df_strategy[STRATEGY]))

# Verify total
total_requested = sum(sampling_dict.values())
print(f"Total samples requested: {total_requested:,}")
assert total_requested == TARGET_CELLS, f"Mismatch: {total_requested} != {TARGET_CELLS}"

# Load metadata
print("\nLoading metadata...")
meta = pd.read_csv(METADATA_FILE)
print(f"Total cells in metadata: {len(meta):,}")

# Sample cells
print("\nSampling cells...")

sampled_indices = defaultdict(list)  # chunk_id -> list of local cell indices

for cell_type, n_samples_needed in tqdm(sampling_dict.items(), desc="Sampling cell types"):
    if n_samples_needed == 0:
        continue

    # Find all cells of this type across chunks
    cells_of_type = meta[meta['cell_type'] == cell_type]

    if len(cells_of_type) == 0:
        print(f"WARNING: No cells found for type '{cell_type}'")
        continue

    # Check if we have enough cells
    if len(cells_of_type) < n_samples_needed:
        print(f"WARNING: Cell type '{cell_type}' has only {len(cells_of_type)} cells, need {n_samples_needed}. Sampling all.")
        n_samples_needed = len(cells_of_type)

    # Sample randomly from all cells of this type
    sampled_cells = cells_of_type.sample(n=n_samples_needed, random_state=RANDOM_SEED)

    # Group by chunk_id and get local indices within each chunk
    for chunk_id, group in sampled_cells.groupby('chunk_id'):
        # Get global indices in the metadata DataFrame
        global_indices = group.index.values

        # Convert to local indices within chunk
        # Cells from chunk_id start at the first occurrence of chunk_id in meta
        chunk_start_idx = meta[meta['chunk_id'] == chunk_id].index[0]
        local_indices = global_indices - chunk_start_idx

        sampled_indices[int(chunk_id)].extend(local_indices.tolist())

# Verify total cells sampled
total_sampled = sum(len(indices) for indices in sampled_indices.values())
print(f"\n" + "="*80)
print(f"Sampling complete!")
print("="*80)
print(f"Total cells sampled: {total_sampled:,}")
print(f"Cells per chunk:")
for chunk_id in sorted(sampled_indices.keys()):
    n_cells = len(sampled_indices[chunk_id])
    print(f"  Chunk {chunk_id:2d}: {n_cells:5,} cells")

# Sort indices within each chunk for efficient access
print("\nSorting indices within chunks...")
for chunk_id in sampled_indices:
    sampled_indices[chunk_id] = sorted(sampled_indices[chunk_id])

# Save to pickle
output_file = os.path.join(OUTPUT_DIR, f"sampling_indices_{STRATEGY}.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(dict(sampled_indices), f)

print(f"\nSaved sampling indices to {output_file}")

# Also save as CSV for human readability
print("\nCreating human-readable CSV...")
rows = []
for chunk_id in sorted(sampled_indices.keys()):
    for local_idx in sampled_indices[chunk_id]:
        # Get cell metadata
        global_idx = meta[meta['chunk_id'] == chunk_id].index[0] + local_idx
        cell_type = meta.loc[global_idx, 'cell_type']
        rows.append({
            'chunk_id': chunk_id,
            'local_index': local_idx,
            'cell_type': cell_type
        })

df_sampled = pd.DataFrame(rows)
csv_file = os.path.join(OUTPUT_DIR, f"sampling_indices_{STRATEGY}.csv")
df_sampled.to_csv(csv_file, index=False)
print(f"Saved human-readable CSV to {csv_file}")

# Verify cell type distribution in sample
print("\n" + "="*80)
print("Verifying sampled cell type distribution:")
print("="*80)
sampled_ct_dist = df_sampled['cell_type'].value_counts().head(20)
print(sampled_ct_dist)

# Compare with requested distribution
print("\n" + "="*80)
print("Comparison with requested distribution (top 10):")
print("="*80)
comparison = []
for ct in df_strategy.nlargest(10, STRATEGY)['cell_type']:
    requested = sampling_dict.get(ct, 0)
    actual = (df_sampled['cell_type'] == ct).sum()
    comparison.append({
        'cell_type': ct,
        'requested': requested,
        'actual': actual,
        'diff': actual - requested
    })

df_comparison = pd.DataFrame(comparison)
print(df_comparison.to_string(index=False))

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

if 'k562_relevance' in df_strategy.columns:
    # Analyze relevance distribution in sample
    sampled_ct_counts = df_sampled['cell_type'].value_counts()
    relevance_breakdown = []

    for relevance_level in [5.0, 3.0, 1.5, 0.5, 0.2]:
        relevant_types = df_strategy[df_strategy['k562_relevance'] == relevance_level]['cell_type'].values
        n_cells_sampled = sampled_ct_counts[sampled_ct_counts.index.isin(relevant_types)].sum()
        relevance_breakdown.append((relevance_level, n_cells_sampled))

    print("\nCells sampled by K562 relevance category:")
    print("  Critical (5.0x): {:6,} cells ({:.1f}%)".format(
        relevance_breakdown[0][1], relevance_breakdown[0][1]/total_sampled*100))
    print("  High     (3.0x): {:6,} cells ({:.1f}%)".format(
        relevance_breakdown[1][1], relevance_breakdown[1][1]/total_sampled*100))
    print("  Medium   (1.5x): {:6,} cells ({:.1f}%)".format(
        relevance_breakdown[2][1], relevance_breakdown[2][1]/total_sampled*100))
    print("  Low      (0.5x): {:6,} cells ({:.1f}%)".format(
        relevance_breakdown[3][1], relevance_breakdown[3][1]/total_sampled*100))
    print("  Minimal  (0.2x): {:6,} cells ({:.1f}%)".format(
        relevance_breakdown[4][1], relevance_breakdown[4][1]/total_sampled*100))

print("\n" + "="*80)
print("Sampling indices creation complete!")
print("="*80)
print(f"\nTo use these indices in training:")
print(f"```python")
print(f"import pickle")
print(f"with open('{output_file}', 'rb') as f:")
print(f"    sampling_indices = pickle.load(f)")
print(f"")
print(f"# Example: Load sampled cells from chunk 0")
print(f"chunk_indices = sampling_indices[0]")
print(f"adata = sc.read_h5ad(h5ad_files[0])")
print(f"sampled_cells = adata[chunk_indices]")
print(f"```")

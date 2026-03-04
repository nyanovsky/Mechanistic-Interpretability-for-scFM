"""Analyze cell type composition in CELLxGENE bone marrow dataset.

Loads all 19 h5ad chunks and computes:
1. Total cells and cell type distribution
2. Sampling strategy for 100k cells to maximize biological diversity
"""

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

# Configuration
DATA_DIR = "/biodata/nyanovsky/datasets/cell_x_gene/bone_marrow/raw/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../plots/cellxgene_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def inspect_metadata(files):
    """Efficiently extract metadata from h5ad files without loading expression data."""
    print(f"Found {len(files)} chunks. Aggregating metadata...")

    combined_obs_list = []

    for i, f in enumerate(files):
        try:
            # Load only the AnnData object (metadata is cheap)
            # We don't touch .X, so this is fast
            ad = sc.read_h5ad(f)

            # Extract just the columns we care about
            # Keep 'soma_joinid' to ensure uniqueness if needed
            cols = ["cell_type", "tissue_general", "assay", "development_stage"]
            # Handle cases where a column might be missing
            available_cols = [c for c in cols if c in ad.obs.columns]

            chunk_obs = ad.obs[available_cols].copy()
            chunk_obs["chunk_id"] = i
            combined_obs_list.append(chunk_obs)

            print(f"Chunk {i}: {len(chunk_obs)} cells processed.")
            del ad

        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Combine all metadata into one DataFrame
    full_meta = pd.concat(combined_obs_list, ignore_index=True)
    print(f"\nTotal Cells: {len(full_meta):,}")

    return full_meta


print("="*80)
print("CELLxGENE Bone Marrow Cell Type Analysis")
print("="*80)

# Find all h5ad files
h5ad_files = sorted(glob(os.path.join(DATA_DIR, "*.h5ad")))
print(f"\nFound {len(h5ad_files)} h5ad files\n")

# Extract metadata efficiently
meta = inspect_metadata(h5ad_files)

# Save full metadata
meta_path = os.path.join(OUTPUT_DIR, "cellxgene_metadata.csv")
meta.to_csv(meta_path, index=False)
print(f"\nSaved full metadata to {meta_path}")

# ===== CELL TYPE ANALYSIS =====
print("\n" + "="*80)
print("CELL TYPE DISTRIBUTION")
print("="*80)

cell_type_counts = meta['cell_type'].value_counts()
total_cells = len(meta)

df_celltypes = pd.DataFrame({
    'cell_type': cell_type_counts.index,
    'count': cell_type_counts.values,
    'percentage': (cell_type_counts.values / total_cells * 100)
})

print(f"\nUnique cell types: {len(df_celltypes)}")
print("\nTop 20 cell types:")
print(df_celltypes.head(20).to_string(index=False))

# Save to CSV
csv_path = os.path.join(OUTPUT_DIR, "celltype_distribution.csv")
df_celltypes.to_csv(csv_path, index=False)
print(f"\nSaved distribution to {csv_path}")

# ===== SAMPLING STRATEGY =====
print("\n" + "="*80)
print("SAMPLING STRATEGY FOR 100K CELLS")
print("="*80)

target_cells = 100000

# Strategy 1: Proportional sampling (maintains distribution)
df_celltypes['proportional_sample'] = (df_celltypes['percentage'] / 100 * target_cells).astype(int)

# Strategy 2: Square root sampling (balances rare/common types)
# Gives more weight to rare types while preserving some proportion
sqrt_weights = np.sqrt(df_celltypes['count'].values)
sqrt_weights_norm = sqrt_weights / sqrt_weights.sum()
df_celltypes['sqrt_sample'] = (sqrt_weights_norm * target_cells).astype(int)

# Strategy 3: Stratified sampling (min 100 cells per type, rest proportional)
min_per_type = 100
stratified_samples = []
remaining_budget = target_cells

# Allocate min samples to all types
for _, row in df_celltypes.iterrows():
    min_sample = min(row['count'], min_per_type)
    stratified_samples.append(min_sample)
    remaining_budget -= min_sample

# Distribute remaining budget proportionally
if remaining_budget > 0:
    common_types_mask = df_celltypes['count'] >= min_per_type
    if common_types_mask.sum() > 0:
        common_counts = df_celltypes.loc[common_types_mask, 'count'].values
        common_proportions = common_counts / common_counts.sum()
        common_additional = (common_proportions * remaining_budget).astype(int)

        common_idx = 0
        for i in range(len(df_celltypes)):
            if common_types_mask.iloc[i]:
                stratified_samples[i] += common_additional[common_idx]
                common_idx += 1

df_celltypes['stratified_sample'] = stratified_samples

# Adjust to ensure total = 100k (handle rounding errors)
for strategy in ['proportional_sample', 'sqrt_sample', 'stratified_sample']:
    diff = target_cells - df_celltypes[strategy].sum()
    if diff != 0:
        # Add/subtract difference to most common type
        df_celltypes.loc[0, strategy] += diff

print("\nStrategy Comparison (Top 20 cell types):")
print(df_celltypes[['cell_type', 'count', 'percentage', 'proportional_sample', 'sqrt_sample', 'stratified_sample']].head(20).to_string(index=False))

# Verify totals
print(f"\nTotal samples per strategy:")
print(f"  Proportional: {df_celltypes['proportional_sample'].sum():,}")
print(f"  Square Root:  {df_celltypes['sqrt_sample'].sum():,}")
print(f"  Stratified:   {df_celltypes['stratified_sample'].sum():,}")

# Save sampling strategies
sampling_csv = os.path.join(OUTPUT_DIR, "sampling_strategies.csv")
df_celltypes.to_csv(sampling_csv, index=False)
print(f"\nSaved sampling strategies to {sampling_csv}")

# ===== VISUALIZATION =====
print("\nGenerating visualizations...")

# 1. Cell type distribution (top 20)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

top_20 = df_celltypes.head(20)

# Bar plot of counts
axes[0].barh(range(len(top_20)), top_20['count'].values)
axes[0].set_yticks(range(len(top_20)))
axes[0].set_yticklabels(top_20['cell_type'].values, fontsize=9)
axes[0].set_xlabel('Number of Cells')
axes[0].set_title('Top 20 Cell Types in CELLxGENE Bone Marrow')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Add count labels
for i, (idx, row) in enumerate(top_20.iterrows()):
    axes[0].text(row['count'] + 1000, i, f"{row['count']:,} ({row['percentage']:.1f}%)",
                va='center', fontsize=8)

# Comparison of sampling strategies (top 20)
x = np.arange(len(top_20))
width = 0.25

axes[1].barh(x - width, top_20['proportional_sample'], width, label='Proportional', alpha=0.8)
axes[1].barh(x, top_20['sqrt_sample'], width, label='Square Root', alpha=0.8)
axes[1].barh(x + width, top_20['stratified_sample'], width, label='Stratified (min 100)', alpha=0.8)

axes[1].set_yticks(x)
axes[1].set_yticklabels(top_20['cell_type'].values, fontsize=9)
axes[1].set_xlabel('Sampled Cells (for 100k total)')
axes[1].set_title('Sampling Strategy Comparison')
axes[1].legend()
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "celltype_distribution_and_sampling.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {plot_path}")
plt.close()

# 2. Pie chart of top types + "Other"
fig, ax = plt.subplots(figsize=(10, 8))

top_10 = df_celltypes.head(10)
other_count = df_celltypes.iloc[10:]['count'].sum()

pie_data = list(top_10['count'].values) + [other_count]
pie_labels = list(top_10['cell_type'].values) + ['Other']

ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
ax.set_title(f'Cell Type Distribution (Total: {total_cells:,} cells)')

pie_path = os.path.join(OUTPUT_DIR, "celltype_pie_chart.png")
plt.savefig(pie_path, dpi=150, bbox_inches='tight')
print(f"Saved pie chart to {pie_path}")
plt.close()

# 3. Shannon diversity analysis
from scipy.stats import entropy

proportions = df_celltypes['count'].values / df_celltypes['count'].sum()
shannon_diversity = entropy(proportions, base=2)
effective_num_types = 2 ** shannon_diversity

print("\n" + "="*80)
print("DIVERSITY METRICS")
print("="*80)
print(f"Shannon Diversity Index: {shannon_diversity:.2f} bits")
print(f"Effective Number of Cell Types: {effective_num_types:.0f}")
print(f"(Dataset behaves as if it has ~{effective_num_types:.0f} equally-abundant cell types)")

# ===== ADDITIONAL METADATA ANALYSIS =====
if 'tissue_general' in meta.columns:
    print("\n" + "="*80)
    print("TISSUE DISTRIBUTION")
    print("="*80)
    tissue_counts = meta['tissue_general'].value_counts()
    print(tissue_counts)

if 'assay' in meta.columns:
    print("\n" + "="*80)
    print("ASSAY DISTRIBUTION")
    print("="*80)
    assay_counts = meta['assay'].value_counts()
    print(assay_counts)

if 'development_stage' in meta.columns:
    print("\n" + "="*80)
    print("DEVELOPMENT STAGE DISTRIBUTION")
    print("="*80)
    dev_stage_counts = meta['development_stage'].value_counts()
    print(dev_stage_counts.head(10))

# ===== RECOMMENDATIONS =====
print("\n" + "="*80)
print("SAMPLING RECOMMENDATIONS")
print("="*80)

print("""
Based on the analysis, here are three sampling strategies:

1. PROPORTIONAL SAMPLING
   - Maintains the natural distribution of cell types
   - Good for learning common biological patterns
   - Rare types remain rare in sample
   - Use case: General-purpose SAE that reflects natural biology

2. SQUARE ROOT SAMPLING (RECOMMENDED)
   - Balances representation of rare and common types
   - sqrt(count) weighting gives more power to rare types
   - Better for discovering diverse biological features
   - Use case: Mechanistic interpretability focused on diverse feature discovery

3. STRATIFIED SAMPLING
   - Guarantees minimum 100 cells per cell type
   - Ensures all types are represented
   - Useful for comprehensive coverage
   - Use case: Need interpretability across ALL cell types

For mechanistic interpretability focused on learning diverse biological features,
I recommend SQUARE ROOT SAMPLING as it balances:
- Coverage of rare cell types (more interpretable features)
- Preservation of common type importance
- Statistical power across the feature space
""")

# ===== CHUNK-WISE DISTRIBUTION =====
print("\n" + "="*80)
print("CHUNK-WISE CELL TYPE DISTRIBUTION")
print("="*80)

# Analyze which chunks contain which cell types
chunk_ct_matrix = pd.crosstab(meta['chunk_id'], meta['cell_type'])
print(f"\nCell types per chunk:")
ct_per_chunk = (chunk_ct_matrix > 0).sum(axis=1)
print(ct_per_chunk)

print(f"\nCells per chunk:")
cells_per_chunk = meta.groupby('chunk_id').size()
print(cells_per_chunk)

# Save chunk distribution
chunk_dist_path = os.path.join(OUTPUT_DIR, "chunk_celltype_matrix.csv")
chunk_ct_matrix.to_csv(chunk_dist_path)
print(f"\nSaved chunk-celltype matrix to {chunk_dist_path}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

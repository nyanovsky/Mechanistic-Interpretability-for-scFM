"""Create K562-informed sampling strategy for CELLxGENE bone marrow.

K562 cells are chronic myeloid leukemia cells (undifferentiated hematopoietic blast cells).
For better transfer learning and interpretability on K562 perturbation data, we should
prioritize sampling cell types most biologically relevant to K562:

1. Hematopoietic stem/progenitor cells (HSCs, MPPs, CMPs, GMPs, MEPs)
2. Myeloid lineage cells (monocytes, granulocytes, myeloblasts)
3. Early/immature cells in all lineages
4. Other immune cells for diversity

This script creates a weighted sampling strategy based on biological relevance to K562.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../plots/cellxgene_analysis")
CELLTYPE_DIST_FILE = os.path.join(OUTPUT_DIR, "celltype_distribution.csv")
TARGET_CELLS = 100000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print("="*80)
print("Creating K562-Informed Sampling Strategy")
print("="*80)
print("\nK562 Background:")
print("  - Chronic myeloid leukemia (CML) cell line")
print("  - Hematopoietic blast cells (undifferentiated)")
print("  - Primarily myeloid lineage with multi-lineage potential")
print("  - Expresses markers of erythroid, myeloid, and megakaryocytic lineages")
print("="*80)

# Load cell type distribution
df_celltypes = pd.read_csv(CELLTYPE_DIST_FILE)
print(f"\nLoaded {len(df_celltypes)} cell types")

# Define biological relevance categories for K562
# Higher priority = more relevant to K562 biology

RELEVANCE_CATEGORIES = {
    'Critical': 5.0,  # Hematopoietic stem/progenitors, very early cells
    'High': 3.0,      # Myeloid lineage, myeloblasts, early committed cells
    'Medium': 1.5,    # Other immune progenitors, monocytes, granulocytes
    'Low': 0.5,       # Mature lymphocytes, terminally differentiated cells
    'Minimal': 0.2    # Very specialized/distant cell types
}

# Categorize cell types by relevance to K562
def assign_relevance(cell_type_name):
    """Assign biological relevance score to K562 based on cell type name."""
    name_lower = cell_type_name.lower()

    # CRITICAL: Stem cells and early progenitors (most similar to K562 blast state)
    critical_keywords = [
        'hematopoietic stem cell',
        'hematopoietic multipotent progenitor',
        'multipotent progenitor',
        'hsc',
        'mpp',
        'common myeloid progenitor',
        'cmp',
        'megakaryocyte-erythroid progenitor',
        'mep',
        'granulocyte monocyte progenitor',
        'gmp',
        'hematopoietic progenitor',
        'hematopoietic oligopotent progenitor',
        'myeloid progenitor',
        'erythroid progenitor',
        'progenitor cell' if 'progenitor cell' == name_lower else None
    ]

    for keyword in critical_keywords:
        if keyword and keyword in name_lower:
            return RELEVANCE_CATEGORIES['Critical']

    # HIGH: Myeloid lineage cells (K562 is myeloid leukemia)
    high_keywords = [
        'monocyte',
        'myeloblast',
        'myeloid',
        'megakaryocyte',
        'erythroblast',
        'basophil',
        'eosinophil',
        'mast cell',
        'dendritic cell',
        'granulocyte',
        'neutrophil',
        'macrophage'
    ]

    for keyword in high_keywords:
        if keyword in name_lower:
            return RELEVANCE_CATEGORIES['High']

    # MEDIUM: Other immune cells, general hematopoietic cells
    medium_keywords = [
        'hematopoietic cell',
        'b cell',
        't cell',
        'nk cell',
        'natural killer',
        'lymphocyte',
        'naive',
        'memory',
        'precursor',
        'immature'
    ]

    for keyword in medium_keywords:
        if keyword in name_lower:
            return RELEVANCE_CATEGORIES['Medium']

    # LOW: Mature/differentiated cells
    low_keywords = [
        'plasma cell',
        'effector',
        'terminally differentiated',
        'activated',
        'regulatory t cell'
    ]

    for keyword in low_keywords:
        if keyword in name_lower:
            return RELEVANCE_CATEGORIES['Low']

    # MINIMAL: Very specialized or distant lineages
    minimal_keywords = [
        'erythrocyte',  # Mature RBCs
        'platelet',
        'unknown'
    ]

    for keyword in minimal_keywords:
        if keyword in name_lower:
            return RELEVANCE_CATEGORIES['Minimal']

    # Default: Low relevance for unclassified
    return RELEVANCE_CATEGORIES['Low']

# Assign relevance scores
print("\nAssigning biological relevance scores...")
df_celltypes['k562_relevance'] = df_celltypes['cell_type'].apply(assign_relevance)

# Show distribution of relevance categories
print("\nRelevance category distribution:")
for category, score in sorted(RELEVANCE_CATEGORIES.items(), key=lambda x: -x[1]):
    n_types = (df_celltypes['k562_relevance'] == score).sum()
    n_cells = df_celltypes[df_celltypes['k562_relevance'] == score]['count'].sum()
    print(f"  {category:12s} ({score:.1f}x): {n_types:3d} cell types, {n_cells:8,} cells")

# Create weighted sampling strategy
print("\nCreating K562-informed weighted sampling...")

# Compute weighted counts
df_celltypes['weighted_count'] = df_celltypes['count'] * df_celltypes['k562_relevance']

# Option 1: Weighted proportional (maintains some natural distribution)
weighted_proportions = df_celltypes['weighted_count'] / df_celltypes['weighted_count'].sum()
df_celltypes['k562_weighted'] = (weighted_proportions * TARGET_CELLS).astype(int)

# Option 2: Weighted sqrt (balances rare/common + relevance boost)
weighted_sqrt = np.sqrt(df_celltypes['count']) * df_celltypes['k562_relevance']
weighted_sqrt_norm = weighted_sqrt / weighted_sqrt.sum()
df_celltypes['k562_weighted_sqrt'] = (weighted_sqrt_norm * TARGET_CELLS).astype(int)

# Option 3: Hybrid - ensure minimum for critical types
MIN_CRITICAL = 500  # Minimum cells for critical cell types
MIN_HIGH = 200
MIN_MEDIUM = 100

hybrid_samples = []
remaining_budget = TARGET_CELLS

for _, row in df_celltypes.iterrows():
    if row['k562_relevance'] == RELEVANCE_CATEGORIES['Critical']:
        min_sample = min(row['count'], MIN_CRITICAL)
    elif row['k562_relevance'] == RELEVANCE_CATEGORIES['High']:
        min_sample = min(row['count'], MIN_HIGH)
    elif row['k562_relevance'] == RELEVANCE_CATEGORIES['Medium']:
        min_sample = min(row['count'], MIN_MEDIUM)
    else:
        min_sample = min(row['count'], 50)  # Minimal guaranteed sampling

    hybrid_samples.append(min_sample)
    remaining_budget -= min_sample

# Distribute remaining budget using weighted proportions
if remaining_budget > 0:
    # Recompute proportions on weighted counts
    additional_props = df_celltypes['weighted_count'] / df_celltypes['weighted_count'].sum()
    additional_samples = (additional_props * remaining_budget).astype(int)
    hybrid_samples = [h + a for h, a in zip(hybrid_samples, additional_samples)]

df_celltypes['k562_hybrid'] = hybrid_samples

# Adjust for rounding errors
for strategy in ['k562_weighted', 'k562_weighted_sqrt', 'k562_hybrid']:
    diff = TARGET_CELLS - df_celltypes[strategy].sum()
    if diff != 0:
        # Add to most abundant type
        df_celltypes.loc[df_celltypes['count'].idxmax(), strategy] += diff

# Verify totals
print("\nTotal samples per strategy:")
print(f"  K562 Weighted:      {df_celltypes['k562_weighted'].sum():,}")
print(f"  K562 Weighted Sqrt: {df_celltypes['k562_weighted_sqrt'].sum():,}")
print(f"  K562 Hybrid:        {df_celltypes['k562_hybrid'].sum():,}")

# Add original strategies for comparison
df_original = pd.read_csv(os.path.join(OUTPUT_DIR, "sampling_strategies.csv"))
df_celltypes = df_celltypes.merge(
    df_original[['cell_type', 'proportional_sample', 'sqrt_sample']],
    on='cell_type',
    how='left'
)

# Save updated sampling strategies
output_file = os.path.join(OUTPUT_DIR, "k562_informed_sampling_strategies.csv")
df_celltypes.to_csv(output_file, index=False)
print(f"\nSaved K562-informed strategies to {output_file}")

# ===== ANALYSIS & VISUALIZATION =====

# Show top cell types by different criteria
print("\n" + "="*80)
print("TOP 20 CELL TYPES BY DIFFERENT STRATEGIES")
print("="*80)

print("\nOriginal (Proportional):")
top_prop = df_celltypes.nlargest(20, 'proportional_sample')[['cell_type', 'count', 'proportional_sample', 'k562_relevance']]
print(top_prop.to_string(index=False))

print("\nK562-Informed (Hybrid) - RECOMMENDED:")
top_k562 = df_celltypes.nlargest(20, 'k562_hybrid')[['cell_type', 'count', 'k562_hybrid', 'k562_relevance']]
print(top_k562.to_string(index=False))

# Compare sampling of critical cell types
print("\n" + "="*80)
print("CRITICAL CELL TYPES (Stem/Progenitor Cells)")
print("="*80)
critical_types = df_celltypes[df_celltypes['k562_relevance'] == RELEVANCE_CATEGORIES['Critical']].copy()
critical_types = critical_types.sort_values('count', ascending=False)
print(critical_types[['cell_type', 'count', 'proportional_sample', 'sqrt_sample', 'k562_hybrid']].to_string(index=False))

total_critical_prop = critical_types['proportional_sample'].sum()
total_critical_k562 = critical_types['k562_hybrid'].sum()
print(f"\nTotal critical cells:")
print(f"  Proportional: {total_critical_prop:,} ({total_critical_prop/TARGET_CELLS*100:.1f}%)")
print(f"  K562 Hybrid:  {total_critical_k562:,} ({total_critical_k562/TARGET_CELLS*100:.1f}%)")
print(f"  Enrichment:   {total_critical_k562/total_critical_prop:.1f}x")

# Visualization
print("\nGenerating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. Comparison of top 20 types
top_20_hybrid = df_celltypes.nlargest(20, 'k562_hybrid')
x = np.arange(len(top_20_hybrid))
width = 0.25

axes[0].barh(x - width, top_20_hybrid['proportional_sample'], width, label='Proportional', alpha=0.8)
axes[0].barh(x, top_20_hybrid['sqrt_sample'], width, label='Sqrt', alpha=0.8)
axes[0].barh(x + width, top_20_hybrid['k562_hybrid'], width, label='K562 Hybrid', alpha=0.8, color='red')

axes[0].set_yticks(x)
axes[0].set_yticklabels(top_20_hybrid['cell_type'].values, fontsize=8)
axes[0].set_xlabel('Sampled Cells (for 100k total)')
axes[0].set_title('Sampling Strategy Comparison: Top 20 Types by K562 Hybrid')
axes[0].legend()
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Color code y-axis labels by relevance
for i, (_, row) in enumerate(top_20_hybrid.iterrows()):
    rel = row['k562_relevance']
    if rel == RELEVANCE_CATEGORIES['Critical']:
        color = 'red'
    elif rel == RELEVANCE_CATEGORIES['High']:
        color = 'orange'
    elif rel == RELEVANCE_CATEGORIES['Medium']:
        color = 'blue'
    else:
        color = 'gray'
    axes[0].get_yticklabels()[i].set_color(color)

# 2. Relevance category enrichment
categories = ['Critical', 'High', 'Medium', 'Low', 'Minimal']
category_scores = [RELEVANCE_CATEGORIES[c] for c in categories]

prop_by_category = []
k562_by_category = []

for score in category_scores:
    mask = df_celltypes['k562_relevance'] == score
    prop_by_category.append(df_celltypes[mask]['proportional_sample'].sum())
    k562_by_category.append(df_celltypes[mask]['k562_hybrid'].sum())

x_cat = np.arange(len(categories))
width = 0.35

axes[1].bar(x_cat - width/2, prop_by_category, width, label='Proportional', alpha=0.8)
axes[1].bar(x_cat + width/2, k562_by_category, width, label='K562 Hybrid', alpha=0.8, color='red')

axes[1].set_xticks(x_cat)
axes[1].set_xticklabels(categories)
axes[1].set_ylabel('Number of Cells Sampled')
axes[1].set_title('Sampling by K562 Relevance Category')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Add enrichment fold-change labels
for i, (p, k) in enumerate(zip(prop_by_category, k562_by_category)):
    if p > 0:
        fold_change = k / p
        axes[1].text(i, max(p, k) + 1000, f'{fold_change:.1f}x', ha='center', fontsize=9, fontweight='bold')

# 3. Scatter: count vs sampling ratio
sampling_ratio_prop = df_celltypes['proportional_sample'] / df_celltypes['count']
sampling_ratio_k562 = df_celltypes['k562_hybrid'] / df_celltypes['count']

colors_by_relevance = df_celltypes['k562_relevance'].map({
    RELEVANCE_CATEGORIES['Critical']: 'red',
    RELEVANCE_CATEGORIES['High']: 'orange',
    RELEVANCE_CATEGORIES['Medium']: 'blue',
    RELEVANCE_CATEGORIES['Low']: 'gray',
    RELEVANCE_CATEGORIES['Minimal']: 'lightgray'
})

axes[2].scatter(df_celltypes['count'], sampling_ratio_prop, alpha=0.5, s=20, label='Proportional', color='blue')
axes[2].scatter(df_celltypes['count'], sampling_ratio_k562, alpha=0.6, s=30, c=colors_by_relevance, label='K562 Hybrid', edgecolors='black', linewidth=0.5)

axes[2].set_xscale('log')
axes[2].set_xlabel('Total Cell Count (log scale)')
axes[2].set_ylabel('Sampling Ratio (sampled / total)')
axes[2].set_title('Sampling Ratio vs Cell Count (colored by K562 relevance)')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=TARGET_CELLS/df_celltypes['count'].sum(), color='black', linestyle='--', alpha=0.5, label='Overall ratio')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='Critical'),
    Patch(facecolor='orange', label='High'),
    Patch(facecolor='blue', label='Medium'),
    Patch(facecolor='gray', label='Low'),
    Patch(facecolor='lightgray', label='Minimal')
]
axes[2].legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "k562_informed_sampling_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {plot_path}")
plt.close()

# ===== RECOMMENDATION =====
print("\n" + "="*80)
print("RECOMMENDATION FOR SAE TRAINING")
print("="*80)
print("""
For training SAEs on bone marrow data to interpret K562 perturbations:

RECOMMENDED STRATEGY: K562 Hybrid Sampling

This strategy:
1. Prioritizes hematopoietic stem/progenitor cells (5x boost)
   - HSCs, MPPs, GMPs, MEPs, CMPs - most similar to K562 blast state
   - {:.1f}% of sample vs {:.1f}% in proportional sampling

2. Enriches for myeloid lineage cells (3x boost)
   - Monocytes, granulocytes, myeloblasts
   - K562 has primarily myeloid character

3. Ensures minimum representation of all cell types
   - Critical types: min 500 cells
   - High relevance: min 200 cells
   - Maintains diversity for comprehensive feature learning

4. Better biological alignment for transfer to K562
   - SAE features learned on progenitor cells will be more interpretable
     when applied to K562 (which are blast-like/progenitor-like)
   - Myeloid-enriched features will capture relevant regulatory programs

Expected benefit: More interpretable steering vectors when predicting
perturbations in K562, as the SAE will have learned features from
biologically similar cell states.

Alternative: If you want maximum generalization, use sqrt_sample.
But for K562-specific interpretability, use k562_hybrid.
""".format(
    total_critical_k562/TARGET_CELLS*100,
    total_critical_prop/TARGET_CELLS*100
))

print("="*80)
print("Analysis complete!")
print("="*80)

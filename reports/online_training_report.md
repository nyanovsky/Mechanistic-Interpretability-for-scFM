# COMPLETED: Online SAE Training for CELLxGENE

## Summary

Successfully implemented online SAE training system to train on 100k CELLxGENE bone marrow cells without storing ~5TB of activations. The system uses a K562-informed sampling strategy to prioritize biologically relevant cell types.

---

## What Was Accomplished

### 1. **Online Training Architecture Design**
   - **File**: `scripts/online_sae_training_design.md`
   - Designed architecture to merge activation extraction and SAE training
   - Memory budget analysis (10.7GB VRAM, 7GB RAM - safe for RTX 3090 + 70GB cluster)
   - Expected training time: ~20 hours for 100k cells

### 2. **Cell Type Analysis**
   - **Script**: `scripts/analyze_cellxgene_celltypes.py`
   - **Output**: `plots/cellxgene_analysis/`
   - Analyzed 915,689 bone marrow cells across 898 unique cell types
   - Shannon diversity: 5.76 bits (effective 54 equally-abundant types)
   - Top cell types: plasma cells (6.5%), classical monocytes (5.6%), NK cells (5.2%)

### 3. **K562-Informed Sampling Strategy**
   - **Script**: `scripts/create_k562_informed_sampling.py`
   - **Output**: `plots/cellxgene_analysis/k562_informed_sampling_strategies.csv`

   **Biological Rationale**:
   - K562 are chronic myeloid leukemia cells (hematopoietic blast-like)
   - Prioritized sampling by relevance to K562 biology:
     - **Critical (5x boost)**: HSCs, MPPs, GMPs, MEPs, CMPs → 33.4% of sample
     - **High (3x boost)**: Myeloid lineage (monocytes, granulocytes) → 32.3%
     - **Medium (1.5x)**: Other immune cells → 28.4%
     - **Low (0.5x)**: Mature differentiated cells → 4.9%
     - **Minimal (0.2x)**: Very specialized/distant → 0.9%

   **Enrichment**: 2.3x more stem/progenitor cells vs. proportional sampling

### 4. **Sampling Indices Generation and h5ad creation**
   - **Script**: `scripts/create_sampling_indices.py`
   - **Output**: `plots/cellxgene_analysis/sampling_indices_k562_hybrid.pkl`
   - Created indices for 99,918 cells (target: 100k)
   - Distributed across all 19 CELLxGENE chunks (~5,400 cells/chunk)
   - Created a single `bm_100k_subset.h5ad` file with the sampled cells

### 5. **Online Training Script**
   - **Scripts**: `scripts/train_sae_online_100k.py`, `scripts/train_sae_online_minimal.py`
   - **Implementation**: `online_sae_training_design.md`

### 6. **Approach validation on PBMC3K**
   - **Script**: `scripts/test_train_sae_online_minimal.py`
   - Ran interpretation script (`script/interpret_sae_parallel.py`) on obtained features
   - Summarized interpretations at `layer_12_summary_online.txt`
   - Compared with offline summary (`layer_12_summary.txt`) and confirmed obtained features are interpretable (got similar amount of significantly enriched features and the obtained features make biological sense). Also the feature scale is almost equal to the offline one, confirmed by feature scale analysis (`scripts/analyze_feature_scale.py` and `scripts/analyze_feature_scale_detailed.py`)


---

## Files Created

```
scripts/
├── online_sae_training_design.md          # Architecture documentation
├── analyze_cellxgene_celltypes.py          # Cell type distribution analysis
├── create_k562_informed_sampling.py        # K562-specific sampling strategy
├── create_sampling_indices.py              # Generate sampling indices
├── train_sae_online_100k.py                     # Main online training script
└── train_sae_online_minimal.py

plots/cellxgene_analysis/
├── cellxgene_metadata.csv                  # Full metadata (915k cells)
├── celltype_distribution.csv              # Cell type counts
├── celltype_distribution_and_sampling.png # Visualization
├── celltype_pie_chart.png                  # Top 10 types + Other
├── sampling_strategies.csv                 # Multiple sampling strategies
├── k562_informed_sampling_strategies.csv  # K562-informed strategies
├── k562_informed_sampling_comparison.png  # Strategy comparison plots
├── sampling_indices_k562_hybrid.pkl       # Final indices (pickle)
├── sampling_indices_k562_hybrid.csv       # Human-readable indices
└── chunk_celltype_matrix.csv              # Chunk-wise distribution
```

---

## How to Use

### Step 1: Train SAE Online (100k cells)

```bash
conda activate aido_env
python scripts/train_sae_online_100k.py \
    --layer 12 \
    --expansion 8 \
    --k 32 \
    --sampling_strategy k562_hybrid
```

**What it does**:
1. Loads 99,918 sampled cells from 19 CELLxGENE chunks
2. Extracts layer 12 activations on-the-fly (no storage)
3. Trains SAE for 10 epochs
4. Saves checkpoints every epoch
5. Saves final model + decoder weights

**Output**:
- `/biodata/nyanovsky/datasets/cell_x_gene/bone_marrow/layer_12/sae_k_32_5120_online/`
  - `topk_sae.pt` - Full model + training history
  - `sae_decoder.pt` - Decoder weights only (for steering)
  - `topk_sae_epoch_*.pt` - Checkpoints
- `plots/sae/layer_12/sae_training_curve_online_k562_hybrid.png`

**Expected time**: ~20 hours on RTX 3090


### Step 2: Interpret SAE Features

After training, interpret features using existing scripts:

```bash
# Interpret features (adjust paths as needed)
python scripts/interpret_sae_parallel.py \
    --sae_path /biodata/nyanovsky/datasets/cell_x_gene/bone_marrow/layer_12/sae_k_32_5120_online/topk_sae.pt \
    --activations_path ... \
    --output_dir ...
```

**Note**: You'll need to extract activations for a subset of cells first for interpretation, or modify `interpret_sae.py` to also work online.


### Step 3: Different Layers or Strategies (optional)

Train on layer 9:
```bash
python scripts/train_sae_online_100k.py --layer 9 --expansion 8 --k 32
```

Use different sampling strategy (square root, less K562-specific):
```bash
python scripts/train_sae_online_100k.py \
    --layer 12 \
    --expansion 8 \
    --k 32 \
    --sampling_strategy sqrt_sample
```

Available strategies:
- `k562_hybrid` - Recommended for K562 perturbation work (33% stem cells)
- `k562_weighted` - Pure weighted sampling
- `k562_weighted_sqrt` - Sqrt with K562 weighting
- `sqrt_sample` - Balanced rare/common (no K562 bias)
- `proportional_sample` - Natural distribution

---

## Memory Usage

### GPU (RTX 3090, 24GB VRAM)
- AIDO.Cell model: ~4GB
- Activation batch (8 cells): ~390MB
- SAE (8x expansion): ~50MB
- SAE activations + gradients: ~6.2GB
- **Total: ~10.7GB** (45% VRAM usage)

### RAM (Cluster, 70GB)
- AnnData chunk loaded: ~5GB
- AIDO preprocessing: ~2GB
- **Total: ~7GB** (10% RAM usage)

---

## Performance Estimates

### Per-batch timing (8 cells):
- AIDO.Cell forward pass: ~0.8 sec
- SAE training step: ~0.2 sec
- **Total: ~1 sec/batch**

### Full training (100k cells, 10 epochs):
- Total batches: ~125k
- **Total time: ~20 hours**

### Breakdown per epoch:
- ~12.5k batches/epoch
- ~2 hours/epoch

---

## Key Advantages

1. **Zero Storage**: No intermediate HDF5 files (~5TB saved)
2. **K562-Informed**: 2.3x enrichment of relevant cell types
3. **Flexible**: Easy to train on different layers, strategies, or cell counts
4. **Reproducible**: Single command, deterministic sampling (seed=42)
5. **Scalable**: Can train on 500k or full 900k cells without storage concerns

---

## Expected Benefits for K562 Perturbation Work

### 1. More Interpretable Features
- SAE learned on stem/progenitor cells (33% of dataset)
- K562 are blast-like cells → better alignment with training distribution
- Features should capture regulatory programs active in undifferentiated states

### 2. Better Transfer Learning
- Myeloid-enriched features (32% of dataset)
- K562 are myeloid leukemia cells
- Steering vectors should be more meaningful when applied to K562

### 3. Relevant Biology
- HSC/MPP/GMP/MEP features → differentiation programs
- Monocyte/granulocyte features → myeloid lineage programs
- These are the exact programs perturbed in K562 experiments

---

## Next Steps

### Short-term:
1. Understand if the epoch by chunk strategy is okay. That is, training 10 epochs per ~400 cell chunk, instead of training 10 epochs on the 100k dataset (currently too slow). You can compare loss curves from training (offline vs online, `sae_training_curve.png` vs `training_curve_online.png` at `plots/sae/layer_12`) to aid in your analysis.
2. **Test on PBMC3K first** (validate implementation)
   ```bash
   # Run train_sae_online_minimal.py
   python scripts/train_sae_online_minimal.py \
       --layer 12 \
       --expansion 8 \
       --k 32 \
   ```

2. **Compare with offline SAE** (if you have PBMC3K offline SAE)
   - Loss curves
   - Reconstruction quality
   - Feature activation patterns


3. **Run full CELLxGENE training** (~20 hours)
   ```bash
   python scripts/train_sae_online_100k.py --layer 12 --expansion 8 --k 32
   ```






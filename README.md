# Mechanistic Interpretability for Single-Cell Biology

## Project Overview

This project focuses on mechanistic interpretability for single-cell biology, specifically by training Sparse Autoencoders (SAEs) on frozen single-cell Foundation Model (scFM) activations to enable interpretable perturbation prediction.

**Goal:** Train a model that "understands" biology by making interpretable predictions. For example, to understand myeloid leukemia:
1.  Train a SAE on activations of a specific layer of a single-cell foundation model (scFM) using a large dataset (CELLxGENE bone marrow).
2.  Perform feature interpretation on the trained SAE (Gene Ontology enrichment, clustering).
3.  Train a small MLP adapter to predict steering vectors for the frozen scFM to match perturbed states (Replogle K562 dataset).

**Pipeline:**
1.  **SAE Training:** Train SAE on scFM layer activations using CELLxGENE bone marrow data (~900k cells).
2.  **Interpretation:** Interpret features via gene ontology enrichment.
3.  **Steering Prediction:** Train MLP adapter at layer (i) of the scFM on Replogle K562 to predict steering vectors:
    $$ x^i = x^i + \text{MLP}(x^i, \text{pert\_id}) \cdot W_{dec} $$
    *   Minimize the MSE between perturbed expression and steered scFM output.
    *   Interpret predictions by analyzing the steering vector $\text{MLP}(x^i, \text{pert\_id}) \cdot W_{dec}$.

## Hardware
*   **GPU:** RTX-3090 (24GB VRAM)
*   **RAM:** Cluster with 70GB RAM

## Repository Structure

```
mech_interp_bio/
├── ModelGenerator/          # AIDO.Cell (GenBio AI) - primary scFM (git ignored)
├── scBERT/                  # Tencent scBERT - alternative scFM (git ignored)
├── scGPT/                   # Reference only (not used - HVG filtering) (git ignored)
├── papers/                  # academic papers with info on models (git ignored)
├── scripts/                 # scripts for straightforward actions: model comparison, inference, etc
├── notebooks/               # notebooks for more complex experiments
├── reports/                 # .md's of the project
├── plots/                   # generated plots from scripts and notebooks
|   ├── sae/                 # plots of saes from different layers                
└── data/                    # (git ignored)
    ├── cell_x_gene/         # CELLxGENE bone marrow
    ├── replogle/            # K562 perturbation dataset
    └── pbmc/                # PBMC3K for testing
```

## Foundation Models

### AIDO.Cell (`/ModelGenerator`)
**Current and only model in use.**
*   Configs in `experiments/AIDO_Cell/` (YAML, composable)
*   Supports DDP/FSDP via Lightning
*   **Repos:**
    *   Full: [GenBio AI ModelGenerator](https://github.com/genbio-ai/ModelGenerator/tree/main)
    *   HuggingFace: [AIDO.Cell HF](https://github.com/genbio-ai/ModelGenerator/tree/main/huggingface/aido.cell)

### scBERT (`/scBERT`)
**Not currently in use.** Discarded due to issues representing cell type clusters in UMAP.
*   **Repo:** [Tencent scBERT](https://github.com/TencentAILabHealthcare/scBERT)

### scGPT (`/scGPT`)
**Not currently in use.** Discarded due to HVG filtering bias.

## Current Progress

### Steering Techniques
1.  **Gene Expression Steering:** Optimized steering vector to increase monocyte marker (S100A9) in B-cells while keeping NK markers low. (`notebooks/aido_steer.ipynb`)
2.  **Contrastive Steering:** Optimized steering vector to move B-cell embeddings closer to monocyte centroid. **Resulted in much better performance.** (`notebooks/aido_contrastive_steer.ipynb`)

### SAE Training and Interpretation
1.  **Training:** Trained SAEs on layers 9, 12, and 15.
2.  **Interpretation:**
    *   **Gene Ontology Enrichment:** Analyzed top activating genes (Participation Ratio) for GO enrichment.
    *   **Feature Scale:** Found variation in feature scale (PR). Negative correlation between PR and max gene activation, and PR and decoder weight norm.
    *   **Coactivation:** Implemented matrix-agnostic coactivation analysis (`scripts/compute_feature_correlations.py`). Found generally low similarity between features, but some pairs show high gene/GO overlap.
    *   **Feature Graphs:** Built graphs connecting features with high gene overlap, finding coherent connected components.
3.  **Online Training:** Implemented and validated an online SAE training scheme (`reports/online_training_report.md`).

## Blockers
*   **Dynamic Range Clamping:** Model underscores high expressed genes and has issues scoring zero-expressed genes. Attempts to fix via decoder retraining were unsuccessful. Likely an inherent model limitation.

## TODOs

### Interpretation
1.  **Feature Steerability:** Experiment with steering strong interpretable features to observe cell fate changes.
    *   *Goal:* Train an MLP to linearly combine features to achieve a specific biological outcome.
2.  **Quantify Interpretability:** Compare against random SAEs or raw activations.
3.  **Negative Steering:** Explore "cell identity" features by negatively steering features and measuring cell movement from centroids.

## Coding Style Guide
*   Write clean code: reuse functions from utils scripts.
*   Keep scripts minimal and extend iteratively.
*   **File Writing:** Always write `.py` files, do not write `.ipynb` files programmatically.

---
*Created on Friday, January 23, 2026.*

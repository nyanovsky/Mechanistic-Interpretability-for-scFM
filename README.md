# Mechanistic Interpretability for Single-Cell Biology

Training Sparse Autoencoders (SAEs) on frozen single-cell Foundation Model (scFM) activations to extract interpretable biological features. These features enable interpretable in-silico experiments by steering the scFM's internal representations using learned feature directions against a desired objective.

**Paper (MLGenX, Gen2 workshops at ICLR 26'):** [Dissecting and Steering Cell Identity in a Single-Cell Foundation Model Using Sparse Autoencoders](https://openreview.net/forum?id=6B3zp01ubG)

## Pipeline

1. **Train SAE** on scFM layer activations (CELLxGENE bone marrow, ~900k cells)
2. **Interpret features** via Gene Ontology enrichment on top-activating genes per feature
3. **Analyze features** — coactivation, feature graphs, scale analysis
4. **Steer** — train adapters to predict which features to steer to achieve an objective in output expression space:


## Repository Structure

```
scripts/
├── utils/               # Shared utilities (SAE model, data, GO, similarity, steering)
├── training/            # SAE training scripts
├── interpretation/      # Feature interpretation & matrix computation
├── analysis/            # Feature analysis, correlations, graphs
├── steering/            # Steering experiments & visualization
└── data_prep/           # Activation extraction from scFM

```

## Requirements

### Foundation Model

This project uses [AIDO.Cell](https://github.com/genbio-ai/ModelGenerator) as the single-cell foundation model. Clone it into the repository root:

```bash
git clone https://github.com/genbio-ai/ModelGenerator.git
```

### Environment

```bash
conda activate aido_env
pip install -r requirements.txt
```

### PBMC3K Dataset

The POC experiments use the PBMC3K dataset. Download via scanpy:

```python
import scanpy as sc

adata = sc.datasets.pbmc3k()          # raw counts
adata.write("data/pbmc/pbmc3k_raw.h5ad")

adata = sc.datasets.pbmc3k_processed() # processed (normalized, annotated. We only use this to annotate cell-types in the raw data.)
adata.write("data/pbmc/pbmc3k_processed.h5ad")
```

### Gene Ontology Data

GO enrichment requires:
- `go-basic.obo` — GO ontology file ([download](http://geneontology.org/ontology/go-basic.obo))
- `goa_human.gaf` — human GO annotations ([download](http://current.geneontology.org/annotations/goa_human.gaf.gz))

### Python Dependencies

See `requirements.txt`. Key packages:
- `torch`, `lightning` — model training
- `scanpy`, `anndata` — single-cell data handling
- `goatools`, `gseapy` — Gene Ontology enrichment
- `networkx` — feature graph analysis
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn` — standard scientific stack

# Mechanistic Interpretability for Single-Cell Biology

Training Sparse Autoencoders (SAEs) on frozen single-cell Foundation Model (scFM) activations to extract interpretable biological features. These features enable interpretable perturbation prediction by steering the scFM's internal representations using learned feature directions.

**Paper:** [Sparse Autoencoders Reveal Interpretable Biological Features in Single-Cell Foundation Models](https://openreview.net/forum?id=6B3zp01ubG)

## Pipeline

1. **Train SAE** on scFM layer activations (CELLxGENE bone marrow, ~900k cells)
2. **Interpret features** via Gene Ontology enrichment on top-activating genes per feature
3. **Analyze features** — coactivation, feature graphs, scale analysis
4. **Steer** — train MLP adapter to predict steering vectors for interpretable perturbation prediction:

$$x^i = x^i + \text{MLP}(x^i, \text{pert\_id}) \cdot W_{dec}$$

## Repository Structure

```
mech_interp_bio/
├── scripts/
│   ├── utils/               # Shared utilities (SAE model, data, GO, similarity, steering)
│   ├── training/            # SAE training scripts
│   ├── interpretation/      # Feature interpretation & matrix computation
│   ├── analysis/            # Feature analysis, correlations, graphs
│   ├── steering/            # Steering experiments & visualization
│   └── data_prep/           # Activation extraction from scFM
├── reports/                 # Experiment reports and project journal
└── data/                    # Datasets (not tracked)
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

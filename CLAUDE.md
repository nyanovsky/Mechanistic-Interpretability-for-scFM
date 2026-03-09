# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Mechanistic interpretability for single-cell biology: training Sparse Autoencoders (SAEs) on frozen single-cell Foundation Model (scFM) activations to extract interpretable biological features, then using those features for interpretable perturbation prediction via steering vectors.

## Repository Structure

```
mech_interp_bio/
├── ModelGenerator/          # AIDO.Cell (GenBio AI) - the scFM (git ignored)
├── scripts/
│   ├── utils/               # Shared utilities (see below)
│   ├── training/            # SAE training scripts
│   ├── interpretation/      # Feature interpretation & matrix computation
│   ├── analysis/            # Feature analysis, correlations, graphs
│   ├── steering/            # Steering experiments & visualization
│   └── data_prep/           # Activation extraction
├── reports/                 # Lab journal (git ignored, see below)
├── notebooks/               # Notebooks (git ignored)
├── plots/                   # Generated plots (git ignored)
├── summaries/               # GO annotation summaries for different SAEs (git ignored)   
└── data/                    # Datasets (git ignored)

```

## How to Run

```bash
conda activate aido_env
```

- AIDO.Cell lives in `ModelGenerator/`. Clone from https://github.com/genbio-ai/ModelGenerator
- Scripts import shared code from `scripts/utils/` via relative imports
- GO enrichment requires `go-basic.obo` and `goa_human.gaf` data files

## Utils Organization

| Module | Contents | Use when... |
|---|---|---|
| `sae_model.py` | `TopKSAE` class, `load_sae()` | Loading or defining SAE models |
| `data_utils.py` | Gene/matrix loading, expression filtering | Loading data, gene lists, activation matrices |
| `go_utils.py` | GO DAG, IC computation, semantic similarity, term overlap | GO enrichment, feature annotation, IC analysis |
| `similarity.py` | Cosine, set-based overlap, pairwise correlations, clustering | Feature comparison, coactivation analysis |
| `steering.py` | `ActivationHook`, `SAESteeringModel`, `SteeringExperiment` | Running steering experiments |

## Coding Conventions

- **Reuse utils**: check `scripts/utils/` before writing new helpers. Add shared logic there.
- **Write .py, not .ipynb**: do not create notebook files programmatically.
- **Keep scripts short**: one clear purpose per script. Start minimal, validate, then extend.
- **Validate incrementally**: run intermediate checks before building on results.

## Lab Journal

`reports/` is the project's lab journal. Read there when you need to gather context for experiment's results or past analysis. Write there for:
- Experiment results and their interpretation
- Design decisions and rationale
- Progress updates on major milestones

Key reports:
- `reports/project_progress.md` — full project history (steering, SAE, coactivation, controls)
- `reports/old_models.md` — discarded models (scBERT, scGPT) and why
- `reports/online_training_report.md` — online SAE training design
- `reports/steering_results.md` — steering experiment results
- `reports/cd4_steering.md` — CD4 T -> CD8 T steering

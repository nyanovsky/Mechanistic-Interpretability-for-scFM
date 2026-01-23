# GEMINI.md

This file provides guidance to GEMINI CLI when working with code in this repository.

## Project Overview

Mechanistic interpretability for single-cell biology: training SAEs on frozen scFM activations for interpretable perturbation prediction.

**Pipeline:**
1. Train SAE on scFM layer activations using CELLxGENE bone marrow data (~900k cells)
2. Interpret features via gene ontology enrichment
3. Train MLP adapter at layer (i) of the scFM on Replogle K562 to predict steering vectors for interpretable predictions: 
   `x^i = x^i + MLP(x^i, pert_id) * W_dec`
   1. Minimize the MSE between perturbed expression and steered scFM output
   2. Interpret predictions by analyzing the steering vector `MLP(x^i, pert_id) * W_dec` 

**Hardware:** RTX-3090 (24GB VRAM), cluster with 70GB RAM

## Repository Structure

```
mech_interp_bio/
├── ModelGenerator/          # AIDO.Cell (GenBio AI) - primary scFM
├── scBERT/                  # Tencent scBERT - alternative scFM
├── scGPT/                   # Reference only (not used - HVG filtering)
├── papers/                  # academic papers with info on models
├── scripts/                 # scripts for straightforward actions: model comparison, inference, etc
├── notebooks/               # notebooks for more complex experiments
├── reports/                 # .md's of the project
├── plots/                   # generated plots from scripts and notebooks
|   ├── sae/                 # plots of saes from different layers                
└── data/
    ├── cell_x_gene/         # CELLxGENE bone marrow
    ├── replogle/            # K562 perturbation dataset
    └── pbmc/                # PBMC3K for testing
```

## Coding style guide

- Write clean code: reuse functions from utils scripts and dont overcomplicate scripts, which should not be so long. If you plan to do a complex task, it is best to start in a minimal fashion, validate with the user, and then extend.

## Foundation Model

### AIDO.Cell (`/ModelGenerator`)

**Current and only model in use**

```bash
# Activate environment
conda activate aido_env
```
- Configs in `experiments/AIDO_Cell/` (YAML, composable)
- Supports DDP/FSDP via Lightning
- Key tutorials: `quickstart.ipynb`, `tutorial_cell_classification.ipynb`
- Academic paper found in `../papers/aido.pdf`
- full github repo: https://github.com/genbio-ai/ModelGenerator/tree/main
- huggingface implementation: https://github.com/genbio-ai/ModelGenerator/tree/main/huggingface/aido.cell

#### Current Progress

I'm doing smalls POCs of my project on the PBMC3K dataset.

**Stering techniques**:
1. Gene expression steering, where I optimized the steering vector so that the expression of a monocyte marker (S100A9) would be high in B-cells while keeping NK markers (GNLY and NKG7) low. Noteobook named `aido_steer.ipynb` at notebooks folder, before and after steer UMAP named `umap_bcell_steering.png` and b-cell distance toward monocytes plot named `bcell_distance_analysis.png` at plots folder.
2. Contrastive steering, where I optimized the steering vector so that the B-cell embeddings would move closer to the monocyte centroid embedding. Notebook named `aido_contrastive_steer.ipynb` at notebooks folder. UMAP named `contrastive_umap_bcell_steering.png` and distance analysis named `contrastive_distance_analysis.ipynb`.
   
The second approach resulted in much better results.

**SAE training and interpretation**:
1. I trained a SAE on the model's 9th, 12th and 15th layer activations, with a script for interpretation in `/scripts/interpret_sae.py` (or its parallelized version `interpret_sae_parallel.py`), and a script for summarizing the findings in the interpretation script in `/scripts/summarize_significant_features.py` 
The interpretation script creates a feature x gene activation matrix, where values correspond to how much a gene activates a feature (averaged across cells), and a feature x cell activation matrix (same as before, but averaged across genes). 
With this, it analyzes the following for the top n_layer activating features:
- Gene ontology enrichment: takes top n_genes_i activating genes for each feature i and does GO enrichment analysis. The number of genes selected is proportional to the number of genes that activate the feature i, where I use the Participation Ratio (PR) as a measure of how many genes activate a certain feature
- Celltype correlation: It analyzes correlation between features and celltypes
I've annotated the top 3000 features for the layers 9 and 15th SAEs, and all the 5120 for the layer 12, finding significantly enriched features in all layers (a bit more that 70% of features have significantly annotated terms). Summaries can be found in `layer_i_summary.txt`

2. Feature scale analysis:
   Using Gene Participation Ratio (PR) as a measure of how many genes activate a certain feature (see `analyze_feature_scale.py` and `analyze_feature_scale_detailed.py` at scripts folder), I found that obtained features vary in this scale (see `feature_scale_distribution.png` at plots folder):
   Feature which activate on very few genes, features which activate on medium sized sets of genes, and so on. Moreso, there is negative correlation between PR and maximum gene activation values, see `feature_scale_vs_peak_activation.png` at each of the SAE plots subfolders. The plots also shows that the most activating features (which are the annotated), are always of higher PR than the 2000 that have not yet been annotated.
   A lower negative correlation also exists between a feature's PR and it's decoder weights norm (see `feature_scale_vs_norm.png`), which may be of use later when steering (maybe we'll need to normalize the steering vector, or bound the scale by which we steer a feature depending on it's PR value) 
2. Layer comparison: I've redacted comparison of the feature interpretations across layers at `sae_interpretation_report.md`
4. I've implemented an online SAE training scheme in advance for the full SAE training on CellxGene bone marrow data and validated the approach with PBMC3K cells. See `reports/online_training_report.md`
5. I've implemented a matrix agnostic (that is, it works with the feature-gene activation matrix, or the feature-cell activation matrix) coactivation script (`compute_feature_correlations.py`, utils at `sae_analysis_utils.py`). I computed pariwise feature similarities with the faeture-gene matrix till now:
   - spearman correlation
   - cosine similarity
   - a custom overlap coefficient: for a given pair of features f_i, f_j, lets call their top activating genes A_i and A_j, which we define as the top p*PR_i (or PR_j, respectively), where p is a proportion between 0 and 1. Then the distance is  |A_i ∩ A_j|/min(|A_i|, |A_j|). That is two features have a high coefficient if any of the two top activating genes is contained in the other.
   - a custom GO overlap coefficient: works the same way as the gene overlap coefficient, but just takes each feature's GO annotations (G_i, G_j) and computes G_i ∩ G_j/min(|G_i|, |G_j|). The coefficient just tells how much of the GO annotations of the smaller feature (smaller in terms of the number of GO annotated terms to it) are present in the bigger feature annotations
   - Lin similarity (sample 2m out of the ~8.4m pairs of features)
   - cosine similarity of decoder weights
  
You can see the similarity distributions at `plots/sae/layer_12/coactivation/`. All of them show little similarity between pairs of features, meaning that features encode mostly different concepts. Still, there are a handful of pairs of features which share similar concepts/top activating genes/have high Lin similarity. I analyzed pairs of varying GO overlap (see `explore_feature_pairs.py` at scripts folder), and found that there doesnt seem to be much relationship between the GO overlap coefficient and the gene overlap coefficient. That is, two features may be annotated to similar GO terms, but activate on different sets of genes (see `reports/feature_pairs_go_overlap_full.txt`). I did the same thing for high gene overlap pairs (see the report at `reports/gene_overlap_analysis.txt`). In this case, features that share top activating genes do have high overlap of GO terms which makes sense.
Given this findings (the fact that high overlaping GO didnt mean high overlapping genes), I feared that the annotations I obtained for the features were not specific (low IC). I analyzed this at `scripts/analyze_go_term_specificity` which ruled out my fear. It is true that features with high overlapping GO terms have less informative annotations than expected (see the `ic_overlay_comparison.png` at the coactivation folder), but this is not true in general (see `feature_ic_bias_overlay.png` at the coactivation folder). 
Lastly, I wanted to see the how was the relationship between a feature's number of significantly annotated terms and their PR/mean IC. It looks like there's a slight negative correlation between PR and number of significantly enriched temrs annotated for a given feature (and the more terms a feature has annotated the more it's mean IC regresses to the mean, see `num_annotations_scatter.png` at the coactivation folder). I interpret this thinking that the more genes that activate a feature, the less specific this feature is (more polysemantic), so it's difficult to annotate it. Moreso, when you can annotate them, their annotations are less specific.

6. Using a stricter gene overlap coefficient (|A_i ∩ A_j|/sqrt(|A_i|)sqrt(|A_j|), to prevent getting pairs with a coefficient of 1 which have, say 5 genes and 500 genes), and looking at it's distribution `coactivation_gene_overlap_sqrt_filtered.png`, I decided to build a graph connecting features if they had overlap bigger than a certain threshold. I went with 0.2 as a threshold given that values lower than that left the graph with a few huge components and very fragmented. I then analyzed the conected components with more than 3 features and got some coherent conected components, which have a mean feature GO overlap distribution much higher than the all-pairs distribution (`graph_analysis/cc_go_overlap_hist_overlap_Sqrt_t0.2.png` at the `plots/sae/layer_12` folder). I generated a summary of the connected components at `graph_analysis/cc_analysis_overlap_sqrt_t0.2.txt`.
 

#### Current blockers

1. Model clamps dynamic range (underescores high expressed genes), and has issues scoring zero-expressed genes, shown at `input_vs_output_expression.png` at plots folder, with specific gene examples in `boxplot_*.png` plots (* being a placeholder for every plot which starts with the preceding string).
I've tried to fix this by training a new decoder on the final gene embeddings to predict expression using the original pretraining objective (MLM reconstruction), but it didn't change anything (see `decoder_input_vs_output.png`).
The issue is probably inherent to the model and can't be fixed. I dont worry too much about this.

#### TODOs:

**Interpretation**   
   
1. Think of experiments to test feature steerability. This can probably be taking a strong interpretable feature, or a combination of features, which we should expect to have a clear result on a cell's fate when steering (some genes end upregulated/downregulated, the whole celltype moves towards another celltype, etc). 
   - Taking this a step further and directly related to the project's goal (this would be the gold standard POC experiment) would be to take a desired outcome (choosing an outcome which should be possible given the biology of PBMC3K cells), and then train a small MLP that linearly combines the obtained features for steering to achieve the outcome. Then analyze the coefficents of the combination to see if the interpretation makes sense.

2. Think of further experiments to quantify interpretability of features. This can possibly include testing a random SAE, and/or testing interpretability of activations directly, instead of features.
   
3. Think of other types of feature interpretation techiniques beyond GO enrichment. One thing that occured to me was to do negative steering with each feature and quantify how much the cells from each celltype move (on average). This would let us discover "cell identity" specific features (if doing negative steering with one feature is not enough, we could train an MLP that steers multiple features to maximize the distance of the cell types to their original centroid) 

## Browser usage
- Whenever accessing a browser page and reading its contents, specifically a github one, output a SUCCESS message if successful. Else, output an ERROR message and do not use the failed page.

## File writing
- Do not write .ipynb files. Always write .py files instead. 

## OLD MODELS

### scBERT (`/scBERT`)

**NOT IN USE**
Discarded due to not being able to represent cell type clusters in UMAP

```bash
# Activate environment
conda activate scbert_env
```

**Preprocessing pipeline:**
```python
# 1. Align genes to Panglao reference (16,906 genes)
# 2. Normalize
sc.pp.normalize_total(data, target_sum=1e4)
sc.pp.log1p(data, base=2)
# 3. Binning is implicit: floor(log2(1 + expr)), capped at 5
```

Key files: `preprocess.py`
Reference data: `data/pretrained/panglao_10000.h5ad`, `gene2vec_16906.npy`

### scGPT (`/scGPT`)

**NOT IN USE**
Discarded due to not being able to ingest full transcriptomes, worked with HVG selection which introduced bias.

```bash
# Activate environment
conda activate scgpt_env
```
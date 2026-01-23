# SAE Interpretation Report: Mechanistic Interpretability in scFM (AIDO.Cell)

This report summarizes the interpretation of Sparse Autoencoders (SAEs) trained on the activations of the AIDO.Cell foundation model at layers 9, 12, and 15. The analysis utilizes the PBMC3K dataset to correlate learned features with specific cell types and biological functions.

## Methodology

SAEs were trained with an 8x expansion factor (K=32, L=5120) on frozen activations. Features were interpreted via:
1.  **Gene Ontology (GO) Enrichment:** Analyzing the top activating genes for each feature (number of genes proportional to the Participation Ratio).
2.  **Cell Type Correlation:** Correlating feature activations across cells with known cell type labels in PBMC3K.

Approximately 2200-2400 features per layer were found to have significant GO annotations.

---

## Layer 9: Structural Components & Basic Lineage Markers

Layer 9 appears to encode fundamental cellular building blocks and early lineage markers. The features often map to physical structures or broad metabolic activities.

### Highlighted Features
*   **Feature 4846 (CD14+ Monocytes, r=0.49):** Strongly associated with **azurophil granules** (GO:0042582) and cytoskeleton. Azurophil granules are lysosomes found in neutrophils and monocytes, making this a highly interpretable structural feature.
*   **Feature 3696 (CD4 T cells, r=0.82):** Captures **CCR1 and CCR5 chemokine receptor binding** (GO:0031726, GO:0031730). These receptors are critical for T cell trafficking, representing a clear lineage-specific surface marker.
*   **Feature 525 (CD4 T cells, r=0.73):** Associated with **MHC class I receptor activity** and oxysterol binding, linking T cell identity with antigen recognition machinery.

### Ambiguous/confusing Features
*   **Feature 2081 (CD14+ Monocytes, r=0.33):** Annotated with **synaptic vesicle transport**. While immune cells utilize secretion machinery similar to neurons, the explicit mapping to "synaptic" terms is biologically inexact for PBMCs, likely reflecting the model's repurposing of general secretion pathways.

---

## Layer 12: Intermediate Regulation & Signaling Pathways

Layer 12 represents a transition zone, showing a mix of metabolic features and increasingly specific signaling pathways.

### Highlighted Features
*   **Feature 3181 (CD14+ Monocytes, r=0.41):** Captures the **Type I interferon signaling pathway** (GO:0060337). This is a distinct and biologically coherent transcriptional program essential for monocyte antiviral response.
*   **Feature 621 (CD14+ Monocytes, r=0.66):** Strongly related to **neutrophil degranulation**. Given the shared myeloid lineage between monocytes and neutrophils, this feature likely encodes shared phagocytic and degranulation machinery.
*   **Feature 108 (B cells, r=0.53):** Linked to **negative regulation of cell growth**. This may capture the quiescent state of naive B cells circulating in the blood.

### Ambiguous/confusing Features
*   **Feature 2749 (CD4 T cells, r=-0.27):** Shows negative correlation with T cells and maps to **Wnt signaling and phototransduction**. The reference to phototransduction is likely a "hallucination" or repurposing of signal transduction genes (like rhodopsin-like GPCRs) used in other contexts.

---

## Layer 15: High-Level Regulation & Response to Stimulus

Layer 15 (a deeper layer) encodes the most complex and specific regulatory circuits, often governing cell fate and active responses to environmental stimuli.

### Highlighted Features
*   **Feature 4847 (CD4 T cells, r=0.77):** **Regulation of interleukin-4 production** (GO:0032753). This is a highly specific feature related to Th2 helper T cell differentiation, indicating the model has resolved specific effector functions.
*   **Feature 3842 (CD14+ Monocytes, r=0.63):** **Response to interferon-gamma** (GO:0071346). Unlike the Type I interferon feature in Layer 12, this captures the distinct response to Type II interferon (IFN-g), a crucial macrophage-activating signal.
*   **Feature 1381 (CD4 T cells, r=0.60):** **Interferon receptor activity** and transmembrane transport, further emphasizing the resolution of cytokine sensing networks in this layer.

### Ambiguous/confusing Features
*   **Feature 2450 (CD14+ Monocytes, r=0.58):** Annotated with **secondary alcohol biosynthetic process** and cholesterol biosynthesis. While lipids are relevant to monocytes (e.g., for foam cell formation), the specific GO term "secondary alcohol" is somewhat generic and chemically abstract compared to the biological reality.

---

## Comparative Analysis: Trends Across Layers

1.  **Complexity Gradient:** There is a discernible shift from physical structures (ribosomes, granules) in Layer 9 to complex signaling (interferon response, cytokine regulation) in Layer 15.
2.  **Specificity:** Later layers show higher correlation coefficients for regulatory features, suggesting the model builds sharper representations of cell states (e.g., "Th2 differentiation") as depth increases.
3.  **Negative Correlations:** Deep layers frequently exhibit features with strong negative correlations to specific cell types, potentially representing lineage-exclusion signals (e.g., genes that *must* be off for a cell to maintain its identity).

---

## Feature Scale & Interpretability Analysis

Analysis of the **Participation Ratio (PR)** and feature activation reveals a critical insight into the model's structure:

*   **PR vs. Activation Order:** There is a strong correlation between a feature's Participation Ratio (how many genes it activates) and its total activation mass across the dataset. The "Annotated" features (top 3000 most activating) consistently have higher PRs than the non-annotated tail.
*   **The "peak" vs. "sum" trade-off:** Although the `feature_scale_representatives` plots show that low-PR features often induce *stronger peak activation* in individual genes, this intensity is not sufficient to compensate for their sparsity.
*   **Implication:** "Interpretable" features in this model are predominantly those that activate broad, co-regulated gene programs (high PR). Features that drive widely distributed changes contribute more to the model's total energy than those that drive intense but sparse changes (low PR). This suggests that the model prioritizes "broad strokes" of gene regulation (e.g., entire pathways) over highly specific single-gene adjustments.

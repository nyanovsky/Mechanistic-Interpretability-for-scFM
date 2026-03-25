"""SAE analysis and steering utilities for mechanistic interpretability of scFMs."""

import os
import sys

# Add AIDO.Cell to path (single canonical location)
_aido_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ModelGenerator', 'huggingface', 'aido.cell')
_aido_path = os.path.abspath(_aido_path)
if _aido_path not in sys.path:
    sys.path.insert(0, _aido_path)

# Core re-exports
from .sae_model import TopKSAE, TopKSAEWithBPre, load_sae
from .data_utils import (
    load_gene_names, get_expressed_genes, get_expressed_genes_mask,
    get_annotated_features, load_go_enrichment, load_go_enrichment_detailed,
    load_feature_statistics, load_feature_attribution_data,
    load_decoder_weights, load_activation_matrices, load_celltype_correlations,
    compute_participation_ratio, get_device
)

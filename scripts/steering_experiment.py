"""
Command-line interface for running SAE steering experiments.

This script provides a CLI for running predefined steering experiments
on AIDO.Cell using trained SAEs.

Usage:
    python steering_experiment.py --experiment viral_defense --data-file data/pbmc/pbmc3k_raw.h5ad

Available experiments:
    - viral_defense: Viral Defense State (Component 4)
    - neutrophil_inflammatory: Neutrophil/Inflammatory Program (Component 7)
    - tcr_signaling: TCR Signaling Module (Component 2A)
    - high_activation_t: High-Activation CD4 T State (Component 2C)
    - bcell_receptor: B-Cell Receptor Signaling (Component 11)
    - cholesterol_metabolism: Cholesterol Metabolism (Component 6)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import anndata as ad

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ModelGenerator/huggingface/aido.cell'))

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata

from steering_utils import (
    SteeringConfig,
    SteeringExperiment,
    load_feature_statistics,
    TopKSAE,
    load_sae
)


def get_experiment_configs() -> dict:
    """Get predefined experiment configurations."""
    configs = {
        'viral_defense': SteeringConfig(
            name="Viral Defense State (Component 4)",
            steering_features=[4367], #[673, 1398, 1790, 2316, 2381, 2593, 3251, 4367, 4827]
            alphas=[0, 0.5, 1, 2, 5],
            layer_idx=12,
            reference='max',
            description="Test if steering interferon/viral response features upregulates ISGs"
        ),

        'neutrophil_inflammatory': SteeringConfig(
            name="Neutrophil/Inflammatory Program (Component 7)",
            steering_features=[799], #[799, 891, 1721, 3101, 4298],
            alphas=[0, 0.5, 1, 2, 5],
            layer_idx=12,
            reference='max',
            description="Test if steering neutrophil features induces inflammatory state"
        ),

        'tcr_signaling': SteeringConfig(
            name="TCR Signaling Module (Component 2A)",
            steering_features=[395, 2628, 4414, 4508],
            alphas=[0, 0.5, 1, 2, 5],
            layer_idx=12,
            reference='max',
            description="Test T-cell activation via TCR signaling features"
        ),

        'high_activation_t': SteeringConfig(
            name="High-Activation CD4 T State (Component 2C)",
            steering_features=[1086, 2706, 3959],
            alphas=[0, 0.5, 1, 2, 5],
            layer_idx=12,
            reference='max',
            description="Test sustained T-cell activation state"
        ),

        'bcell_receptor': SteeringConfig(
            name="B-Cell Receptor Signaling (Component 11)",
            steering_features=[3170], #[469, 997, 1741, 3170]
            alphas=[0, 0.5, 1, 2, 5],
            layer_idx=12,
            reference='max',
            description="Test B-cell identity/activation via BCR features"
        ),

        'cholesterol_metabolism': SteeringConfig(
            name="Cholesterol Metabolism (Component 6)",
            steering_features=[206, 282, 2188, 2600, 4188],
            alphas=[0, 0.5, 1, 2, 5],
            layer_idx=12,
            reference='max',
            description="Test metabolic rewiring without cell identity change"
        ),
    }

    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Run SAE steering experiments on AIDO.Cell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Experiment configuration
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['viral_defense', 'neutrophil_inflammatory', 'tcr_signaling',
                'high_activation_t', 'bcell_receptor', 'cholesterol_metabolism'],
        help='Which experiment to run'
    )

    # Model and data paths
    parser.add_argument(
        '--model-name',
        type=str,
        default='genbio-ai/AIDO.Cell-100M',
        help='AIDO.Cell model name or path'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to input .h5ad file'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=12,
        help='Layer to steer (0-indexed)'
    )
    parser.add_argument(
        '--sae-dir',
        type=str,
        help='Path to SAE directory (default: auto-detect from layer)'
    )

    # Steering parameters
    parser.add_argument(
        '--reference',
        type=str,
        default='max',
        choices=['max', 'mean', 'median', 'p90'],
        help='Reference statistic for steering'
    )
    parser.add_argument(
        '--alphas',
        type=float,
        nargs='+',
        help='Alpha values to test (overrides config defaults)'
    )
    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip baseline (no steering) run'
    )

    # Processing parameters
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--max-cells',
        type=int,
        help='Maximum number of cells to process (for testing)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/steering',
        help='Directory to save results'
    )

    args = parser.parse_args()

    # Setup paths
    if args.sae_dir is None:
        BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
        args.sae_dir = f"{BASE_DIR}/sae_k_32_5120"

    INTERPRETATION_DIR = f"{args.sae_dir}/interpretations_filter_zero_expressed"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("SAE STEERING EXPERIMENT")
    print("="*70)
    print(f"Experiment: {args.experiment}")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_file}")
    print(f"Layer: {args.layer}")
    print(f"SAE directory: {args.sae_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print("="*70)

    # Load model
    print("\n1. Loading AIDO.Cell model...")
    config = CellFoundationConfig.from_pretrained(args.model_name)
    model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config)
    model = model.to(args.device)

    if args.device == "cuda":
        model = model.to(torch.bfloat16)

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print("✓ Model loaded")

    # Load SAE
    print("\n2. Loading SAE...")
    sae = load_sae(args.sae_dir, args.device)
    print("✓ SAE loaded")

    # Load feature statistics
    print("\n3. Loading feature statistics...")
    feature_stats = load_feature_statistics(INTERPRETATION_DIR)
    print("✓ Feature statistics loaded")

    # Load and align data
    print("\n4. Loading and aligning data...")
    adata = ad.read_h5ad(args.data_file)
    print(f"Loaded data with shape: {adata.shape}")

    adata_aligned, attention_mask = align_adata(adata)
    print(f"Aligned data with shape: {adata_aligned.shape}")

    # Optionally subset cells
    cell_indices = None
    if args.max_cells is not None and args.max_cells < adata_aligned.n_obs:
        print(f"Subsampling to {args.max_cells} cells...")
        import numpy as np
        cell_indices = np.random.choice(adata_aligned.n_obs, args.max_cells, replace=False)

    # Get experiment config
    configs = get_experiment_configs()
    config = configs[args.experiment]

    # Override layer and alphas if specified
    config.layer_idx = args.layer
    config.reference = args.reference
    if args.alphas is not None:
        config.alphas = args.alphas

    # Initialize experiment
    print("\n5. Initializing experiment...")
    experiment = SteeringExperiment(
        model=model,
        sae=sae,
        feature_stats=feature_stats,
        attention_mask=attention_mask,
        layer_idx=args.layer,
        device=args.device,
        batch_size=args.batch_size
    )
    print("✓ Experiment initialized")

    # Run experiment
    print("\n6. Running steering experiment...")
    results = experiment.run_experiment(
        config,
        adata_aligned,
        cell_indices=cell_indices,
        include_baseline=not args.no_baseline
    )

    # Save results
    output_file = os.path.join(
        args.output_dir,
        f"{args.experiment}_layer{args.layer}_results.pt"
    )
    print(f"\n7. Saving results to {output_file}...")
    torch.save({
        'config': config,
        'results': results,
        'args': vars(args)
    }, output_file)
    print("✓ Results saved")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults structure:")
    print(f"  - Results keys: {list(results.keys())}")
    print(f"  - Each result contains: logits, embeddings, cell_indices, alpha")
    print(f"\nNext steps:")
    print(f"  1. Analyze gene expression changes (logits)")
    print(f"  2. Visualize embedding shifts (UMAP)")
    print(f"  3. Identify differentially expressed genes")


if __name__ == "__main__":
    main()

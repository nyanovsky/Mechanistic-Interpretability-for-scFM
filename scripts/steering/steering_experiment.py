"""
Command-line interface for running SAE steering experiments.

Usage:
    # Run with explicit features and alphas:
    python steering_experiment.py --features 3079 --alphas 0 0.5 1 2 5 \
        --data-file data/pbmc/pbmc3k_raw.h5ad

    # Run with a name for the experiment:
    python steering_experiment.py --features 3079 4687 --alphas 0 2 5 \
        --name "cytotoxic_features" --data-file data/pbmc/pbmc3k_raw.h5ad

    # Run on specific cell types only:
    python steering_experiment.py --features 3079 --alphas 0 2 5 \
        --celltypes "CD4 T cells" "CD8 T cells" --data-file data/pbmc/pbmc3k_raw.h5ad
"""

import os
import sys
import argparse
import torch
import anndata as ad
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_sae
from utils.steering import SteeringConfig, SteeringExperiment

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata


def main():
    parser = argparse.ArgumentParser(
        description='Run SAE steering experiments on AIDO.Cell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Steering configuration
    parser.add_argument('--features', type=int, nargs='+', required=True,
                        help='Feature IDs to steer')
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0, 0.5, 1, 2, 5],
                        help='Alpha values to test (default: 0 0.5 1 2 5)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: auto-generated from features)')

    # Model and data paths
    parser.add_argument('--model-name', type=str, default='genbio-ai/AIDO.Cell-100M',
                        help='AIDO.Cell model name or path')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to input .h5ad file')
    parser.add_argument('--layer', type=int, default=12,
                        help='Layer to steer (0-indexed, default: 12)')
    parser.add_argument('--sae-dir', type=str, default=None,
                        help='Path to SAE directory (default: auto-detect from layer)')

    # Cell type filtering
    parser.add_argument('--celltypes', type=str, nargs='+', default=None,
                        help='Cell types to include (default: all cells)')
    parser.add_argument('--processed-file', type=str,
                        default='data/pbmc/pbmc3k_processed.h5ad',
                        help='Processed .h5ad with celltype annotations (used with --celltypes)')

    # Processing parameters
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: auto)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for processing (default: 4)')
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Maximum number of cells to process')

    # Output
    parser.add_argument('--output-dir', type=str, default='results/steering',
                        help='Directory to save results')

    args = parser.parse_args()

    # Setup paths
    if args.sae_dir is None:
        BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
        args.sae_dir = f"{BASE_DIR}/sae_k_32_5120"

    # Auto-generate experiment name
    if args.name is None:
        features_str = "_".join(map(str, args.features))
        args.name = f"feat_{features_str}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SAE STEERING EXPERIMENT")
    print("=" * 70)
    print(f"Name: {args.name}")
    print(f"Features: {args.features}")
    print(f"Alphas: {args.alphas}")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_file}")
    print(f"Layer: {args.layer}")
    print(f"SAE: {args.sae_dir}")
    print(f"Cell types: {args.celltypes if args.celltypes else 'all'}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    # Load model
    print("\n1. Loading AIDO.Cell model...")
    config = CellFoundationConfig.from_pretrained(args.model_name)
    model = CellFoundationForMaskedLM.from_pretrained(args.model_name, config=config)
    model = model.to(args.device)

    if args.device == "cuda":
        model = model.to(torch.bfloat16)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("  Model loaded")

    # Load SAE
    print("\n2. Loading SAE...")
    sae = load_sae(args.sae_dir, args.device)
    print("  SAE loaded")

    # Load and align data
    print("\n3. Loading and aligning data...")
    adata = ad.read_h5ad(args.data_file)
    print(f"  Loaded: {adata.shape}")

    adata_aligned, attention_mask = align_adata(adata)
    print(f"  Aligned: {adata_aligned.shape}")

    # Optionally filter by cell type
    cell_indices = None
    if args.celltypes is not None:
        print(f"\n  Filtering to cell types: {args.celltypes}")
        adata_processed = ad.read_h5ad(args.processed_file)

        if 'celltype' in adata_processed.obs.columns:
            cell_type_col = 'celltype'
        elif 'louvain' in adata_processed.obs.columns:
            cell_type_col = 'louvain'
        else:
            raise ValueError("No celltype column found in processed file")

        # Match cells between aligned and processed data
        common_cells = [name for name in adata_aligned.obs_names if name in adata_processed.obs_names]
        processed_types = adata_processed.obs.loc[common_cells, cell_type_col]
        matching_cells = processed_types[processed_types.isin(args.celltypes)].index.tolist()

        # Convert to indices in aligned adata
        name_to_idx = {name: i for i, name in enumerate(adata_aligned.obs_names)}
        cell_indices = np.array([name_to_idx[name] for name in matching_cells])

        print(f"  Found {len(cell_indices)} cells matching requested types")
        for ct in args.celltypes:
            n = (processed_types.loc[[c for c in matching_cells if c in processed_types.index]] == ct).sum()
            print(f"    {ct}: {n}")

    # Optionally subsample
    if args.max_cells is not None:
        if cell_indices is None:
            if args.max_cells < adata_aligned.n_obs:
                print(f"  Subsampling to {args.max_cells} cells...")
                cell_indices = np.random.choice(adata_aligned.n_obs, args.max_cells, replace=False)
        elif args.max_cells < len(cell_indices):
            print(f"  Subsampling to {args.max_cells} cells...")
            cell_indices = np.random.choice(cell_indices, args.max_cells, replace=False)

    # Build steering config
    steering_config = SteeringConfig(
        name=args.name,
        steering_features=args.features,
        alphas=args.alphas,
        layer_idx=args.layer,
    )

    # Initialize experiment
    print("\n4. Initializing experiment...")
    experiment = SteeringExperiment(
        model=model,
        sae=sae,
        attention_mask=attention_mask,
        layer_idx=args.layer,
        device=args.device,
        batch_size=args.batch_size
    )
    print("  Experiment initialized")

    # Run experiment
    print("\n5. Running steering experiment...")
    results = experiment.run_experiment(
        steering_config,
        adata_aligned,
        cell_indices=cell_indices,
    )

    # Save results
    output_file = os.path.join(
        args.output_dir,
        f"{args.name}_layer{args.layer}_results.pt"
    )
    print(f"\n6. Saving results to {output_file}...")
    torch.save({
        'config': steering_config,
        'results': results,
        'args': vars(args)
    }, output_file)
    print("  Results saved")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results keys: {list(results.keys())}")
    print(f"Each result contains: logits, embeddings, cell_indices, alpha")


if __name__ == "__main__":
    main()

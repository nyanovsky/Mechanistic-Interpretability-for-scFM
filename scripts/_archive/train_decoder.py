import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
# Import ModelGenerator tasks
from modelgenerator.tasks import Embed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ModelGenerator/huggingface/aido.cell'))

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata

RAW_DATA_FILE = "../data/pbmc/pbmc3k_raw.h5ad"
PROCESSED_DATA_FILE = "../data/pbmc/pbmc3k_processed.h5ad"
OUTPUT_DIR = "../data/pbmc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
BATCH_SIZE = 4  # Small due to memory constraints
ACCUMULATION_STEPS = 8  # Effective batch size = 4 * 8 = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
MASK_RATIO = 0.15
MASK_VALUE = 1.0  # Small positive count as mask token (distinguishable from true zeros after normalization)
HIDDEN_SIZE = 640
INTERMEDIATE_SIZE = 256

print(f"Device: {DEVICE}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, RAW_DATA_FILE)
    processed_data_path = os.path.join(script_dir, PROCESSED_DATA_FILE)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)

    print("Loading data...")
    adata_processed = ad.read_h5ad(processed_data_path)
    adata_raw = ad.read_h5ad(raw_data_path)

    # Get common cells
    common_cells = adata_raw.obs_names.intersection(adata_processed.obs_names)
    adata_raw = adata_raw[common_cells].copy()

    # Align to AIDO.Cell vocab
    adata_aligned, attention_mask = align_adata(adata_raw)
    print(f"Aligned data shape: {adata_aligned.shape}")
    print(f"Non-zero genes: {attention_mask.sum()}")

    model = Embed.from_config({
        "model.backbone": "aido_cell_100m",
        "model.batch_size": BATCH_SIZE
    }).to(DEVICE).to(torch.bfloat16)

    model.configure_model()
    model.eval()

    class MLPExpressionDecoder(nn.Module):
        """MLP decoder that predicts expression from per-gene hidden states."""
        
        def __init__(self, hidden_size=640, intermediate_size=256):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.LayerNorm(intermediate_size),
                nn.Linear(intermediate_size, 1)
            )
        
        def forward(self, hidden_states):
            # hidden_states: [batch, n_genes, hidden_size]
            return self.decoder(hidden_states).squeeze(-1)  # [batch, n_genes]
        
    # Prepare data
    n_cells = adata_aligned.n_obs
    n_genes = adata_aligned.n_vars
    n_valid_genes = int(attention_mask.sum())  # Number of genes in AIDO.Cell vocab

    # Train/val split
    train_idx, val_idx = train_test_split(
        list(range(n_cells)),
        test_size=0.2,
        random_state=42
    )

    print(f"Training cells: {len(train_idx)}")
    print(f"Validation cells: {len(val_idx)}")
    print(f"Valid genes (in vocab): {n_valid_genes}")

    # Prepare attention mask tensor
    attn_mask_bool = torch.tensor(attention_mask, dtype=torch.bool)

    # Initialize decoder
    decoder = MLPExpressionDecoder(
        hidden_size=HIDDEN_SIZE, 
        intermediate_size=INTERMEDIATE_SIZE
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Training loop
    print("="*60)
    print("Training MLP Expression Decoder")
    print("="*60)
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Accumulation steps: {ACCUMULATION_STEPS} (effective batch = {BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"Mask ratio: {MASK_RATIO}")
    print(f"Mask value: {MASK_VALUE} (distinguishes masked from true zeros)")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*60)

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        # Training
        decoder.train()
        epoch_train_loss = 0
        n_train_batches = 0
        optimizer.zero_grad()

        # Shuffle training indices
        np.random.shuffle(train_idx)

        n_batches = (len(train_idx) + BATCH_SIZE - 1) // BATCH_SIZE
        pbar = tqdm(enumerate(range(0, len(train_idx), BATCH_SIZE)),
                    total=n_batches,
                    desc=f"Epoch {epoch+1:2d}/{NUM_EPOCHS}")

        for batch_idx, batch_start in pbar:
            batch_indices = train_idx[batch_start:batch_start+BATCH_SIZE]
            batch_counts = adata_aligned.X[batch_indices].toarray()

            # Create mask for valid genes only (mask MASK_RATIO of valid genes per cell)
            gene_mask = torch.rand(len(batch_indices), n_valid_genes) < MASK_RATIO

            # Expand mask back to full gene space for masking input
            full_gene_mask = torch.zeros(len(batch_indices), n_genes, dtype=torch.bool)
            full_gene_mask[:, attn_mask_bool] = gene_mask

            # Mask the input counts (use MASK_VALUE instead of 0 to distinguish from true zeros)
            masked_counts = batch_counts.copy()
            masked_counts[full_gene_mask.numpy()] = MASK_VALUE

            # Forward through frozen encoder with masked input
            with torch.no_grad():
                batch_tensor = torch.from_numpy(masked_counts).to(torch.bfloat16).to(DEVICE)
                batch_transformed = model.transform({'sequences': batch_tensor})

                hidden_states = model(batch_transformed).last_hidden_state
                hidden_states = hidden_states[:, attention_mask.astype(bool), :]

                if epoch == 0 and batch_start == 0:
                    print(f"Shape of hidden_states: {hidden_states.shape}")

            # Create target as log1p(CPM) for valid genes only (no depth tokens)
            batch_counts_tensor = torch.from_numpy(batch_counts).float().to(DEVICE)
            total_counts = batch_counts_tensor.sum(dim=1, keepdim=True)
            target = torch.log1p(batch_counts_tensor / total_counts * 10000)
            target = torch.clamp(target, max=20)
            target = target[:, attn_mask_bool]  # [batch, n_valid_genes]

            # Forward through decoder
            pred = decoder(hidden_states.detach().float())

            # Loss only on masked positions (gene_mask is already [batch, n_valid_genes])
            gene_mask_device = gene_mask.to(DEVICE)
            loss = F.mse_loss(pred[gene_mask_device], target[gene_mask_device])

            # Gradient accumulation
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_loss += loss.item() * ACCUMULATION_STEPS  # Scale back for logging
            n_train_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}")

        # Handle remaining gradients at end of epoch
        if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        avg_train_loss = epoch_train_loss / n_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation (predict ALL genes, compute loss on all valid genes)
        decoder.eval()
        epoch_val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch_start in range(0, len(val_idx), BATCH_SIZE):
                batch_indices = val_idx[batch_start:batch_start+BATCH_SIZE]
                batch_counts = adata_aligned.X[batch_indices].toarray()

                batch_tensor = torch.from_numpy(batch_counts).to(torch.bfloat16).to(DEVICE)
                batch_transformed = model.transform({'sequences': batch_tensor})

                hidden_states = model(batch_transformed).last_hidden_state
                hidden_states = hidden_states[:, attention_mask.astype(bool), :]

                # Create target as log1p(CPM) for valid genes only
                batch_counts_tensor = torch.from_numpy(batch_counts).float().to(DEVICE)
                total_counts = batch_counts_tensor.sum(dim=1, keepdim=True)
                target = torch.log1p(batch_counts_tensor / total_counts * 10000)
                target = torch.clamp(target, max=20)
                target = target[:, attn_mask_bool]  # [batch, n_valid_genes]

                pred = decoder(hidden_states.float())

                # Loss on all valid genes (pred and target are both [batch, n_valid_genes])
                loss = F.mse_loss(pred, target)
                
                epoch_val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = epoch_val_loss / n_val_batches
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    print("\nTraining complete!")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MLP Decoder Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(script_dir, '../plots/decoder_training_curves.png'), dpi=150)
    plt.show()

    # Evaluation: Input vs Output scatter plot
    print("Evaluating decoder on validation set...")

    all_targets = []
    all_preds = []

    decoder.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(val_idx), BATCH_SIZE), desc="Evaluating"):
            batch_indices = val_idx[batch_start:batch_start+BATCH_SIZE]
            batch_counts = adata_aligned.X[batch_indices].toarray()

            # Create target as log1p(CPM) for valid genes only
            batch_counts_tensor = torch.from_numpy(batch_counts).float().to(DEVICE)
            total_counts = batch_counts_tensor.sum(dim=1, keepdim=True)
            target = torch.log1p(batch_counts_tensor / total_counts * 10000)
            target = torch.clamp(target, max=20)
            target = target[:, attn_mask_bool]  # [batch, n_valid_genes]

            batch_tensor = torch.from_numpy(batch_counts).to(torch.bfloat16).to(DEVICE)
            batch_transformed = model.transform({'sequences': batch_tensor})

            hidden_states = model(batch_transformed).last_hidden_state
            hidden_states = hidden_states[:, attention_mask.astype(bool), :]

            pred = decoder(hidden_states.float())

            # Both pred and target are [batch, n_valid_genes]
            for i in range(len(batch_indices)):
                all_targets.append(target[i].cpu().numpy())
                all_preds.append(pred[i].cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Compute metrics
    mse = np.mean((all_targets - all_preds) ** 2)
    pearson_r, _ = pearsonr(all_targets, all_preds)

    print(f"\nMSE: {mse:.4f}")
    print(f"Pearson r: {pearson_r:.4f}")


    # Scatter plot: Input vs Output
    fig, ax = plt.subplots(figsize=(8, 8))

    # Subsample for plotting (too many points)
    n_points = min(100000, len(all_targets))
    idx = np.random.choice(len(all_targets), n_points, replace=False)

    ax.scatter(all_targets[idx], all_preds[idx], alpha=0.1, s=1)
    ax.plot([all_targets.min(), all_targets.max()], 
            [all_targets.min(), all_targets.max()], 
            'r--', linewidth=2, label='y=x')

    ax.set_xlabel('Input Expression (log1p(CPM))')
    ax.set_ylabel('Predicted Expression')
    ax.set_title(f'MLP Decoder: Input vs Output\nMSE = {mse:.4f}, Pearson r = {pearson_r:.4f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, '../plots/decoder_input_vs_output.png'), dpi=150)
    plt.show()


        # Analyze zero-expression predictions
    zero_mask = all_targets == 0
    nonzero_mask = all_targets > 0

    print("Zero-expression analysis:")
    print(f"  Zero inputs: {zero_mask.sum():,}")
    print(f"  Non-zero inputs: {nonzero_mask.sum():,}")
    print(f"  Mean prediction for zero inputs: {all_preds[zero_mask].mean():.4f}")
    print(f"  Std prediction for zero inputs: {all_preds[zero_mask].std():.4f}")

    # Plot histogram of predictions for zero inputs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(all_preds[zero_mask], bins=50, alpha=0.7)
    axes[0].axvline(0, color='r', linestyle='--', label='Target (0)')
    axes[0].set_xlabel('Predicted Expression')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Predictions for Zero-Expression Genes')
    axes[0].legend()

    # Predictions vs targets for high expression genes
    high_expr_mask = all_targets > 3
    axes[1].scatter(all_targets[high_expr_mask], all_preds[high_expr_mask], alpha=0.3, s=5)
    axes[1].plot([3, all_targets.max()], [3, all_targets.max()], 'r--', label='y=x')
    axes[1].set_xlabel('Input Expression')
    axes[1].set_ylabel('Predicted Expression')
    axes[1].set_title('Predictions for High-Expression Genes (>3)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, '../plots/decoder_zero_and_high_analysis.png'), dpi=150)
    plt.show()


    # Save decoder
    decoder_path = os.path.join(output_dir, 'mlp_decoder.pt')
    torch.save({
        'model_state_dict': decoder.state_dict(),
        'hidden_size': HIDDEN_SIZE,
        'intermediate_size': INTERMEDIATE_SIZE,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_mse': mse,
        'final_pearson_r': pearson_r
    }, decoder_path)

    print(f"Decoder saved to {decoder_path}")


if __name__ == "__main__":
    main()
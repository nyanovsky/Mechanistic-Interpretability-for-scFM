#%%
import os
import torch
import scanpy as sc
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
# Import ModelGenerator tasks
from modelgenerator.tasks import Embed
import sys
import torch
import torch.nn as nn
import anndata as ad
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
sys.path.insert(0, '../ModelGenerator/huggingface/aido.cell')

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts

#sys.path.insert(0, '../ModelGenerator/')


#%%
MODEL_NAME = "genbio-ai/AIDO.Cell-100M"
RAW_DATA_FILE = "../data/pbmc/pbmc3k_raw.h5ad"
PROCESSED_DATA_FILE = "../data/pbmc/pbmc3k_processed.h5ad"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Small batch size to prevent OOM

adata_processed = ad.read_h5ad(PROCESSED_DATA_FILE)
adata_raw = ad.read_h5ad(RAW_DATA_FILE)

cell_types = adata_processed.obs['louvain']

common_cells = adata_raw.obs_names.intersection(adata_processed.obs_names)

# Filter raw data to common cells and get cell types for those cells
adata_raw = adata_raw[common_cells].copy()
cell_types = cell_types.loc[common_cells]

adata_aligned, attention_mask = align_adata(adata_raw)

adata_aligned.obs["cell_type"] = cell_types.values

### model with decoder ###
config = CellFoundationConfig.from_pretrained(MODEL_NAME)
model = CellFoundationForMaskedLM.from_pretrained(MODEL_NAME, config=config)
model = model.to(DEVICE)

if DEVICE == "cuda":
    model = model.to(torch.bfloat16)

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

model.eval()
print("Model with decoder loaded and frozen")

### only embedding model ###
model_embed = Embed.from_config({
        "model.backbone": "aido_cell_100m",
        "model.batch_size": BATCH_SIZE
    }).to(DEVICE).to(torch.bfloat16)


model_embed.configure_model()
model_embed.eval() 
print("Embedding model loaded and frozen")

                ###------------------ CELL EMBEDDINGS FROM DECODER MODEL ----------###
# Get hidden size from config
hidden_size = config.hidden_size
print(f"Hidden size: {hidden_size}")

# Prepare attention mask
n_genes = adata_aligned.shape[1]
attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(DEVICE)

# Run monocytes through model to get target S100A9 level
print("Running selected genes through model to get expression levels after forward pass...")
cell_embeddings_from_dec = []

hidden_states = {}

def capture_hook(module, input, output):
    hidden_states['last_hidden_state'] = output

  # Register hook on the final layer norm of the encoder
hook_handle = model.bert.encoder.ln.register_forward_hook(capture_hook)

try:
    with torch.no_grad():
        for i in tqdm(range(0,adata_aligned.n_obs, BATCH_SIZE), desc="Processing monocytes"):

            batch_counts = adata_aligned[i:i+BATCH_SIZE].X.toarray()

            batch_processed = preprocess_counts(batch_counts, device=DEVICE)
            batch_attn_mask = attn_mask_tensor.repeat(batch_processed.shape[0], 1)
            depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=DEVICE)
            batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

            outputs = model(input_ids=batch_processed, attention_mask=batch_attn_mask, return_dict=True)

            last_hidden_states = hidden_states["last_hidden_state"][:,:-2,:]
            last_hidden_states = last_hidden_states[:, attention_mask.astype(bool), :]
            if i == 0:
                print(f"Shape of last_hidden_states: {last_hidden_states.shape}")

            cell_embeddings_from_dec.append(last_hidden_states.mean(dim=1).float().cpu().numpy())
finally:
    hook_handle.remove()
# Calculate mean expression for each gene

cell_embeddings_from_dec = np.concatenate(cell_embeddings_from_dec, axis=0)


            ###----------------CELL EMBEDDINGS FROM EMBEDDER-------------###

cell_embeddings_from_emb = []
with torch.no_grad():
    for i in tqdm(range(0, adata_aligned.n_obs, BATCH_SIZE), desc="Embedding cells"):
        batch_np = adata_aligned[i: i+BATCH_SIZE].X.toarray()
        batch_tensor = torch.from_numpy(batch_np).to(torch.bfloat16).to(DEVICE)
        batch_transformed = model_embed.transform({'sequences': batch_tensor})

        embs = model_embed(batch_transformed)
        embs = embs[:, attention_mask.astype(bool), :].last_hidden_state

        if i == 0:
            print(f"Shape of embs: {embs.shape}")
        

        cell_emb = embs.float().mean(dim=1).cpu().numpy()
        cell_embeddings_from_emb.append(cell_emb)

cell_embeddings_from_emb = np.concatenate(cell_embeddings_from_emb, axis=0)


# Get L2 distance between cells, plot distribution and save figure
distances = np.linalg.norm(cell_embeddings_from_dec - cell_embeddings_from_emb, axis=1)

plt.figure(figsize=(10, 6))
plt.hist(distances, bins=50, density=True, alpha=0.7, color='blue')
plt.title('Distribution of L2 Distances Between Cell Embeddings')
plt.xlabel('L2 Distance')
plt.ylabel('Density')
plt.savefig('../plots/embed_aido_comp.png', dpi=150)
plt.show()
plt.close()

# %%

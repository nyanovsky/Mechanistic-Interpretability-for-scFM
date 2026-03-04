"""Create a random (untrained) SAE baseline with default PyTorch init (Kaiming uniform)."""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import TopKSAE


parser = argparse.ArgumentParser(description="Create a random (untrained) SAE baseline")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the random SAE")
parser.add_argument("--expansion", type=int, default=2, help="Expansion factor (default: 2)")
parser.add_argument("--input-dim", type=int, default=640, help="Input dimension (default: 640)")
parser.add_argument("--k", type=int, default=32, help="Top-K sparsity (default: 32)")
parser.add_argument("--b-pre", action="store_true",
                    help="Use learned pre-encoder centering bias")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

sae = TopKSAE(input_dim=args.input_dim, expansion=args.expansion, k=args.k, use_b_pre=args.b_pre)

checkpoint = {
    "input_dim": args.input_dim,
    "expansion": args.expansion,
    "k": args.k,
    "use_b_pre": args.b_pre,
    "model_state_dict": sae.state_dict(),
}

path = os.path.join(args.output_dir, "topk_sae.pt")
torch.save(checkpoint, path)
print(f"Saved random SAE to {path} (input_dim={args.input_dim}, expansion={args.expansion}, k={args.k}, b_pre={args.b_pre})")

"""Validate adaptive gene selection by analyzing GO enrichment success by PR range.

Uses existing GO enrichment results to identify which PR ranges have low success rates.

Usage:
    python validate_adaptive_selection.py --layer 12
"""

import os
import argparse
import numpy as np
import pandas as pd
import glob as glob_module


def parse_args():
    parser = argparse.ArgumentParser(description='Validate adaptive gene selection')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--expansion', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    latent_dim = 640 * args.expansion
    base_dir = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}/sae_k_32_{latent_dim}"
    interp_dir = f"{base_dir}/interpretations_filter_zero_expressed"

    # Load PR values
    pr_path = os.path.join(interp_dir, 'feature_participation_ratios.npy')
    if not os.path.exists(pr_path):
        print(f"ERROR: {pr_path} not found")
        return
    pr_values = np.load(pr_path)

    # Find all enrichment directories
    feature_dirs = glob_module.glob(os.path.join(interp_dir, "feature_*_enrichr"))

    # Check which features have significant enrichment and collect all feature IDs
    enriched_features = set()
    all_feature_ids = set()

    for fdir in feature_dirs:
        feat_id = int(fdir.split('_')[-2])
        all_feature_ids.add(feat_id)

        # Check for significant results
        txt_files = glob_module.glob(os.path.join(fdir, "*.txt"))
        has_enrichment = False
        for txt in txt_files:
            try:
                df = pd.read_csv(txt, sep='\t')
                if 'Adjusted P-value' in df.columns and (df['Adjusted P-value'] < 0.05).any():
                    has_enrichment = True
                    break
            except:
                continue

        if has_enrichment:
            enriched_features.add(feat_id)

    # Analyze by PR bins
    pr_bins = [0, 5, 20, 100, 500, 10000]
    pr_labels = ['<5', '5-20', '20-100', '100-500', '>500']

    results = []
    for feat_id in all_feature_ids:
        if feat_id >= len(pr_values):
            continue
        pr = pr_values[feat_id]
        has_go = feat_id in enriched_features

        # Find PR bin
        pr_bin = None
        for i, (low, high) in enumerate(zip(pr_bins[:-1], pr_bins[1:])):
            if low <= pr < high:
                pr_bin = pr_labels[i]
                break

        results.append({
            'feature_id': feat_id,
            'pr': pr,
            'pr_bin': pr_bin,
            'has_go': has_go
        })

    df = pd.DataFrame(results)

    # Print results
    print("="*70)
    print(f"Annotation Success by PR Range (Layer {args.layer})")
    print("="*70)
    print(f"{'PR Range':<15} {'Success':<10} {'Total':<10} {'Rate':<10}")
    print("-" * 70)

    for pr_range in pr_labels:
        subset = df[df['pr_bin'] == pr_range]
        if len(subset) == 0:
            continue
        success = subset['has_go'].sum()
        total = len(subset)
        rate = success / total * 100
        print(f"{pr_range:<15} {success:<10} {total:<10} {rate:>6.1f}%")

    overall = df['has_go'].sum() / len(df) * 100
    print("-" * 70)
    print(f"{'Overall':<15} {df['has_go'].sum():<10} {len(df):<10} {overall:>6.1f}%")
    print("="*70)

    # Adaptive gene count suggestions
    print("\nAdaptive gene counts (scale_factor=0.6, bounds=[10,100]):")
    for pr_range in pr_labels:
        subset = df[df['pr_bin'] == pr_range]
        if len(subset) > 0:
            mean_pr = subset['pr'].mean()
            suggested = int(mean_pr * 0.6)
            suggested = max(10, min(suggested, 100))
            print(f"  PR {pr_range:<10} (mean={mean_pr:>6.1f}) → {suggested:>3} genes (vs fixed 30)")


if __name__ == '__main__':
    main()

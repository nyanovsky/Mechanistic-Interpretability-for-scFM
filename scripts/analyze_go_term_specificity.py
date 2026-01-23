"""Analyze Information Content of GO terms.

This script:
1. Tests whether features with high GO term overlap share specific (high IC) or generic (low IC) terms
2. Tests whether features are biased toward generic terms in general
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from goatools.semantic import get_info_content
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# Import utility functions
sys.path.insert(0, os.path.dirname(__file__))
from sae_analysis_utils import (
    load_go_enrichment,
    load_go_dag_and_associations
)


def compute_all_go_dag_ics(godag, term_counts):
    """Compute information content for ALL GO terms in the DAG.

    Args:
        godag: GO DAG object
        term_counts: TermCounts object

    Returns:
        dict: {GO_ID: IC_value}
    """
    print("\nComputing IC for ALL GO terms in DAG...")

    all_go_ids = list(godag.keys())
    print(f"Found {len(all_go_ids)} GO terms in DAG")

    term_ics = {}
    for go_id in tqdm(all_go_ids, desc="Computing IC for all GO terms"):
        ic = get_info_content(go_id, term_counts)
        term_ics[go_id] = ic

    print(f"Computed IC for {len(term_ics)} GO terms")

    return term_ics


def compute_feature_ic_statistics(go_enrichments, all_term_ics):
    """Compute mean, median, and max IC for each feature's annotated terms.

    Args:
        go_enrichments: dict {feature_id: set(GO_IDs)}
        all_term_ics: dict {GO_ID: IC_value}

    Returns:
        tuple: (mean_ics_dict, median_ics_dict, max_ics_dict)
    """
    print("\nComputing mean, median, and max IC per feature...")

    feature_mean_ics = {}
    feature_median_ics = {}
    feature_max_ics = {}

    for feature_id, go_terms in tqdm(go_enrichments.items(), desc="Processing features"):
        # Get ICs for this feature's terms
        ics = [all_term_ics[term] for term in go_terms if term in all_term_ics]

        if len(ics) > 0:
            feature_mean_ics[feature_id] = np.mean(ics)
            feature_median_ics[feature_id] = np.median(ics)
            feature_max_ics[feature_id] = np.max(ics)

    print(f"Computed statistics for {len(feature_mean_ics)} features")

    return feature_mean_ics, feature_median_ics, feature_max_ics


def analyze_overlap_pairs(go_enrichments, go_overlap_file, indices_file,
                          term_ics, overlap_threshold=0.9):
    """Analyze IC of shared terms in high-overlap pairs.

    Args:
        go_enrichments: dict {feature_id: set(GO_IDs)}
        go_overlap_file: Path to GO overlap .npy file
        indices_file: Path to feature indices .npy file
        term_ics: dict {GO_ID: IC_value}
        overlap_threshold: Threshold for "high overlap" (default: 0.9)

    Returns:
        dict with analysis results
    """
    print(f"\nAnalyzing feature pairs with GO overlap > {overlap_threshold}...")

    # Load overlap matrix and feature indices
    overlaps = np.load(go_overlap_file)
    feature_indices = np.load(indices_file)

    print(f"Loaded {len(overlaps):,} pairwise overlaps for {len(feature_indices)} features")

    # Find high-overlap pairs
    high_overlap_mask = overlaps > overlap_threshold
    n_high_overlap = high_overlap_mask.sum()

    print(f"Found {n_high_overlap:,} pairs with overlap > {overlap_threshold} ({100*n_high_overlap/len(overlaps):.2f}%)")

    # Reconstruct pairs and compute IC stats
    shared_term_ics = []  # IC values for all shared terms in high-overlap pairs

    idx = 0
    for i in range(len(feature_indices)):
        for j in range(i + 1, len(feature_indices)):
            if high_overlap_mask[idx]:
                feat_i = feature_indices[i]
                feat_j = feature_indices[j]

                # Get shared terms
                terms_i = go_enrichments.get(feat_i, set())
                terms_j = go_enrichments.get(feat_j, set())
                shared_terms = terms_i & terms_j

                if len(shared_terms) > 0:
                    # Get ICs for shared terms
                    ics = [term_ics[term] for term in shared_terms if term in term_ics]
                    shared_term_ics.extend(ics)

            idx += 1

    print(f"Collected IC values for {len(shared_term_ics)} shared term instances")

    return {
        'shared_term_ics': np.array(shared_term_ics),
        'n_high_overlap_pairs': n_high_overlap
    }


def plot_ic_distributions(all_term_ics, analysis_results, feature_mean_ics, feature_median_ics, feature_max_ics, output_dir, overlap_threshold):
    """Plot IC distributions and comparisons.

    Args:
        all_term_ics: dict {GO_ID: IC_value} for all terms in DAG
        analysis_results: dict from analyze_overlap_pairs
        feature_mean_ics: dict {feature_id: mean_IC}
        feature_median_ics: dict {feature_id: median_IC}
        feature_max_ics: dict {feature_id: max_IC}
        output_dir: Directory to save plots
        overlap_threshold: Overlap threshold used
    """
    shared_ics = analysis_results['shared_term_ics']

    # Convert to arrays
    all_ics = np.array(list(all_term_ics.values()))
    feature_mean_ics_arr = np.array(list(feature_mean_ics.values()))
    feature_median_ics_arr = np.array(list(feature_median_ics.values()))
    feature_max_ics_arr = np.array(list(feature_max_ics.values()))

    # === Plot 1: Side-by-side comparison ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Baseline: All terms
    ax = axes[0]
    ax.hist(all_ics, bins=50, edgecolor='black', alpha=0.7, color='steelblue', density=True)
    ax.axvline(np.mean(all_ics), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_ics):.2f}')
    ax.axvline(np.median(all_ics), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(all_ics):.2f}')
    ax.set_xlabel('Information Content', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('IC Distribution: ALL GO Terms in DAG', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    # High-overlap pairs: Shared terms
    ax = axes[1]
    if len(shared_ics) > 0:
        ax.hist(shared_ics, bins=50, edgecolor='black', alpha=0.7, color='coral', density=True)
        ax.axvline(np.mean(shared_ics), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(shared_ics):.2f}')
        ax.axvline(np.median(shared_ics), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(shared_ics):.2f}')
        ax.set_xlabel('Information Content', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'IC Distribution: Shared Terms in High-Overlap Pairs (overlap > {overlap_threshold})',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'No shared terms in high-overlap pairs',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ic_distribution_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved IC distribution comparison to {output_path}")
    plt.close()

    # === Plot 2: Overlay comparison ===
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(all_ics, bins=50, edgecolor='black', alpha=0.5, color='steelblue',
            density=True, label='All terms (baseline)')

    if len(shared_ics) > 0:
        ax.hist(shared_ics, bins=50, edgecolor='black', alpha=0.5, color='coral',
                density=True, label=f'Shared terms (overlap > {overlap_threshold})')

    ax.axvline(np.mean(all_ics), color='blue', linestyle='--', linewidth=2, alpha=0.7,
               label=f'All terms mean: {np.mean(all_ics):.2f}')

    if len(shared_ics) > 0:
        ax.axvline(np.mean(shared_ics), color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Shared terms mean: {np.mean(shared_ics):.2f}')

    ax.set_xlabel('Information Content', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('IC Distribution: All Terms vs Shared Terms in High-Overlap Pairs',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ic_overlap_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved IC overlap comparison to {output_path}")
    plt.close()

    # === Plot 3: Feature Bias Analysis - Overlay with all terms (mean, median, max) ===
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(all_ics, bins=50, edgecolor='black', alpha=0.25, color='steelblue',
            density=True, label='All GO terms (baseline)')
    ax.hist(feature_mean_ics_arr, bins=50, edgecolor='black', alpha=0.25, color='coral',
            density=True, label='Feature mean IC')
    #ax.hist(feature_median_ics_arr, bins=50, edgecolor='black', alpha=0.25, color='green',
    #        density=True, label='Feature median IC')
    ax.hist(feature_max_ics_arr, bins=50, edgecolor='black', alpha=0.25, color='green',
            density=True, label='Feature max IC')

    ax.axvline(np.mean(all_ics), color='blue', linestyle='--', linewidth=2, alpha=0.7,
               label=f'All terms mean: {np.mean(all_ics):.2f}')
    ax.axvline(np.mean(feature_mean_ics_arr), color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Feature mean: {np.mean(feature_mean_ics_arr):.2f}')
    #ax.axvline(np.mean(feature_median_ics_arr), color='darkgreen', linestyle='--', linewidth=2, alpha=0.7,
    #           label=f'Feature median: {np.mean(feature_median_ics_arr):.2f}')
    ax.axvline(np.mean(feature_max_ics_arr), color='darkviolet', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Feature max: {np.mean(feature_max_ics_arr):.2f}')

    ax.set_xlabel('Information Content', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('IC Distribution: All GO Terms vs Feature Mean/Max IC',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_ic_bias_overlay.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature IC bias overlay to {output_path}")
    plt.close()


def plot_pr_vs_ic(feature_mean_ics, feature_median_ics, feature_max_ics, pr_values, output_dir):
    """Plot PR vs IC scatter plots to analyze scale vs specificity.

    Args:
        feature_mean_ics: dict {feature_id: mean_IC}
        feature_median_ics: dict {feature_id: median_IC}
        feature_max_ics: dict {feature_id: max_IC}
        pr_values: dict {feature_id: PR_value}
        output_dir: Directory to save plots
    """
    # Align PR values with IC values
    # IMPORTANT: Sort to ensure consistent ordering across all arrays
    common_features = sorted(set(feature_max_ics.keys()) & set(pr_values.keys()))

    prs = np.array([pr_values[f] for f in common_features])
    max_ics = np.array([feature_max_ics[f] for f in common_features])
    median_ics = np.array([feature_median_ics[f] for f in common_features])

    print(f"\nPlotting PR vs IC for {len(common_features)} features...")

    # Compute correlations
    corr_pr_max_spearman, p_pr_max_spearman = spearmanr(prs, max_ics)

    corr_pr_median_spearman, p_pr_median_spearman = spearmanr(prs, median_ics)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: PR vs Max IC
    ax = axes[0]
    ax.scatter(prs, max_ics, alpha=0.5, s=30, color='steelblue',
               edgecolors='black', linewidth=0.3)

    ax.set_xlabel('Participation Ratio (PR)', fontsize=12)
    ax.set_ylabel('Max IC', fontsize=12)
    ax.set_title(f'Feature Scale vs Max Specificity\n' +
                 f'Spearman ρ={corr_pr_max_spearman:.3f} (p={p_pr_max_spearman:.2e})',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xscale('log')

    # Plot 2: PR vs Median IC
    ax = axes[1]
    ax.scatter(prs, median_ics, alpha=0.5, s=30, color='steelblue',
               edgecolors='black', linewidth=0.3)

    ax.set_xlabel('Participation Ratio (PR)', fontsize=12)
    ax.set_ylabel('Median IC', fontsize=12)
    ax.set_title(f'Feature Scale vs Median Specificity\n' +
                 f'Spearman ρ={corr_pr_median_spearman:.3f} (p={p_pr_median_spearman:.2e})',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xscale('log')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pr_vs_ic_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved PR vs IC scatter plots to {output_path}")
    plt.close()

    print("="*70)


def plot_num_annotations_vs_pr_and_ics(pr_values, go_enrichments, feature_mean_ics, output_dir):
    common_features = sorted(set(go_enrichments.keys()) & set(pr_values.keys()) & set(feature_mean_ics.keys()))

    prs = np.array([pr_values[f] for f in common_features])
    num_enrichments = np.array([len(go_enrichments[f]) for f in common_features])
    mean_ics = np.array([feature_mean_ics[f] for f in common_features])

    # Create scatter plot
    print(f"\nPlotting PR vs IC for {len(common_features)} features...")

    # Compute correlations
    corr_num_mean_spearman, p_num_mean_spearman = spearmanr(num_enrichments, mean_ics)

    corr_pr_num_spearman, p_pr_num_spearman = spearmanr(num_enrichments, prs)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: PR vs mean IC
    ax = axes[0]
    ax.scatter(num_enrichments, mean_ics, alpha=0.5, s=30, color='steelblue',
               edgecolors='black', linewidth=0.3)

    ax.set_xlabel('Number of enrichments', fontsize=12)
    ax.set_ylabel('Mean IC', fontsize=12)
    ax.set_title(f'Number of enrichments vs Mean IC\n' +
                 f'Spearman ρ={corr_num_mean_spearman:.3f} (p={p_num_mean_spearman:.2e})',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xscale('log')

    # Plot 2: num_annotations vs mean IC
    ax = axes[1]
    ax.scatter(num_enrichments, prs, alpha=0.5, s=30, color='steelblue',
               edgecolors='black', linewidth=0.3)

    ax.set_xlabel('Number of enrichments', fontsize=12)
    ax.set_ylabel('Participation Ratio (PR)', fontsize=12)
    ax.set_title(f'Number of enrichments vs PR\n' +
                 f'Spearman ρ={corr_pr_num_spearman:.3f} (p={p_pr_num_spearman:.2e})',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')


    plt.tight_layout()
    output_path = os.path.join(output_dir, 'num_annotations_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved PR vs IC scatter plots to {output_path}")
    plt.close()

    print("="*70)


def plot_num_annotations_hist(go_enrichments, output_dir):
    num_annottaions = np.array([len(go_enrichments[f]) for f in go_enrichments])

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(num_annottaions, bins=30, edgecolor='black', alpha=0.7, color='steelblue', density=False)

    ax.set_xlabel('Number of enrichments', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Number of GO Enrichments per Feature',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'num_annotations_hist.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved num annotations histogram to {output_path}")
    plt.close()

    print("="*70)


def save_feature_ic_statistics(feature_mean_ics, feature_median_ics, feature_max_ics, output_dir):
    """Save IC statistics for each feature to a CSV file.

    Args:
        feature_mean_ics: dict {feature_id: mean_IC}
        feature_median_ics: dict {feature_id: median_IC}
        feature_max_ics: dict {feature_id: max_IC}
        output_dir: Directory to save the CSV file
    """
    print("\nSaving IC statistics to CSV...")

    # Get all features that have IC statistics
    all_features = sorted(set(feature_mean_ics.keys()) |
                         set(feature_median_ics.keys()) |
                         set(feature_max_ics.keys()))

    # Create DataFrame
    data = {
        'feature_id': all_features,
        'mean_ic': [feature_mean_ics.get(f, np.nan) for f in all_features],
        'median_ic': [feature_median_ics.get(f, np.nan) for f in all_features],
        'max_ic': [feature_max_ics.get(f, np.nan) for f in all_features]
    }

    df = pd.DataFrame(data)
    df.set_index('feature_id', inplace=True)

    # Save to CSV
    output_file = os.path.join(output_dir, "feature_ic_statistics.csv")
    df.to_csv(output_file)

    print(f"Saved IC statistics to {output_file}")
    print(f"Total features with IC statistics: {len(all_features)}")
    print(f"\nPreview:")
    print(df.head(10))


def print_statistics(all_term_ics, analysis_results, feature_mean_ics, feature_median_ics, feature_max_ics):
    """Print summary statistics."""
    all_ics = np.array(list(all_term_ics.values()))
    shared_ics = analysis_results['shared_term_ics']
    feature_mean_ics_arr = np.array(list(feature_mean_ics.values()))
    feature_median_ics_arr = np.array(list(feature_median_ics.values()))
    feature_max_ics_arr = np.array(list(feature_max_ics.values()))

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print("\n[1] All GO Terms in DAG (Baseline):")
    print(f"  N terms: {len(all_ics)}")
    print(f"  Mean IC: {np.mean(all_ics):.4f}")
    print(f"  Median IC: {np.median(all_ics):.4f}")
    print(f"  Std IC: {np.std(all_ics):.4f}")
    print(f"  Min IC: {np.min(all_ics):.4f}")
    print(f"  Max IC: {np.max(all_ics):.4f}")

    if len(shared_ics) > 0:
        print("\n[2] Shared Terms in High-Overlap Pairs:")
        print(f"  N term instances: {len(shared_ics)}")
        print(f"  Mean IC: {np.mean(shared_ics):.4f}")
        print(f"  Median IC: {np.median(shared_ics):.4f}")
        print(f"  Std IC: {np.std(shared_ics):.4f}")
        print(f"  Min IC: {np.min(shared_ics):.4f}")
        print(f"  Max IC: {np.max(shared_ics):.4f}")

        # Statistical comparison
        print("\n  Comparison (shared vs all):")
        diff_mean = np.mean(shared_ics) - np.mean(all_ics)
        print(f"    Difference in means: {diff_mean:.4f}")

        # Percentile of shared mean in all distribution
        percentile = (all_ics < np.mean(shared_ics)).sum() / len(all_ics) * 100
        print(f"    Shared mean at {percentile:.1f}th percentile of all terms")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(all_ics)**2 + np.std(shared_ics)**2) / 2)
        cohens_d = diff_mean / pooled_std
        print(f"    Effect size (Cohen's d): {cohens_d:.4f}")

        if cohens_d < -0.5:
            print("    → Shared terms are LESS specific (lower IC)")
        elif cohens_d > 0.5:
            print("    → Shared terms are MORE specific (higher IC)")
        else:
            print("    → Shared terms have similar specificity")
    else:
        print("\n[2] No shared terms found in high-overlap pairs")

    print("\n[3] Feature Mean ICs:")
    print(f"  N features: {len(feature_mean_ics_arr)}")
    print(f"  Mean: {np.mean(feature_mean_ics_arr):.4f}")
    print(f"  Median: {np.median(feature_mean_ics_arr):.4f}")
    print(f"  Std: {np.std(feature_mean_ics_arr):.4f}")
    print(f"  Min: {np.min(feature_mean_ics_arr):.4f}")
    print(f"  Max: {np.max(feature_mean_ics_arr):.4f}")

    # Statistical comparison
    print("\n  Comparison (feature means vs all):")
    diff_mean = np.mean(feature_mean_ics_arr) - np.mean(all_ics)
    print(f"    Difference in means: {diff_mean:.4f}")

    # Percentile of feature mean in all distribution
    percentile = (all_ics < np.mean(feature_mean_ics_arr)).sum() / len(all_ics) * 100
    print(f"    Feature mean at {percentile:.1f}th percentile of all terms")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(all_ics)**2 + np.std(feature_mean_ics_arr)**2) / 2)
    cohens_d = diff_mean / pooled_std
    print(f"    Effect size (Cohen's d): {cohens_d:.4f}")

    print("\n[4] Feature Median ICs:")
    print(f"  N features: {len(feature_median_ics_arr)}")
    print(f"  Mean: {np.mean(feature_median_ics_arr):.4f}")
    print(f"  Median: {np.median(feature_median_ics_arr):.4f}")
    print(f"  Std: {np.std(feature_median_ics_arr):.4f}")
    print(f"  Min: {np.min(feature_median_ics_arr):.4f}")
    print(f"  Max: {np.max(feature_median_ics_arr):.4f}")

    # Statistical comparison for medians
    print("\n  Comparison (feature medians vs all):")
    diff_mean_median = np.mean(feature_median_ics_arr) - np.mean(all_ics)
    print(f"    Difference in means: {diff_mean_median:.4f}")

    # Percentile of feature median mean in all distribution
    percentile_median = (all_ics < np.mean(feature_median_ics_arr)).sum() / len(all_ics) * 100
    print(f"    Feature median mean at {percentile_median:.1f}th percentile of all terms")

    # Effect size (Cohen's d) for medians
    pooled_std_median = np.sqrt((np.std(all_ics)**2 + np.std(feature_median_ics_arr)**2) / 2)
    cohens_d_median = diff_mean_median / pooled_std_median
    print(f"    Effect size (Cohen's d): {cohens_d_median:.4f}")

    print("\n[5] Feature Max ICs:")
    print(f"  N features: {len(feature_max_ics_arr)}")
    print(f"  Mean: {np.mean(feature_max_ics_arr):.4f}")
    print(f"  Median: {np.median(feature_max_ics_arr):.4f}")
    print(f"  Std: {np.std(feature_max_ics_arr):.4f}")
    print(f"  Min: {np.min(feature_max_ics_arr):.4f}")
    print(f"  Max: {np.max(feature_max_ics_arr):.4f}")

    # Statistical comparison for max
    print("\n  Comparison (feature maxes vs all):")
    diff_mean_max = np.mean(feature_max_ics_arr) - np.mean(all_ics)
    print(f"    Difference in means: {diff_mean_max:.4f}")

    # Percentile of feature max mean in all distribution
    percentile_max = (all_ics < np.mean(feature_max_ics_arr)).sum() / len(all_ics) * 100
    print(f"    Feature max mean at {percentile_max:.1f}th percentile of all terms")

    # Effect size (Cohen's d) for max
    pooled_std_max = np.sqrt((np.std(all_ics)**2 + np.std(feature_max_ics_arr)**2) / 2)
    cohens_d_max = diff_mean_max / pooled_std_max
    print(f"    Effect size (Cohen's d): {cohens_d_max:.4f}")

    print("\n  Interpretation:")
    # Use the max analysis for interpretation (most telling for specificity)
    if cohens_d_max < -0.5:
        print("    → Even the MOST SPECIFIC terms per feature are below average")
        print("    → Features lack specific biological annotations - CONCERN VALID")
    elif cohens_d_max > 0.5:
        print("    → Features have at least one HIGHLY SPECIFIC term each")
        print("    → Annotations capture specific biological concepts - GOOD")
    else:
        print("    → Feature max specificity similar to general GO terms")
        print("    → Mixed: some features may have specific terms, others may not")

    print(f"\n  Summary across metrics:")
    print(f"    Mean IC:   {np.mean(feature_mean_ics_arr):.2f} (Cohen's d: {cohens_d:.2f})")
    print(f"    Median IC: {np.mean(feature_median_ics_arr):.2f} (Cohen's d: {cohens_d_median:.2f})")
    print(f"    Max IC:    {np.mean(feature_max_ics_arr):.2f} (Cohen's d: {cohens_d_max:.2f})")

    if np.mean(feature_median_ics_arr) > np.mean(feature_mean_ics_arr):
        print("\n    → Median > Mean: Features have some low-IC outliers pulling mean down")
    elif np.mean(feature_median_ics_arr) < np.mean(feature_mean_ics_arr):
        print("\n    → Median < Mean: Features have some high-IC outliers pulling mean up")

    if cohens_d_max > 1.0:
        print("    → Strong evidence: Features encode SPECIFIC concepts")
    elif cohens_d_max > 0.5:
        print("    → Moderate evidence: Features encode SPECIFIC concepts")
    elif cohens_d_max > 0:
        print("    → Weak evidence: Features may encode specific concepts")
    else:
        print("    → Features do NOT systematically encode specific concepts")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze information content of GO terms in high-overlap pairs'
    )
    parser.add_argument('--layer', type=int, default=12,
                        help='AIDO.Cell layer (9, 12, or 15)')
    parser.add_argument('--expansion', type=int, default=8,
                        help='SAE expansion factor')
    parser.add_argument('--k', type=int, default=32,
                        help='Top-K sparsity')
    parser.add_argument('--overlap-threshold', type=float, default=0.9,
                        help='Threshold for high GO overlap (default: 0.9)')

    args = parser.parse_args()

    # Paths
    INPUT_DIM = 640
    LATENT_DIM = INPUT_DIM * args.expansion
    BASE_DIR = f"/biodata/nyanovsky/datasets/pbmc3k/layer_{args.layer}"
    SAE_DIR = f"{BASE_DIR}/sae_k_{args.k}_{LATENT_DIM}"
    INTERPRETATION_DIR = f"{SAE_DIR}/interpretations_filter_zero_expressed"
    COACTIVATION_DIR = f"plots/sae/layer_{args.layer}/coactivation"

    # GO overlap files
    GO_OVERLAP_FILE = os.path.join(COACTIVATION_DIR, "feature_go_overlap_filtered_similarities.npy")
    INDICES_FILE = os.path.join(COACTIVATION_DIR, "feature_go_overlap_filtered_indices.npy")

    # Check files exist
    if not os.path.exists(GO_OVERLAP_FILE):
        print(f"ERROR: GO overlap file not found: {GO_OVERLAP_FILE}")
        print("Run compute_feature_correlations.py with --metric=go_overlap first")
        return

    if not os.path.exists(INDICES_FILE):
        print(f"ERROR: Indices file not found: {INDICES_FILE}")
        return

    print("="*70)
    print("GO TERM SPECIFICITY ANALYSIS")
    print("="*70)
    print(f"Layer: {args.layer}")
    print(f"Expansion: {args.expansion}x")
    print(f"Top-K: {args.k}")
    print(f"Overlap threshold: {args.overlap_threshold}")
    print("="*70)

    # Load GO enrichment
    print(f"\nLoading GO enrichment from {INTERPRETATION_DIR}")
    go_enrichments = load_go_enrichment(INTERPRETATION_DIR, p_threshold=0.05, mode='ids')
    print(f"Loaded enrichment for {len(go_enrichments)} features")

    # Load GO DAG and associations
    print("\nLoading GO DAG and associations...")
    godag, term_counts = load_go_dag_and_associations()

    # Compute IC for ALL GO terms in DAG
    all_term_ics = compute_all_go_dag_ics(godag, term_counts)

    # Compute mean, median, and max IC per feature
    feature_mean_ics, feature_median_ics, feature_max_ics = compute_feature_ic_statistics(go_enrichments, all_term_ics)

    # Save IC statistics to CSV
    save_feature_ic_statistics(feature_mean_ics, feature_median_ics, feature_max_ics, INTERPRETATION_DIR)

    # Load PR values for PR vs IC analysis
    PR_FILE = os.path.join(INTERPRETATION_DIR, "feature_participation_ratios.npy")
    if os.path.exists(PR_FILE):
        print(f"\nLoading participation ratios from {PR_FILE}")
        pr_values_array = np.load(PR_FILE)
        # Create dict mapping feature_id to PR value
        pr_values = {i: pr_values_array[i] for i in range(len(pr_values_array))}
        print(f"Loaded PR values for {len(pr_values)} features")
    else:
        print(f"\nWarning: PR file not found at {PR_FILE}")
        print("Skipping PR vs IC analysis")
        pr_values = None

    # Analyze high-overlap pairs
    analysis_results = analyze_overlap_pairs(
        go_enrichments,
        GO_OVERLAP_FILE,
        INDICES_FILE,
        all_term_ics,
        overlap_threshold=args.overlap_threshold
    )

    # Print statistics
    print_statistics(all_term_ics, analysis_results, feature_mean_ics, feature_median_ics, feature_max_ics)

    # Create plots
    print("\nGenerating plots...")
    plot_ic_distributions(all_term_ics, analysis_results, feature_mean_ics, feature_median_ics, feature_max_ics, COACTIVATION_DIR, args.overlap_threshold)

    # Plot PR vs IC if PR values available
    if pr_values is not None:
        plot_pr_vs_ic(feature_mean_ics, feature_median_ics, feature_max_ics, pr_values, COACTIVATION_DIR)
        plot_num_annotations_vs_pr_and_ics(pr_values, go_enrichments, feature_mean_ics, COACTIVATION_DIR)
    plot_num_annotations_hist(go_enrichments, COACTIVATION_DIR)


    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

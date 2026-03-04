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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import load_go_enrichment
from utils.go_utils import load_go_dag_and_associations, compute_all_go_dag_ics


def get_reachable_go_ics(interp_dir, all_term_ics):
    """Get IC values for GO terms that appear in enrichment results (all tested terms).

    Loads enrichment output files with no p-value filter to collect the full set of
    testable GO terms, then intersects with IC values.

    Args:
        interp_dir: Path to interpretation directory with enrichment results
        all_term_ics: dict {GO_ID: IC_value}

    Returns:
        dict {GO_ID: IC_value} filtered to testable terms
    """
    print("Collecting reachable GO terms from enrichment output files...")
    all_tested = load_go_enrichment(interp_dir, p_threshold=1.0, mode='ids')

    # Union across all features
    reachable_terms = set()
    for terms in all_tested.values():
        reachable_terms |= terms

    reachable_ics = {t: all_term_ics[t] for t in reachable_terms
                     if t in all_term_ics and all_term_ics[t] > 0}

    print(f"  Reachable GO terms (IC > 0): {len(reachable_ics)} / {len(all_term_ics)} total")

    return reachable_ics


def compute_null_max_ic(go_enrichments, ic_pool, n_samples=1000, seed=42):
    """Compute null distribution of max IC by random sampling.

    For each feature with K enriched terms, sample K terms from the IC pool,
    take the max, repeat n_samples times, and store the average max IC.

    Args:
        go_enrichments: dict {feature_id: set(GO_IDs)}
        ic_pool: dict {GO_ID: IC_value} - the pool to sample from
        n_samples: Number of random draws per feature (default: 1000)
        seed: Random seed

    Returns:
        null_max_ics: dict {feature_id: expected_max_IC_under_null}
    """
    rng = np.random.default_rng(seed)
    pool_ics = np.array(list(ic_pool.values()))

    null_max_ics = {}
    for feature_id, go_terms in tqdm(go_enrichments.items(), desc="Computing null max IC"):
        # Use only terms that have a valid IC (matching compute_feature_ic_statistics)
        k = sum(1 for t in go_terms if t in ic_pool)
        if k == 0:
            continue
        samples = rng.choice(pool_ics, size=(n_samples, k), replace=True)
        null_max_ics[feature_id] = samples.max(axis=1).mean()

    return null_max_ics


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


def plot_ic_distributions(all_term_ics, analysis_results, feature_mean_ics, feature_median_ics, feature_max_ics, output_dir, overlap_threshold, null_max_ics=None):
    """Plot IC distributions and comparisons.

    Args:
        all_term_ics: dict {GO_ID: IC_value} for all terms in DAG
        analysis_results: dict from analyze_overlap_pairs
        feature_mean_ics: dict {feature_id: mean_IC}
        feature_median_ics: dict {feature_id: median_IC}
        feature_max_ics: dict {feature_id: max_IC}
        output_dir: Directory to save plots
        overlap_threshold: Overlap threshold used
        null_max_ics: dict {feature_id: expected_max_IC_under_null} (optional)
    """
    shared_ics = analysis_results['shared_term_ics'] if analysis_results else np.array([])

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

    # === Plot 3: Feature Max IC vs Null Model ===
    if null_max_ics is not None:
        # Align: only features present in both
        common = sorted(set(feature_max_ics.keys()) & set(null_max_ics.keys()))
        observed_max = np.array([feature_max_ics[f] for f in common])
        null_max = np.array([null_max_ics[f] for f in common])

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(null_max, bins=50, edgecolor='black', alpha=0.35, color='steelblue',
                density=True, label=f'Null (random sampling)')
        ax.hist(observed_max, bins=50, edgecolor='black', alpha=0.35, color='darkorange',
                density=True, label='Observed feature max IC')

        ax.axvline(np.mean(null_max), color='steelblue', linestyle='--', linewidth=2,
                   label=f'Null mean: {np.mean(null_max):.2f}')
        ax.axvline(np.mean(observed_max), color='darkorange', linestyle='--', linewidth=2,
                   label=f'Observed mean: {np.mean(observed_max):.2f}')

        ax.set_xlabel('Information Content', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Feature Max IC: Observed vs Null Model', fontsize=13)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=12)

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'feature_ic_bias_overlay.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature IC bias overlay to {output_path}")
        plt.close()

        # Print comparison stats
        diff = observed_max - null_max
        print(f"\n  Null model comparison ({len(common)} features):")
        print(f"    Observed mean max IC: {np.mean(observed_max):.3f}")
        print(f"    Null mean max IC:     {np.mean(null_max):.3f}")
        print(f"    Mean difference:      {np.mean(diff):.3f}")
        print(f"    % features above null: {100 * (diff > 0).sum() / len(diff):.1f}%")
    else:
        # Fallback: old plot without null model
        fig, ax = plt.subplots(figsize=(10, 6))
        all_ics_filtered = all_ics[all_ics > 0]

        ax.hist(all_ics_filtered, bins=50, edgecolor='black', alpha=0.35, color='steelblue',
                density=True, label='All GO terms (IC > 0)')
        ax.hist(feature_max_ics_arr, bins=50, edgecolor='black', alpha=0.35, color='darkorange',
                density=True, label='Feature max IC')
        ax.axvline(np.mean(all_ics_filtered), color='steelblue', linestyle='--', linewidth=2,
                   label=f'Baseline mean: {np.mean(all_ics_filtered):.2f}')

        ax.set_xlabel('Information Content', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Feature Max IC vs GO Baseline', fontsize=13)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=12)

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

    parser.add_argument('--interp_dir', type=str)
    parser.add_argument('--plot_suffix', type=str, default='')
    parser.add_argument('--null-samples', type=int, default=0,
                        help='Number of null samples for max IC comparison (0 = disable, e.g. 1000)')


    args = parser.parse_args()

    # Paths
    INPUT_DIM = 640
    LATENT_DIM = INPUT_DIM * args.expansion
    INTERPRETATION_DIR = args.interp_dir
    COACTIVATION_DIR = f"plots/sae/layer_{args.layer}{args.plot_suffix}/coactivation"

    # GO overlap files (optional, for high-overlap pair analysis)
    GO_OVERLAP_FILE = os.path.join(COACTIVATION_DIR, "feature_go_overlap_filtered_similarities.npy")
    INDICES_FILE = os.path.join(COACTIVATION_DIR, "feature_go_overlap_filtered_indices.npy")
    has_overlap_files = os.path.exists(GO_OVERLAP_FILE) and os.path.exists(INDICES_FILE)
    if not has_overlap_files:
        print(f"Warning: GO overlap files not found in {COACTIVATION_DIR}")
        print("Skipping high-overlap pair analysis (plots 1-2). Run compute_feature_correlations.py to enable.")

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

    # Compute null model if requested
    null_max_ics = None
    if args.null_samples > 0:
        reachable_ics = get_reachable_go_ics(INTERPRETATION_DIR, all_term_ics)
        print(f"\nComputing null max IC distribution ({args.null_samples} samples per feature)...")
        print(f"  Sampling from {len(reachable_ics)} reachable GO terms (not full DAG)")
        null_max_ics = compute_null_max_ic(go_enrichments, reachable_ics, n_samples=args.null_samples)

        # Save null max ICs
        null_path = os.path.join(INTERPRETATION_DIR, 'null_max_ics.npy')
        null_arr = np.array([[f, null_max_ics[f]] for f in sorted(null_max_ics.keys())])
        np.save(null_path, null_arr)
        print(f"Saved null max ICs to {null_path}")

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

    # Analyze high-overlap pairs (if overlap files available)
    analysis_results = None
    if has_overlap_files:
        analysis_results = analyze_overlap_pairs(
            go_enrichments,
            GO_OVERLAP_FILE,
            INDICES_FILE,
            all_term_ics,
            overlap_threshold=args.overlap_threshold
        )
        print_statistics(all_term_ics, analysis_results, feature_mean_ics, feature_median_ics, feature_max_ics)

    # Create plots
    os.makedirs(COACTIVATION_DIR, exist_ok=True)
    print("\nGenerating plots...")
    plot_ic_distributions(all_term_ics, analysis_results, feature_mean_ics, feature_median_ics, feature_max_ics, COACTIVATION_DIR, args.overlap_threshold, null_max_ics=null_max_ics)

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

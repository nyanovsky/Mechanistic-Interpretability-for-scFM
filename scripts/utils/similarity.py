"""Numeric similarity metrics, set-based comparisons, and clustering utilities."""

import numpy as np
import torch
from tqdm import tqdm

from .data_utils import get_device, compute_participation_ratio


# =============================================================================
# Gene-Feature Set Computation
# =============================================================================

def compute_gene_feature_sets(fg_matrix, min_features=5, max_features=200):
    """Compute top features per gene using gene-centric PR-adaptive thresholding.

    Transposes the feature-gene matrix to (n_genes, n_features), computes PR
    per gene, then uses PR-adaptive thresholding to get top features per gene.

    Args:
        fg_matrix: [n_features, n_genes] array
        min_features: Minimum features per gene
        max_features: Maximum features per gene

    Returns:
        tuple: (feature_sets, gene_pr)
            feature_sets: list of sets, each containing feature indices for that gene
            gene_pr: [n_genes] array of gene participation ratios
    """
    gene_matrix = fg_matrix.T  # (n_genes, n_features)

    print("Computing gene-centric participation ratios...")
    gene_pr = compute_participation_ratio(gene_matrix, axis=1)
    print(f"  Gene PR range: [{gene_pr.min():.1f}, {gene_pr.max():.1f}], "
          f"median: {np.median(gene_pr):.1f}")

    print("Computing top features per gene (PR-adaptive)...")
    feature_sets = get_top_k_genes_per_feature(
        gene_matrix, gene_pr, pr_scale=1,
        min_genes=min_features, max_genes=max_features
    )

    set_sizes = [len(s) for s in feature_sets]
    print(f"  Feature set sizes: min={min(set_sizes)}, max={max(set_sizes)}, "
          f"median={np.median(set_sizes):.0f}, mean={np.mean(set_sizes):.1f}")

    return feature_sets, gene_pr


# =============================================================================
# Vector Similarity
# =============================================================================

def compute_cosine_similarity(features, device=None):
    """Compute pairwise cosine similarity matrix.

    Args:
        features: (n_samples, n_dims) numpy array or torch tensor
        device: 'cuda' or 'cpu'

    Returns:
        numpy array: (n_samples, n_samples) similarity matrix
    """
    if device is None:
        device = get_device()

    print(f"Computing cosine similarity on {device}...")

    if isinstance(features, np.ndarray):
        features_t = torch.from_numpy(features).float()
    else:
        features_t = features.float()

    features_t = features_t.to(device)

    # Normalize
    features_normed = torch.nn.functional.normalize(features_t, p=2, dim=1)

    # Compute similarity
    similarity = features_normed @ features_normed.T

    return similarity.cpu().numpy()


def find_nearest_neighbors(similarity_matrix, k=10, device=None):
    """Find k nearest neighbors from similarity matrix.

    Args:
        similarity_matrix: (n, n) matrix
        k: number of neighbors

    Returns:
        tuple: (indices, values) numpy arrays
    """
    if device is None:
        device = get_device()

    print(f"Finding {k} nearest neighbors on {device}...")

    if isinstance(similarity_matrix, np.ndarray):
        sim_t = torch.from_numpy(similarity_matrix).float()
    else:
        sim_t = similarity_matrix.float()

    sim_t = sim_t.to(device)

    # Exclude self by setting diagonal to -inf
    sim_t.fill_diagonal_(-float('inf'))

    values, indices = torch.topk(sim_t, k, dim=1)

    return indices.cpu().numpy(), values.cpu().numpy()


# =============================================================================
# Set-Based Similarity
# =============================================================================

def get_top_k_genes_per_feature(feature_gene_matrix, pr_values, pr_scale=1, min_genes=10, max_genes=100):
    """Extract top-k genes for each feature based on PR-adaptive thresholding.

    Args:
        feature_gene_matrix: [n_features, n_genes] array
        pr_values: [n_features] array of participation ratios
        pr_scale: Scale factor for PR (default: 1)
        min_genes: Minimum genes per feature
        max_genes: Maximum genes per feature

    Returns:
        List of sets, where each set contains gene indices for that feature
    """
    n_features = feature_gene_matrix.shape[0]
    gene_sets = []

    for i in range(n_features):
        k = int(pr_values[i] * pr_scale)
        k = np.clip(k, min_genes, max_genes)

        top_k_indices = np.argsort(feature_gene_matrix[i])[::-1][:k]
        gene_sets.append(set(top_k_indices))

    return gene_sets


def compute_set_based_similarities(gene_sets, metric='jaccard'):
    """Compute pairwise set-based similarities.

    Args:
        gene_sets: List of sets (gene indices for each feature)
        metric: 'jaccard', 'overlap', 'overlap_sqrt', 'intersection', or 'containment'
               'overlap_sqrt' uses Ochiai coefficient: |A n B| / sqrt(|A| * |B|)

    Returns:
        Array of similarity values for all pairs
    """
    n_feat = len(gene_sets)
    n_pairs = (n_feat * (n_feat - 1)) // 2

    print(f"Computing {metric} for {n_feat} features ({n_pairs:,} pairs)...")

    similarities = np.zeros(n_pairs)
    idx = 0

    for i in tqdm(range(n_feat), desc=f"Computing {metric}"):
        for j in range(i + 1, n_feat):
            set_i = gene_sets[i]
            set_j = gene_sets[j]

            intersection = len(set_i & set_j)

            if metric == 'jaccard':
                union = len(set_i | set_j)
                sim = intersection / union if union > 0 else 0.0
            elif metric == 'overlap':
                min_size = min(len(set_i), len(set_j))
                sim = intersection / min_size if min_size > 0 else 0.0
            elif metric == 'overlap_sqrt':
                geometric_mean = np.sqrt(len(set_i) * len(set_j))
                sim = intersection / geometric_mean if geometric_mean > 0 else 0.0
            elif metric == 'intersection':
                sim = intersection
            elif metric == 'containment':
                cont_i = intersection / len(set_i) if len(set_i) > 0 else 0.0
                cont_j = intersection / len(set_j) if len(set_j) > 0 else 0.0
                sim = (cont_i + cont_j) / 2
            else:
                raise ValueError(f"Unknown set metric: {metric}")

            similarities[idx] = sim
            idx += 1

    return similarities


def compute_pairwise_correlations(feature_matrix, metric='pearson', pr_values=None,
                                   pr_scale=0.6, min_genes=10, max_genes=100):
    """Compute pairwise correlations/similarities between features.

    Args:
        feature_matrix: [n_features, n_dim] array (features x genes or features x cells)
        metric: Similarity metric (pearson, cosine, spearman, jaccard, overlap, overlap_sqrt,
                intersection, containment)
        pr_values: [n_features] array of participation ratios (required for set-based metrics)
        pr_scale: Scale factor for PR-based top-k selection (for set-based metrics)
        min_genes: Minimum genes per feature (for set-based metrics)
        max_genes: Maximum genes per feature (for set-based metrics)

    Returns:
        Array of correlation/similarity values for all pairs
    """
    n_feat, n_dim = feature_matrix.shape
    n_pairs = (n_feat * (n_feat - 1)) // 2

    # Set-based metrics require gene matrix and PR values
    set_based_metrics = ['jaccard', 'overlap', 'overlap_sqrt', 'intersection', 'containment']

    if metric in set_based_metrics:
        if pr_values is None:
            raise ValueError(f"{metric} requires pr_values argument")

        print(f"Extracting top-k genes per feature (PR scale={pr_scale}, min={min_genes}, max={max_genes})...")
        gene_sets = get_top_k_genes_per_feature(feature_matrix, pr_values, pr_scale, min_genes, max_genes)

        set_sizes = [len(s) for s in gene_sets]
        print(f"Gene set sizes: min={min(set_sizes)}, max={max(set_sizes)}, mean={np.mean(set_sizes):.1f}")

        return compute_set_based_similarities(gene_sets, metric)

    # Continuous metrics
    print(f"Computing {metric} for {n_feat} features ({n_pairs:,} pairs)...")

    if metric == 'pearson':
        mean = feature_matrix.mean(axis=1, keepdims=True)
        std = feature_matrix.std(axis=1, keepdims=True)
        standardized = (feature_matrix - mean) / (std + 1e-8)
        sim_matrix = (standardized @ standardized.T) / n_dim
    elif metric == 'cosine':
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        normalized = feature_matrix / (norms + 1e-8)
        sim_matrix = normalized @ normalized.T
    elif metric == 'spearman':
        from scipy.stats import rankdata
        print("  Converting to ranks...")
        ranks = np.apply_along_axis(rankdata, 1, feature_matrix).astype(float)
        mean = ranks.mean(axis=1, keepdims=True)
        std = ranks.std(axis=1, keepdims=True)
        standardized = (ranks - mean) / (std + 1e-8)
        sim_matrix = (standardized @ standardized.T) / n_dim
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Extract upper triangle (excluding diagonal)
    similarities = sim_matrix[np.triu_indices(n_feat, k=1)]

    return similarities


# =============================================================================
# Clustering
# =============================================================================

def perform_hierarchical_clustering(similarity_matrix, n_clusters=None, method='ward', metric='euclidean'):
    """Perform hierarchical clustering on a similarity/correlation matrix.

    Args:
        similarity_matrix: [n_features, n_features] symmetric matrix or [n_pairs] condensed distance matrix
        n_clusters: Number of clusters (if None, returns linkage matrix only)
        method: Linkage method ('ward', 'average', 'complete', 'single')
        metric: Distance metric (default: 'euclidean')

    Returns:
        If n_clusters provided: cluster_labels array
        Otherwise: (linkage_matrix, distance_matrix)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Convert similarity to distance
    if similarity_matrix.ndim == 2:
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        condensed_dist = squareform(distance_matrix, checks=False)
    else:
        condensed_dist = 1 - similarity_matrix

    print(f"Performing hierarchical clustering (method={method})...")
    linkage_matrix = linkage(condensed_dist, method=method)

    if n_clusters is not None:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        print(f"Assigned {len(np.unique(cluster_labels))} clusters")
        return cluster_labels

    return linkage_matrix, condensed_dist


def determine_optimal_clusters(similarity_matrix, max_clusters=None, method='silhouette'):
    """Determine optimal number of clusters.

    Args:
        similarity_matrix: [n_features, n_features] or condensed distance matrix
        max_clusters: Maximum clusters to test (default: sqrt(n_features))
        method: 'silhouette' or 'elbow' (davies-bouldin)

    Returns:
        dict: {'n_clusters': int, 'scores': array, 'method': str}
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    if similarity_matrix.ndim == 2:
        n_features = similarity_matrix.shape[0]
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        condensed_dist = squareform(distance_matrix, checks=False)
    else:
        n_pairs = len(similarity_matrix)
        n_features = int(np.ceil(np.sqrt(2 * n_pairs)))
        condensed_dist = 1 - similarity_matrix
        distance_matrix = squareform(condensed_dist)

    if max_clusters is None:
        max_clusters = int(np.sqrt(n_features))
        print(f"Using adaptive max_clusters = sqrt({n_features}) = {max_clusters}")

    linkage_matrix = linkage(condensed_dist, method='ward')

    scores = []
    n_clusters_range = range(2, min(max_clusters + 1, n_features))

    print(f"Testing {len(n_clusters_range)} cluster configurations...")
    for n in n_clusters_range:
        labels = fcluster(linkage_matrix, n, criterion='maxclust')

        if method == 'silhouette':
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
        else:  # davies_bouldin
            score = -davies_bouldin_score(distance_matrix, labels)

        scores.append(score)

    scores = np.array(scores)
    optimal_idx = np.argmax(scores)
    optimal_n = list(n_clusters_range)[optimal_idx]

    print(f"Optimal clusters: {optimal_n} (score={scores[optimal_idx]:.4f})")

    return {
        'n_clusters': optimal_n,
        'scores': scores,
        'n_clusters_range': list(n_clusters_range),
        'method': method
    }


def analyze_cluster_coherence(cluster_labels, go_enrichments, godag=None, term_counts=None,
                              go_metric='lin', overlap_metric='jaccard'):
    """Analyze GO enrichment coherence within vs between clusters.

    Args:
        cluster_labels: array of cluster assignments for each feature
        go_enrichments: dict {feature_id: set(GO_IDs)}
        godag: GO DAG (required for semantic similarity)
        term_counts: TermCounts (required for semantic similarity)
        go_metric: 'lin' or 'resnik' for semantic similarity (None to skip)
        overlap_metric: 'jaccard' or 'overlap' for term overlap

    Returns:
        dict with 'within_cluster', 'between_cluster', and 'cluster_sizes' keys
    """
    from .go_utils import compute_pairwise_go_similarity, compute_go_term_overlap

    feature_ids = np.arange(len(cluster_labels))

    enriched_features = set(go_enrichments.keys())
    valid_mask = np.array([fid in enriched_features for fid in feature_ids])

    feature_ids = feature_ids[valid_mask]
    cluster_labels_filtered = cluster_labels[valid_mask]

    print(f"Analyzing {len(feature_ids)} features with GO enrichment across {len(np.unique(cluster_labels_filtered))} clusters")

    within_pairs = []
    between_pairs = []

    for i in range(len(feature_ids)):
        for j in range(i + 1, len(feature_ids)):
            feat_i, feat_j = feature_ids[i], feature_ids[j]

            if cluster_labels_filtered[i] == cluster_labels_filtered[j]:
                within_pairs.append((feat_i, feat_j))
            else:
                between_pairs.append((feat_i, feat_j))

    print(f"  Within-cluster pairs: {len(within_pairs)}")
    print(f"  Between-cluster pairs: {len(between_pairs)}")

    results = {
        'within_cluster': {},
        'between_cluster': {},
        'cluster_sizes': np.bincount(cluster_labels_filtered)
    }

    if go_metric is not None and godag is not None and term_counts is not None:
        print("Computing within-cluster GO semantic similarity...")
        within_go_sim = compute_pairwise_go_similarity(
            go_enrichments, godag, term_counts, metric=go_metric,
            feature_pairs=within_pairs, verbose=True
        )

        print("Computing between-cluster GO semantic similarity...")
        between_go_sim = compute_pairwise_go_similarity(
            go_enrichments, godag, term_counts, metric=go_metric,
            feature_pairs=between_pairs, verbose=True
        )

        results['within_cluster']['go_sim'] = list(within_go_sim.values())
        results['between_cluster']['go_sim'] = list(between_go_sim.values())

    print("Computing within-cluster GO term overlap...")
    within_overlap = compute_go_term_overlap(
        go_enrichments, feature_pairs=within_pairs,
        metric=overlap_metric, verbose=True
    )

    print("Computing between-cluster GO term overlap...")
    between_overlap = compute_go_term_overlap(
        go_enrichments, feature_pairs=between_pairs,
        metric=overlap_metric, verbose=True
    )

    results['within_cluster']['overlap'] = list(within_overlap.values())
    results['between_cluster']['overlap'] = list(between_overlap.values())

    return results

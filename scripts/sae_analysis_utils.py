"""Utility functions for SAE analysis scripts."""

import os
import re
from glob import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from goatools.obo_parser import GODag
from goatools.associations import read_gaf
from goatools.semantic import TermCounts, get_info_content
from goatools.semantic import resnik_sim, lin_sim
import sys
import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ModelGenerator/huggingface/aido.cell'))
from aido_cell.utils import align_adata

def load_gene_names(raw_data_path):
    """Load gene names from original data aligned with activations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, raw_data_path)

    adata_raw = ad.read_h5ad(raw_path)
    adata_aligned, attention_mask = align_adata(adata_raw)

    # Get gene names for valid genes
    gene_names = adata_aligned.var_names[attention_mask.astype(bool)]
    return gene_names.tolist()


def load_decoder_weights(sae_dir):
    """
    Load decoder weights as (n_features, hidden_dim) array.
    Tries loading 'sae_decoder.pt' first, then falls back to 'topk_sae.pt'.
    """
    decoder_path = os.path.join(sae_dir, 'sae_decoder.pt')
    W_dec = None
    
    # Try standalone decoder file
    if os.path.exists(decoder_path):
        print(f"Loading decoder from {decoder_path}")
        try:
            decoder = torch.load(decoder_path, map_location='cpu')
            if isinstance(decoder, dict) and 'weight' in decoder:
                 W_dec = decoder['weight'].numpy()
            elif isinstance(decoder, torch.Tensor):
                 W_dec = decoder.numpy()
            else:
                # Try generic state dict access if structure is unknown but likely dict
                keys = decoder.keys() if isinstance(decoder, dict) else []
                if 'state_dict' in keys:
                     W_dec = decoder['state_dict']['decoder.weight'].numpy()
        except Exception as e:
            print(f"Warning: Failed to load {decoder_path}: {e}")

    # Fallback to checkpoint
    if W_dec is None:
        checkpoint_path = os.path.join(sae_dir, 'topk_sae.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading decoder from checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                W_dec = checkpoint['model_state_dict']['decoder.weight'].numpy()
            elif 'state_dict' in checkpoint:
                W_dec = checkpoint['state_dict']['decoder.weight'].numpy()
    
    if W_dec is None:
        raise FileNotFoundError(f"Could not load decoder weights from {sae_dir}")

    # Standardize shape: ensure (n_features, hidden_dim)
    # Heuristic: usually number of features (latent_dim) > hidden_dim
    # Existing scripts imply stored weights are (hidden, features) and need transposing
    if W_dec.shape[0] < W_dec.shape[1]: 
         W_dec = W_dec.T
    
    return W_dec

def load_go_enrichment(interpretations_dir, p_threshold=0.05, mode='terms'):
    """
    Load GO enrichment results.
    
    Args:
        interpretations_dir (str): Directory containing interpretation results
        p_threshold (float): Adjusted p-value threshold
        mode (str): 'terms' (returns full term strings) or 'ids' (returns GO:XXXXX IDs)
        
    Returns: 
        dict: {feature_id: set(terms/ids)}
    """
    feature_data = {}
    pattern = os.path.join(interpretations_dir, 'feature_*_enrichr')
    feature_dirs = glob(pattern)
    
    go_id_regex = re.compile(r'\(GO:(\d+)\)')

    for feature_dir in feature_dirs:
        match = re.search(r'feature_(\d+)_enrichr', feature_dir)
        if not match:
            continue
        feature_id = int(match.group(1))
        items = set()

        for go_file in glob(os.path.join(feature_dir, 'GO_*_2021.human.enrichr.reports.txt')):
            try:
                df = pd.read_csv(go_file, sep='\t')
                # Check required columns
                if 'Adjusted P-value' not in df.columns or 'Term' not in df.columns:
                    continue
                    
                significant = df[df['Adjusted P-value'] < p_threshold]
                
                for term in significant['Term']:
                    if mode == 'ids':
                        id_match = go_id_regex.search(term)
                        if id_match:
                            items.add(f"GO:{id_match.group(1)}")
                    else:
                        items.add(term)
            except Exception:
                continue

        if items:
            feature_data[feature_id] = items

    print(f"Loaded GO enrichment ({mode}) for {len(feature_data)} features")
    return feature_data

def load_celltype_correlations(interpretations_dir):
    """Load feature-celltype correlation matrix."""
    corr_path = os.path.join(interpretations_dir, 'feature_celltype_correlations.csv')
    if os.path.exists(corr_path):
        df = pd.read_csv(corr_path, index_col=0)
        print(f"Loaded celltype correlations: {df.shape}")
        return df
    print(f"Warning: Celltype correlations not found at {corr_path}")
    return None

def load_activation_matrices(interpretations_dir):
    """
    Load feature-gene and cell-feature activation matrices.

    Returns:
        tuple: (feature_gene_matrix, cell_feature_matrix)
    """
    feature_gene_path = os.path.join(interpretations_dir, 'feature_gene_matrix.npy')
    cell_feature_path = os.path.join(interpretations_dir, 'cell_feature_matrix.npy')

    fg_mat = None
    cf_mat = None

    if os.path.exists(feature_gene_path):
        print(f"Loading feature-gene matrix from {feature_gene_path}...")
        fg_mat = np.load(feature_gene_path)
        print(f"  Shape: {fg_mat.shape}")
    else:
        print(f"Warning: {feature_gene_path} not found")

    if os.path.exists(cell_feature_path):
        print(f"Loading cell-feature matrix from {cell_feature_path}...")
        cf_mat = np.load(cell_feature_path)
        print(f"  Shape: {cf_mat.shape}")
    else:
        print(f"Warning: {cell_feature_path} not found")

    return fg_mat, cf_mat


def get_expressed_genes_mask(raw_data_path, min_mean_expr=0.01, min_pct_cells=0.5):
    """Return boolean mask for genes with sufficient expression.

    Filters out genes with zero/low expression to avoid spurious correlations.

    Args:
        raw_data_path: Path to raw h5ad file
        min_mean_expr: Minimum mean expression across cells (default: 0.01)
        min_pct_cells: Minimum % of cells with nonzero expression (default: 0.5%)

    Returns:
        expressed_mask: Boolean array of shape (n_genes,)
    """
    adata_raw = ad.read_h5ad(raw_data_path)
    adata_aligned, attention_mask = align_adata(adata_raw)

    X = adata_aligned.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X_filtered = X[:, attention_mask.astype(bool)]

    # Compute expression stats per gene
    mean_expr = X_filtered.mean(axis=0)
    pct_nonzero = (X_filtered > 0).sum(axis=0) / X_filtered.shape[0] * 100

    # Filter: mean expression > threshold OR expressed in > N% of cells
    expressed_mask = (mean_expr > min_mean_expr) | (pct_nonzero > min_pct_cells)

    return expressed_mask

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def compute_cosine_similarity(features, device=None):
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        features: (n_samples, n_dims) numpy array or torch tensor
        device: 'cuda' or 'cpu'
        
    Returns:
        numpy array: (n_samples, n_samples) similarity matrix
    """
    if device is None:
        device = get_device()
        
    print(f"Computing cosine similarity on {device}...")
    
    # Ensure torch tensor
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
    """
    Find k nearest neighbors from similarity matrix.
    
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
# GO & Semantic Similarity Utilities
# =============================================================================

OBO_PATH = "/biodata/nyanovsky/datasets/GO/go-basic.obo"
GAF_PATH = "/biodata/nyanovsky/datasets/GO/goa_human.gaf"
NAMESPACES = ['biological_process', 'cellular_component', 'molecular_function']
NS_ABBREV = {'biological_process': 'BP', 'cellular_component': 'CC', 'molecular_function': 'MF'}

def load_go_dag_and_associations():
    """
    Load GO DAG and associations for all namespaces.
    
    Returns:
        tuple: (godag, term_counts)
    """
    print(f"Loading GO DAG from {OBO_PATH}...")
    godag = GODag(OBO_PATH)

    print(f"Loading Associations from {GAF_PATH}...")
    # read_gaf defaults to namespace='BP', so we must load each explicitly
    assoc_bp = read_gaf(GAF_PATH, namespace='BP', prt=None)
    assoc_cc = read_gaf(GAF_PATH, namespace='CC', prt=None)
    assoc_mf = read_gaf(GAF_PATH, namespace='MF', prt=None)

    # Merge all associations
    assoc = {}
    for ns_assoc in [assoc_bp, assoc_cc, assoc_mf]:
        for gene, terms in ns_assoc.items():
            if gene in assoc:
                assoc[gene] = assoc[gene] | terms
            else:
                assoc[gene] = terms
    
    print(f"  Merged: {len(assoc)} genes with GO annotations")
    
    print("Computing Information Content...")
    term_counts = TermCounts(godag, assoc)
    
    return godag, term_counts

def separate_by_namespace(go_ids, godag):
    """Separate GO IDs by namespace (BP, CC, MF)."""
    by_ns = defaultdict(set)
    for go_id in go_ids:
        if go_id in godag:
            ns = godag[go_id].namespace
            by_ns[ns].add(go_id)
    return dict(by_ns)

def compute_max_sim_within_namespace(gos_a_ns, gos_b_ns, godag, term_counts, metric='lin'):
    """
    Compute max similarity within each namespace, then aggregate.

    Returns:
        dict: {'max_overall': float, 'by_namespace': {ns: score}}
    """
    results = {'max_overall': 0.0, 'by_namespace': {}}

    for ns in NAMESPACES:
        valid_a = gos_a_ns.get(ns, set())
        valid_b = gos_b_ns.get(ns, set())

        if not valid_a or not valid_b:
            continue

        max_s = 0.0
        for go_a in valid_a:
            for go_b in valid_b:
                try:
                    if metric == 'lin':
                        s = lin_sim(go_a, go_b, godag, term_counts)
                    elif metric == 'resnik':
                        s = resnik_sim(go_a, go_b, godag, term_counts)
                    else:
                        s = 0
                    if s is not None and s > max_s:
                        max_s = s
                except Exception:
                    continue

        if max_s > 0:
            results['by_namespace'][ns] = max_s
            if max_s > results['max_overall']:
                results['max_overall'] = max_s

    return results


# =============================================================================
# Clustering Analysis
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
    # For correlation: distance = 1 - correlation
    if similarity_matrix.ndim == 2:
        # Square matrix - convert to distance
        distance_matrix = 1 - similarity_matrix
        # Ensure diagonal is 0
        np.fill_diagonal(distance_matrix, 0)
        # Convert to condensed form for linkage
        condensed_dist = squareform(distance_matrix, checks=False)
    else:
        # Already condensed
        condensed_dist = 1 - similarity_matrix

    # Perform hierarchical clustering
    print(f"Performing hierarchical clustering (method={method})...")
    linkage_matrix = linkage(condensed_dist, method=method)

    if n_clusters is not None:
        # Cut tree to get cluster assignments
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

    # Convert to distance matrix
    if similarity_matrix.ndim == 2:
        n_features = similarity_matrix.shape[0]
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        condensed_dist = squareform(distance_matrix, checks=False)
    else:
        # Infer n_features from condensed matrix
        n_pairs = len(similarity_matrix)
        n_features = int(np.ceil(np.sqrt(2 * n_pairs)))
        condensed_dist = 1 - similarity_matrix
        distance_matrix = squareform(condensed_dist)

    # Set default max_clusters adaptively
    if max_clusters is None:
        max_clusters = int(np.sqrt(n_features))
        print(f"Using adaptive max_clusters = sqrt({n_features}) = {max_clusters}")

    # Perform clustering once
    linkage_matrix = linkage(condensed_dist, method='ward')

    # Test different numbers of clusters
    scores = []
    n_clusters_range = range(2, min(max_clusters + 1, n_features))

    print(f"Testing {len(n_clusters_range)} cluster configurations...")
    for n in n_clusters_range:
        labels = fcluster(linkage_matrix, n, criterion='maxclust')

        if method == 'silhouette':
            # Higher is better
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
        else:  # davies_bouldin
            # Lower is better, so negate
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


# Global variables for worker processes (initialized once, shared via fork)
_worker_godag = None
_worker_term_counts = None

def _init_worker(godag, term_counts):
    """Initialize global GO DAG and term_counts in worker process."""
    global _worker_godag, _worker_term_counts
    _worker_godag = godag
    _worker_term_counts = term_counts

def _compute_similarity_worker(args):
    """Worker function for parallel GO similarity computation.

    Uses global _worker_godag and _worker_term_counts initialized via fork.
    """
    pairs_chunk, go_enrichments, metric, worker_id = args

    # Pre-compute namespace separation for all features in this chunk (cache)
    # This avoids redundant separations when features appear in multiple pairs
    unique_features = set()
    for feat_i, feat_j in pairs_chunk:
        if feat_i in go_enrichments:
            unique_features.add(feat_i)
        if feat_j in go_enrichments:
            unique_features.add(feat_j)

    # Cache namespace separations
    namespace_cache = {}
    for feat in unique_features:
        namespace_cache[feat] = separate_by_namespace(go_enrichments[feat], _worker_godag)

    # Compute similarities using cached namespace separations
    similarities = {}
    for feat_i, feat_j in pairs_chunk:
        if feat_i not in namespace_cache or feat_j not in namespace_cache:
            continue

        gos_i_ns = namespace_cache[feat_i]
        gos_j_ns = namespace_cache[feat_j]

        # Compute max similarity
        result = compute_max_sim_within_namespace(gos_i_ns, gos_j_ns, _worker_godag, _worker_term_counts, metric)
        similarities[(feat_i, feat_j)] = result['max_overall']

    return similarities


def compute_pairwise_go_similarity(go_enrichments, godag, term_counts, metric='lin',
                                   feature_pairs=None, verbose=True, n_workers=None):
    """Compute pairwise GO semantic similarity between features.

    Args:
        go_enrichments: dict {feature_id: set(GO_IDs)}
        godag: GO DAG object
        term_counts: TermCounts object for semantic similarity
        metric: 'lin' or 'resnik'
        feature_pairs: List of (feature_i, feature_j) tuples. If None, computes all pairs.
        verbose: Print progress
        n_workers: Number of parallel workers (None = use all CPUs)

    Returns:
        dict: {(feature_i, feature_j): similarity_score}
    """
    from tqdm import tqdm
    import multiprocessing as mp

    if feature_pairs is None:
        # Generate all pairs
        features = sorted(go_enrichments.keys())
        feature_pairs = [(features[i], features[j])
                        for i in range(len(features))
                        for j in range(i + 1, len(features))]

    # Use multiprocessing if n_workers specified or > 1000 pairs
    if n_workers is None:
        n_workers = mp.cpu_count()

    if len(feature_pairs) < 1000 or n_workers == 1:
        # Serial computation for small workloads
        similarities = {}
        iterator = tqdm(feature_pairs, desc=f"Computing {metric} similarity") if verbose else feature_pairs

        for feat_i, feat_j in iterator:
            if feat_i not in go_enrichments or feat_j not in go_enrichments:
                continue

            gos_i = go_enrichments[feat_i]
            gos_j = go_enrichments[feat_j]

            # Separate by namespace
            gos_i_ns = separate_by_namespace(gos_i, godag)
            gos_j_ns = separate_by_namespace(gos_j, godag)

            # Compute max similarity
            result = compute_max_sim_within_namespace(gos_i_ns, gos_j_ns, godag, term_counts, metric)
            similarities[(feat_i, feat_j)] = result['max_overall']

        return similarities

    # Parallel computation
    if verbose:
        print(f"Using {n_workers} workers for parallel computation...")
        print(f"Workers will share GO DAG via fork (copy-on-write)...")

    # Split pairs into MANY small chunks for frequent progress updates
    # Use chunks of ~1000 pairs each so progress bar updates frequently
    pairs_per_chunk = 1000
    n_chunks = (len(feature_pairs) + pairs_per_chunk - 1) // pairs_per_chunk

    if verbose:
        print(f"Splitting {len(feature_pairs):,} pairs into {n_chunks:,} chunks of ~{pairs_per_chunk} pairs each")

    chunks = []
    for i in range(n_chunks):
        start_idx = i * pairs_per_chunk
        end_idx = min(start_idx + pairs_per_chunk, len(feature_pairs))
        chunks.append(feature_pairs[start_idx:end_idx])

    # Prepare arguments for workers
    worker_args = [(chunk, go_enrichments, metric, i) for i, chunk in enumerate(chunks)]

    # Use 'fork' context to share GO DAG via copy-on-write (default on Linux)
    # Workers will inherit godag and term_counts without copying
    ctx = mp.get_context('fork')

    # Run parallel computation with initializer to set global variables
    # Use imap_unordered for incremental results
    similarities = {}

    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(godag, term_counts)) as pool:
        if verbose:
            # Use imap_unordered with chunksize=1 to get results as they complete
            results_iter = pool.imap_unordered(_compute_similarity_worker, worker_args, chunksize=1)

            # Progress bar shows actual pairs processed (not just chunks)
            with tqdm(total=len(feature_pairs), desc=f"Computing {metric} similarity", unit="pairs") as pbar:
                for result_dict in results_iter:
                    similarities.update(result_dict)
                    pbar.update(len(result_dict))  # Update by number of pairs in this chunk
        else:
            results = pool.map(_compute_similarity_worker, worker_args)
            for result_dict in results:
                similarities.update(result_dict)

    return similarities


def _compute_go_term_overlap_worker(args):
    """Worker function for parallel GO term overlap computation."""
    pairs_chunk, go_enrich, met, worker_id = args
    overlaps_local = {}
    for feat_i, feat_j in pairs_chunk:
        if feat_i not in go_enrich or feat_j not in go_enrich:
            continue

        set_i = go_enrich[feat_i]
        set_j = go_enrich[feat_j]
        intersection = len(set_i & set_j)

        if met == 'jaccard':
            union = len(set_i | set_j)
            score = intersection / union if union > 0 else 0.0
        elif met == 'overlap':
            min_size = min(len(set_i), len(set_j))
            score = intersection / min_size if min_size > 0 else 0.0
        else:
            raise ValueError(f"Unknown metric: {met}")

        overlaps_local[(feat_i, feat_j)] = score
    return overlaps_local


def compute_go_term_overlap(go_enrichments, feature_pairs=None, metric='overlap', verbose=True, n_workers=None):
    """Compute GO term set overlap between features.

    Args:
        go_enrichments: dict {feature_id: set(GO_IDs)}
        feature_pairs: List of (feature_i, feature_j) tuples. If None, computes all pairs.
        metric: 'jaccard' or 'overlap' (default: overlap coefficient for containment)
        verbose: Print progress
        n_workers: Number of parallel workers (None = use all CPUs)

    Returns:
        dict: {(feature_i, feature_j): overlap_score}
    """
    from tqdm import tqdm
    import multiprocessing as mp

    if feature_pairs is None:
        features = sorted(go_enrichments.keys())
        feature_pairs = [(features[i], features[j])
                        for i in range(len(features))
                        for j in range(i + 1, len(features))]

    # Use multiprocessing if n_workers specified or > 1000 pairs
    if n_workers is None:
        n_workers = mp.cpu_count()

    if len(feature_pairs) < 1000 or n_workers == 1:
        # Serial computation
        overlaps = {}
        iterator = tqdm(feature_pairs, desc=f"Computing GO {metric}") if verbose else feature_pairs

        for feat_i, feat_j in iterator:
            if feat_i not in go_enrichments or feat_j not in go_enrichments:
                continue

            set_i = go_enrichments[feat_i]
            set_j = go_enrichments[feat_j]
            intersection = len(set_i & set_j)

            if metric == 'jaccard':
                union = len(set_i | set_j)
                score = intersection / union if union > 0 else 0.0
            elif metric == 'overlap':
                min_size = min(len(set_i), len(set_j))
                score = intersection / min_size if min_size > 0 else 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            overlaps[(feat_i, feat_j)] = score
        return overlaps

    # Parallel computation (same pattern as Lin similarity)
    if verbose:
        print(f"Using {n_workers} workers for parallel computation...")
        print(f"Splitting {len(feature_pairs):,} pairs into chunks of ~1000 pairs each")

    pairs_per_chunk = 1000
    n_chunks = (len(feature_pairs) + pairs_per_chunk - 1) // pairs_per_chunk
    chunks = [feature_pairs[i*pairs_per_chunk : min((i+1)*pairs_per_chunk, len(feature_pairs))]
              for i in range(n_chunks)]

    worker_args = [(chunk, go_enrichments, metric, i) for i, chunk in enumerate(chunks)]
    ctx = mp.get_context('fork')

    overlaps = {}
    with ctx.Pool(n_workers) as pool:
        if verbose:
            results_iter = pool.imap_unordered(_compute_go_term_overlap_worker, worker_args, chunksize=1)
            with tqdm(total=len(feature_pairs), desc=f"Computing GO {metric}", unit="pairs") as pbar:
                for result_dict in results_iter:
                    overlaps.update(result_dict)
                    pbar.update(len(result_dict))
        else:
            results = pool.map(_compute_go_term_overlap_worker, worker_args)
            for result_dict in results:
                overlaps.update(result_dict)

    return overlaps


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
        dict: {
            'within_cluster': {'go_sim': scores, 'overlap': scores},
            'between_cluster': {'go_sim': scores, 'overlap': scores},
            'cluster_sizes': array
        }
    """
    # Get feature IDs (assume they're indices matching cluster_labels)
    feature_ids = np.arange(len(cluster_labels))

    # Filter to only features with GO enrichment
    enriched_features = set(go_enrichments.keys())
    valid_mask = np.array([fid in enriched_features for fid in feature_ids])

    feature_ids = feature_ids[valid_mask]
    cluster_labels_filtered = cluster_labels[valid_mask]

    print(f"Analyzing {len(feature_ids)} features with GO enrichment across {len(np.unique(cluster_labels_filtered))} clusters")

    # Generate within-cluster and between-cluster pairs
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

    # Compute GO semantic similarity
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

    # Compute GO term overlap
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


# =============================================================================
# Feature Co-activation Analysis
# =============================================================================

def get_top_k_genes_per_feature(feature_gene_matrix, pr_values, pr_scale=0.6, min_genes=10, max_genes=100):
    """Extract top-k genes for each feature based on PR-adaptive thresholding.

    Args:
        feature_gene_matrix: [n_features, n_genes] array
        pr_values: [n_features] array of participation ratios
        pr_scale: Scale factor for PR (default: 0.6 = 60%)
        min_genes: Minimum genes per feature
        max_genes: Maximum genes per feature

    Returns:
        List of sets, where each set contains gene indices for that feature
    """
    from tqdm import tqdm
    n_features = feature_gene_matrix.shape[0]
    gene_sets = []

    for i in range(n_features):
        # Compute adaptive k based on PR
        k = int(pr_values[i] * pr_scale)
        k = np.clip(k, min_genes, max_genes)

        # Get top-k gene indices
        top_k_indices = np.argsort(feature_gene_matrix[i])[::-1][:k]
        gene_sets.append(set(top_k_indices))

    return gene_sets


def compute_set_based_similarities(gene_sets, metric='jaccard'):
    """Compute pairwise set-based similarities.

    Args:
        gene_sets: List of sets (gene indices for each feature)
        metric: 'jaccard', 'overlap', 'overlap_sqrt', 'intersection', or 'containment'
               'overlap_sqrt' uses Ochiai coefficient: |A ∩ B| / sqrt(|A| * |B|)

    Returns:
        Array of similarity values for all pairs
    """
    from tqdm import tqdm
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
                # Overlap coefficient: intersection / min(|A|, |B|)
                min_size = min(len(set_i), len(set_j))
                sim = intersection / min_size if min_size > 0 else 0.0
            elif metric == 'overlap_sqrt':
                # Ochiai coefficient: intersection / sqrt(|A| * |B|)
                # Penalizes asymmetric set sizes
                geometric_mean = np.sqrt(len(set_i) * len(set_j))
                sim = intersection / geometric_mean if geometric_mean > 0 else 0.0
            elif metric == 'intersection':
                # Raw intersection count
                sim = intersection
            elif metric == 'containment':
                # Asymmetric: average of both directions
                # (|A ∩ B| / |A| + |A ∩ B| / |B|) / 2
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
        metric: Similarity metric (pearson, cosine, spearman, jaccard, overlap, overlap_sqrt, intersection, containment)
        pr_values: [n_features] array of participation ratios (required for set-based metrics)
        pr_scale: Scale factor for PR-based top-k selection (for set-based metrics)
        min_genes: Minimum genes per feature (for set-based metrics)
        max_genes: Maximum genes per feature (for set-based metrics)

    Returns:
        Array of correlation/similarity values for all pairs
    """
    from scipy.stats import pearsonr, spearmanr

    n_feat, n_dim = feature_matrix.shape
    n_pairs = (n_feat * (n_feat - 1)) // 2

    # Set-based metrics require gene matrix and PR values
    set_based_metrics = ['jaccard', 'overlap', 'overlap_sqrt', 'intersection', 'containment']

    if metric in set_based_metrics:
        if pr_values is None:
            raise ValueError(f"{metric} requires pr_values argument")

        print(f"Extracting top-k genes per feature (PR scale={pr_scale}, min={min_genes}, max={max_genes})...")
        gene_sets = get_top_k_genes_per_feature(feature_matrix, pr_values, pr_scale, min_genes, max_genes)

        # Print some stats
        set_sizes = [len(s) for s in gene_sets]
        print(f"Gene set sizes: min={min(set_sizes)}, max={max(set_sizes)}, mean={np.mean(set_sizes):.1f}")

        return compute_set_based_similarities(gene_sets, metric)

    # Continuous metrics
    print(f"Computing {metric} for {n_feat} features ({n_pairs:,} pairs)...")

    if metric == 'pearson':
        # Standardize features (z-score normalization)
        mean = feature_matrix.mean(axis=1, keepdims=True)
        std = feature_matrix.std(axis=1, keepdims=True)
        standardized = (feature_matrix - mean) / (std + 1e-8)
        # Compute correlation matrix via dot product
        sim_matrix = (standardized @ standardized.T) / n_dim
    elif metric == 'cosine':
        # Normalize to unit vectors
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        normalized = feature_matrix / (norms + 1e-8)
        # Compute cosine similarity via dot product
        sim_matrix = normalized @ normalized.T
    elif metric == 'spearman':
        # Compute Spearman via ranking then Pearson
        from scipy.stats import rankdata
        print("  Converting to ranks...")
        # Apply ranking along each feature (axis=1)
        ranks = np.apply_along_axis(rankdata, 1, feature_matrix).astype(float)
        # Standardize ranks
        mean = ranks.mean(axis=1, keepdims=True)
        std = ranks.std(axis=1, keepdims=True)
        standardized = (ranks - mean) / (std + 1e-8)
        sim_matrix = (standardized @ standardized.T) / n_dim
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Extract upper triangle (excluding diagonal)
    similarities = sim_matrix[np.triu_indices(n_feat, k=1)]

    return similarities

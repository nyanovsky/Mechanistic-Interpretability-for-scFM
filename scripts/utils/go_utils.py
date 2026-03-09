"""Gene Ontology enrichment, semantic similarity, and term overlap utilities."""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from goatools.obo_parser import GODag
from goatools.associations import read_gaf
from goatools.semantic import TermCounts, get_info_content
from goatools.semantic import resnik_sim, lin_sim

try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False


# =============================================================================
# Constants
# =============================================================================

OBO_PATH = "/biodata/nyanovsky/datasets/GO/go-basic.obo"
GAF_PATH = "/biodata/nyanovsky/datasets/GO/goa_human.gaf"
NAMESPACES = ['biological_process', 'cellular_component', 'molecular_function']
NS_ABBREV = {'biological_process': 'BP', 'cellular_component': 'CC', 'molecular_function': 'MF'}


# =============================================================================
# GO Enrichment Analysis
# =============================================================================

GO_GENE_SETS = ['GO_Biological_Process_2021', 'GO_Molecular_Function_2021', 'GO_Cellular_Component_2021']


def run_go_enrichment(gene_list, output_dir, background=None, identifier=None, verbose=True):
    """Run GO enrichment on a gene list using gseapy.

    Uses gp.enrich() (offline hypergeometric test) when background is provided,
    gp.enrichr() (online API) otherwise.

    Args:
        gene_list: List of gene names
        output_dir: Directory for output files
        background: Optional background gene list for offline enrichment
        identifier: String used in output naming (e.g. 'feature_42', 'Up_Alpha5_vs_Baseline')
        verbose: Print errors (set False for parallel contexts)

    Returns:
        DataFrame of concatenated results, or None if no results
    """
    if not HAS_GSEAPY or not gene_list:
        return None

    all_results = []
    subdir = f'{identifier}_enrichr' if identifier else 'enrichr'

    for gs in GO_GENE_SETS:
        try:
            if background is not None:
                enr = gp.enrich(
                    gene_list=gene_list,
                    gene_sets=gs,
                    background=background,
                    outdir=os.path.join(output_dir, subdir),
                    cutoff=0.05
                )
            else:
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=gs,
                    organism='human',
                    outdir=os.path.join(output_dir, subdir),
                    cutoff=0.05
                )
            if enr.results is not None and not enr.results.empty:
                all_results.append(enr.results)
        except Exception as e:
            if verbose:
                print(f"  GO enrichment for {gs} failed: {e}")
            continue

    if not all_results:
        return None

    full_results_df = pd.concat(all_results, ignore_index=True)

    if identifier:
        summary_file = os.path.join(output_dir, f"go_summary_{identifier}.csv")
        full_results_df.to_csv(summary_file, index=False)

    return full_results_df


# =============================================================================
# GO DAG Loading & Information Content
# =============================================================================

def load_go_dag_and_associations():
    """Load GO DAG and associations for all namespaces.

    Returns:
        tuple: (godag, term_counts)
    """
    print(f"Loading GO DAG from {OBO_PATH}...")
    godag = GODag(OBO_PATH)

    print(f"Loading Associations from {GAF_PATH}...")
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


def separate_by_namespace(go_ids, godag):
    """Separate GO IDs by namespace (BP, CC, MF)."""
    by_ns = defaultdict(set)
    for go_id in go_ids:
        if go_id in godag:
            ns = godag[go_id].namespace
            by_ns[ns].add(go_id)
    return dict(by_ns)


# =============================================================================
# Semantic Similarity
# =============================================================================

def compute_max_sim_within_namespace(gos_a_ns, gos_b_ns, godag, term_counts, metric='lin'):
    """Compute max similarity within each namespace, then aggregate.

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
# Parallel Workers
# =============================================================================

# Global variables for worker processes (initialized once, shared via fork)
_worker_godag = None
_worker_term_counts = None

def _init_worker(godag, term_counts):
    """Initialize global GO DAG and term_counts in worker process."""
    global _worker_godag, _worker_term_counts
    _worker_godag = godag
    _worker_term_counts = term_counts

def _compute_similarity_worker(args):
    """Worker function for parallel GO similarity computation."""
    pairs_chunk, go_enrichments, metric, worker_id = args

    unique_features = set()
    for feat_i, feat_j in pairs_chunk:
        if feat_i in go_enrichments:
            unique_features.add(feat_i)
        if feat_j in go_enrichments:
            unique_features.add(feat_j)

    namespace_cache = {}
    for feat in unique_features:
        namespace_cache[feat] = separate_by_namespace(go_enrichments[feat], _worker_godag)

    similarities = {}
    for feat_i, feat_j in pairs_chunk:
        if feat_i not in namespace_cache or feat_j not in namespace_cache:
            continue

        gos_i_ns = namespace_cache[feat_i]
        gos_j_ns = namespace_cache[feat_j]

        result = compute_max_sim_within_namespace(gos_i_ns, gos_j_ns, _worker_godag, _worker_term_counts, metric)
        similarities[(feat_i, feat_j)] = result['max_overall']

    return similarities


# =============================================================================
# Pairwise GO Similarity
# =============================================================================

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
    import multiprocessing as mp

    if feature_pairs is None:
        features = sorted(go_enrichments.keys())
        feature_pairs = [(features[i], features[j])
                        for i in range(len(features))
                        for j in range(i + 1, len(features))]

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

            gos_i_ns = separate_by_namespace(gos_i, godag)
            gos_j_ns = separate_by_namespace(gos_j, godag)

            result = compute_max_sim_within_namespace(gos_i_ns, gos_j_ns, godag, term_counts, metric)
            similarities[(feat_i, feat_j)] = result['max_overall']

        return similarities

    # Parallel computation
    if verbose:
        print(f"Using {n_workers} workers for parallel computation...")
        print(f"Workers will share GO DAG via fork (copy-on-write)...")

    pairs_per_chunk = 1000
    n_chunks = (len(feature_pairs) + pairs_per_chunk - 1) // pairs_per_chunk

    if verbose:
        print(f"Splitting {len(feature_pairs):,} pairs into {n_chunks:,} chunks of ~{pairs_per_chunk} pairs each")

    chunks = []
    for i in range(n_chunks):
        start_idx = i * pairs_per_chunk
        end_idx = min(start_idx + pairs_per_chunk, len(feature_pairs))
        chunks.append(feature_pairs[start_idx:end_idx])

    worker_args = [(chunk, go_enrichments, metric, i) for i, chunk in enumerate(chunks)]

    ctx = mp.get_context('fork')

    similarities = {}

    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(godag, term_counts)) as pool:
        if verbose:
            results_iter = pool.imap_unordered(_compute_similarity_worker, worker_args, chunksize=1)

            with tqdm(total=len(feature_pairs), desc=f"Computing {metric} similarity", unit="pairs") as pbar:
                for result_dict in results_iter:
                    similarities.update(result_dict)
                    pbar.update(len(result_dict))
        else:
            results = pool.map(_compute_similarity_worker, worker_args)
            for result_dict in results:
                similarities.update(result_dict)

    return similarities


# =============================================================================
# GO Term Overlap
# =============================================================================

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
        elif met == 'ochiai':
            denom = np.sqrt(len(set_i) * len(set_j))
            score = intersection / denom if denom > 0 else 0.0
        else:
            raise ValueError(f"Unknown metric: {met}")

        overlaps_local[(feat_i, feat_j)] = score
    return overlaps_local


def compute_go_term_overlap(go_enrichments, feature_pairs=None, metric='overlap', verbose=True, n_workers=None):
    """Compute GO term set overlap between features.

    Args:
        go_enrichments: dict {feature_id: set(GO_IDs)}
        feature_pairs: List of (feature_i, feature_j) tuples. If None, computes all pairs.
        metric: 'jaccard', 'overlap', or 'ochiai'
        verbose: Print progress
        n_workers: Number of parallel workers (None = use all CPUs)

    Returns:
        dict: {(feature_i, feature_j): overlap_score}
    """
    import multiprocessing as mp

    if feature_pairs is None:
        features = sorted(go_enrichments.keys())
        feature_pairs = [(features[i], features[j])
                        for i in range(len(features))
                        for j in range(i + 1, len(features))]

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
            elif metric == 'ochiai':
                denom = np.sqrt(len(set_i) * len(set_j))
                score = intersection / denom if denom > 0 else 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            overlaps[(feat_i, feat_j)] = score
        return overlaps

    # Parallel computation
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

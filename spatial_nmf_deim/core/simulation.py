"""
Simulation data generator for SpatialNMF-DEIM benchmarking.

Generates synthetic spatial transcriptomics data with:
- Grid-based spatial layout with Voronoi tissue domains
- Known ground-truth marker genes
- Controllable parameters (K, FC, dropout, etc.)
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Dict, Optional


def generate_simulation(
    K: int = 10,
    G: int = 5000,
    N_per_type: int = 300,
    markers_per_type: int = 30,
    fold_change: float = 8.0,
    dropout_rate: float = 0.3,
    delta: float = 1.0,
    seed: int = 42,
) -> Dict:
    """
    Generate simulated spatial transcriptomics data with ground truth.

    Gene categories (G total):
      - Marker genes:  K * markers_per_type, each assigned to 1 type
      - Housekeeping:  200 genes, moderate uniform expression
      - Noise genes:   remainder, random expression

    Spatial layout: grid-based with K-1 tissue Voronoi domains
    + 1 peripheral background cluster.

    Parameters
    ----------
    K : int
        Number of types (including background).
    G : int
        Total number of genes.
    N_per_type : int
        Target spots per type (approximate for grid layout).
    markers_per_type : int
        Marker genes per tissue type.
    fold_change : float
        Mean expression ratio of marker in target vs non-target type.
    dropout_rate : float
        Probability of zeroing out a count (simulates ST dropout).
    delta : float
        Grid spacing.
    seed : int
        Random seed.

    Returns
    -------
    result : dict with keys:
        'Y' : ndarray (N, G) raw counts
        'labels' : ndarray (N,) type labels 0..K-1 (K-1 = background)
        'coords' : ndarray (N, 2) spatial coordinates
        'gene_names' : ndarray (G,) gene names "Gene_0", ...
        'true_markers' : dict str(k) -> set of marker gene names
        'n_tissue_types' : int = K - 1
    """
    rng = np.random.RandomState(seed)
    n_tissue = K - 1
    N = None  # determined by grid

    # --- 1. Generate grid spatial layout ---
    target_per_type = N_per_type
    side = int(np.sqrt((target_per_type * K) * 1.4)) + 1

    xs = np.arange(side) * delta
    ys = np.arange(side) * delta
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    center = np.array([(side - 1) * delta / 2, (side - 1) * delta / 2])
    max_radius = (side - 1) * delta / 2 * 0.72

    dist_to_center = np.linalg.norm(grid_points - center, axis=1)
    interior_mask = dist_to_center <= max_radius
    interior_pts = grid_points[interior_mask]

    # KMeans partition into tissue domains
    kmeans = KMeans(n_clusters=n_tissue, random_state=seed, n_init=10, max_iter=300)
    domain_labels = kmeans.fit_predict(interior_pts)

    labels = np.full(len(grid_points), K - 1, dtype=int)
    labels[interior_mask] = domain_labels

    # Subsample if too many points
    target_total = N_per_type * K
    if len(labels) > target_total * 1.5:
        keep = []
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            n_keep = min(N_per_type, len(idx_k))
            keep.append(rng.choice(idx_k, n_keep, replace=False))
        keep_idx = np.concatenate(keep)
        keep_mask = np.zeros(len(labels), dtype=bool)
        keep_mask[keep_idx] = True
        grid_points = grid_points[keep_mask]
        labels = labels[keep_mask]

    N = len(labels)
    coords = grid_points

    # --- 2. Generate expression (centroid matrix) ---
    gene_names = np.array([f"Gene_{i}" for i in range(G)])

    # Assign marker genes
    n_markers_total = n_tissue * markers_per_type
    marker_gene_idx = rng.permutation(G)[:n_markers_total].copy()
    true_marker_idx = {}
    for k in range(n_tissue):
        s = k * markers_per_type
        true_marker_idx[k] = set(marker_gene_idx[s:s + markers_per_type].tolist())

    H_true = np.zeros((K, G))

    # Marker genes: lognormal with heterogeneity
    for k in range(n_tissue):
        for g in true_marker_idx[k]:
            H_true[k, g] = rng.lognormal(mean=np.log(0.5 * fold_change), sigma=0.3)
            for k2 in range(K):
                if k2 != k:
                    H_true[k2, g] = rng.lognormal(mean=np.log(0.5), sigma=0.3)

    # Housekeeping genes
    non_marker = np.array(sorted(set(range(G)) - set(marker_gene_idx)))
    n_hk = min(200, len(non_marker))
    for g in non_marker[:n_hk]:
        H_true[:, g] = rng.uniform(2.5, 4.0, K)

    # Noise genes
    for g in non_marker[n_hk:]:
        H_true[:, g] = rng.lognormal(mean=1.2, sigma=0.5, size=K)

    # Background type: low expression everywhere
    H_true[K - 1, :] = rng.lognormal(mean=np.log(0.3), sigma=0.3, size=G)

    # --- 3. Spot-level counts (Poisson sampling) ---
    H_prop = H_true / H_true.sum(axis=1, keepdims=True)
    lib_sizes = rng.lognormal(mean=7.5, sigma=0.5, size=N)
    lambdas = H_prop[labels] * lib_sizes[:, None]
    Y_counts = rng.poisson(np.maximum(lambdas, 0.01))

    # --- 4. Dropout ---
    drop_mask = rng.random((N, G)) < dropout_rate
    Y_counts[drop_mask] = 0

    # --- 5. Build true_markers dict ---
    true_markers = {}
    for k in range(n_tissue):
        true_markers[str(k)] = set(
            str(gene_names[g]) if hasattr(gene_names[g], 'item') else str(gene_names[g])
            for g in true_marker_idx[k]
        )
    # Background has no markers
    true_markers[str(K - 1)] = set()

    return {
        'Y': Y_counts,
        'labels': labels,
        'coords': coords,
        'gene_names': gene_names,
        'true_markers': true_markers,
        'n_tissue_types': n_tissue,
        'K': K,
        'G': G,
        'N': N,
    }
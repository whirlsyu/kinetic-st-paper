"""
Spatial graph construction utilities.

Replaces the flashdeconv.utils.graph dependency with a local implementation.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from typing import Optional


def coords_to_adjacency(
    coords: np.ndarray,
    method: str = "knn",
    k: int = 6,
    radius: Optional[float] = None,
) -> sparse.spmatrix:
    """
    Build a spatial adjacency matrix from coordinates.

    Parameters
    ----------
    coords : ndarray of shape (n_spots, 2) or (n_spots, 3)
        Spatial coordinates.
    method : str
        Graph construction method:
        - "knn": K-nearest neighbors (undirected, symmetric)
        - "radius": Radius-based neighbors
        - "grid": Grid-based (4-neighbor connectivity for delta-spaced grids)
    k : int
        Number of neighbors for KNN.
    radius : float, optional
        Radius for radius-based graph.

    Returns
    -------
    A : sparse.csr_matrix of shape (n_spots, n_spots)
        Binary adjacency matrix.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]

    if method == "grid":
        return _grid_adjacency(coords)
    elif method == "radius":
        if radius is None:
            raise ValueError("radius must be specified for method='radius'")
        return _radius_adjacency(coords, radius)
    elif method == "knn":
        return _knn_adjacency(coords, k)
    else:
        raise ValueError(f"Unknown method: {method}")


def _knn_adjacency(coords: np.ndarray, k: int) -> sparse.csr_matrix:
    """KNN graph (symmetrized)."""
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)  # +1 to exclude self

    rows, cols = [], []
    for i in range(len(coords)):
        for j in indices[i, 1:]:  # skip self
            rows.append(i)
            cols.append(int(j))

    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(coords), len(coords)))
    # Symmetrize
    A = A.maximum(A.T)
    return A.tocsr()


def _radius_adjacency(coords: np.ndarray, radius: float) -> sparse.csr_matrix:
    """Radius-based graph."""
    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius, output_type='ndarray')

    if len(pairs) == 0:
        return sparse.csr_matrix((len(coords), len(coords)))

    rows = pairs[:, 0]
    cols = pairs[:, 1]
    data = np.ones(len(rows))
    A = sparse.coo_matrix((data, (rows, cols)),
                          shape=(len(coords), len(coords)))
    return A.tocsr()


def _grid_adjacency(coords: np.ndarray) -> sparse.csr_matrix:
    """Grid-based adjacency (4-neighbor) for delta-spaced grids."""
    n = len(coords)
    # Estimate delta from median nearest-neighbor distance
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    delta_est = np.median(dists[:, 1])

    # Connect points within 1.5 * delta (covers grid neighbors)
    radius = delta_est * 1.5
    return _radius_adjacency(coords, radius)
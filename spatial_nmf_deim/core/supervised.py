"""
Supervised DEIM for marker gene discovery.

When anatomical region labels are known, DEIM operates on the type centroid
matrix to select genes that best distinguish tissue types. This is a
two-stage procedure:
  1. Gene pool selection: SVD + DEIM on the centroid matrix
  2. Specificity ranking: z-score ranking within the pool

The supervised setting has an analytical solution for NMF:
  W = one-hot label matrix (known), H = group means (analytical).
DEIM's role is to select which genes to report as markers.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List

from spatial_nmf_deim.core.deim import compute_deim_scores


def specificity_rank(
    H_full: np.ndarray,
    k: int,
    gene_idx: np.ndarray,
) -> np.ndarray:
    """
    Rank genes in pool by specificity z-score for type k.

    Parameters
    ----------
    H_full : ndarray (K, G)
        Group mean expression matrix.
    k : int
        Target type index.
    gene_idx : ndarray
        Indices of genes in the pool.

    Returns
    -------
    z : ndarray (len(gene_idx),)
        Specificity z-scores.
    """
    H_sub = H_full[:, gene_idx]
    Ks = H_sub.shape[0]
    mean_k = H_sub[k]
    mean_rest = (H_sub.sum(axis=0) - H_sub[k]) / (Ks - 1)
    z = (mean_k - mean_rest) / (np.std(H_sub, axis=0) + 1e-10)
    return z


class SupervisedDEIM:
    """
    Supervised marker gene discovery via DEIM on type centroids.

    Parameters
    ----------
    pool_size : int, default=2000
        Number of genes to select in the DEIM pool.
    n_top : int, default=20
        Number of top marker genes per type.
    svd_rank : int, optional
        Rank for SVD on centroid matrix. Default: K-1.

    Attributes
    ----------
    selected_pool_ : ndarray
        Gene indices in the DEIM pool.
    markers_ : dict
        markers_[type_label] = list of top gene names.

    Example
    -------
    >>> sd = SupervisedDEIM(pool_size=2000)
    >>> sd.fit(Y, labels, gene_names)
    >>> markers = sd.markers_
    """

    def __init__(
        self,
        pool_size: int = 2000,
        n_top: int = 20,
        svd_rank: Optional[int] = None,
    ):
        self.pool_size = pool_size
        self.n_top = n_top
        self.svd_rank = svd_rank

        self.selected_pool_ = None
        self.markers_ = None
        self.H_ = None
        self.deim_scores_ = None
        self._fitted = False

    def fit(
        self,
        Y: np.ndarray,
        labels: np.ndarray,
        gene_names: Optional[np.ndarray] = None,
    ) -> "SupervisedDEIM":
        """
        Fit supervised DEIM and discover marker genes.

        Parameters
        ----------
        Y : ndarray (n_spots, n_genes)
            Raw count matrix.
        labels : ndarray (n_spots,)
            Type labels (integers or strings).
        gene_names : ndarray, optional
            Gene names.

        Returns
        -------
        self
        """
        unique_labels = np.unique(labels)
        K = len(unique_labels)
        n_spots, n_genes = Y.shape

        # Preprocess
        lib_size = np.maximum(Y.sum(axis=1, keepdims=True), 1.0)
        Y_logcpm = np.log1p(Y / lib_size * 1e4)

        # Group means
        H_full = np.zeros((K, n_genes))
        for k, l in enumerate(unique_labels):
            H_full[k] = Y[labels == l].mean(axis=0)

        H_log = np.log1p(H_full)
        self.H_ = H_full

        # SVD on centroid matrix
        U_h, s_h, Vt_h = np.linalg.svd(H_log, full_matrices=False)
        rank = self.svd_rank if self.svd_rank else min(K - 1, 20, len(s_h))
        V_h = Vt_h[:rank, :].T

        # DEIM scores
        deim_scores = compute_deim_scores(V_h, s_h[:rank])
        self.deim_scores_ = deim_scores

        # Select pool
        ng = min(self.pool_size, n_genes)
        self.selected_pool_ = np.argsort(deim_scores)[::-1][:ng]

        # Per-type specificity ranking
        gene_names_arr = np.array(gene_names) if gene_names is not None else \
            np.array([f"Gene_{i}" for i in range(n_genes)])

        self.markers_ = {}
        for k, l in enumerate(unique_labels):
            z = specificity_rank(H_full, k, self.selected_pool_)
            top_k = np.argsort(z)[::-1][:self.n_top]
            self.markers_[str(l)] = [str(gene_names_arr[self.selected_pool_[i]])
                                     for i in top_k]

        self._fitted = True
        return self

    def evaluate(
        self,
        true_markers: Dict[str, set],
    ) -> Dict:
        """
        Evaluate marker discovery against ground truth.

        Parameters
        ----------
        true_markers : dict
            true_markers[type_label] = set of known marker gene names.

        Returns
        -------
        results : dict
            'per_type': dict of hit counts,
            'mean_hits': float,
            'total_hits': int,
        """
        if not self._fitted:
            raise RuntimeError("Not fitted.")

        per_type = {}
        total = 0
        n_types = 0
        for l, top_genes in self.markers_.items():
            known = true_markers.get(l, set())
            if len(known) == 0:
                continue
            hits = sum(1 for g in top_genes if g in known)
            per_type[l] = hits
            total += hits
            n_types += 1

        return {
            'per_type': per_type,
            'mean_hits': total / n_types if n_types > 0 else 0.0,
            'total_hits': total,
        }
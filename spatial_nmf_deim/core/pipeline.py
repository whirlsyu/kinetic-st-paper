"""
High-level pipeline: SpatialNMFDEIM

Integrates DEIM gene selection + Graph-regularized NMF into a single
user-friendly interface. This is the main class users interact with.

Comparison with FlashDeconv
----------------------------
FlashDeconv (supervised):
    Y (spots x genes) + X (reference K x genes) + coords
    -> HVG + markers -> leverage scores -> CountSketch -> BCD solver
    -> beta (N x K) cell type proportions

SpatialNMFDEIM (unsupervised):
    Y (spots x genes) + coords
    -> DEIM selects REAL genes -> log-CPM -> graph-regularized NMF
    -> W (N x R) spatial loadings + H (R x G) gene module signatures

Key differences:
1. No scRNA-seq reference needed (unsupervised)
2. DEIM selects actual genes (interpretable), not CountSketch projections
3. Output includes gene module signatures (H), not just proportions
4. Discoveries spatial gene co-expression modules
"""

import numpy as np
from scipy import sparse
from typing import Optional, Dict, Any, Union

from spatial_nmf_deim.core.deim import deim_select
from spatial_nmf_deim.core.nmf_solver import GraphRegularizedNMF, compute_module_genes
from spatial_nmf_deim.core.graph import coords_to_adjacency


class SpatialNMFDEIM:
    """
    Unsupervised spatial gene module discovery via DEIM + Graph-NMF.

    Discovers spatially coherent gene expression modules directly from
    spatial transcriptomics data, without requiring a scRNA-seq reference.

    Parameters
    ----------
    n_modules : int, default=10
        Number of spatial gene modules to discover.
    n_genes : int, default=500
        Number of genes to select via DEIM.
    deim_method : str, default="deim"
        DEIM selection strategy:
        - "deim": Full greedy DEIM (most interpretable, deterministic)
        - "leverage": Leverage score ranking (fast, deterministic)
        - "deim_leverage": Combined DEIM-leverage scores
    lambda_spatial : float or "auto", default="auto"
        Spatial regularization strength. "auto" tunes based on data scale.
    spatial_method : str, default="knn"
        Spatial graph method ("knn", "radius", "grid").
    k_neighbors : int, default=6
        Number of neighbors for KNN graph.
    radius : float, optional
        Radius for radius-based graph.
    max_iter : int, default=200
        Maximum NMF iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    deim_rank : int, optional
        Effective rank for DEIM SVD. Auto-estimated if None.
    verbose : bool, default=False
        Print progress.
    random_state : int, default=0
        Random seed.

    Attributes
    ----------
    W_ : ndarray of shape (n_spots, n_modules)
        Spatial loadings. W_[i, k] = membership of spot i in module k.
    H_ : ndarray of shape (n_modules, n_genes_selected)
        Gene module signatures. H_[k, g] = weight of gene g in module k.
    selected_genes_ : ndarray
        Indices of DEIM-selected genes.
    proportions_ : ndarray of shape (n_spots, n_modules)
        Normalized spatial loadings (sum to 1 per spot).

    Example
    -------
    >>> from spatial_nmf_deim import SpatialNMFDEIM
    >>> model = SpatialNMFDEIM(n_modules=10, n_genes=500)
    >>> model.fit(Y, coords)
    >>> # Get spatial domain assignments
    >>> domains = model.get_dominant_module()
    >>> # Get module genes (real gene names!)
    >>> modules = model.get_module_genes(gene_names, n_top=20)
    """

    def __init__(
        self,
        n_modules: int = 10,
        n_genes: int = 500,
        deim_method: str = "deim",
        lambda_spatial: Union[float, str] = "auto",
        spatial_method: str = "knn",
        k_neighbors: int = 6,
        radius: Optional[float] = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        deim_rank: Optional[int] = None,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.n_modules = n_modules
        self.n_genes = n_genes
        self.deim_method = deim_method
        self.lambda_spatial = lambda_spatial
        self.spatial_method = spatial_method
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.max_iter = max_iter
        self.tol = tol
        self.deim_rank = deim_rank
        self.verbose = verbose
        self.random_state = random_state

        # Fitted attributes
        self.W_ = None
        self.H_ = None
        self.selected_genes_ = None
        self.proportions_ = None
        self.info_ = None
        self._fitted = False

    def fit(
        self,
        Y,
        coords: np.ndarray,
        gene_names: Optional[np.ndarray] = None,
    ) -> "SpatialNMFDEIM":
        """
        Fit the model and discover spatial gene modules.

        Parameters
        ----------
        Y : array-like of shape (n_spots, n_genes)
            Spatial transcriptomics count matrix (sparse or dense).
        coords : ndarray of shape (n_spots, 2) or (n_spots, 3)
            Spatial coordinates.
        gene_names : ndarray, optional
            Gene names for interpretability.

        Returns
        -------
        self
        """
        n_spots, n_genes_total = Y.shape

        if self.verbose:
            print(f"SpatialNMF-DEIM: {n_spots} spots x {n_genes_total} genes")
            print(f"  Target: {self.n_modules} modules, {self.n_genes} genes via {self.deim_method}")

        self.n_spots_ = n_spots
        self.n_genes_total_ = n_genes_total
        self.gene_names_ = gene_names
        self._Y_raw = Y

        # ===== Step 1: DEIM Gene Selection =====
        if self.verbose:
            print("Step 1: DEIM gene selection...")

        selected_idx, deim_scores, V = deim_select(
            Y,
            n_genes=self.n_genes,
            rank=self.deim_rank,
            method=self.deim_method,
            preprocess=True,  # log-CPM inside DEIM
        )

        self.selected_genes_ = selected_idx
        self.deim_scores_ = deim_scores

        if self.verbose:
            print(f"  Selected {len(selected_idx)} genes via {self.deim_method}")

        # ===== Step 2: Subset and preprocess =====
        if self.verbose:
            print("Step 2: Preprocessing selected genes...")

        # Subset to selected genes
        Y_subset = Y[:, selected_idx]
        if sparse.issparse(Y_subset):
            Y_subset = Y_subset.toarray()

        # Log-CPM normalization (same as FlashDeconv)
        lib_size = Y_subset.sum(axis=1, keepdims=True)
        lib_size = np.maximum(lib_size, 1.0)
        Y_norm = np.log1p(Y_subset / lib_size * 1e4)

        # ===== Step 3: Build spatial graph =====
        if self.verbose:
            print("Step 3: Building spatial graph...")

        A = coords_to_adjacency(
            coords,
            method=self.spatial_method,
            k=self.k_neighbors,
            radius=self.radius,
        )

        avg_neighbors = np.mean(np.asarray(A.sum(axis=1)).flatten())
        if self.verbose:
            print(f"  Average neighbors: {avg_neighbors:.1f}")

        # ===== Step 4: Auto-tune lambda =====
        if self.lambda_spatial == "auto":
            # Scale lambda relative to data magnitude
            data_scale = np.mean(Y_norm ** 2)
            avg_deg = np.mean(np.asarray(A.sum(axis=1)).flatten())
            lambda_ = 0.01 * data_scale / max(avg_deg, 1.0)
            if self.verbose:
                print(f"Step 4: Auto-tuned lambda = {lambda_:.6f}")
        else:
            lambda_ = float(self.lambda_spatial)
            if self.verbose:
                print(f"Step 4: Using lambda = {lambda_:.6f}")

        self.lambda_used_ = lambda_

        # ===== Step 5: Graph-regularized NMF =====
        if self.verbose:
            print(f"Step 5: Graph-regularized NMF ({self.n_modules} modules)...")

        solver = GraphRegularizedNMF(
            n_modules=self.n_modules,
            lambda_spatial=lambda_,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        W, H, nmf_info = solver.solve(Y_norm, A, verbose=self.verbose)

        # Use raw W for module assignment (column-scaled W can distort argmax)
        W_raw = nmf_info.get("W_raw", W)
        self.W_ = W_raw
        self.H_ = H
        self.info_ = nmf_info

        # Normalize to proportions
        row_sums = W_raw.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        self.proportions_ = W_raw / row_sums

        self._fitted = True

        if self.verbose:
            print("SpatialNMF-DEIM: Done!")

        return self

    def fit_transform(
        self,
        Y,
        coords: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Fit and return spatial loadings."""
        self.fit(Y, coords, **kwargs)
        return self.W_

    def get_dominant_module(self) -> np.ndarray:
        """
        Get the dominant module assignment for each spot.

        Returns
        -------
        domains : ndarray of shape (n_spots,)
            Module index (0 to n_modules-1) for each spot.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return np.argmax(self.proportions_, axis=1)

    def get_module_genes(
        self,
        gene_names: Optional[np.ndarray] = None,
        n_top: int = 20,
        min_score: float = 0.0,
    ) -> Dict[int, Dict]:
        """
        Get top genes for each discovered module.

        Parameters
        ----------
        gene_names : ndarray, optional
            Gene names. If None, uses stored gene_names_ or integer indices.
        n_top : int
            Number of top genes per module.
        min_score : float
            Minimum score threshold.

        Returns
        -------
        modules : dict
            modules[k] = {'genes': [...], 'scores': [...]}
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Map selected gene indices back to original gene names
        if gene_names is not None:
            selected_names = gene_names[self.selected_genes_]
        elif self.gene_names_ is not None:
            selected_names = np.array(self.gene_names_)[self.selected_genes_]
        else:
            selected_names = None

        return compute_module_genes(self.H_, selected_names, n_top, min_score)

    def get_module_enrichment(self, gene_sets: Dict[str, np.ndarray]) -> Dict[int, Dict]:
        """
        Compute enrichment of known gene sets in discovered modules.

        Parameters
        ----------
        gene_sets : dict
            gene_sets[name] = array of gene names or indices.

        Returns
        -------
        enrichment : dict
            enrichment[module_k][gene_set_name] = {
                'overlap': int, 'p_value': float, 'genes': list
            }
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # This is a placeholder for Fisher's exact test or hypergeometric test
        # Implementation depends on the specific use case
        from scipy.stats import hypergeom

        selected_gene_set = set(self.selected_genes_)
        n_selected = len(selected_gene_set)

        enrichment = {}
        for k in range(self.n_modules):
            # Get module genes (top 100 by score)
            module_genes = set(np.argsort(self.H_[k])[::-1][:100])
            enrichment[k] = {}

            for set_name, set_genes in gene_sets.items():
                set_indices = set(set_genes) if not isinstance(set_genes[0], int) else set(int(g) for g in set_genes)
                overlap = module_genes & set_indices
                enrichment[k][set_name] = {
                    "overlap_count": len(overlap),
                    "overlap_genes": list(overlap),
                }

                # Hypergeometric test
                M = n_selected  # total genes in selection
                n = len(module_genes)  # draws (module genes)
                N = len(set_indices)  # success states (gene set members)
                k_val = len(overlap)  # observed successes

                if M > 0 and n > 0 and N > 0:
                    p_value = float(hypergeom.sf(k_val - 1, M, N, n))
                    enrichment[k][set_name]["p_value"] = p_value
                else:
                    enrichment[k][set_name]["p_value"] = 1.0

        return enrichment

    def summary(self) -> Dict[str, Any]:
        """Get model summary."""
        if not self._fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_spots": self.n_spots_,
            "n_genes_total": self.n_genes_total_,
            "n_genes_selected": len(self.selected_genes_),
            "n_modules": self.n_modules,
            "deim_method": self.deim_method,
            "lambda_spatial": self.lambda_used_,
            "converged": self.info_["converged"],
            "n_iterations": self.info_["n_iterations"],
            "final_objective": self.info_.get("final_objective", 0.0),
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"SpatialNMFDEIM(n_modules={self.n_modules}, "
            f"n_genes={self.n_genes}, deim='{self.deim_method}', "
            f"status={status})"
        )
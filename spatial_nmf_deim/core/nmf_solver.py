"""
Graph-regularized NMF solver for spatial gene module discovery.

Solves:
    min_{W >= 0, H >= 0}  ||Y - W @ H||_F^2 + lambda * Tr(H^T L H)

where:
    Y : (n_spots, n_genes) - spatial expression (after DEIM gene selection)
    W : (n_spots, n_modules) - spatial loadings (domain assignments)
    H : (n_modules, n_genes) - gene module signatures
    L : (n_spots, n_spots) - graph Laplacian from spatial coordinates
    lambda : spatial regularization strength

The key insight: this is the SAME optimization as FlashDeconv's BCD solver,
but:
- W replaces beta (spatial loadings instead of cell type proportions)
- H replaces X_sketch (discovered gene modules instead of reference signatures)
- H is LEARNED, not given as reference
- Gene space uses DEIM-selected real genes, not CountSketch projections

Both W and H are updated via multiplicative update rules with spatial
regularization on H (the spatial factor).
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Dict


def graph_regularized_nmf(
    Y: np.ndarray,
    A: sparse.spmatrix,
    n_modules: int = 10,
    lambda_spatial: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-4,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Graph-regularized Non-negative Matrix Factorization.

    Factorizes Y ≈ W @ H with spatial smoothness on W (via graph Laplacian).

    The spatial regularization Tr(W^T L W) encourages neighboring spots
    to have similar module membership, producing spatially coherent domains.

    Parameters
    ----------
    Y : ndarray of shape (n_spots, n_genes)
        Preprocessed spatial expression matrix (after DEIM gene selection).
    A : sparse matrix of shape (n_spots, n_spots)
        Spatial adjacency matrix.
    n_modules : int, default=10
        Number of gene modules (rank of factorization).
    lambda_spatial : float, default=0.1
        Spatial regularization strength. Higher = smoother spatial domains.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance (relative change in objective).
    W_init : ndarray, optional
        Initial spatial loadings (n_spots, n_modules).
    H_init : ndarray, optional
        Initial gene signatures (n_modules, n_genes).
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    W : ndarray of shape (n_spots, n_modules)
        Spatial loadings. W[i, k] indicates how strongly spot i
        belongs to module k. Can be used for spatial domain assignment.
    H : ndarray of shape (n_modules, n_genes)
        Gene module signatures. H[k, g] indicates how strongly gene g
        contributes to module k. Can identify marker genes per module.
    info : dict
        Optimization information.
    """
    n_spots, n_genes = Y.shape
    n_modules = min(n_modules, n_spots, n_genes)

    # --- Compute graph Laplacian ---
    degrees = np.asarray(A.sum(axis=1)).flatten()
    D_inv_sqrt = np.zeros(n_spots)
    nonzero = degrees > 0
    D_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    # L_norm = I - D^{-1/2} A D^{-1/2}, stored as L = D^{-1/2} A D^{-1/2}
    # Tr(W^T L_norm W) = sum_{i,j} A_ij ||w_i/sqrt(d_i) - w_j/sqrt(d_j)||^2
    # This is equivalent to the normalized cut objective.

    # For the multiplicative update, we need D^{-1/2} A D^{-1/2}
    D_inv_sqrt_mat = sparse.diags(D_inv_sqrt)
    L_smooth = D_inv_sqrt_mat @ A @ D_inv_sqrt_mat  # (N, N) sparse

    # --- Initialization ---
    if W_init is not None:
        W = W_init.copy()
    else:
        # Initialize with SVD-based NMF initialization
        W = _svd_init(Y, n_modules, which="W")

    if H_init is not None:
        H = H_init.copy()
    else:
        H = _svd_init(Y, n_modules, which="H")

    # Ensure non-negativity
    W = np.maximum(W, 1e-10)
    H = np.maximum(H, 1e-10)

    # --- Precompute spatial smoothing term ---
    # The gradient of Tr(W^T L_norm W) w.r.t. W is 2 * L_norm @ W
    # In the multiplicative update, we need the positive and negative parts:
    #   [L_norm @ W]_+ and [L_norm @ W]_-
    # Since L_smooth = D^{-1/2} A D^{-1/2} has non-negative entries,
    # L_smooth @ W is already non-negative, so [L_smooth @ W]_- = 0

    # --- Iterative updates ---
    prev_obj = np.inf
    objectives = []
    converged = False

    for iteration in range(max_iter):
        # ---- Update H (gene signatures) ----
        # Standard NMF multiplicative update for H:
        # H <- H * (W^T Y) / (W^T W H + eps)
        WtY = W.T @ Y           # (K, G)
        WtW = W.T @ W           # (K, K)
        WtWH = WtW @ H          # (K, G)
        H = H * (WtY / (WtWH + 1e-10))
        H = np.maximum(H, 1e-10)

        # ---- Update W (spatial loadings) with graph regularization ----
        # Standard NMF: W <- W * (Y H^T) / (W H H^T + eps)
        # With graph regularization on W:
        # W <- W * (Y H^T) / (W H H^T + lambda * L_smooth @ W + eps)
        YHt = Y @ H.T           # (N, K)
        HHt = H @ H.T           # (G, G) -> actually (K, K)
        WHHt = W @ HHt          # (N, K)
        Lsmooth_W = L_smooth @ W  # (N, K) - spatial smoothing

        W = W * (YHt / (WHHt + lambda_spatial * Lsmooth_W + 1e-10))
        W = np.maximum(W, 1e-10)

        # --- Convergence check ---
        if iteration % 5 == 0 or iteration == max_iter - 1:
            # Compute objective: ||Y - WH||^2 + lambda * Tr(W^T L W)
            recon = W @ H
            fidelity = np.sum((Y - recon) ** 2)
            # Tr(W^T (I - L_smooth) W) = ||W||^2 - Tr(W^T L_smooth W)
            spatial_term = np.sum(W * Lsmooth_W)
            # Note: L_smooth = D^{-1/2} A D^{-1/2}, and L_norm = I - L_smooth
            # So Tr(W^T L_norm W) = ||W||^2_F - Tr(W^T L_smooth W)
            # = ||W||^2_F - spatial_term
            # We want to MINIMIZE Tr(W^T L_norm W), which is equivalent to
            # MAXIMIZING Tr(W^T L_smooth W).
            # In the objective: minimize fidelity + lambda * Tr(W^T L_norm W)
            obj = fidelity + lambda_spatial * (np.sum(W ** 2) - spatial_term)
            objectives.append(obj)

            rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1e-10)

            if verbose and (iteration % 20 == 0 or iteration == max_iter - 1):
                print(f"  Iter {iteration}: obj={obj:.4f}, rel_change={rel_change:.2e}")

            if rel_change < tol and iteration > 10:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break

            prev_obj = obj

    # --- Post-processing: normalize modules ---
    # Normalize each module's gene signature to sum to 1
    H_norms = H.sum(axis=1, keepdims=True)
    H_norms = np.maximum(H_norms, 1e-10)
    H_normalized = H / H_norms
    W_scaled = W * H_norms.T  # Adjust W to compensate

    info = {
        "converged": converged,
        "n_iterations": iteration + 1,
        "final_objective": objectives[-1] if objectives else 0.0,
        "objectives": objectives,
        "W_raw": W,  # raw W for correct module assignment (column-scaled W distorts argmax)
    }

    return W_scaled, H_normalized, info


class GraphRegularizedNMF:
    """
    Graph-regularized NMF for spatial gene module discovery.

    This is the core solver class, analogous to FlashDeconv's BCD solver,
    but operates in an unsupervised setting without reference signatures.

    Parameters
    ----------
    n_modules : int, default=10
        Number of spatial gene modules to discover.
    lambda_spatial : float or "auto", default="auto"
        Spatial regularization strength.
    max_iter : int, default=200
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    """

    def __init__(
        self,
        n_modules: int = 10,
        lambda_spatial: float = 0.1,
        max_iter: int = 200,
        tol: float = 1e-4,
    ):
        self.n_modules = n_modules
        self.lambda_spatial = lambda_spatial
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        Y: np.ndarray,
        A: sparse.spmatrix,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve the graph-regularized NMF problem.

        Parameters
        ----------
        Y : ndarray of shape (n_spots, n_genes)
            Spatial expression matrix (DEIM-selected genes).
        A : sparse matrix of shape (n_spots, n_spots)
            Spatial adjacency matrix.
        verbose : bool
            Print progress.

        Returns
        -------
        W, H, info : see graph_regularized_nmf()
        """
        return graph_regularized_nmf(
            Y=Y,
            A=A,
            n_modules=self.n_modules,
            lambda_spatial=self.lambda_spatial,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=verbose,
        )


def _svd_init(
    Y: np.ndarray,
    rank: int,
    which: str = "W",
) -> np.ndarray:
    """
    SVD-based initialization for NMF (NNDSVD-like).

    Uses the truncated SVD to initialize factors with non-negative values.

    Parameters
    ----------
    Y : ndarray of shape (n_spots, n_genes)
        Input matrix.
    rank : int
        Factorization rank.
    which : str
        Which factor to initialize ("W" or "H").

    Returns
    -------
    factor : ndarray
        Initialized factor matrix.
    """
    n_spots, n_genes = Y.shape

    # Truncated SVD
    k = min(rank, n_spots, n_genes)
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    U = U[:, :k]
    S = np.diag(s[:k])
    V = Vt[:k, :].T

    # NNDSVD-like initialization: take absolute values
    if which == "W":
        # W = |U| @ sqrt(S)
        factor = np.abs(U) @ np.sqrt(S)
    else:
        # H = sqrt(S) @ |V|^T
        factor = np.sqrt(S) @ np.abs(V).T

    # Add small noise to break symmetry
    factor += 1e-4 * np.abs(factor).max() * np.random.rand(*factor.shape)

    return factor


def compute_module_genes(
    H: np.ndarray,
    gene_names: Optional[np.ndarray] = None,
    n_top: int = 20,
    min_score: float = 0.0,
) -> Dict[int, Dict]:
    """
    Extract top genes for each module from the gene signature matrix.

    Parameters
    ----------
    H : ndarray of shape (n_modules, n_genes)
        Gene module signatures from NMF.
    gene_names : ndarray, optional
        Gene names. If None, uses integer indices.
    n_top : int
        Number of top genes per module.
    min_score : float
        Minimum score threshold for gene inclusion.

    Returns
    -------
    modules : dict
        modules[k] = {
            'genes': list of gene names/indices,
            'scores': list of corresponding scores,
        }
    """
    n_modules = H.shape[0]
    modules = {}

    for k in range(n_modules):
        scores = H[k, :]
        top_idx = np.argsort(scores)[::-1]

        genes = []
        gene_scores = []
        for idx in top_idx:
            if scores[idx] < min_score:
                break
            if gene_names is not None:
                genes.append(str(gene_names[idx]))
            else:
                genes.append(int(idx))
            gene_scores.append(float(scores[idx]))
            if len(genes) >= n_top:
                break

        modules[k] = {"genes": genes, "scores": gene_scores}

    return modules
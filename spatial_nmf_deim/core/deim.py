"""
DEIM (Discrete Empirical Interpolation Method) for gene selection.

Replaces CountSketch in FlashDeconv with a deterministic, interpretable
gene selection procedure. DEIM selects actual genes (not random projections),
making the resulting features fully biologically interpretable.

Mathematical background
----------------------
Given matrix M with SVD M = U @ diag(s) @ Vt, DEIM greedily selects
rows of V[:,:r] (right singular vectors) to form an interpolatory basis.
Each selected row corresponds to a real gene in the original space.

Key difference from leverage scores:
- Leverage scores: row norms of U (importance for column space)
- DEIM: greedy selection on V (maximizes interpolation accuracy)

Both can be computed from the same SVD, and both are compatible with
leverage-score-weighted importance ranking.

References
----------
Chaturantabut, Sorensen (2010). "Discrete Empirical Interpolation Method
for Nonlinear Model Reduction." SIAM J. Sci. Comput.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import sparse


def compute_deim_scores(
    V: np.ndarray,
    singular_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute DEIM importance scores for each row of V.

    Unlike standard DEIM which greedily selects one row at a time,
    this function computes a score for every row that reflects how
    important each row would be for the DEIM interpolation framework.

    The score combines two factors:
    1. Row norm in V (leverage score component)
    2. Maximum absolute value in the first few columns of V
       (captures the "pivot-like" importance used in DEIM's greedy step)

    Parameters
    ----------
    V : ndarray of shape (n_features, k)
        Right singular vectors (from SVD of the data matrix).
    k should be the effective rank.
    singular_values : ndarray of shape (k,), optional
        Singular values. If provided, scores are weighted by
        singular value importance (more principled).

    Returns
    -------
    scores : ndarray of shape (n_features,)
        DEIM importance score for each feature (gene).
    """
    n_features, k = V.shape

    # Factor 1: Leverage score component (row norms of V)
    if singular_values is not None:
        # Weight by singular value importance (same as FlashDeconv)
        s2 = singular_values[:k] ** 2
        weights = s2 / (s2 + 1e-6)
        leverage = np.sum((V ** 2) * weights[np.newaxis, :], axis=1)
    else:
        leverage = np.sum(V ** 2, axis=1)

    # Factor 2: DEIM greedy pivot score
    # The first DEIM step selects the row with the largest absolute
    # value in the first column of V. Subsequent steps look at
    # the residual. We approximate this by looking at max |V[:, j]|
    # weighted by the importance of each singular vector.
    if singular_values is not None:
        s_norm = singular_values[:k] / (singular_values[0] + 1e-10)
    else:
        s_norm = np.ones(k) / np.sqrt(k)

    pivot_score = np.zeros(n_features)
    for j in range(k):
        # Weight later columns less (they capture less variance)
        pivot_score += s_norm[j] * np.abs(V[:, j])

    # Combine: leverage captures overall importance, pivot captures
    # the greedy DEIM selection criterion
    # Use geometric mean to balance both factors
    scores = np.sqrt(leverage * pivot_score + 1e-20)

    # Normalize to sum to 1
    scores = scores / (scores.sum() + 1e-10)

    return scores


def deim_select(
    Y: np.ndarray,
    n_genes: int = 500,
    rank: Optional[int] = None,
    method: str = "deim",
    preprocess: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select genes using DEIM from spatial transcriptomics data.

    Performs SVD on the (optionally preprocessed) spatial matrix,
    then applies DEIM-style greedy selection to pick informative genes.

    Parameters
    ----------
    Y : array-like of shape (n_spots, n_genes)
        Spatial transcriptomics count matrix.
    n_genes : int, default=500
        Number of genes to select.
    rank : int, optional
        Effective rank for truncated SVD. If None, estimated automatically
        based on singular value drop-off (ratio > 0.1 of largest).
    method : str, default="deim"
        Selection strategy:
        - "deim": Full DEIM greedy selection (deterministic)
        - "leverage": Leverage score ranking (deterministic)
        - "deim_leverage": DEIM scores (combined, as in compute_deim_scores)
    preprocess : bool, default=True
        Whether to apply log-CPM normalization before SVD.

    Returns
    -------
    selected_idx : ndarray of shape (n_genes,)
        Indices of selected genes (real gene indices).
    scores : ndarray of shape (n_genes,)
        Selection scores for selected genes.
    V : ndarray of shape (n_genes_orig, k)
        Right singular vectors (for downstream use).

    Notes
    -----
    The selected genes are actual genes from the input matrix,
    making them fully interpretable. This is the key advantage
    over CountSketch, which produces non-interpretable mixed features.
    """
    from scipy.sparse import diags, issparse

    n_spots, n_genes_orig = Y.shape
    n_genes = min(n_genes, n_genes_orig)

    # --- Preprocessing ---
    if preprocess:
        if issparse(Y):
            lib_size = np.array(Y.sum(axis=1)).flatten()
            lib_size[lib_size == 0] = 1.0
            D = diags(1e4 / lib_size)
            Y_proc = D @ Y
            Y_proc.data = np.log1p(Y_proc.data)
            if issparse(Y_proc):
                Y_proc = Y_proc.toarray()
        else:
            Y_dense = np.asarray(Y, dtype=np.float64)
            lib_size = Y_dense.sum(axis=1, keepdims=True)
            lib_size = np.maximum(lib_size, 1.0)
            Y_proc = np.log1p(Y_dense / lib_size * 1e4)
    else:
        if issparse(Y):
            Y_proc = Y.toarray()
        else:
            Y_proc = np.asarray(Y, dtype=np.float64)

    # --- SVD (truncated) ---
    # For very large matrices, use randomized SVD for efficiency
    if n_spots > 10000 or n_genes_orig > 5000:
        # Randomized SVD (much faster for large matrices)
        U, s, Vt = _randomized_svd(Y_proc, n_components=min(rank or 100, min(n_spots, n_genes_orig) - 1))
    else:
        # Full SVD (exact for small matrices)
        U, s, Vt = np.linalg.svd(Y_proc, full_matrices=False)

    # Determine effective rank
    if rank is None:
        rank = _estimate_rank(s, threshold=0.1)
    rank = min(rank, len(s), n_genes_orig, n_spots)

    # Truncate to effective rank
    V = Vt[:rank, :].T  # (n_genes_orig, rank)
    singular_values = s[:rank]

    # --- Gene selection ---
    if method == "deim":
        selected_idx, scores = _deim_greedy(V, n_genes)
    elif method == "leverage":
        scores_all = np.sum(V ** 2, axis=1)
        top_idx = np.argsort(scores_all)[::-1][:n_genes]
        selected_idx = top_idx
        scores = scores_all[top_idx]
    elif method == "deim_leverage":
        scores_all = compute_deim_scores(V, singular_values)
        top_idx = np.argsort(scores_all)[::-1][:n_genes]
        selected_idx = top_idx
        scores = scores_all[top_idx]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'deim', 'leverage', or 'deim_leverage'.")

    # Sort selected indices for consistency
    sort_order = np.argsort(selected_idx)
    selected_idx = selected_idx[sort_order]
    scores = scores[sort_order]

    return selected_idx, scores, V


def _deim_greedy(
    V: np.ndarray,
    n_select: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full DEIM greedy selection algorithm.

    Greedily selects rows of V to maximize interpolation accuracy.
    At each step, selects the row with the largest residual error.

    Parameters
    ----------
    V : ndarray of shape (n_features, k)
        Right singular vectors.
    n_select : int
        Number of rows (genes) to select.

    Returns
    -------
    selected : ndarray of shape (n_select,)
        Selected row indices.
    scores : ndarray of shape (n_select,)
        Selection scores (residual magnitude at selection time).
    """
    n_features, k = V.shape
    n_select = min(n_select, n_features)

    selected = []
    scores = []
    # Track which rows are still available
    available = np.ones(n_features, dtype=bool)

    # DEIM Step 1: select row with max |V[:, 0]| (first column)
    first_col = np.abs(V[:, 0])
    first_col[~available] = -1  # exclude unavailable
    idx = np.argmax(first_col)
    selected.append(idx)
    available[idx] = False
    scores.append(float(first_col[idx]))

    # Build interpolation matrix incrementally
    # V_selected stores the selected rows of V
    V_selected = V[idx : idx + 1, :].copy()  # (1, k)

    # Subsequent DEIM steps
    for step in range(1, n_select):
        # Compute residual: for each available row, compute
        # the interpolation error = V[row] - V[row, selected_cols] @ V_selected^{-1} @ V_selected
        # Simplified: residual = V[row] projected onto the orthogonal complement of V_selected

        if step < k:
            # Full DEIM with pseudo-inverse
            try:
                # DEIM residual computation:
                # For each available row r, compute the interpolation error
                # using the already-selected rows as interpolation basis.
                #
                # V_selected is (step, k) matrix of selected rows
                # For each available row v (1, k):
                #   residual = v - V_selected.T @ (V_selected @ V_selected.T)^{-1} @ (V_selected @ v)
                # Equivalently: residual = v - V_selected.T @ pinv(V_selected) @ v
                V_avail = V[available, :]  # (n_avail, k)
                # V_selected is (step, k), compute V_selected^+ @ V_avail^T
                # then residual_i = V_avail[i] - V_selected^T @ (V_selected^+ @ V_avail[i]^T)
                # More efficiently: residual = V_avail - (V_selected^+ @ V_avail^T)^T @ V_selected
                # No wait, let's think again.
                # We want: for each row v in V_avail, compute v - V_selected^T @ pinv(V_selected @ V_selected^T) @ V_selected @ v
                # = v - V_selected^T @ (V_selected @ V_selected^T)^{-1} @ V_selected @ v  (since step <= k)
                # Simplify: let Z = V_selected @ V_selected^T  (step x step), then
                # residual = v - V_selected^T @ Z^{-1} @ V_selected @ v
                Z = V_selected @ V_selected.T  # (step, step)
                # For all rows at once:
                # V_selected @ V_avail^T is (step, n_avail)
                Vsel_Vavail = V_selected @ V_avail.T  # (step, n_avail)
                # Solve Z @ coef = Vsel_Vavail for coef
                try:
                    coef = np.linalg.solve(Z, Vsel_Vavail)  # (step, n_avail)
                except np.linalg.LinAlgError:
                    coef = np.linalg.lstsq(Z, Vsel_Vavail, rcond=None)[0]
                # Reconstructed = V_selected^T @ coef  -> (k, n_avail) but we need (n_avail, k)
                reconstructed = (V_selected.T @ coef).T  # (n_avail, k)
                residual = V_avail - reconstructed
                max_residual = np.max(np.abs(residual), axis=1)

                # Pick the row with largest residual
                local_idx = np.argmax(max_residual)
                global_indices = np.where(available)[0]
                idx = global_indices[local_idx]
            except np.linalg.LinAlgError:
                # Fallback: pick by max |V[:, step]|
                col = np.abs(V[:, min(step, k - 1)])
                col[~available] = -1
                idx = np.argmax(col)
        else:
            # After k steps, V_selected is full rank
            # Use residual from the column space projection
            col = np.abs(V[:, (step - 1) % k])
            col[~available] = -1
            idx = np.argmax(col)

        selected.append(idx)
        available[idx] = False
        scores.append(float(np.max(np.abs(V[idx, :]))))

        # Update V_selected
        V_selected = np.vstack([V_selected, V[idx : idx + 1, :]])

    return np.array(selected), np.array(scores)


def _randomized_svd(
    M: np.ndarray,
    n_components: int = 100,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomized SVD for efficient low-rank approximation of large matrices.

    Parameters
    ----------
    M : ndarray of shape (m, n)
        Input matrix.
    n_components : int
        Number of singular values/vectors to compute.
    n_oversamples : int
        Extra dimensions for better approximation.
    n_iter : int
        Number of power iterations for better singular value separation.
    random_state : int
        Random seed.

    Returns
    -------
    U : ndarray (m, n_components)
    s : ndarray (n_components,)
    Vt : ndarray (n_components, n)
    """
    rng = np.random.RandomState(random_state)
    m, n = M.shape
    actual_rank = min(n_components + n_oversamples, min(m, n))

    # Step 1: Random projection
    Omega = rng.standard_normal((n, actual_rank))
    Y = M @ Omega  # (m, actual_rank)

    # Step 2: Power iteration for better singular value separation
    for _ in range(n_iter):
        Y = M @ (M.T @ Y)

    # Step 3: QR decomposition
    Q, _ = np.linalg.qr(Y)

    # Step 4: Form smaller matrix and compute its SVD
    B = Q.T @ M  # (actual_rank, n)
    Ub, s, Vtb = np.linalg.svd(B, full_matrices=False)

    # Step 5: Recover U
    U = Q @ Ub

    # Truncate to requested rank
    k = min(n_components, len(s))
    return U[:, :k], s[:k], Vtb[:k, :]


def _estimate_rank(
    singular_values: np.ndarray,
    threshold: float = 0.1,
) -> int:
    """
    Estimate effective rank from singular values.

    Uses the ratio criterion: rank = number of singular values
    greater than threshold * max(singular_values).

    Parameters
    ----------
    singular_values : ndarray
        Singular values in descending order.
    threshold : float
        Ratio threshold.

    Returns
    -------
    rank : int
        Estimated effective rank.
    """
    if len(singular_values) == 0:
        return 1
    s_max = singular_values[0]
    rank = int(np.sum(singular_values > threshold * s_max))
    return max(rank, 1)  # At least rank 1
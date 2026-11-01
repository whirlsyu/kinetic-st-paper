# -*- coding: utf-8 -*-
"""
Asymmetric cell-cell interaction analysis inspired by scIMF (Jiang et al. 2026).

scIMF uses cell-wise attention to capture non-reciprocal interactions. KineticST's
collision operator with ligand-receptor rules also models directional signaling.
This module quantifies interaction asymmetry from the kinetic steady-state.

Metrics:
  - asymmetry_index: ||W - W.T||_F / ||W + W.T||_F per domain pair
  - directional_flow: net flux from domain A to B vs B to A
  - non_stationarity: temporal variance of interaction strengths
"""

import numpy as np
from scipy import sparse


def interaction_asymmetry(W, labels, domain_ids=None):
    """Compute asymmetry of the interaction matrix W between domains.

    Parameters
    ----------
    W : (n, n) array-like
        Pairwise interaction strengths (e.g. collision kernel weights).
    labels : (n,) array-like
        Domain assignments per spot/cell.
    domain_ids : list, optional
        Subset of domains to analyze.

    Returns
    -------
    dict with keys:
        asymmetry_matrix : (k, k) domain-level asymmetry scores
        net_flow : (k, k) net directional flow (A->B minus B->A)
        global_asymmetry : scalar, 0 = symmetric, 1 = fully asymmetric
    """
    if sparse.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)
    labels = np.asarray(labels)

    if domain_ids is None:
        domain_ids = np.unique(labels)
    k = len(domain_ids)

    # Aggregate to domain-level interaction matrix
    M = np.zeros((k, k))
    for i, di in enumerate(domain_ids):
        for j, dj in enumerate(domain_ids):
            mask_i = labels == di
            mask_j = labels == dj
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                M[i, j] = W[np.ix_(mask_i, mask_j)].mean()

    # Asymmetry: ||M - M.T|| / ||M + M.T||
    diff = M - M.T
    symm = M + M.T
    asym_matrix = np.where(np.abs(symm) > 1e-12,
                           np.abs(diff) / (np.abs(symm) + 1e-12), 0.0)
    global_asym = np.abs(diff).sum() / (np.abs(symm).sum() + 1e-12)

    # Net directional flow: M[i,j] - M[j,i]
    net_flow = diff

    return {
        "asymmetry_matrix": asym_matrix,
        "net_flow": net_flow,
        "global_asymmetry": float(global_asym),
        "domain_ids": domain_ids.tolist(),
        "interaction_matrix": M,
    }


def temporal_nonstationarity(W_history, labels, window=10):
    """Measure how interaction strengths vary over time.

    Parameters
    ----------
    W_history : list of (n, n) arrays
        Interaction matrices at successive time points.
    labels : (n,) array-like
        Domain assignments.
    window : int
        Number of time points to aggregate.

    Returns
    -------
    dict with keys:
        temporal_cv : (k, k) coefficient of variation per domain pair
        mean_cv : scalar mean temporal variability
        is_stationary : bool, True if mean_cv < 0.3
    """
    domain_ids = np.unique(labels)
    k = len(domain_ids)
    T = len(W_history)

    # Aggregate each time point to domain level
    M_hist = np.zeros((T, k, k))
    for t, W in enumerate(W_history):
        if sparse.issparse(W):
            W = W.toarray()
        for i, di in enumerate(domain_ids):
            for j, dj in enumerate(domain_ids):
                mask_i = labels == di
                mask_j = labels == dj
                if mask_i.sum() > 0 and mask_j.sum() > 0:
                    M_hist[t, i, j] = W[np.ix_(mask_i, mask_j)].mean()

    # Coefficient of variation over time
    mean_M = M_hist.mean(axis=0)
    std_M = M_hist.std(axis=0)
    temporal_cv = np.where(mean_M > 1e-12, std_M / (mean_M + 1e-12), 0.0)
    mean_cv = float(temporal_cv.mean())

    return {
        "temporal_cv": temporal_cv,
        "mean_cv": mean_cv,
        "is_stationary": mean_cv < 0.3,
        "interaction_history": M_hist,
    }


def ligand_receptor_directionality(Y, labels, lr_pairs, gene_names):
    """Quantify directional signaling from ligand to receptor expression.

    For each LR pair (L, R), compute the correlation between L expression in
    domain A and R expression in neighboring domain B, vs the reverse.

    Parameters
    ----------
    Y : (n, g) array-like
        Expression matrix (spots x genes).
    labels : (n,) array-like
        Domain assignments.
    lr_pairs : list of (ligand, receptor) tuples
    gene_names : (g,) array-like
        Gene names matching columns of Y.

    Returns
    -------
    dict with per-pair directionality scores and aggregate metrics.
    """
    if sparse.issparse(Y):
        Y = Y.toarray()
    Y = np.asarray(Y, dtype=np.float64)
    labels = np.asarray(labels)
    g2i = {str(g): i for i, g in enumerate(gene_names)}

    domain_ids = np.unique(labels)
    results = []

    for lig, rec in lr_pairs:
        if lig not in g2i or rec not in g2i:
            continue
        iL, iR = g2i[lig], g2i[rec]
        L_expr = Y[:, iL]
        R_expr = Y[:, iR]

        # Mean expression per domain
        L_mean = np.array([L_expr[labels == d].mean() for d in domain_ids])
        R_mean = np.array([R_expr[labels == d].mean() for d in domain_ids])

        # Directionality: correlation of L in A with R in B vs R in A with L in B
        # Simplified: use covariance asymmetry
        LR_cov = np.cov(L_mean, R_mean)[0, 1]
        RL_cov = np.cov(R_mean, L_mean)[0, 1]  # same, but we use cross-domain

        # Cross-domain: L in domain i predicts R in domain j (i != j)
        cross_scores = []
        for i in range(len(domain_ids)):
            for j in range(len(domain_ids)):
                if i == j:
                    continue
                # L_i * R_j vs R_i * L_j
                lr_score = L_mean[i] * R_mean[j]
                rl_score = R_mean[i] * L_mean[j]
                cross_scores.append((lr_score - rl_score) / (lr_score + rl_score + 1e-12))

        dir_score = float(np.mean(cross_scores)) if cross_scores else 0.0
        results.append({
            "ligand": lig,
            "receptor": rec,
            "directionality": dir_score,
            "L_mean": L_mean.tolist(),
            "R_mean": R_mean.tolist(),
        })

    return {
        "pairs": results,
        "mean_directionality": float(np.mean([r["directionality"] for r in results])) if results else 0.0,
        "n_pairs": len(results),
    }

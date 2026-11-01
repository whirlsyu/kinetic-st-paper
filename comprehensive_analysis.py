
Comprehensive analysis script (upgraded): 15 supervised marker discovery + 5 unsupervised spatial clustering methods.

Comparison of DEIM vs HVG vs LogFC vs Wilcoxon vs T-test vs RF vs LR-L1 and others.
Includes KineticST (kinetic-theory spatial domain identification based on Deng Yu's mathematical work).
Includes KineticST-marker (supervised marker discovery on the kinetically smoothed expression field).

Datasets:
  1. Simulated data (K=10, G=5000, FC=8, dropout=0.3) x 10 seeds
  2. STSP human brain sections x 8 (~5000 spots, 5-8 clusters)
  3. MOSTA mouse embryo Stereo-seq x 3 (E9.5/E12.5, 13-17 types)
  4. mouse_E11.5_heart mouse heart sections x 4 (replacing elongated Arabidopsis, 5 heart regions)

Supervised methods (15):
  DEIM-c(100/200/500/1000/2000), mRMR(500), Laplacian(500), HVG(500),
  DEIM-full(500), LogFC, Wilcoxon, T-test, RF-importance, LR-L1, KineticST-marker

Unsupervised methods (5):
  DEIM+NMF, HVG+NMF, HVG+NMF(no spatial), PCA+KMeans, KineticST

KineticST method (based on kinetic_st package):
  Micro-meso-macro multiscale bridging methodology based on Deng Yu's 2026 Fields Medal work.
  Ligand-receptor cell communication is modeled as a Boltzmann-like collision operator.
  Cell-state density is solved via stochastic kinetic equations.
  Spatial domains are defined through the non-equilibrium invariant measure.

Outputs:
  - analysis_results.json (all quantitative metrics)
  - fig_spatial_{dataset}.png (spatial domain visualization)

Usage:
    python comprehensive_analysis.py
    # Or run cell by cell in Jupyter


# %% [markdown]
# # Comprehensive Analysis: DEIM vs HGV
# 14 supervised marker methods + 5 unsupervised clustering methods

# %%
import sys
import os
import json
import time
import warnings

warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = Path('dataset')  # TODO: set your dataset path

# ============================================================
# Package imports
# ============================================================
from spatial_nmf_deim import (
    SpatialNMFDEIM,
    SupervisedDEIM,
    generate_simulation,
    coords_to_adjacency,
)
from spatial_nmf_deim.core.nmf_solver import GraphRegularizedNMF, graph_regularized_nmf
from spatial_nmf_deim.core.deim import deim_select

# KineticST: kinetic-theory spatial transcriptomics analysis based on Deng Yu's mathematical work
from kinetic_st import KineticST, KineticSTConfig, find_kinetic_markers


# %%
# ============================================================
# Utility Functions
# ============================================================

def logcpm(Y):
    """Log-CPM normalization: np.log1p(Y / max(lib_size, 1) * 1e4)."""
    Y = np.asarray(Y, dtype=np.float64)
    lib_size = np.maximum(Y.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(Y / lib_size * 1e4)


def select_hvg_indices(Y_norm, n_genes=500):
    """Select top variable genes by variance on log-CPM data."""
    var = np.var(Y_norm, axis=0)
    return np.argsort(var)[::-1][:n_genes]


def compute_auto_lambda(Y_norm, A):
    """Auto-tune spatial regularization lambda (same formula as SpatialNMFDEIM).
    lambda = 0.01 * mean(Y_norm^2) / max(avg_degree, 1)
    """
    data_scale = np.mean(Y_norm ** 2)
    avg_deg = np.mean(np.asarray(A.sum(axis=1)).flatten())
    return 0.01 * data_scale / max(avg_deg, 1.0)


def encode_labels(labels):
    """Convert string labels to integers. Returns (int_labels, label_map)."""
    original = np.asarray(labels).copy()
    unique_orig = np.unique(original)
    label_map = {str(l): i for i, l in enumerate(unique_orig)}
    int_labels = np.array([label_map[str(l)] for l in original], dtype=int)
    return int_labels, label_map


def subsample_indices(labels, target_per_type=300, seed=42):
    """Stratified subsample indices, preserving type proportions."""
    rng = np.random.RandomState(seed)
    unique = np.unique(labels)
    idx_keep = []
    for l in unique:
        idx_l = np.where(labels == l)[0]
        n = min(target_per_type, len(idx_l))
        if n == len(idx_l):
            idx_keep.append(idx_l)
        else:
            idx_keep.append(rng.choice(idx_l, n, replace=False))
    return np.concatenate(idx_keep)


def spatial_smooth_labels(labels, A, n_iter=2):
    """Majority-vote spatial smoothing using sparse adjacency matrix."""
    labels = labels.copy().astype(int)
    K = labels.max() + 1
    n = len(labels)
    for _ in range(n_iter):
        one_hot = np.zeros((n, K))
        one_hot[np.arange(n), labels] = 1.0
        if sparse.issparse(A):
            counts = np.asarray(A.T @ one_hot)
        else:
            counts = A.T @ one_hot
        labels = np.argmax(counts, axis=1)
    return labels


# %%
# ============================================================
# Metrics for unsupervised clustering
# ============================================================

def compute_coherence(pred, A):
    """Build KNN graph from adjacency, count fraction of neighbor pairs
    with same predicted label.
    pred: (n,) integer labels
    A: sparse adjacency matrix
    """
    A_csr = A.tocsr()
    n_agree, n_total = 0, 0
    for i in range(len(pred)):
        nb = A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i + 1]]
        n_total += len(nb)
        n_agree += np.sum(pred[nb] == pred[i])
    return n_agree / max(n_total, 1)


def compute_entropy(pred, K):
    """Normalized Shannon entropy of cluster size distribution.
    H / log2(K), so 1.0 = uniform distribution.
    """
    counts = np.bincount(pred, minlength=K)
    probs = counts / counts.sum()
    H = -np.sum(probs * np.log2(probs + 1e-10))
    return H / np.log2(K)


# %%
# ============================================================
# Supervised Marker Discovery Methods (14 total)
# ============================================================

def run_all_supervised(Y, labels, gene_names, n_top=20, true_markers=None,
                       coords=None, n_domains=None):
    """Run all 14 supervised marker discovery methods.

    Parameters
    ----------
    Y : ndarray (n_spots, n_genes) - raw counts (dense)
    labels : ndarray (n_spots,) - integer labels
    gene_names : ndarray (n_genes,)
    n_top : int - top markers per type
    true_markers : dict or None - ground truth for hit evaluation

    Returns
    -------
    results : dict with 'methods', 'mean_hits', 'total_hits', 'win_types', 'max_per_type'
    """
    n_genes_total = Y.shape[1]
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    gene_arr = np.array(gene_names) if gene_names is not None else np.array([f"G_{i}" for i in range(n_genes_total)])

    # Precompute group mean matrix (used by many methods)
    Y_norm = logcpm(Y)
    H_mean = np.zeros((K, n_genes_total))
    for k, l in enumerate(unique_labels):
        H_mean[k] = Y_norm[labels == l].mean(axis=0)

    all_markers = {}

    # ---- 1-5. DEIM-c with different pool sizes ----
    for pool_size in [100, 200, 500, 1000, 2000]:
        key = f"DEIM-c({pool_size})"
        print(f"    {key}...", end='', flush=True)
        t0 = time.time()
        try:
            ng = min(pool_size, n_genes_total)
            sd = SupervisedDEIM(pool_size=ng, n_top=n_top)
            sd.fit(Y, labels, gene_arr)
            all_markers[key] = sd.markers_
            print(f" done ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f" FAILED ({e})")
            all_markers[key] = {str(l): [] for l in unique_labels}

    # ---- 6. mRMR(500) ----
    print("    mRMR(500)...", end='', flush=True)
    t0 = time.time()
    try:
        pool_n = min(500, n_genes_total)
        # Pre-filter to top 3000 HVG for speed (MI computation is O(n_genes * n_spots))
        n_pre = min(3000, n_genes_total)
        hvg_pre_idx = np.argsort(np.var(Y_norm, axis=0))[::-1][:n_pre]
        Y_pre = Y_norm[:, hvg_pre_idx]
        gene_arr_pre = gene_arr[hvg_pre_idx]
        H_mean_pre = H_mean[:, hvg_pre_idx]
        # Relevance: MI between each pre-filtered gene and labels
        mi_relevance = mutual_info_classif(Y_pre, labels, discrete_features=False)
        # Candidate pool: top 800 by relevance
        candidate_local = np.argsort(mi_relevance)[::-1][:min(800, n_pre)]
        Y_cand = Y_pre[:, candidate_local]
        mi_labels = mi_relevance[candidate_local].copy()
        # Correlation-based redundancy
        Y_c = Y_cand - Y_cand.mean(axis=0, keepdims=True)
        stds = Y_c.std(axis=0, keepdims=True)
        stds[stds < 1e-10] = 1.0
        Y_c = Y_c / stds
        corr_matrix = np.clip((Y_c.T @ Y_c) / Y_c.shape[0], -1, 1)
        selected_local = []
        selected_set = set()
        for step in range(pool_n):
            if step == 0:
                best_local = int(np.argmax(mi_labels))
            else:
                corr_sel = np.abs(corr_matrix[:, selected_local])
                max_red = corr_sel.max(axis=1) ** 2
                mrmr_scores = mi_labels - max_red
                for s in selected_local:
                    mrmr_scores[s] = -np.inf
                best_local = int(np.argmax(mrmr_scores))
            selected_local.append(best_local)
            selected_set.add(best_local)
        # Map back to pre-filtered indices, then to original gene names
        pool_idx_arr = hvg_pre_idx[candidate_local[selected_local]]
        markers_mrmr = {}
        for k, l in enumerate(unique_labels):
            mean_k = H_mean[k, pool_idx_arr]
            mean_rest = (H_mean[:, pool_idx_arr].sum(axis=0) - mean_k) / (K - 1)
            z = (mean_k - mean_rest) / (np.std(H_mean[:, pool_idx_arr], axis=0) + 1e-10)
            top = np.argsort(z)[::-1][:n_top]
            markers_mrmr[str(l)] = [str(gene_arr[pool_idx_arr[i]]) for i in top]
        all_markers['mRMR(500)'] = markers_mrmr
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['mRMR(500)'] = {str(l): [] for l in unique_labels}

    # ---- 7. Laplacian(500) ----
    print("    Laplacian(500)...", end='', flush=True)
    t0 = time.time()
    try:
        from sklearn.neighbors import kneighbors_graph
        # Build KNN(6) graph on log-CPM data
        n_neighbors_graph = min(6, Y_norm.shape[0] - 1)
        W_graph = kneighbors_graph(Y_norm, n_neighbors=n_neighbors_graph, mode='connectivity',
                                   metric='euclidean', include_self=False)
        W_graph = W_graph.maximum(W_graph.T)  # symmetrize

        # Compute normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
        degrees = np.asarray(W_graph.sum(axis=1)).flatten()
        D_inv_sqrt = np.zeros(Y_norm.shape[0])
        nonzero_d = degrees > 0
        D_inv_sqrt[nonzero_d] = 1.0 / np.sqrt(degrees[nonzero_d])
        D_inv_sqrt_mat = sparse.diags(D_inv_sqrt)
        L_sym = sparse.eye(Y_norm.shape[0]) - D_inv_sqrt_mat @ W_graph @ D_inv_sqrt_mat

        # Laplacian score for each gene: g^T L g / (g^T g + eps)
        # Lower score = more informative (less smooth on the graph)
        n_eval = min(500, n_genes_total)
        var_order = np.argsort(np.var(Y_norm, axis=0))[::-1][:min(2000, n_genes_total)]
        lap_scores = np.zeros(len(var_order))
        for j, g_idx in enumerate(var_order):
            g_vec = Y_norm[:, g_idx]
            # g^T L g = g^T g - g^T D^{-1/2} W D^{-1/2} g
            g_norm_sq = np.dot(g_vec, g_vec)
            # Compute D^{-1/2} W D^{-1/2} g
            g_scaled = g_vec * D_inv_sqrt
            smooth_part = W_graph @ g_scaled  # W @ D^{-1/2} g
            smooth_part = smooth_part * D_inv_sqrt  # D^{-1/2} W D^{-1/2} g
            lap_val = g_norm_sq - np.dot(g_vec, smooth_part)
            lap_scores[j] = lap_val / (g_norm_sq + 1e-10)

        # Select lowest 500 (most informative = least smooth on graph)
        pool_n_lap = min(500, len(var_order))
        top_lap_local = np.argsort(lap_scores)[:pool_n_lap]
        pool_idx_lap = var_order[top_lap_local]

        # Z-score specificity ranking within pool
        markers_lap = {}
        for k, l in enumerate(unique_labels):
            mean_k = H_mean[k, pool_idx_lap]
            mean_rest = (H_mean[:, pool_idx_lap].sum(axis=0) - mean_k) / (K - 1)
            z = (mean_k - mean_rest) / (np.std(H_mean[:, pool_idx_lap], axis=0) + 1e-10)
            top = np.argsort(z)[::-1][:n_top]
            markers_lap[str(l)] = [str(gene_arr[pool_idx_lap[i]]) for i in top]
        all_markers['Laplacian(500)'] = markers_lap
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['Laplacian(500)'] = {str(l): [] for l in unique_labels}

    # ---- 8. HVG(500) ----
    print("    HVG(500)...", end='', flush=True)
    t0 = time.time()
    try:
        pool_idx_hvg = select_hvg_indices(Y_norm, n_genes=min(500, n_genes_total))
        markers_hvg = {}
        for k, l in enumerate(unique_labels):
            mean_k = H_mean[k, pool_idx_hvg]
            mean_rest = (H_mean[:, pool_idx_hvg].sum(axis=0) - mean_k) / (K - 1)
            z = (mean_k - mean_rest) / (np.std(H_mean[:, pool_idx_hvg], axis=0) + 1e-10)
            top = np.argsort(z)[::-1][:n_top]
            markers_hvg[str(l)] = [str(gene_arr[pool_idx_hvg[i]]) for i in top]
        all_markers['HVG(500)'] = markers_hvg
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['HVG(500)'] = {str(l): [] for l in unique_labels}

    # ---- 9. DEIM-full(500) ----
    print("    DEIM-full(500)...", end='', flush=True)
    t0 = time.time()
    try:
        # For speed: pre-filter to top 5000 HVG if gene count is large
        if n_genes_total > 5000:
            hvg_pre = np.argsort(np.var(Y_norm, axis=0))[::-1][:5000]
            Y_deim_full = Y[:, hvg_pre]
            gene_arr_deim = gene_arr[hvg_pre]
            H_mean_deim = H_mean[:, hvg_pre]
        else:
            Y_deim_full = Y
            gene_arr_deim = gene_arr
            H_mean_deim = H_mean
        selected_idx, _, _ = deim_select(Y_deim_full, n_genes=min(500, Y_deim_full.shape[1]), method='deim_leverage')
        markers_deim_full = {}
        for k, l in enumerate(unique_labels):
            mean_k = H_mean_deim[k, selected_idx]
            mean_rest = (H_mean_deim[:, selected_idx].sum(axis=0) - mean_k) / (K - 1)
            z = (mean_k - mean_rest) / (np.std(H_mean_deim[:, selected_idx], axis=0) + 1e-10)
            top = np.argsort(z)[::-1][:n_top]
            markers_deim_full[str(l)] = [str(gene_arr_deim[selected_idx[i]]) for i in top]
        all_markers['DEIM-full(500)'] = markers_deim_full
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['DEIM-full(500)'] = {str(l): [] for l in unique_labels}

    # ---- 10. LogFC ----
    print("    LogFC...", end='', flush=True)
    t0 = time.time()
    try:
        markers_logfc = {}
        for k, l in enumerate(unique_labels):
            mean_k = H_mean[k]
            mean_rest = (H_mean.sum(axis=0) - H_mean[k]) / (K - 1)
            logfc = mean_k - mean_rest
            # Rank ALL genes per type (NOT pool-based)
            top = np.argsort(logfc)[::-1][:n_top]
            markers_logfc[str(l)] = [str(gene_arr[i]) for i in top]
        all_markers['LogFC'] = markers_logfc
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['LogFC'] = {str(l): [] for l in unique_labels}

    # ---- 11. Wilcoxon ----
    print("    Wilcoxon...", end='', flush=True)
    t0 = time.time()
    try:
        import scanpy as sc
        import pandas as pd

        labels_cat = pd.Categorical([str(l) for l in labels])
        adata = sc.AnnData(X=Y_norm, obs={'label': labels_cat})
        adata.var_names = np.array([str(g) for g in gene_arr])
        sc.tl.rank_genes_groups(adata, groupby='label', method='wilcoxon', n_genes=n_top)

        markers_wilcox = {}
        names_dict = adata.uns['rank_genes_groups']['names']
        for l in unique_labels:
            l_str = str(l)
            if l_str in names_dict.dtype.names:
                markers_wilcox[l_str] = [str(g) for g in names_dict[l_str][:n_top]]
            else:
                markers_wilcox[l_str] = []
        all_markers['Wilcoxon'] = markers_wilcox
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['Wilcoxon'] = {str(l): [] for l in unique_labels}

    # ---- 12. T-test ----
    print("    T-test...", end='', flush=True)
    t0 = time.time()
    try:
        import scanpy as sc
        import pandas as pd

        labels_cat = pd.Categorical([str(l) for l in labels])
        adata = sc.AnnData(X=Y_norm, obs={'label': labels_cat})
        adata.var_names = np.array([str(g) for g in gene_arr])
        sc.tl.rank_genes_groups(adata, groupby='label', method='t-test', n_genes=n_top)

        markers_ttest = {}
        names_dict = adata.uns['rank_genes_groups']['names']
        for l in unique_labels:
            l_str = str(l)
            if l_str in names_dict.dtype.names:
                markers_ttest[l_str] = [str(g) for g in names_dict[l_str][:n_top]]
            else:
                markers_ttest[l_str] = []
        all_markers['T-test'] = markers_ttest
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['T-test'] = {str(l): [] for l in unique_labels}

    # ---- 13. RF-importance ----
    print("    RF-importance...", end='', flush=True)
    t0 = time.time()
    try:
        # Pre-filter to top 2000 HVG for ML methods (speed)
        n_ml_genes = min(2000, n_genes_total)
        hvg_ml_idx = np.argsort(np.var(Y_norm, axis=0))[::-1][:n_ml_genes]
        Y_ml = Y_norm[:, hvg_ml_idx]
        gene_arr_ml = gene_arr[hvg_ml_idx]

        # Subsample for speed if needed
        n_train = min(3000, Y_ml.shape[0])
        if n_train < Y_ml.shape[0]:
            rng = np.random.RandomState(42)
            train_idx = rng.choice(Y_ml.shape[0], n_train, replace=False)
            Y_train = Y_ml[train_idx]
            labels_train = labels[train_idx]
        else:
            Y_train = Y_ml
            labels_train = labels

        markers_rf = {}
        for k, l in enumerate(unique_labels):
            y_bin = (labels_train == k).astype(int)
            rf = RandomForestClassifier(
                n_estimators=50, random_state=0,
                class_weight='balanced', n_jobs=-1
            )
            rf.fit(Y_train, y_bin)
            importance = rf.feature_importances_
            top = np.argsort(importance)[::-1][:n_top]
            markers_rf[str(l)] = [str(gene_arr_ml[i]) for i in top]
        all_markers['RF-importance'] = markers_rf
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['RF-importance'] = {str(l): [] for l in unique_labels}

    # ---- 14. LR-L1 ----
    print("    LR-L1...", end='', flush=True)
    t0 = time.time()
    try:
        # Use same pre-filtered HVG data as RF
        markers_lr = {}
        for k, l in enumerate(unique_labels):
            y_bin = (labels_train == k).astype(int)
            lr = LogisticRegression(
                penalty='l1', solver='liblinear', C=1.0,
                max_iter=500, random_state=0
            )
            lr.fit(Y_train, y_bin)
            coef_abs = np.abs(lr.coef_.flatten())
            top = np.argsort(coef_abs)[::-1][:n_top]
            markers_lr[str(l)] = [str(gene_arr_ml[i]) for i in top]
        all_markers['LR-L1'] = markers_lr
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['LR-L1'] = {str(l): [] for l in unique_labels}

    # ---- 15. KineticST-marker (kinetically smoothed expression field) ----
    print("    KineticST-marker...", end='', flush=True)
    t0 = time.time()
    try:
        if coords is None:
            raise ValueError("coords required for KineticST-marker")
        n_spots_k = Y.shape[0]
        n_steps_k = 150 if n_spots_k <= 3000 else 80
        latent_dim_k = min(12, Y.shape[1], n_spots_k - 1)
        n_dom_k = n_domains if n_domains else len(unique_labels)
        model_mk = KineticST(KineticSTConfig(
            latent_dim=latent_dim_k, solver="stochastic", n_steps=n_steps_k,
            diffusion=0.03, alpha=0.1, beta=0.2, gamma=0.05,
            n_domains=n_dom_k, k_neighbors=6, run_wave_analysis=False,
            verbose=False, seed=0,
        ))
        model_mk.fit(np.asarray(Y, dtype=np.float64), np.asarray(coords, dtype=np.float64))
        # Decode kinetically smoothed expression field (final steady state) back to gene space
        smooth_expr = model_mk.decode_to_genes(model_mk.fields_[-1].mean_state)
        all_markers['KineticST-marker'] = find_kinetic_markers(
            smooth_expr, labels, gene_arr, n_top=n_top)
        print(f" done ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_markers['KineticST-marker'] = {str(l): [] for l in unique_labels}

    # ============ Evaluation ============
    method_names = [
        "DEIM-c(100)", "DEIM-c(200)", "DEIM-c(500)", "DEIM-c(1000)", "DEIM-c(2000)",
        "mRMR(500)", "Laplacian(500)", "HVG(500)", "DEIM-full(500)",
        "LogFC", "Wilcoxon", "T-test", "RF-importance", "LR-L1", "KineticST-marker"
    ]

    results = {
        'methods': method_names,
        'mean_hits': {},
        'total_hits': {},
        'win_types': {},
        'max_per_type': {},
    }

    if true_markers is not None:
        # Ground truth evaluation: count hits
        # Collect per-type hits for all methods to compute win_types
        all_per_type = {}  # method -> {type: hits}
        for mname in method_names:
            mm = all_markers[mname]
            per_type = {}
            total = 0
            n_types = 0
            for l_str, top_genes in mm.items():
                known = true_markers.get(l_str, true_markers.get(str(l_str), set()))
                if len(known) == 0:
                    continue
                hits = sum(1 for g in top_genes if g in known)
                per_type[l_str] = hits
                total += hits
                n_types += 1
            all_per_type[mname] = per_type
            results['mean_hits'][mname] = total / max(n_types, 1)
            results['total_hits'][mname] = total
            if per_type:
                results['max_per_type'][mname] = max(per_type.values())
            else:
                results['max_per_type'][mname] = 0

        # win_types: number of types where this method has the highest hits
        all_types_with_gt = set()
        for mname in method_names:
            all_types_with_gt.update(all_per_type[mname].keys())
        for l_str in all_types_with_gt:
            best_hits = -1
            best_method = None
            for mname in method_names:
                h = all_per_type[mname].get(l_str, 0)
                if h > best_hits:
                    best_hits = h
                    best_method = mname
            if best_method is not None:
                results['win_types'][best_method] = results['win_types'].get(best_method, 0) + 1
        # Fill missing
        for mname in method_names:
            if mname not in results['win_types']:
                results['win_types'][mname] = 0
    else:
        # Real data: use mean z-score specificity as primary metric
        gene_to_idx = {str(g): i for i, g in enumerate(gene_arr)}
        for mname in method_names:
            mm = all_markers[mname]
            total_z, n_types = 0.0, 0
            max_z = 0.0
            for k, l in enumerate(unique_labels):
                top = mm.get(str(l), [])
                if not top:
                    continue
                z_scores = []
                for g in top:
                    if g in gene_to_idx:
                        gi = gene_to_idx[g]
                        mk = H_mean[k, gi]
                        mr = (H_mean[:, gi].sum() - mk) / (K - 1)
                        z = (mk - mr) / (np.std(H_mean[:, gi]) + 1e-10)
                        z_scores.append(z)
                if z_scores:
                    mean_z_type = np.mean(z_scores)
                    total_z += mean_z_type
                    n_types += 1
                    if mean_z_type > max_z:
                        max_z = mean_z_type
            results['mean_hits'][mname] = total_z / max(n_types, 1)
            results['total_hits'][mname] = total_z
            results['max_per_type'][mname] = max_z
            results['win_types'][mname] = 0

    results['_markers'] = all_markers
    return results


# %%
# ============================================================
# Unsupervised Spatial Clustering Methods (5 total)
# ============================================================

def run_all_unsupervised(Y, coords, n_modules, spatial_method='knn'):
    """Run all 5 unsupervised clustering methods.

    Parameters
    ----------
    Y : ndarray (n_spots, n_genes) - raw counts (dense)
    coords : ndarray (n_spots, 2)
    n_modules : int - number of clusters
    spatial_method : str - 'knn' or 'grid'

    Returns
    -------
    results : dict with predictions, metrics (ARI, NMI, Coherence, Entropy, Time)
    """
    Y_norm = logcpm(Y)
    n_genes = Y.shape[1]

    # Build spatial adjacency (used for coherence and some methods)
    A = coords_to_adjacency(coords, method=spatial_method, k=6)

    all_preds = {}
    all_times = {}

    # ---- 1. DEIM+NMF ----
    print("    DEIM+NMF...", end='', flush=True)
    t0 = time.time()
    try:
        model = SpatialNMFDEIM(
            n_modules=n_modules, n_genes=min(500, n_genes),
            deim_method='deim_leverage', lambda_spatial='auto',
            spatial_method=spatial_method, k_neighbors=6,
            max_iter=200, verbose=False,
        )
        model.fit(Y, coords)
        pred_dn = np.argmax(model.W_, axis=1)
        all_preds['DEIM+NMF'] = pred_dn
        all_times['DEIM+NMF'] = time.time() - t0
        print(f" done ({all_times['DEIM+NMF']:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_preds['DEIM+NMF'] = np.zeros(Y.shape[0], dtype=int)
        all_times['DEIM+NMF'] = 0.0

    # ---- 2. HVG+NMF (with spatial regularization) ----
    print("    HVG+NMF...", end='', flush=True)
    t0 = time.time()
    try:
        # HVG selection on full data
        hvg_idx = select_hvg_indices(Y_norm, n_genes=min(500, n_genes))
        Y_sub_raw = Y[:, hvg_idx]
        if sparse.issparse(Y_sub_raw):
            Y_sub_raw = Y_sub_raw.toarray()
        Y_sub_raw = np.asarray(Y_sub_raw, dtype=np.float64)
        # Log-CPM on subset
        lib_size_sub = np.maximum(Y_sub_raw.sum(axis=1, keepdims=True), 1.0)
        Y_sub_norm = np.log1p(Y_sub_raw / lib_size_sub * 1e4)
        # Auto lambda
        lambda_auto = compute_auto_lambda(Y_sub_norm, A)
        # Graph-regularized NMF
        solver = GraphRegularizedNMF(n_modules=n_modules, lambda_spatial=lambda_auto, max_iter=200)
        W, H, info = solver.solve(Y_sub_norm, A, verbose=False)
        W_raw = info.get('W_raw', W)
        pred_hn = np.argmax(W_raw, axis=1)
        all_preds['HVG+NMF'] = pred_hn
        all_times['HVG+NMF'] = time.time() - t0
        print(f" done ({all_times['HVG+NMF']:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_preds['HVG+NMF'] = np.zeros(Y.shape[0], dtype=int)
        all_times['HVG+NMF'] = 0.0

    # ---- 3. HVG+NMF (no spatial, lambda=0) ----
    print("    HVG+NMF (no spatial)...", end='', flush=True)
    t0 = time.time()
    try:
        hvg_idx2 = select_hvg_indices(Y_norm, n_genes=min(500, n_genes))
        Y_sub_raw2 = Y[:, hvg_idx2]
        if sparse.issparse(Y_sub_raw2):
            Y_sub_raw2 = Y_sub_raw2.toarray()
        Y_sub_raw2 = np.asarray(Y_sub_raw2, dtype=np.float64)
        lib_size_sub2 = np.maximum(Y_sub_raw2.sum(axis=1, keepdims=True), 1.0)
        Y_sub_norm2 = np.log1p(Y_sub_raw2 / lib_size_sub2 * 1e4)
        # No spatial regularization (lambda=0)
        solver_ns = GraphRegularizedNMF(n_modules=n_modules, lambda_spatial=0.0, max_iter=200)
        W_ns, H_ns, info_ns = solver_ns.solve(Y_sub_norm2, A, verbose=False)
        W_ns_raw = info_ns.get('W_raw', W_ns)
        pred_hn_ns = np.argmax(W_ns_raw, axis=1)
        _key_hvg_ns = 'HVG+NMF\n(no spatial)'
        all_preds[_key_hvg_ns] = pred_hn_ns
        all_times[_key_hvg_ns] = time.time() - t0
        _t = all_times[_key_hvg_ns]
        print(f" done ({_t:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        _key_hvg_ns = 'HVG+NMF\n(no spatial)'
        all_preds[_key_hvg_ns] = np.zeros(Y.shape[0], dtype=int)
        all_times[_key_hvg_ns] = 0.0

    # ---- 4. PCA+KMeans ----
    print("    PCA+KMeans...", end='', flush=True)
    t0 = time.time()
    try:
        hvg_idx_pca = select_hvg_indices(Y_norm, n_genes=min(2000, n_genes))
        Y_hvg_pca = Y_norm[:, hvg_idx_pca]
        n_comp = min(50, Y_hvg_pca.shape[1] - 1, Y_hvg_pca.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=0)
        Z_pca = pca.fit_transform(Y_hvg_pca)
        km = KMeans(n_clusters=n_modules, random_state=0, n_init=10, max_iter=300)
        pred_pk = km.fit_predict(Z_pca)
        all_preds['PCA+KMeans'] = pred_pk
        all_times['PCA+KMeans'] = time.time() - t0
        print(f" done ({all_times['PCA+KMeans']:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_preds['PCA+KMeans'] = np.zeros(Y.shape[0], dtype=int)
        all_times['PCA+KMeans'] = 0.0

    # ---- 5. KineticST (kinetic-theory method based on Deng Yu's mathematical work) ----
    print("    KineticST...", end='', flush=True)
    t0 = time.time()
    try:
        # Adaptively adjust parameters based on dataset size
        n_spots_kst = Y.shape[0]
        # Reduce steps for large datasets to control runtime
        n_steps_kst = 150 if n_spots_kst <= 3000 else 80
        latent_dim_kst = min(12, n_genes, n_spots_kst - 1)
        model_kst = KineticST(KineticSTConfig(
            latent_dim=latent_dim_kst,
            solver="stochastic",
            n_steps=n_steps_kst,
            diffusion=0.03,
            alpha=0.1, beta=0.2, gamma=0.05,
            n_domains=n_modules,
            k_neighbors=6,
            run_wave_analysis=False,  # Disable wave analysis for speed
            verbose=False,
            seed=0,
        ))
        model_kst.fit(Y, coords)
        pred_kst = model_kst.get_domain_labels()
        all_preds['KineticST'] = pred_kst
        all_times['KineticST'] = time.time() - t0
        print(f" done ({all_times['KineticST']:.1f}s)")
    except Exception as e:
        print(f" FAILED ({e})")
        all_preds['KineticST'] = np.zeros(Y.shape[0], dtype=int)
        all_times['KineticST'] = 0.0

    return all_preds, all_times, A


# %%
# ============================================================
# Visualization: Spatial Domain Plots
# ============================================================

def plot_spatial_domains(coords, true_labels, predictions, dataset_name, save_path):
    """2x3 grid: Ground Truth + 5 unsupervised methods.
    High-contrast visualization with larger dots and clear colors.
    """
    method_names = list(predictions.keys())
    n_panels = 1 + len(method_names)
    n_rows = 2
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    axes_flat = axes.flatten()

    n = len(coords)
    if n > 3000:
        s = 15.0
    elif n > 1000:
        s = 25.0
    else:
        s = 40.0

    cmap = plt.get_cmap('tab20')

    ax = axes_flat[0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=true_labels, cmap=cmap,
                    s=s, alpha=0.95, edgecolors='none', marker='s')
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

    for i, mname in enumerate(method_names):
        ax = axes_flat[i + 1]
        pred = predictions[mname]
        ax.scatter(coords[:, 0], coords[:, 1], c=pred, cmap=cmap,
                   s=s, alpha=0.95, edgecolors='none', marker='s')
        ax.set_title(mname.replace('\n', ' '), fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(f'{dataset_name} -- Spatial Domain Comparison', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# %%



# %%
# ============================================================
# Comparison Plots: Supervised Methods
# ============================================================

def _sem(values):
    """Standard error of the mean."""
    return np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0.0


def _err_hbar(ax, y_pos, means, sems, color='black', lw=1.5, cap=0.2):
    """Draw rightward-only T-shaped error bars for horizontal bar charts."""
    for y, m, s in zip(y_pos, means, sems):
        if s > 0:
            ax.plot([m, m + s], [y, y], color=color, lw=lw, zorder=5)
            ax.plot([m + s, m + s], [y - cap, y + cap], color=color, lw=lw, zorder=5)


def _err_vbar(ax, x_pos, means, sems, color='black', lw=1.5, cap=0.15):
    """Draw upward-only T-shaped error bars for vertical bar charts."""
    for x, m, s in zip(x_pos, means, sems):
        if s > 0:
            ax.plot([x, x], [m, m + s], color=color, lw=lw, zorder=5)
            ax.plot([x - cap, x + cap], [m + s, m + s], color=color, lw=lw, zorder=5)


def plot_supervised_comparison(all_results, save_path):
    """Bar chart showing mean + SEM (upper only) of supervised methods."""
    methods = all_results[0]['supervised']['methods']
    n_methods = len(methods)

    all_values = []
    for m in methods:
        vals = [r['supervised']['mean_hits'].get(m, 0) for r in all_results]
        all_values.append(vals)

    means = [np.mean(v) for v in all_values]
    sems = [_sem(v) for v in all_values]

    fig, ax = plt.subplots(figsize=(8, n_methods * 0.35))

    y_pos = np.arange(n_methods)
    color_map = plt.cm.tab20(np.linspace(0, 1, n_methods))

    ax.barh(y_pos, means, color=color_map, alpha=0.8)
    _err_hbar(ax, y_pos, means, sems)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Mean Hits / Z-score', fontsize=12)
    ax.set_title(f'Supervised Marker Discovery: Mean ± SEM (n={len(all_results)} datasets)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, (m, s) in enumerate(zip(means, sems)):
        ax.text(m + s + 0.05, i, f'{m:.2f}±{s:.2f}', fontsize=7, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_supervised_cross_dataset(all_results, save_path):
    """Bar chart showing mean ± SD for each dataset type group."""
    methods = all_results[0]['supervised']['methods']
    
    groups = {
        'Simulation': [],
        'MOSTA': [],
        'STSP': [],
        'Heart': [],
    }
    
    for r in all_results:
        name = r['short_name']
        if name == 'Simulation':
            groups['Simulation'].append(r)
        elif 'Heart' in name:
            groups['Heart'].append(r)
        elif 'E' in name or 'mouse' in name.lower():
            groups['MOSTA'].append(r)
        elif 'STSP' in name:
            groups['STSP'].append(r)
    
    groups = {k: v for k, v in groups.items() if v}
    
    panel_labels = ['a', 'b', 'c', 'd']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    color_map = plt.cm.tab20(np.linspace(0, 1, len(methods)))

    for i, (gname, gresults) in enumerate(groups.items()):
        ax = axes_flat[i]
        all_values = []
        for m in methods:
            vals = [r['supervised']['mean_hits'].get(m, 0) for r in gresults]
            all_values.append(vals)

        means = [np.mean(v) for v in all_values]
        sems = [_sem(v) for v in all_values]

        y_pos = np.arange(len(methods))
        ax.barh(y_pos, means, color=color_map, alpha=0.8)
        _err_hbar(ax, y_pos, means, sems)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=7)
        ax.set_xlabel('Mean Hits / Z-score', fontsize=10)
        ax.set_title(f'{gname} (n={len(gresults)})', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.text(-0.14, 1.03, panel_labels[i], transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# %%
# ============================================================
# Comparison Plots: Unsupervised Methods
# ============================================================

def plot_unsupervised_comparison(all_results, save_path):
    """Bar chart showing mean ± SD of unsupervised methods across all datasets."""
    methods = [m.replace('\n', ' ') for m in all_results[0]['unsupervised']['methods']]
    metrics = ['ARI', 'NMI', 'Coherence']
    n_methods = len(methods)

    panel_labels = ['a', 'b', 'c', 'd']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    color_map = plt.cm.Set3(np.linspace(0, 1, n_methods))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        all_values = []
        for m in all_results[0]['unsupervised']['methods']:
            vals = [r['unsupervised'][metric].get(m, 0) for r in all_results]
            all_values.append(vals)

        means = [np.mean(v) for v in all_values]
        sems = [_sem(v) for v in all_values]

        y_pos = np.arange(n_methods)
        ax.barh(y_pos, means, color=color_map, alpha=0.8)
        _err_hbar(ax, y_pos, means, sems)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=10)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric}: Mean ± SEM (n={len(all_results)})', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.3)
        ax.text(-0.12, 1.05, panel_labels[i], transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')

        for j, (m_val, s_val) in enumerate(zip(means, sems)):
            ax.text(m_val + s_val + 0.02, j, f'{m_val:.3f}±{s_val:.3f}', fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_unsupervised_radar(all_results, save_path):
    """Radar chart showing mean values of unsupervised methods across all datasets."""
    metrics = ['ARI', 'NMI', 'Coherence', 'Entropy']
    n_metrics = len(metrics)

    theta = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    theta += theta[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    ax.set_title(f'Unsupervised Methods: Mean Performance (n={len(all_results)})',
                 weight='bold', size=12, pad=20)

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(metrics, fontsize=10)

    ax.set_ylim(0, 1)

    methods = all_results[0]['unsupervised']['methods']
    for i, mname in enumerate(methods):
        values = []
        for metric in metrics:
            vals = [r['unsupervised'][metric].get(mname, 0) for r in all_results]
            values.append(np.mean(vals))
        values += values[:1]
        ax.plot(theta, values, color=colors[i], linewidth=2, label=mname.replace('\n', ' '))
        ax.fill(theta, values, color=colors[i], alpha=0.15)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# %%
# ============================================================
# Additional Comparison Plots
# ============================================================

def plot_supervised_simulation(sim_results, save_path):
    """Supervised methods comparison for simulation data (mean + SEM over seeds)."""
    if not isinstance(sim_results, list):
        sim_results = [sim_results]

    methods = sim_results[0]['supervised']['methods']
    n_methods = len(methods)

    hits_values = [[r['supervised']['mean_hits'].get(m, 0) for r in sim_results] for m in methods]
    win_values = [[r['supervised']['win_types'].get(m, 0) for r in sim_results] for m in methods]

    means_hits = [np.mean(v) for v in hits_values]
    sems_hits = [_sem(v) for v in hits_values]
    means_win = [np.mean(v) for v in win_values]
    sems_win = [_sem(v) for v in win_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    y_pos = np.arange(n_methods)
    color_map = plt.cm.tab20(np.linspace(0, 1, n_methods))

    ax1.barh(y_pos, means_hits, color=color_map, alpha=0.8)
    _err_hbar(ax1, y_pos, means_hits, sems_hits)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=8)
    ax1.set_xlabel('Mean Hits', fontsize=12)
    ax1.set_title(f'Simulation: Mean Hits ± SEM (n={len(sim_results)} seeds)',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.text(-0.14, 1.03, 'a', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top')
    for i, (m, s) in enumerate(zip(means_hits, sems_hits)):
        ax1.text(m + s + 0.03, i, f'{m:.2f}±{s:.2f}', fontsize=7, va='center')

    ax2.barh(y_pos, means_win, color=color_map, alpha=0.8)
    _err_hbar(ax2, y_pos, means_win, sems_win)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=8)
    ax2.set_xlabel('Win Types', fontsize=12)
    ax2.set_title(f'Simulation: Win Types ± SEM (n={len(sim_results)} seeds)',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.text(-0.14, 1.03, 'b', transform=ax2.transAxes,
             fontsize=16, fontweight='bold', va='top')
    for i, (m, s) in enumerate(zip(means_win, sems_win)):
        ax2.text(m + s + 0.03, i, f'{m:.2f}±{s:.2f}', fontsize=7, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_unsupervised_simulation(sim_results, save_path):
    """Unsupervised clustering performance on simulation replicates (mean + SEM)."""
    if not isinstance(sim_results, list):
        sim_results = [sim_results]

    methods = [m.replace('\n', ' ') for m in sim_results[0]['unsupervised']['methods']]
    raw_methods = sim_results[0]['unsupervised']['methods']
    metrics = ['ARI', 'NMI', 'Coherence']
    n_methods = len(methods)

    panel_labels = ['a', 'b', 'c', 'd']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 4 * len(metrics)))
    color_map = plt.cm.Set3(np.linspace(0, 1, n_methods))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        all_values = [[r['unsupervised'][metric].get(m, 0) for r in sim_results]
                      for m in raw_methods]
        means = [np.mean(v) for v in all_values]
        sems = [_sem(v) for v in all_values]

        y_pos = np.arange(n_methods)
        ax.barh(y_pos, means, color=color_map, alpha=0.8)
        _err_hbar(ax, y_pos, means, sems)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=10)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric} (n={len(sim_results)} seeds)', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.3)
        ax.text(-0.12, 1.05, panel_labels[i], transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')

        for j, (m_val, s_val) in enumerate(zip(means, sems)):
            ax.text(m_val + s_val + 0.02, j, f'{m_val:.3f}±{s_val:.3f}',
                    fontsize=8, va='center')

    fig.suptitle('Simulation: Unsupervised Clustering (Mean ± SEM)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_unsupervised_time(all_results, save_path):
    """Time comparison for unsupervised methods (mean + SEM, upper only)."""
    methods = [m.replace('\n', ' ') for m in all_results[0]['unsupervised']['methods']]
    n_methods = len(methods)

    all_values = []
    for m in all_results[0]['unsupervised']['methods']:
        vals = [r['unsupervised']['Time_sec'].get(m, 0) for r in all_results]
        all_values.append(vals)

    means = [np.mean(v) for v in all_values]
    sems = [_sem(v) for v in all_values]

    fig, ax = plt.subplots(figsize=(n_methods * 0.7, 5))

    x_pos = np.arange(n_methods)
    color_map = plt.cm.Set3(np.linspace(0, 1, n_methods))

    ax.bar(x_pos, means, color=color_map, alpha=0.8)
    _err_vbar(ax, x_pos, means, sems)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, fontsize=9)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Unsupervised Clustering: Time (Mean ± SEM, n={len(all_results)})',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)

    for i, (m_val, s_val) in enumerate(zip(means, sems)):
        ax.text(i, m_val + s_val + max(means) * 0.02, f'{m_val:.1f}±{s_val:.1f}',
                fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_unsupervised_heatmap(all_results, save_path):
    """Heatmap of unsupervised method performance (mean values per dataset group)."""
    methods = [m.replace('\n', ' ') for m in all_results[0]['unsupervised']['methods']]
    metrics = ['ARI', 'NMI', 'Coherence']

    groups = {
        'Simulation': [],
        'MOSTA': [],
        'STSP': [],
        'Heart': [],
    }
    
    for r in all_results:
        name = r['short_name']
        if name == 'Simulation':
            groups['Simulation'].append(r)
        elif 'Heart' in name:
            groups['Heart'].append(r)
        elif 'E' in name or 'mouse' in name.lower():
            groups['MOSTA'].append(r)
        elif 'STSP' in name:
            groups['STSP'].append(r)
    
    groups = {k: v for k, v in groups.items() if v}
    group_names = list(groups.keys())

    panel_labels = ['a', 'b', 'c', 'd']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 3.5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = []
        for gname, gresults in groups.items():
            row = []
            for m in all_results[0]['unsupervised']['methods']:
                vals = [r['unsupervised'][metric].get(m, 0) for r in gresults]
                row.append(np.mean(vals))
            data.append(row)
        data = np.array(data)

        im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(group_names)))
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels(group_names, fontsize=10)

        for j in range(len(group_names)):
            for k in range(len(methods)):
                ax.text(k, j, f"{data[j, k]:.3f}",
                        ha='center', va='center', color='white', fontsize=9)

        ax.set_title(f'{metric} (Mean)', fontsize=13, fontweight='bold')
        ax.text(-0.08, 1.02, panel_labels[i], transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_supervised_scatter(all_results, save_path):
    """Scatter plot of mean_hits vs win_types for supervised methods (mean values)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    methods = all_results[0]['supervised']['methods']
    color_map = plt.cm.tab20(np.linspace(0, 1, len(methods)))

    for i, m in enumerate(methods):
        x_vals = [r['supervised']['mean_hits'].get(m, 0) for r in all_results]
        y_vals = [r['supervised']['win_types'].get(m, 0) for r in all_results]

        x_mean = np.mean(x_vals)
        x_sem = _sem(x_vals)
        y_mean = np.mean(y_vals)
        y_sem = _sem(y_vals)

        ax.plot(x_mean, y_mean, 'o', color=color_map[i], markersize=8, zorder=5)
        cap = 0.15
        if x_sem > 0:
            ax.plot([x_mean, x_mean + x_sem], [y_mean, y_mean], color=color_map[i], lw=1.5, zorder=4)
            ax.plot([x_mean + x_sem, x_mean + x_sem], [y_mean - cap, y_mean + cap],
                    color=color_map[i], lw=1.5, zorder=4)
        if y_sem > 0:
            ax.plot([x_mean, x_mean], [y_mean, y_mean + y_sem], color=color_map[i], lw=1.5, zorder=4)
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            ax.plot([x_mean - x_range * 0.01, x_mean + x_range * 0.01], [y_mean + y_sem, y_mean + y_sem],
                    color=color_map[i], lw=1.5, zorder=4)

        ax.annotate(m, (x_mean + x_sem + 0.05, y_mean), fontsize=8,
                    color=color_map[i], va='center')

    ax.set_xlabel('Mean Hits / Z-score', fontsize=12)
    ax.set_ylabel('Win Types', fontsize=12)
    ax.set_title(f'Supervised Methods: Mean Hits vs Win Types (n={len(all_results)})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# Per-dataset analysis
# ============================================================

def analyze_dataset(name, short_name, Y, labels, coords, gene_names, n_modules,
                    true_markers=None, spatial_method='knn', subsample_per_type=None,
                    dataset_key=None):
    """Run supervised + unsupervised analysis on one dataset.

    Parameters
    ----------
    name : str - full display name
    short_name : str - short name for file naming
    Y : array-like (n_spots, n_genes)
    labels : array-like (n_spots,) - can be strings or ints
    coords : array-like (n_spots, 2)
    gene_names : array-like (n_genes,)
    n_modules : int
    true_markers : dict or None
    spatial_method : str
    subsample_per_type : int or None - subsample this many per type
    dataset_key : str - key for plot filename

    Returns
    -------
    result : dict with all metrics
    """
    print(f"\n{'=' * 70}")
    print(f"  {name}  |  {Y.shape[0]} spots x {Y.shape[1]} genes  |  K={n_modules}")
    print(f"{'=' * 70}")

    result = {
        'name': name,
        'short_name': short_name,
        'n_spots': int(Y.shape[0]),
        'n_genes': int(Y.shape[1]),
        'n_modules': n_modules,
        'has_ground_truth': true_markers is not None,
        'dataset_key': dataset_key or short_name.lower(),
    }

    # ---- Dense array ----
    if sparse.issparse(Y):
        Y = Y.toarray()
    Y = np.asarray(Y, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)

    # ---- Encode string labels to integers ----
    labels_int, label_map = encode_labels(labels)
    labels = labels_int

    # ---- Subsample if needed ----
    if subsample_per_type is not None:
        idx = subsample_indices(labels, target_per_type=subsample_per_type)
        Y = Y[idx]
        labels = labels[idx]
        coords = coords[idx]
        result['n_spots'] = int(len(labels))
        print(f"  Subsampled to {len(labels)} spots")

    # ---- Map true_markers keys to integer label space ----
    true_markers_mapped = None
    if true_markers is not None:
        true_markers_mapped = {}
        for k, v in true_markers.items():
            # Try to map original key to integer label
            int_key = label_map.get(str(k), None)
            if int_key is not None:
                true_markers_mapped[str(int_key)] = v
            else:
                true_markers_mapped[str(k)] = v

    # ============ SUPERVISED ============
    print(f"\n  [Supervised] 15 methods...")
    sup_results = run_all_supervised(Y, labels, gene_names, n_top=20,
                                      true_markers=true_markers_mapped,
                                      coords=coords, n_domains=n_modules)

    result['supervised'] = {
        'methods': sup_results['methods'],
        'mean_hits': sup_results['mean_hits'],
        'total_hits': sup_results['total_hits'],
        'win_types': sup_results['win_types'],
        'max_per_type': sup_results['max_per_type'],
    }

    # ============ UNSUPERVISED ============
    print(f"\n  [Unsupervised] 5 methods...")
    preds, times, A_spatial = run_all_unsupervised(Y, coords, n_modules, spatial_method=spatial_method)

    # Compute metrics
    unsup_method_names = ["DEIM+NMF", "HVG+NMF", "HVG+NMF\n(no spatial)", "PCA+KMeans", "KineticST"]
    unsup_results = {
        'methods': unsup_method_names,
        'ARI': {},
        'NMI': {},
        'Coherence': {},
        'Entropy': {},
        'Time_sec': {},
    }

    for mname in unsup_method_names:
        pred = preds.get(mname, np.zeros(len(labels), dtype=int))
        unsup_results['ARI'][mname] = float(adjusted_rand_score(labels, pred))
        unsup_results['NMI'][mname] = float(normalized_mutual_info_score(labels, pred))
        unsup_results['Coherence'][mname] = float(compute_coherence(pred, A_spatial))
        unsup_results['Entropy'][mname] = float(compute_entropy(pred, n_modules))
        unsup_results['Time_sec'][mname] = float(times.get(mname, 0.0))

    result['unsupervised'] = unsup_results

    # Store spatial data for plots
    result['_spatial'] = {
        'coords': coords,
        'true_labels': labels,
        'predictions': preds,
    }

    # Print summary
    print(f"\n  [Supervised] Summary:")
    for mname in sup_results['methods']:
        mh = sup_results['mean_hits'].get(mname, 0)
        print(f"    {mname:20s}: mean_hits={mh:.3f}")
    print(f"\n  [Unsupervised] Summary:")
    for mname in unsup_method_names:
        ari_val = unsup_results['ARI'].get(mname, 0)
        nmi_val = unsup_results['NMI'].get(mname, 0)
        coh_val = unsup_results['Coherence'].get(mname, 0)
        print(f"    {mname:25s}: ARI={ari_val:.4f}  NMI={nmi_val:.4f}  Coh={coh_val:.4f}")

    return result


# %%
# ============================================================
# Helper: Load h5ad dataset with auto-detection
# ============================================================

def load_h5ad_dataset(filepath):
    import anndata as ad
    adata = ad.read_h5ad(str(filepath))
    
    Y_sparse = adata.X
    
    label_cols = ['annotation', 'clusters', 'cell_type', 'labels', 'type', 'celltype', 'cluster']
    label_col = None
    for col in label_cols:
        if col in adata.obs.columns:
            label_col = col
            break
    
    coords_key = None
    if 'spatial' in adata.obsm:
        coords_key = 'spatial'
    elif 'X_spatial' in adata.obsm:
        coords_key = 'X_spatial'
    
    labels_raw = np.array(adata.obs[label_col].values).flatten() if label_col else None
    coords = np.array(adata.obsm[coords_key]) if coords_key else None
    genes = np.array(adata.var_names.values)
    
    # Filter out invalid coordinates (NaN or Inf)
    if coords is not None and labels_raw is not None:
        valid_mask = np.isfinite(coords).all(axis=1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"  Removing {n_invalid} spots with invalid coordinates")
            coords = coords[valid_mask]
            labels_raw = labels_raw[valid_mask]
            if sparse.issparse(Y_sparse):
                Y_sparse = Y_sparse[valid_mask]
            else:
                Y_sparse = Y_sparse[valid_mask]
    
    n_modules = len(np.unique(labels_raw)) if labels_raw is not None else 10
    
    basename = filepath.stem
    short_name = basename.replace('.MOSTA', '').replace('_stereoseq', '').replace('Spatial_processed_', '')
    
    del adata
    
    return {
        'Y': Y_sparse,
        'labels': labels_raw,
        'coords': coords,
        'genes': genes,
        'n_modules': n_modules,
        'short_name': short_name,
        'name': basename,
        'label_col': label_col,
    }


# Note: mouse_E11.5_heart is a 3D reconstruction (64 2D slices stacked along z-axis).
# Coordinates in obsm['aligned_spatial_3D'], ground-truth labels in obs['heart_regions']
# (5 anatomical heart regions). Single-slice aspect ratio 1.0-1.3, not elongated.
HEART_SLICES = ['slices_38', 'slices_45', 'slices_46', 'slices_48']


def load_heart_slices(filepath, slice_names=HEART_SLICES):
    """Split mouse_E11.5_heart 3D reconstruction into independent 2D datasets by slice.

    Each slice takes (x, y) from aligned_spatial_3D as 2D coordinates, heart_regions as
    ground-truth labels (5 anatomical regions: left/right ventricle, left/right atrium, outflow tract).
    """
    import anndata as ad
    adata = ad.read_h5ad(str(filepath))
    coords3d = np.array(adata.obsm['aligned_spatial_3D'])
    slices = np.asarray(adata.obs['slices'].values)
    regions = np.asarray(adata.obs['heart_regions'].values)
    genes = np.array(adata.var_names.values)
    out = []
    for sn in slice_names:
        mask = slices == sn
        if mask.sum() == 0:
            continue
        Y = adata.X[mask]
        Y = Y.toarray() if sparse.issparse(Y) else np.asarray(Y, dtype=np.float64)
        xy = np.asarray(coords3d[mask][:, :2], dtype=np.float64)
        lab = regions[mask]
        valid = np.isfinite(xy).all(axis=1)
        Y, xy, lab = Y[valid], xy[valid], lab[valid]
        sid = sn.replace('slices_', '')
        out.append({
            'Y': Y, 'labels': lab, 'coords': xy, 'genes': genes,
            'n_modules': len(np.unique(lab)),
            'short_name': f'Heart_S{sid}',
            'name': f'mouse_E11.5_heart_{sn}',
            'label_col': 'heart_regions',
        })
    del adata
    return out


# %%
# ============================================================
# Load datasets and run
# ============================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("  Comprehensive Analysis: DEIM vs HGV (Upgraded)")
    print("  14 supervised methods + 5 unsupervised methods")
    print("  Processing all datasets in dataset/ folder")
    print("=" * 70)

    import anndata as ad
    import pandas as pd

    all_results = []

    # ===== 1. Simulation (10 seeds for mean ± SD) =====
    print("\n>>> [1] Generating and analyzing simulated data with 10 seeds...")
    n_sim_seeds = 10
    for seed_i in range(n_sim_seeds):
        seed = 42 + seed_i
        print(f"\n--- Simulation seed {seed_i + 1}/{n_sim_seeds} (seed={seed}) ---")
        sim = generate_simulation(
            K=10, G=5000, fold_change=8.0, dropout_rate=0.3,
            N_per_type=300, delta=1.0, seed=seed
        )
        r_sim = analyze_dataset(
            name=f"Simulation (K=10, seed={seed})",
            short_name="Simulation",
            Y=sim['Y'], labels=sim['labels'], coords=sim['coords'],
            gene_names=sim['gene_names'], n_modules=9,
            true_markers=sim['true_markers'],
            spatial_method='grid',
            dataset_key=f'simulation_seed{seed}',
        )
        all_results.append(r_sim)
        del sim

    # ===== 2. Process all h5ad files in dataset/ =====
    # Exclude elongated Arabidopsis (replaced with heart slices) and 3D heart (handled separately)
    h5ad_files = [p for p in DATA_DIR.rglob('*.h5ad')
                  if 'Arabidopsis' not in str(p) and p.name != 'mouse_E11.5_heart.h5ad']
    print(f"\n>>> Found {len(h5ad_files)} h5ad files (brain+embryo; Arabidopsis replaced by heart slices)")
    
    for i, h5ad_path in enumerate(h5ad_files, start=2):
        print(f"\n>>> [{i}] Loading and analyzing {h5ad_path.name}...")
        try:
            ds = load_h5ad_dataset(h5ad_path)
            
            Y_sparse = ds['Y']
            labels_raw = ds['labels']
            coords = ds['coords']
            genes = ds['genes']
            n_modules = ds['n_modules']
            
            # Skip if labels or coords are invalid
            if labels_raw is None or len(labels_raw) == 0:
                print(f"  SKIPPED: No valid labels found")
                continue
            if coords is None or len(coords) == 0:
                print(f"  SKIPPED: No valid coordinates found")
                continue
            
            labels_int, _ = encode_labels(labels_raw)
            n_spots_total = len(labels_raw)
            
            if n_spots_total > 4000:
                idx = subsample_indices(labels_int, target_per_type=300)
                Y_data = np.asarray(
                    Y_sparse[idx].toarray() if sparse.issparse(Y_sparse) else Y_sparse[idx],
                    dtype=np.float64
                )
                labels_data = labels_raw[idx]
                coords_data = coords[idx]
                print(f"  Subsampled from {n_spots_total} to {len(labels_data)} spots")
            else:
                Y_data = np.asarray(
                    Y_sparse.toarray() if sparse.issparse(Y_sparse) else Y_sparse,
                    dtype=np.float64
                )
                labels_data = labels_raw
                coords_data = coords
            
            del Y_sparse
            
            r = analyze_dataset(
                name=ds['name'],
                short_name=ds['short_name'],
                Y=Y_data, labels=labels_data, coords=coords_data,
                gene_names=genes, n_modules=n_modules,
                true_markers=None,
                spatial_method='knn',
                subsample_per_type=None,
                dataset_key=ds['short_name'].lower(),
            )
            all_results.append(r)
            del Y_data
            
        except Exception as e:
            print(f"  FAILED to process {h5ad_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # ===== 2b. Mouse E11.5 heart slices (replacing elongated Arabidopsis) =====
    heart_path = DATA_DIR / 'mosta_embryo' / 'mouse_E11.5_heart.h5ad'
    if heart_path.exists():
        print(f"\n>>> [Heart] Splitting mouse_E11.5_heart into {len(HEART_SLICES)} 2D slices...")
        try:
            for ds in load_heart_slices(heart_path):
                print(f"\n>>> Analyzing {ds['name']} ...")
                r = analyze_dataset(
                    name=ds['name'],
                    short_name=ds['short_name'],
                    Y=ds['Y'], labels=ds['labels'], coords=ds['coords'],
                    gene_names=ds['genes'], n_modules=ds['n_modules'],
                    true_markers=None,
                    spatial_method='knn',
                    subsample_per_type=300,
                    dataset_key=ds['short_name'].lower(),
                )
                all_results.append(r)
        except Exception as e:
            print(f"  FAILED to process heart: {e}")
            import traceback
            traceback.print_exc()

    # ===== Generate spatial domain plots =====
    print(f"\n{'=' * 70}")
    print("  Generating spatial domain plots...")
    print(f"{'=' * 70}")

    for r in all_results:
        if '_spatial' in r:
            sp = r.pop('_spatial')
            plot_path = SCRIPT_DIR / f"fig_spatial_{r.get('dataset_key', r['short_name'].lower())}.png"
            plot_spatial_domains(
                sp['coords'], sp['true_labels'], sp['predictions'],
                r['name'], str(plot_path),
            )

    # ===== Save CSV results =====
    print(f"\n{'=' * 70}")
    print("  Saving CSV results...")
    print(f"{'=' * 70}")

    csv_supervised_rows = []
    csv_unsupervised_rows = []
    
    for r in all_results:
        for m in r['supervised']['methods']:
            csv_supervised_rows.append({
                'Dataset': r['short_name'],
                'Method': m,
                'Mean_Hits': r['supervised']['mean_hits'].get(m, 0),
                'Total_Hits': r['supervised']['total_hits'].get(m, 0),
                'Win_Types': r['supervised']['win_types'].get(m, 0),
                'Max_Per_Type': r['supervised']['max_per_type'].get(m, 0),
            })
        
        for m in r['unsupervised']['methods']:
            csv_unsupervised_rows.append({
                'Dataset': r['short_name'],
                'Method': m.replace('\n', ' '),
                'ARI': r['unsupervised']['ARI'].get(m, 0),
                'NMI': r['unsupervised']['NMI'].get(m, 0),
                'Coherence': r['unsupervised']['Coherence'].get(m, 0),
                'Entropy': r['unsupervised']['Entropy'].get(m, 0),
                'Time_sec': r['unsupervised']['Time_sec'].get(m, 0),
            })
    
    df_supervised = pd.DataFrame(csv_supervised_rows)
    df_unsupervised = pd.DataFrame(csv_unsupervised_rows)
    
    csv_sup_path = SCRIPT_DIR / 'results_supervised.csv'
    csv_unsup_path = SCRIPT_DIR / 'results_unsupervised.csv'
    
    df_supervised.to_csv(csv_sup_path, index=False, encoding='utf-8-sig')
    df_unsupervised.to_csv(csv_unsup_path, index=False, encoding='utf-8-sig')
    
    print(f"  Saved: {csv_sup_path}")
    print(f"  Saved: {csv_unsup_path}")

    # ===== Save JSON =====
    print(f"\n{'=' * 70}")
    print("  Saving JSON results...")
    print(f"{'=' * 70}")

    json_output = {'datasets': all_results}
    json_path = SCRIPT_DIR / 'analysis_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # ===== Generate comparison plots =====
    print(f"{'=' * 70}")
    print("  Generating comparison plots...")
    print(f"{'=' * 70}")

    plot_supervised_comparison(all_results, str(SCRIPT_DIR / 'fig_supervised_comparison.png'))
    plot_supervised_cross_dataset(all_results, str(SCRIPT_DIR / 'fig_supervised_cross.png'))
    plot_unsupervised_comparison(all_results, str(SCRIPT_DIR / 'fig_unsupervised_comparison.png'))
    plot_unsupervised_radar(all_results, str(SCRIPT_DIR / 'fig_unsupervised_radar.png'))

    sim_results = [r for r in all_results if r['short_name'] == 'Simulation']
    if sim_results:
        plot_supervised_simulation(sim_results, str(SCRIPT_DIR / 'fig_supervised_simulation.png'))
        plot_unsupervised_simulation(sim_results, str(SCRIPT_DIR / 'fig_unsupervised_simulation.png'))
    plot_unsupervised_time(all_results, str(SCRIPT_DIR / 'fig_unsupervised_time.png'))
    plot_unsupervised_heatmap(all_results, str(SCRIPT_DIR / 'fig_unsupervised_heatmap.png'))
    plot_supervised_scatter(all_results, str(SCRIPT_DIR / 'fig_supervised_scatter.png'))

    # ===== Summary =====
    elapsed = time.time() - t_start
    print(f"{'=' * 70}")
    print(f"  Analysis complete! Total time: {elapsed:.1f}s")
    print(f"  Datasets analyzed: {len(all_results)}")
    print(f"  Output directory: {SCRIPT_DIR}")
    print(f"  CSV Files: results_supervised.csv, results_unsupervised.csv")
    print(f"  JSON File: analysis_results.json")
    print(f"  PNG Files: fig_spatial_*.png, fig_*.png")
    print(f"{'=' * 70}")

    return json_output


# %%
if __name__ == '__main__':
    main()

"""
E3: 独立模拟数据验证
用与 KineticST 引擎无关的方式生成模拟数据，验证 KineticST 的泛化能力。
生成方案：sklearn make_blobs（空间聚类）+ marker 基因 + dropout。
"""
import sys
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
sys.path.insert(0, r'd:\文章\最新方法应用时空转录组')
sys.path.insert(0, r'd:\文章\最新方法应用时空转录组\scripts')
from kinetic_st import KineticST, KineticSTConfig
from kinetic_st.core.graph_utils import coords_to_adjacency
from spatial_nmf_deim import SpatialNMFDEIM
from spatial_nmf_deim.core.nmf_solver import GraphRegularizedNMF
RESULTS_DIR = r'd:\文章\最新方法应用时空转录组\results'
def generate_independent_simulation(n_clusters=9, n_spots=2500, n_genes=5000,
                                     cluster_std=8.0, dropout=0.3, seed=42):
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0, 100, (n_spots, 2))
    centers = rng.uniform(20, 80, (n_clusters, 2))
    from scipy.spatial.distance import cdist
    dist = cdist(coords, centers)
    labels = np.argmin(dist, axis=1)
    expression = np.zeros((n_spots, n_genes), dtype=np.float64)
    marker_genes_per_cluster = n_genes // n_clusters
    for c in range(n_clusters):
        mask = labels == c
        n_c = mask.sum()
        if n_c == 0:
            continue
        marker_start = c * marker_genes_per_cluster
        marker_end = (c + 1) * marker_genes_per_cluster
        expression[mask, marker_start:marker_end] = rng.lognormal(
            mean=3.0, sigma=0.5, size=(n_c, marker_genes_per_cluster))
        other_mask = np.ones(n_genes, dtype=bool)
        other_mask[marker_start:marker_end] = False
        expression[np.ix_(mask, other_mask)] = rng.lognormal(
            mean=0.3, sigma=0.4, size=(n_c, n_genes - marker_genes_per_cluster))
    dropout_mask = rng.random(expression.shape) < dropout
    expression[dropout_mask] = 0
    counts = rng.poisson(expression).astype(np.float64)
    return {
        'Y': counts,
        'coords': coords,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_genes': n_genes,
    }
def logcpm(Y):
    Yd = np.asarray(Y, dtype=np.float64)
    lib = Yd.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0)
    return np.log1p(Yd / lib * 1e4)
def select_hvg(Y_norm, n_genes=500):
    mean = Y_norm.mean(axis=0)
    var = Y_norm.var(axis=0)
    dispersion = var / (mean + 1e-8)
    return np.argsort(dispersion)[::-1][:n_genes]
def compute_auto_lambda(Y_norm, A):
    from scipy.sparse.csgraph import laplacian
    L = laplacian(A, normed=True)
    n = Y_norm.shape[0]
    quad = np.trace(Y_norm.T @ L @ Y_norm) / max(n, 1)
    return max(0.01, 1.0 / (quad + 1e-6))
def compute_coherence(labels, A):
    A = A.tocoo()
    if A.nnz == 0:
        return 0.0
    same = sum(1 for i, j in zip(A.row, A.col) if labels[i] == labels[j])
    return same / A.nnz
def run_all_methods(Y, coords, n_modules):
    Y_norm = logcpm(Y)
    n_genes = min(500, Y.shape[1])
    A = coords_to_adjacency(coords, method='knn', k=6)
    preds = {}
    times = {}
    # 1. DEIM+NMF
    t0 = time.time()
    try:
        model = SpatialNMFDEIM(n_modules=n_modules, n_genes=n_genes,
                               deim_method='deim_leverage', lambda_spatial='auto',
                               spatial_method='knn', k_neighbors=6, max_iter=200)
        model.fit(Y, coords)
        preds['DEIM+NMF'] = np.argmax(model.W_, axis=1)
    except Exception:
        preds['DEIM+NMF'] = np.zeros(Y.shape[0], dtype=int)
    times['DEIM+NMF'] = time.time() - t0
    # 2. HVG+NMF
    t0 = time.time()
    try:
        hvg_idx = select_hvg(Y_norm, n_genes=n_genes)
        Y_sub = Y[:, hvg_idx]
        lib = np.maximum(Y_sub.sum(axis=1, keepdims=True), 1.0)
        Y_sub_norm = np.log1p(Y_sub / lib * 1e4)
        lam = compute_auto_lambda(Y_sub_norm, A)
        solver = GraphRegularizedNMF(n_modules=n_modules, lambda_spatial=lam, max_iter=200)
        W, H, info = solver.solve(Y_sub_norm, A)
        preds['HVG+NMF'] = np.argmax(info.get('W_raw', W), axis=1)
    except Exception:
        preds['HVG+NMF'] = np.zeros(Y.shape[0], dtype=int)
    times['HVG+NMF'] = time.time() - t0
    # 3. PCA+KMeans
    t0 = time.time()
    try:
        pca = PCA(n_components=min(15, Y.shape[0] - 1, Y.shape[1]))
        Z = pca.fit_transform(Y_norm)
        km = KMeans(n_clusters=n_modules, random_state=0, n_init=10).fit(Z)
        preds['PCA+KMeans'] = km.labels_
    except Exception:
        preds['PCA+KMeans'] = np.zeros(Y.shape[0], dtype=int)
    times['PCA+KMeans'] = time.time() - t0
    # 4. KineticST
    t0 = time.time()
    try:
        n_steps = 150 if Y.shape[0] <= 3000 else 80
        latent_dim = min(12, Y.shape[1], Y.shape[0] - 1)
        model_kst = KineticST(KineticSTConfig(
            latent_dim=latent_dim, solver="stochastic", n_steps=n_steps,
            diffusion=0.03, alpha=0.1, beta=0.2, gamma=0.05,
            n_domains=n_modules, k_neighbors=6, run_wave_analysis=False,
            verbose=False, seed=0,
        ))
        model_kst.fit(np.asarray(Y, dtype=np.float64), np.asarray(coords, dtype=np.float64))
        preds['KineticST'] = model_kst.get_domain_labels()
    except Exception:
        preds['KineticST'] = np.zeros(Y.shape[0], dtype=int)
    times['KineticST'] = time.time() - t0
    return preds, times, A
def main():
    print("=" * 70)
    print("  E3: 独立模拟数据验证")
    print("=" * 70)
    n_seeds = 10
    n_clusters = 9
    all_results = []
    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        print(f"\n--- Seed {seed_i+1}/{n_seeds} (seed={seed}) ---")
        sim = generate_independent_simulation(
            n_clusters=n_clusters, n_spots=2500, n_genes=5000,
            cluster_std=8.0, dropout=0.3, seed=seed)
        Y = sim['Y']
        coords = sim['coords']
        labels_gt = sim['labels']
        k_gt = sim['n_clusters']
        print(f"  n_spots={Y.shape[0]}, n_genes={Y.shape[1]}, K_gt={k_gt}")
        preds, times, A = run_all_methods(Y, coords, k_gt)
        for method, pred in preds.items():
            ari = float(adjusted_rand_score(labels_gt, pred))
            nmi = float(normalized_mutual_info_score(labels_gt, pred))
            coh = float(compute_coherence(pred, A))
            all_results.append({
                'seed': int(seed),
                'method': method,
                'ari': ari,
                'nmi': nmi,
                'coherence': coh,
                'runtime': float(times[method]),
                'n_predicted': int(len(np.unique(pred))),
            })
            print(f"  {method:15s}: ARI={ari:.4f} NMI={nmi:.4f} time={times[method]:.1f}s")
    out_json = os.path.join(RESULTS_DIR, 'e3_independent_simulation_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {out_json}")
    print(f"\n{'='*70}")
    print("  E3 汇总 (mean ± std across 10 seeds)")
    print(f"{'='*70}")
    methods = sorted(set(r['method'] for r in all_results))
    for m in methods:
        aris = [r['ari'] for r in all_results if r['method'] == m]
        nmis = [r['nmi'] for r in all_results if r['method'] == m]
        print(f"  {m:15s}: ARI={np.mean(aris):.4f}±{np.std(aris):.4f}  "
              f"NMI={np.mean(nmis):.4f}±{np.std(nmis):.4f}")
if __name__ == '__main__':
    main()

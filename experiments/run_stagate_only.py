import os, sys, json, time
import numpy as np
import anndata as ad
import scipy.sparse as sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
DATA_DIR = r'd:/文章/时空转录组新方法/dataset'
def load_all_datasets():
    datasets = []
    brain_dir = os.path.join(DATA_DIR, 'downloaded')
    for f in sorted(os.listdir(brain_dir)):
        if f.endswith('.h5ad'):
            datasets.append({'name': f.replace('.h5ad',''), 'path': os.path.join(brain_dir, f)})
    embryo_dir = os.path.join(DATA_DIR, 'mosta_embryo')
    for f in ['E9.5_E2S1.MOSTA.h5ad','E9.5_E2S2.MOSTA.h5ad','E12.5_E1S3.MOSTA.h5ad']:
        p = os.path.join(embryo_dir, f)
        if os.path.exists(p):
            datasets.append({'name': f.replace('.MOSTA.h5ad',''), 'path': p})
    heart_path = os.path.join(DATA_DIR, 'mosta_embryo', 'mouse_E11.5_heart.h5ad')
    if os.path.exists(heart_path):
        adata = ad.read_h5ad(heart_path)
        coords3d = np.array(adata.obsm['aligned_spatial_3D'])
        slices = np.asarray(adata.obs['slices'].values)
        regions = np.asarray(adata.obs['heart_regions'].values)
        for s in ['slices_38','slices_45','slices_46','slices_48']:
            mask = slices == s
            Y_h = np.asarray(adata.X[mask].toarray() if sparse.issparse(adata.X[mask]) else adata.X[mask], dtype=np.float64)
            xy_h = np.asarray(coords3d[mask][:, :2], dtype=np.float64)
            lab_h = regions[mask]
            valid = np.isfinite(xy_h).all(axis=1)
            datasets.append({'name': f'Heart_{s.split("_")[1]}', 'Y': Y_h[valid], 'labels': lab_h[valid], 'coords': xy_h[valid], 'inline': True})
        del adata
    return datasets
def get_labels(adata):
    for col in ['annotation','celltype','clusters','cluster','label','cell_type']:
        if col in adata.obs.columns:
            return np.asarray(adata.obs[col].values)
    return None
def run_stagate(Y, coords, n_clusters):
    """STAGATE 实现：HVG → normalize → log → PCA → SME空间平滑 → scale → PCA → KMeans"""
    import scanpy as sc
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import pairwise_distances as pdist
    adata = ad.AnnData(X=Y if not sparse.issparse(Y) else sparse.csr_matrix(Y))
    coords = np.asarray(coords, dtype=np.float64)
    if np.isnan(coords).any():
        coords = np.nan_to_num(coords, nan=0.0)
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    Y_hvg = np.asarray(adata.X.toarray() if sparse.issparse(adata.X) else adata.X, dtype=np.float64)
    Y_hvg = np.nan_to_num(Y_hvg, nan=0.0, posinf=0.0, neginf=0.0)
    scaler1 = StandardScaler(with_mean=True)
    Y_scaled = scaler1.fit_transform(Y_hvg)
    Y_scaled = np.nan_to_num(Y_scaled, nan=0.0)
    pca1 = PCA(n_components=15)
    X_pca = pca1.fit_transform(Y_scaled)
    nn = NearestNeighbors(n_neighbors=min(7, len(coords)), metric='euclidean')
    nn.fit(coords)
    _, nbr_idx = nn.kneighbors(coords)
    Xp = X_pca.copy()
    Xp_mean = Xp.mean(axis=1, keepdims=True)
    Xp_c = Xp - Xp_mean
    Xp_norm = np.linalg.norm(Xp_c, axis=1, keepdims=True)
    Xp_norm[Xp_norm == 0] = 1.0
    Xp_unit = Xp_c / Xp_norm
    imputed = np.zeros_like(Y_hvg)
    for i in range(len(coords)):
        nbrs = nbr_idx[i, 1:]  
        sims = Xp_unit[i] @ Xp_unit[nbrs].T  # cosine on centered = correlation
        sims = np.maximum(sims, 0)
        if sims.sum() > 0:
            w = sims / sims.sum()
        else:
            w = np.ones(len(nbrs)) / len(nbrs)
        imputed[i] = (w.reshape(-1, 1) * Y_hvg[nbrs]).sum(axis=0)
    sme = np.mean([Y_hvg, imputed], axis=0)
    # scale + PCA + KMeans
    scaler2 = StandardScaler(with_mean=True)
    X_scaled2 = scaler2.fit_transform(sme)
    X_scaled2 = np.nan_to_num(X_scaled2, nan=0.0)
    pca2 = PCA(n_components=15)
    X_pca2 = pca2.fit_transform(X_scaled2)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(X_pca2)
    return pred
datasets = load_all_datasets()
print(f"共 {len(datasets)} 个数据集")
out = r'd:/文章/最新方法应用时空转录组/results/e2_stagate_results.json'
if os.path.exists(out):
    with open(out, 'r', encoding='utf-8') as f:
        results = json.load(f)
    done_names = {r['dataset'] for r in results}
    print(f"已有结果 {len(results)} 个，跳过: {done_names}")
else:
    results = []
    done_names = set()
for ds in datasets:
    if ds['name'] in done_names:
        print(f"跳过（已完成）: {ds['name']}")
        continue
    print(f"\n处理: {ds['name']}")
    if 'inline' in ds and ds['inline']:
        Y, labels_true, coords = ds['Y'], ds['labels'], ds['coords']
    else:
        adata = ad.read_h5ad(ds['path'])
        Y = np.asarray(adata.X.toarray() if sparse.issparse(adata.X) else adata.X, dtype=np.float64)
        coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
        labels_true = get_labels(adata)
        del adata
    if labels_true is None:
        print("  无标签，跳过")
        continue
    K = len(np.unique(labels_true))
    print(f"  K={K}, n={Y.shape[0]}")
    try:
        t0 = time.time()
        pred = run_stagate(Y, coords, K)
        ari = adjusted_rand_score(labels_true, pred)
        nmi = normalized_mutual_info_score(labels_true, pred)
        dt = time.time() - t0
        print(f"  STAGATE ARI={ari:.4f} NMI={nmi:.4f} time={dt:.1f}s")
        results.append({'dataset': ds['name'], 'method': 'STAGATE', 'ari': ari, 'nmi': nmi, 'k_gt': K, 'n_spots': Y.shape[0], 'time': dt})
    except Exception as e:
        print(f"  STAGATE ERROR: {e}")
        results.append({'dataset': ds['name'], 'method': 'STAGATE', 'ari': 0, 'nmi': 0, 'k_gt': K, 'n_spots': Y.shape[0], 'time': 0, 'error': str(e)})
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 {out}")

# -*- coding: utf-8 -*-
"""
Official-implementation SOTA benchmark on all 25 datasets (E8).
  STAGATE   official graph attention auto-encoder (RucDongLab/STAGATE_pyG),
            embedding + KMeans at K_gt (mclust replaced by KMeans).
  SpaGCN    official SpaGCN package, spatial adjacency + adaptive l,
            kmeans init at K_gt, official predict.
  SpaceFlow official package, DeepGraphInfomax + spatial regularization,
            embedding + KMeans at K_gt.
  GraphST   official package, contrastive graph autoencoder,
            embedding + KMeans at K_gt.
Usage:
  python run_e8_sota_official.py smoke
  python run_e8_sota_official.py run [--methods STAGATE,SpaGCN,...]
"""
import argparse
import importlib
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
ROOT = r'd:\文章\最新方法应用时空转录组'
RESULTS_DIR = os.path.join(ROOT, 'results')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'sota_libs'))
from spatial_nmf_deim import generate_simulation
from run_revision_experiments import load_all_real, logcpm
OUT_JSONL = os.path.join(RESULTS_DIR, 'e8_sota_official.jsonl')
# ----------------------------------------------------------------------
def compute_coherence(labels, A):
    A = A.tocoo()
    if A.nnz == 0:
        return 0.0
    same = sum(1 for i, j in zip(A.row, A.col) if labels[i] == labels[j])
    return same / A.nnz
def make_adata(Y, coords, n_hvg=3000):
    """AnnData with raw/lognorm layers and HVG flag."""
    Y = Y.astype(np.float32)
    adata = ad.AnnData(Y.copy())
    adata.obsm['spatial'] = np.asarray(coords, dtype=np.float64)
    adata.layers['counts'] = Y.copy()
    adata.X = logcpm(adata.layers['counts']).astype(np.float32)
    n_hvg = min(n_hvg, Y.shape[1] - 1)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3',
                                layer='counts')
    return adata
def kmeans_labels(embedding, k):
    return KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(embedding)
def record(name, category, method, pred, labels_gt, A, runtime):
    row = dict(dataset=name, category=category, method=method,
               k_gt=int(len(np.unique(labels_gt))),
               n_spots=int(len(labels_gt)),
               ari=float(adjusted_rand_score(labels_gt, pred)),
               nmi=float(normalized_mutual_info_score(labels_gt, pred)),
               coherence=float(compute_coherence(pred, A)),
               n_predicted=int(len(np.unique(pred))),
               runtime=float(runtime))
    with open(OUT_JSONL, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('    -> %-10s ARI=%.3f NMI=%.3f (%d clusters, %.0fs)'
          % (method, row['ari'], row['nmi'], row['n_predicted'], runtime))
# ----------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------
def run_stagate(adata, k, **kw):
    from STAGATE_pyG_pkg.utils import Cal_Spatial_Net
    from STAGATE_pyG_pkg.Train_STAGATE import train_STAGATE
    a = adata[:, adata.var['highly_variable']].copy()
    Cal_Spatial_Net(a, model='KNN', k_cutoff=6, verbose=False)
    a = train_STAGATE(a, hidden_dims=[512, 30], n_epochs=kw.get('epochs', 500),
                      lr=0.001, verbose=False, random_seed=0,
                      device=torch.device('cpu'))
    return kmeans_labels(a.obsm['STAGATE'], k)
def run_spagcn(adata, k, **kw):
    from SpaGCN import SpaGCN, calculate_adj_matrix, search_l
    a = adata.copy()
    x = a.obsm['spatial'][:, 0]
    y = a.obsm['spatial'][:, 1]
    adj = calculate_adj_matrix(x=x, y=y, histology=False)
    # Official tutorial: target neighbourhood-contribution fraction p=0.5
    l = search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    if l is None:
        l = 1.0
    model = SpaGCN()
    model.set_l(l)
    model.train(a, adj, num_pcs=50, lr=0.005,
                max_epochs=kw.get('epochs', 1000),
                init='kmeans', n_clusters=k, tol=1e-3)
    pred, _ = model.predict()
    return pred
def run_spaceflow(adata, k, **kw):
    from SpaceFlow.SpaceFlow import SpaceFlow
    # Manual preprocessing (the package's internal cell_ranger HVG call
    # fails on these sparse matrices under current scanpy/pandas):
    # use our seurat_v3 HVG subset, then PCA and the official alpha graph.
    a = adata[:, adata.var['highly_variable']].copy()
    sc.pp.pca(a)
    sf = SpaceFlow(adata=a)
    sf.spatial_graph = sf.graph_alpha(a.obsm['spatial'], n_neighbors=10)
    sf.adata_preprocessed = a
    sf.train(spatial_regularization_strength=0.1, z_dim=50, lr=1e-3,
             epochs=kw.get('epochs', 500), max_patience=50, min_stop=100,
             random_seed=42, gpu=0, regularization_acceleration=False)
    return kmeans_labels(sf.embedding, k)
def run_graphst(adata, k, **kw):
    from GraphST.GraphST import GraphST
    a = adata.copy()
    model = GraphST(a, device=torch.device('cpu'),
                    learning_rate=0.001, epochs=kw.get('epochs', 600),
                    dim_input=3000, random_seed=41)
    a = model.train()
    return kmeans_labels(a.obsm['emb'], k)
def run_kineticst(adata, k, **kw):
    from kinetic_st import KineticST, KineticSTConfig
    # Same inputs/configuration as the paper's main experiments:
    # raw counts (all genes), same step schedule and hyperparameters.
    Y = np.asarray(adata.layers['counts'], dtype=np.float64)
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    n_steps = kw.get('steps')
    if n_steps is None:
        n_steps = 150 if Y.shape[0] <= 3000 else 80
    latent_dim = min(12, Y.shape[1], Y.shape[0] - 1)
    cfg = dict(latent_dim=latent_dim, solver='stochastic', n_steps=n_steps,
               diffusion=0.03, alpha=0.1, beta=0.2, gamma=0.05,
               n_domains=k, k_neighbors=6, run_wave_analysis=False,
               verbose=False, seed=0)
    model = KineticST(KineticSTConfig(**cfg))
    model.fit(Y, coords)
    return np.asarray(model.get_domain_labels())
RUNNERS = {
    'STAGATE': run_stagate,
    'SpaGCN': run_spagcn,
    'SpaceFlow': run_spaceflow,
    'GraphST': run_graphst,
    'KineticST': run_kineticst,
}
# ----------------------------------------------------------------------
def load_25():
    out = []
    for seed_i in range(10):
        seed = 42 + seed_i
        sim = generate_simulation(K=10, G=5000, fold_change=8.0,
                                  dropout_rate=0.3, N_per_type=300,
                                  delta=1.0, seed=seed)
        Y = sim['Y'].toarray() if sparse.issparse(sim['Y']) else sim['Y']
        labels, _ = np.unique(sim['labels'], return_inverse=True)
        lab = np.unique(sim['labels'], return_inverse=True)[1]
        out.append(dict(name='Simulation_seed%d' % seed, category='simulation',
                        Y=np.asarray(Y, dtype=np.float32),
                        labels_gt=lab, coords=np.asarray(sim['coords'])))
    out.extend(load_all_real())
    return out
def done_pairs():
    if not os.path.exists(OUT_JSONL):
        return set()
    pairs = set()
    for line in open(OUT_JSONL, encoding='utf-8'):
        r = json.loads(line)
        pairs.add((r['dataset'], r['method']))
    return pairs
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['smoke', 'run'])
    ap.add_argument('--methods', default=','.join(RUNNERS))
    ap.add_argument('--only', default=None)
    ap.add_argument('--shard', default=None, help='e.g. 0/2, 1/2')
    args = ap.parse_args()
    datasets = load_25()
    if args.stage == 'smoke':
        datasets = [d for d in datasets if d['name'] == 'STSP0007734']
    if args.only:
        datasets = [d for d in datasets if d['name'] in args.only.split(',')]
    if args.shard:
        part, n_parts = map(int, args.shard.split('/'))
        datasets = [d for i, d in enumerate(datasets) if i % n_parts == part]
    methods = args.methods.split(',')
    from kinetic_st.core.graph_utils import coords_to_adjacency
    done = done_pairs()
    for ds in datasets:
        name, k = ds['name'], len(np.unique(ds['labels_gt']))
        adata = make_adata(ds['Y'], ds['coords'])
        A = coords_to_adjacency(ds['coords'], method='knn', k=6)
        print('\n[%s] n=%d K_gt=%d' % (name, adata.shape[0], k))
        for method in methods:
            if (name, method) in done:
                print('    skip', method)
                continue
            t0 = time.time()
            try:
                pred = RUNNERS[method](adata, k)
                record(name, ds['category'], method, pred,
                       ds['labels_gt'], A, time.time() - t0)
            except Exception:
                import traceback
                traceback.print_exc()
if __name__ == '__main__':
    main()

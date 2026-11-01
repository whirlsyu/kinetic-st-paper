# -*- coding: utf-8 -*-
"""Export the 25 datasets for BayesSpace: MatrixMarket logCPM + metadata CSV."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import sparse
from scipy.io import mmwrite
import scanpy as sc
import anndata as ad
ROOT = r'd:\文章\最新方法应用时空转录组'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from run_e8_sota_official import load_25
from run_revision_experiments import logcpm
# ASCII path: current scipy fast_matrix_market C++ writer silently
# fails on non-ASCII (Chinese) paths on Windows.
OUT = r'D:\kst_bayesspace_data'
os.makedirs(OUT, exist_ok=True)
manifest = []
for ds in load_25():
    name = ds['name']
    Y = np.asarray(ds['Y'], dtype=np.float64)
    X = logcpm(Y)
    # HVG subset on raw values for tractable matrices in R
    a = ad.AnnData(Y.astype(np.float32))
    a.layers['counts'] = Y.astype(np.float32)
    n_hvg = min(3000, Y.shape[1] - 1)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor='seurat_v3',
                                layer='counts')
    sel = np.asarray(a.var['highly_variable'])
    Xs = sparse.csr_matrix(X[:, sel])
    safe = name.replace('.', '_')
    base = os.path.join(OUT, safe)
    mmwrite(base + '_logcounts.mtx', Xs.T)  # genes x spots for R
    np.savetxt(base + '_coords.csv', ds['coords'], delimiter=',')
    np.savetxt(base + '_labels.csv', ds['labels_gt'], fmt='%d', delimiter=',')
    k = int(len(np.unique(ds['labels_gt'])))
    manifest.append(dict(file=safe, name=name, category=ds['category'],
                         k_gt=k, n_spots=int(Y.shape[0]),
                         n_genes=int(sel.sum())))
    print(name, Xs.shape, 'K=%d' % k)
import json
with open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print('Exported %d datasets to %s' % (len(manifest), OUT))

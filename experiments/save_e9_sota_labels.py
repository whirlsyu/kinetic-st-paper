# -*- coding: utf-8 -*-
"""Re-run the six benchmarked methods on the representative datasets
shown in the paper's spatial-map panels and SAVE the predicted labels,
so that SOTA domain maps can be drawn without re-running.
Default datasets (match Figs. 2D, 3C, 3D, 4C, 4D):
    Simulation_seed42, STSP0007728, STSP0007732, E9.5_E2S1, Heart_S45
For every dataset a single CSV is written to results/e9_sota_labels/:
    x, y, gt, KineticST, STAGATE, SpaGCN, SpaceFlow, GraphST, BayesSpace
plus a small JSON with per-method ARI (for map annotation).
The five Python methods reuse the exact E8 runners (same protocol,
K = K_gt, same seeds). BayesSpace labels are merged from the official
MCMC predictions saved during E8 (D:\\kst_bayesspace_data) -- no re-run
needed; if a dataset's BayesSpace file is missing the column is skipped
with a warning.
Resume-safe: existing per-method columns are kept and not recomputed.
Usage:
    python save_e9_sota_labels.py                      # all defaults
    python save_e9_sota_labels.py --datasets STSP0007732,Heart_S45
    python save_e9_sota_labels.py --methods KineticST,STAGATE
"""
import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
ROOT = r'd:\文章\最新方法应用时空转录组'
EXP_DIR = os.path.join(ROOT, 'scripts', 'experiments')
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'sota_libs'))
from run_e8_sota_official import RUNNERS, make_adata, load_25  # noqa: E402
OUT_DIR = os.path.join(ROOT, 'results', 'e9_sota_labels')
BS_DATA = r'D:\kst_bayesspace_data'
DEFAULT_DATASETS = ['Simulation_seed42', 'STSP0007728', 'STSP0007732',
                    'E9.5_E2S1', 'Heart_S45']
PY_METHODS = ['KineticST', 'STAGATE', 'SpaGCN', 'SpaceFlow', 'GraphST']
def load_bayesspace_labels(name):
    """Return the saved BayesSpace MCMC labels for dataset `name`, or None."""
    manifest = pd.read_json(os.path.join(BS_DATA, 'manifest.json'))
    hit = manifest[manifest.name == name]
    if not len(hit):
        return None
    pred_p = os.path.join(BS_DATA, hit.iloc[0]['file'] + '_bayesspace_pred.csv')
    if not os.path.exists(pred_p):
        return None
    return pd.read_csv(pred_p).iloc[:, 0].values
def save_dataset(ds, methods):
    from sklearn.metrics import adjusted_rand_score
    name = ds['name']
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_p = os.path.join(OUT_DIR, name + '.csv')
    json_p = os.path.join(OUT_DIR, name + '_ari.json')
    coords = np.asarray(ds['coords'])
    gt = np.asarray(ds['labels_gt'])
    if os.path.exists(csv_p):
        out = pd.read_csv(csv_p)
    else:
        out = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'gt': gt})
    k = len(np.unique(gt))
    adata = None  # built lazily on first python-method run
    ari = json.load(open(json_p)) if os.path.exists(json_p) else {}
    for method in methods:
        if method in out.columns:
            print('    skip %s (already saved)' % method)
            continue
        if method == 'BayesSpace':
            pred = load_bayesspace_labels(name)
            if pred is None or len(pred) != len(gt):
                print('    WARN: no BayesSpace labels for %s' % name)
                continue
        else:
            if adata is None:
                adata = make_adata(ds['Y'], coords)
            t0 = time.time()
            pred = np.asarray(RUNNERS[method](adata, k))
            print('    -> %-10s done in %.0fs' % (method, time.time() - t0))
        out[method] = pred
        ari[method] = float(adjusted_rand_score(gt, pred))
        out.to_csv(csv_p, index=False)
        with open(json_p, 'w') as f:
            json.dump(ari, f, indent=2)
        print('    saved %s column (ARI=%.3f)' % (method, ari[method]))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DEFAULT_DATASETS))
    ap.add_argument('--methods',
                    default=','.join(PY_METHODS + ['BayesSpace']))
    args = ap.parse_args()
    wanted = args.datasets.split(',')
    methods = args.methods.split(',')
    datasets = {d['name']: d for d in load_25()}
    for name in wanted:
        if name not in datasets:
            print('UNKNOWN dataset: %s' % name)
            continue
        print('\n[%s]' % name)
        save_dataset(datasets[name], methods)
    print('\nAll done. Labels in %s' % OUT_DIR)
if __name__ == '__main__':
    main()

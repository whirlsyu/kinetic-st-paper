# -*- coding: utf-8 -*-
"""Combine E8 results: 4 (5 incl. KineticST) Python methods + BayesSpace."""
import json
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT = r'd:\文章\最新方法应用时空转录组'
DATA = r'D:\kst_bayesspace_data'
rows = []
jsonl = os.path.join(ROOT, 'results', 'e8_sota_official.jsonl')
if os.path.exists(jsonl):
    for line in open(jsonl, encoding='utf-8'):
        rows.append(json.loads(line))
manifest = pd.read_json(os.path.join(DATA, 'manifest.json'))
bs_csv = os.path.join(ROOT, 'results', 'e8_bayesspace_results.csv')
if os.path.exists(bs_csv):
    bs = pd.read_csv(bs_csv)
    for _, m in manifest.iterrows():
        pred_p = os.path.join(DATA, m['file'] + '_bayesspace_pred.csv')
        gt_p = os.path.join(DATA, m['file'] + '_labels.csv')
        if not os.path.exists(pred_p):
            continue
        pred = pd.read_csv(pred_p).iloc[:, 0].values
        gt = pd.read_csv(gt_p, header=None).iloc[:, 0].values
        rt = bs.loc[bs['dataset'] == m['name'], 'runtime']
        rows.append(dict(
            dataset=m['name'], category=m['category'], method='BayesSpace',
            k_gt=int(m['k_gt']), n_spots=int(m['n_spots']),
            ari=float(adjusted_rand_score(gt, pred)),
            nmi=float(normalized_mutual_info_score(gt, pred)),
            n_predicted=int(len(np.unique(pred))),
            runtime=float(rt.iloc[0]) if len(rt) else np.nan))
df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=['dataset', 'method'], keep='last')
methods = ['KineticST', 'STAGATE', 'SpaGCN', 'SpaceFlow', 'GraphST',
           'BayesSpace']
methods = [m for m in methods if m in set(df['method'])]
print('records:', len(df), '| datasets per method:')
print(df.groupby('method')['dataset'].nunique().reindex(methods))
cats = ['overall', 'simulation', 'brain', 'embryo', 'heart']
summary = {}
for cat in cats:
    sub = df if cat == 'overall' else df[df['category'] == cat]
    tbl = []
    for m in methods:
        v = sub[sub['method'] == m]
        if len(v) == 0:
            continue
        tbl.append(dict(method=m, n=len(v),
                        ari_mean=v['ari'].mean(), ari_sd=v['ari'].std(),
                        nmi_mean=v['nmi'].mean(),
                        runtime_mean=v['runtime'].mean()))
    summary[cat] = tbl
    print('\n=== %s (n=%d datasets) ===' % (cat,
          sub['dataset'].nunique()))
    print(pd.DataFrame(tbl).to_string(index=False, float_format=lambda x:
          '%.3f' % x))
# pairwise wins on ARI (matched datasets)
wide = df.pivot_table(index='dataset', columns='method', values='ari')
win_tbl = []
for m in methods:
    wins = 0
    for other in methods:
        if other == m:
            continue
        both = wide[[m, other]].dropna()
        wins += int((both[m] > both[other]).sum())
    win_tbl.append(dict(method=m, pairwise_wins=wins))
print('\n=== pairwise ARI wins ===')
print(pd.DataFrame(win_tbl).to_string(index=False))
out = os.path.join(ROOT, 'results', 'e8_combined_summary.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(dict(per_record=df.to_dict('records'), by_category=summary,
                   pairwise_wins=win_tbl), f, indent=2, ensure_ascii=False)
df.to_csv(os.path.join(ROOT, 'results', 'e8_combined_table.csv'), index=False)
print('\nSaved:', out)

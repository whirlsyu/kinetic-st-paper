import json
import numpy as np
from collections import defaultdict
with open(r'd:/文章/最新方法应用时空转录组/results/e2_sota_comparison_results.json', 'r', encoding='utf-8') as f:
    e2_results = json.load(f)
with open(r'd:/文章/最新方法应用时空转录组/results/e2_stagate_results.json', 'r', encoding='utf-8') as f:
    stagate_results = json.load(f)
e2_merged = [r for r in e2_results if not (r['method'] == 'STAGATE')]
e2_merged.extend(stagate_results)
with open(r'd:/文章/最新方法应用时空转录组/results/e2_sota_comparison_results.json', 'w', encoding='utf-8') as f:
    json.dump(e2_merged, f, indent=2, ensure_ascii=False)
datasets = sorted(set(r['dataset'] for r in e2_merged))
methods = sorted(set(r['method'] for r in e2_merged))
by_ds = defaultdict(dict)
for r in e2_merged:
    by_ds[r['dataset']][r['method']] = r
print(f"数据集: {len(datasets)}, 方法: {methods}")
print()
print(f"{'='*110}")
print("E2 完整结果（含 STAGATE）")
print(f"{'='*110}")
print(f"{'数据集':<30} {'K':>4} {'n':>6} {'KineticST':>11} {'STAGATE':>10} {'PCA+KM':>10} {'DEIM+NMF':>10}")
print("-" * 110)
for ds in datasets:
    info = by_ds[ds]
    k_gt = info.get('KineticST', {}).get('k_gt', '?')
    n = info.get('KineticST', {}).get('n_spots', '?')
    vals = []
    for m in ['KineticST', 'STAGATE', 'PCA+KMeans', 'DEIM+NMF']:
        vals.append(info.get(m, {}).get('ari', 0))
    print(f"{ds:<30} {k_gt:>4} {n:>6} {vals[0]:>11.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")
print(f"\n{'='*110}")
print("方法汇总")
print(f"{'='*110}")
print(f"{'方法':<20} {'平均ARI':>10} {'±std':>10} {'中位数':>10} {'脑(n=8)':>10} {'胚胎(n=3)':>10} {'心脏(n=4)':>10}")
print("-" * 110)
brain_ds = [d for d in datasets if 'Spatial' in d]
embryo_ds = [d for d in datasets if d.startswith('E')]
heart_ds = [d for d in datasets if 'Heart' in d]
for m in ['KineticST', 'STAGATE', 'PCA+KMeans', 'DEIM+NMF']:
    all_ari = [by_ds[ds].get(m, {}).get('ari', 0) for ds in datasets]
    brain_ari = [by_ds[ds].get(m, {}).get('ari', 0) for ds in brain_ds]
    embryo_ari = [by_ds[ds].get(m, {}).get('ari', 0) for ds in embryo_ds]
    heart_ari = [by_ds[ds].get(m, {}).get('ari', 0) for ds in heart_ds]
    print(f"{m:<20} {np.mean(all_ari):>10.4f} {np.std(all_ari):>10.4f} {np.median(all_ari):>10.4f} "
          f"{np.mean(brain_ari):>10.4f} {np.mean(embryo_ari):>10.4f} {np.mean(heart_ari):>10.4f}")
print(f"\n{'='*60}")
print("KineticST vs 各方法 胜负统计")
print(f"{'='*60}")
for competitor in ['STAGATE', 'PCA+KMeans', 'DEIM+NMF']:
    wins = sum(1 for ds in datasets if by_ds[ds].get('KineticST', {}).get('ari', 0) > by_ds[ds].get(competitor, {}).get('ari', 0))
    print(f"  KineticST vs {competitor}: {wins}胜/{len(datasets)-wins}负")

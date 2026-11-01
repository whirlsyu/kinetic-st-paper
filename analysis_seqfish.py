# -*- coding: utf-8 -*-
"""
seqFISH+ 补充分析 (Eng et al., Nature 2019):
  - seqfishplus_OB.h5ad        2050 cells x 10000 genes, louvain K=25
  - seqfishplus_cortex_svz.h5ad  913 cells x 10000 genes, louvain K=26
使用与原文相同的分析流程 (comprehensive_analysis.analyze_dataset)。
输出: 合并写入 scripts/seqfish_results.json (不覆盖 new_platforms_results.json)
Usage: python analysis_seqfish.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
import anndata as ad
from comprehensive_analysis import analyze_dataset
DATA_DIR = Path(r'd:\文章\时空转录组新方法\dataset')
SF_DIR = DATA_DIR / 'seqfishplus_mouse_brain'
DATASETS = [
    ('seqfishplus_OB.h5ad', 'SeqFISHplus_OB', 'seqfishplus_ob'),
    ('seqfishplus_cortex_svz.h5ad', 'SeqFISHplus_CortexSVZ', 'seqfishplus_cortexsvz'),
]
def load_seqfish(path):
    adata = ad.read_h5ad(str(path))
    Y = adata.X
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    genes = np.asarray(adata.var_names.values).astype(str)
    labels = np.asarray(adata.obs['louvain'].values).flatten().astype(str)
    valid = np.isfinite(coords).all(axis=1)
    valid &= np.array([l not in ('nan', 'None', '', 'NA') for l in labels])
    Y, coords, labels = Y[valid], coords[valid], labels[valid]
    n = len(np.unique(labels))
    del adata
    return Y, labels, coords, genes, n
def main():
    t0 = time.time()
    print("=" * 70)
    print("  seqFISH+ 补充分析 (OB / Cortex-SVZ, louvain 标签)")
    print("=" * 70)
    results = []
    for fname, short, key in DATASETS:
        p = SF_DIR / fname
        if not p.exists():
            print(f">>> 跳过 {short}: 文件不存在")
            continue
        print(f"\n>>> [seqFISH+] {short}")
        try:
            Y, labels, coords, genes, n_modules = load_seqfish(p)
            print(f"  {Y.shape[0]} cells x {Y.shape[1]} genes, K={n_modules}")
            r = analyze_dataset(
                name=f"{short} (seqFISH+)",
                short_name=short,
                Y=Y, labels=labels, coords=coords, gene_names=genes,
                n_modules=n_modules, subsample_per_type=300,
                dataset_key=key)
            r['platform'] = 'seqFISH+'
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e!r}")
    out = SCRIPT_DIR / 'seqfish_results.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump([{k: v for k, v in r.items() if k != '_spatial'}
                   for r in results], f, ensure_ascii=False, indent=2)
    print(f"\n完成! 用时 {(time.time()-t0)/60:.1f} 分钟, 共 {len(results)} 数据集 -> {out}")
if __name__ == '__main__':
    main()

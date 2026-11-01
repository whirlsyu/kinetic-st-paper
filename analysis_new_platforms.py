# -*- coding: utf-8 -*-
"""
新平台数据补充分析：Slide-seqV2 / MERFISH / seqFISH+
背景：原综合分析 (comprehensive_analysis.py) 覆盖的真实数据平台为
  - 10x Visium   (STSP 人脑切片 x 8)
  - Stereo-seq   (MOSTA 小鼠胚胎 x 3 + 小鼠 E11.5 心脏 2D 切片 x 4)
本脚本补齐其余三个主流平台，使用与原文相同的分析流程：
  带标签数据: 15 种监督 marker 方法 + 5 种无监督空间聚类方法 (ARI/NMI/Coherence/Entropy)
  无标签数据 (Slide-seqV2): 5 种无监督方法 + Coherence/Entropy 无参考评估
数据集 (均位于 d:/文章/时空转录组新方法/dataset/):
  1. merfish_mouse_brain_ABC/merfish_ABC_sec{30,32}.h5ad
     MERFISH 小鼠全脑冠状切片 (Yao et al., Nature 2023, Allen Brain Cell Atlas)
     550 基因 panel, 标签 = ABC 'class' 细胞大类
  2. seqfishplus_mouse_brain/seqfishplus_{OB,cortex_svz}.h5ad
     seqFISH+ 小鼠嗅球 / 皮层+SVZ (Eng et al., Nature 2019, CaiGroup 官方源数据)
     10000 基因, 标签 = 原文 Louvain 聚类
  3. slideseqv2_mouse_kidney/slideseqv2_kidney_191109_{06,07,08}.h5ad
     Slide-seqV2 健康小鼠肾 (GSE190094, BTBR-wt/wt), 无细胞类型标签
输出 (本目录):
  - new_platforms_results.json / new_platforms_unsupervised.csv / new_platforms_supervised.csv
  - fig_newplatforms_{key}_maps.png / fig_newplatforms_unsupervised.png
Usage:
    python analysis_new_platforms.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ============================================================
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
import anndata as ad
import pandas as pd
from comprehensive_analysis import (
    analyze_dataset,
    encode_labels,
    subsample_indices,
    plot_spatial_domains,
    run_all_unsupervised,
    compute_coherence,
    compute_entropy,
)
DATA_DIR = Path(r'd:\文章\时空转录组新方法\dataset')
UNSUP_METHODS = ["DEIM+NMF", "HVG+NMF", "HVG+NMF\n(no spatial)", "PCA+KMeans", "KineticST"]
# ============================================================
# ============================================================
DATASETS = [
    {
        'path': DATA_DIR / 'merfish_mouse_brain_ABC' / 'merfish_ABC_sec30_region.h5ad',
        'platform': 'MERFISH',
        'short_name': 'MERFISH_s30',
        'label_cols': ['region'],
    },
    {
        'path': DATA_DIR / 'merfish_mouse_brain_ABC' / 'merfish_ABC_sec32_region.h5ad',
        'platform': 'MERFISH',
        'short_name': 'MERFISH_s32',
        'label_cols': ['region'],
    },
    {
        'path': DATA_DIR / 'slideseqv2_mouse_kidney' / 'slideseqv2_kidney_06.h5ad',
        'platform': 'Slide-seqV2',
        'short_name': 'SlideSeqV2_Kidney06',
        'label_cols': [],   
        'n_modules': 10,
    },
    {
        'path': DATA_DIR / 'slideseqv2_mouse_kidney' / 'slideseqv2_kidney_07.h5ad',
        'platform': 'Slide-seqV2',
        'short_name': 'SlideSeqV2_Kidney07',
        'label_cols': [],
        'n_modules': 10,
    },
    {
        'path': DATA_DIR / 'slideseqv2_mouse_kidney' / 'slideseqv2_kidney_08.h5ad',
        'platform': 'Slide-seqV2',
        'short_name': 'SlideSeqV2_Kidney08',
        'label_cols': [],
        'n_modules': 10,
    },
]
def load_new_platform_dataset(cfg):
    adata = ad.read_h5ad(str(cfg['path']))
    label_col = None
    for col in cfg['label_cols']:
        if col in adata.obs.columns:
            label_col = col
            break
    if cfg['label_cols'] and label_col is None:
        raise ValueError(f"{cfg['path'].name}: 未找到标签列, obs={list(adata.obs.columns)}")
    coords_key = 'spatial' if 'spatial' in adata.obsm else ('X_spatial' if 'X_spatial' in adata.obsm else None)
    if coords_key is None:
        raise ValueError(f"{cfg['path'].name}: 未找到空间坐标")
    Y = adata.X
    coords = np.asarray(adata.obsm[coords_key], dtype=np.float64)
    genes = np.asarray(adata.var_names.values).astype(str)
    labels = np.asarray(adata.obs[label_col].values).flatten().astype(str) if label_col else None
    valid = np.isfinite(coords).all(axis=1)
    if labels is not None:
        valid &= np.array([l not in ('nan', 'None', '', 'NA') for l in labels])
    n_drop = int((~valid).sum())
    if n_drop:
        print(f"  过滤 {n_drop} 个无效点")
    Y = Y[valid]
    coords = coords[valid]
    if labels is not None:
        labels = labels[valid]
    n_modules = len(np.unique(labels)) if labels is not None else cfg.get('n_modules', 10)
    del adata
    return Y, labels, coords, genes, n_modules, label_col
def run_unlabeled(cfg, Y, coords, n_modules):
    Y_dense = np.asarray(Y.toarray() if sparse.issparse(Y) else Y, dtype=np.float64)
    print(f"\n  [Unsupervised-only] 5 methods (K={n_modules})...")
    preds, times, A = run_all_unsupervised(Y_dense, coords, n_modules, spatial_method='knn')
    unsup = {'methods': UNSUP_METHODS, 'ARI': {}, 'NMI': {},
             'Coherence': {}, 'Entropy': {}, 'Time_sec': {}}
    for m in UNSUP_METHODS:
        pred = preds.get(m, np.zeros(Y_dense.shape[0], dtype=int))
        unsup['ARI'][m] = None
        unsup['NMI'][m] = None
        unsup['Coherence'][m] = float(compute_coherence(pred, A))
        unsup['Entropy'][m] = float(compute_entropy(pred, n_modules))
        unsup['Time_sec'][m] = float(times.get(m, 0.0))
    result = {
        'name': f"{cfg['short_name']} ({cfg['platform']})",
        'short_name': cfg['short_name'],
        'platform': cfg['platform'],
        'n_spots': int(Y_dense.shape[0]),
        'n_genes': int(Y_dense.shape[1]),
        'n_modules': n_modules,
        'has_ground_truth': False,
        'dataset_key': cfg['short_name'].lower(),
        'supervised': None,
        'unsupervised': unsup,
        '_spatial': {'coords': coords, 'true_labels': preds.get('DEIM+NMF'),
                     'predictions': preds},
    }
    print(f"\n  [Unsupervised] Summary:")
    for m in UNSUP_METHODS:
        print(f"    {m:25s}: Coh={unsup['Coherence'][m]:.4f}  "
              f"Ent={unsup['Entropy'][m]:.4f}  t={unsup['Time_sec'][m]:.1f}s")
    return result
def plot_cross_platform_bars(all_results, save_path):
    colors = {'DEIM+NMF': '#A9D6B8', 'HVG+NMF': '#9BBFE0',
              'HVG+NMF\n(no spatial)': '#C5B6DD', 'PCA+KMeans': '#F2C9A4',
              'KineticST': '#E8A2B0'}
    labeled = [r for r in all_results if r['has_ground_truth']]
    unlabeled = [r for r in all_results if not r['has_ground_truth']]
    n_panels = (2 if labeled else 0) + (1 if unlabeled else 0)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(5.2 * max(n_panels, 1), 4.4))
    if n_panels == 1:
        axes = [axes]
    pi = 0
    if labeled:
        names = [r['short_name'] for r in labeled]
        for ax, metric in zip(axes[:2], ['ARI', 'NMI']):
            x = np.arange(len(names))
            width = 0.16
            for i, m in enumerate(UNSUP_METHODS):
                vals = [r['unsupervised'][metric].get(m) or 0 for r in labeled]
                ax.bar(x + (i - 2) * width, vals, width, label=m.replace('\n', ' '),
                       color=colors[m], edgecolor='gray', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{n}\n({r['platform']})" for n, r in zip(names, labeled)], fontsize=8)
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1.0)
            ax.set_title(f'Labeled New Platforms: {metric}', fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        axes[1].legend(fontsize=7.5, loc='upper right')
        pi = 2
    if unlabeled:
        ax = axes[pi]
        names = [r['short_name'] for r in unlabeled]
        x = np.arange(len(names))
        width = 0.16
        for i, m in enumerate(UNSUP_METHODS):
            vals = [r['unsupervised']['Coherence'].get(m) or 0 for r in unlabeled]
            ax.bar(x + (i - 2) * width, vals, width, label=m.replace('\n', ' '),
                   color=colors[m], edgecolor='gray', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n({r['platform']})" for n, r in zip(names, unlabeled)], fontsize=8)
        ax.set_ylabel('Coherence')
        ax.set_ylim(0, 1.0)
        ax.set_title('Unlabeled (Slide-seqV2): Spatial Coherence', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
def main():
    t_start = time.time()
    print("=" * 70)
    print("  新平台补充分析: Slide-seqV2 / MERFISH / seqFISH+")
    print("=" * 70)
    all_results = []
    for cfg in DATASETS:
        if not cfg['path'].exists():
            print(f"\n>>> 跳过 {cfg['short_name']}: 文件不存在 ({cfg['path']})")
            continue
        print(f"\n>>> [{cfg['platform']}] {cfg['path'].name}")
        try:
            Y, labels, coords, genes, n_modules, label_col = load_new_platform_dataset(cfg)
            print(f"  标签列: {label_col}  |  {Y.shape[0]} spots x {Y.shape[1]} genes  |  K={n_modules}")
            if labels is not None:
                labels_int, _ = encode_labels(labels)
                if len(labels) > 4000:
                    idx = subsample_indices(labels_int, target_per_type=300)
                    Y = Y[idx]
                    labels = labels[idx]
                    coords = coords[idx]
                    print(f"  分层抽样 -> {len(labels)} spots")
                Y_dense = np.asarray(Y.toarray() if sparse.issparse(Y) else Y, dtype=np.float64)
                r = analyze_dataset(
                    name=f"{cfg['short_name']} ({cfg['platform']})",
                    short_name=cfg['short_name'],
                    Y=Y_dense, labels=labels, coords=coords,
                    gene_names=genes, n_modules=n_modules,
                    true_markers=None,
                    spatial_method='knn',
                    subsample_per_type=None,
                    dataset_key=cfg['short_name'].lower(),
                )
                r['platform'] = cfg['platform']
                all_results.append(r)
                del Y_dense
            else:
                if Y.shape[0] > 6000:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(Y.shape[0], 6000, replace=False)
                    idx.sort()
                    Y = Y[idx]
                    coords = coords[idx]
                    print(f"  随机抽样 -> 6000 spots")
                r = run_unlabeled(cfg, Y, coords, n_modules)
                all_results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    if not all_results:
        print("\n没有成功分析任何数据集!")
        return
    print(f"\n{'=' * 70}\n  生成空间域图...\n{'=' * 70}")
    for r in all_results:
        sp = r.pop('_spatial', None)
        if sp is None:
            continue
        try:
            plot_spatial_domains(
                sp['coords'], sp['true_labels'], sp['predictions'],
                f"{r['short_name']} ({r['platform']})",
                str(SCRIPT_DIR / f"fig_newplatforms_{r['dataset_key']}_maps.png"),
            )
        except Exception as e:
            print(f"  绘图失败 {r['short_name']}: {e}")
    unsup_rows, sup_rows = [], []
    for r in all_results:
        for m in r['unsupervised']['methods']:
            unsup_rows.append({
                'Dataset': r['short_name'], 'Platform': r['platform'], 'Method': m,
                'Has_labels': r['has_ground_truth'],
                'ARI': r['unsupervised']['ARI'].get(m),
                'NMI': r['unsupervised']['NMI'].get(m),
                'Coherence': r['unsupervised']['Coherence'].get(m, 0),
                'Entropy': r['unsupervised']['Entropy'].get(m, 0),
                'Time_sec': r['unsupervised']['Time_sec'].get(m, 0),
            })
        if r['supervised']:
            for m in r['supervised']['methods']:
                sup_rows.append({
                    'Dataset': r['short_name'], 'Platform': r['platform'], 'Method': m,
                    'Mean_hits': r['supervised']['mean_hits'].get(m, 0),
                    'Total_hits': r['supervised']['total_hits'].get(m, 0),
                    'Win_types': r['supervised']['win_types'].get(m, 0),
                })
    df_unsup = pd.DataFrame(unsup_rows)
    df_unsup.to_csv(SCRIPT_DIR / 'new_platforms_unsupervised.csv', index=False, encoding='utf-8-sig')
    if sup_rows:
        pd.DataFrame(sup_rows).to_csv(SCRIPT_DIR / 'new_platforms_supervised.csv',
                                      index=False, encoding='utf-8-sig')
    plot_cross_platform_bars(all_results, str(SCRIPT_DIR / 'fig_newplatforms_unsupervised.png'))
    # ===== JSON =====
    with open(SCRIPT_DIR / 'new_platforms_results.json', 'w', encoding='utf-8') as f:
        json.dump({'datasets': all_results}, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n{'=' * 70}\n  无监督结果汇总\n{'=' * 70}")
    show = df_unsup.copy()
    for c in ['ARI', 'NMI', 'Coherence', 'Entropy']:
        show[c] = show[c].astype(float).round(3)
    print(show.pivot_table(index=['Platform', 'Dataset'], columns='Method',
                           values='ARI').round(3).to_string())
    print()
    print(show.pivot_table(index=['Platform', 'Dataset'], columns='Method',
                           values='Coherence').round(3).to_string())
    print(f"\n  总耗时: {time.time() - t_start:.1f}s")
if __name__ == '__main__':
    main()

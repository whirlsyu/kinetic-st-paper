# -*- coding: utf-8 -*-
"""Figure 9 (E8): head-to-head benchmark of six methods on 25 datasets.
Panel A: per-dataset ARI by category (mean bar + jittered points).
Panel B: per-dataset NMI by category (same layout).
All values read from results/e8_combined_table.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DF = pd.read_csv(r'd:\文章\最新方法应用时空转录组\results'
                 r'\e8_combined_table.csv')
OUT = r'd:\文章\最新方法应用时空转录组\fig\fig9_e8_sota.png'
METHODS = ['KineticST', 'STAGATE', 'SpaGCN', 'SpaceFlow', 'BayesSpace',
           'GraphST']
COLORS = {'KineticST': '#F2C9A4', 'STAGATE': '#9BBFE0',
          'SpaGCN': '#A9D6B8', 'SpaceFlow': '#C5B6DD',
          'BayesSpace': '#F0B6B6', 'GraphST': '#A9D3D0'}
CATS = ['simulation', 'brain', 'embryo', 'heart']
CAT_LABEL = {'simulation': 'Simulation (n=10)', 'brain': 'Brain (n=8)',
             'embryo': 'Embryo (n=3)', 'heart': 'Heart (n=4)'}
plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans',
                     'axes.edgecolor': '#444444',
                     'axes.linewidth': 0.8})
fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.2), sharex=False)
rng = np.random.default_rng(0)
for row, metric, mlabel in [(0, 'ari', 'ARI'), (1, 'nmi', 'NMI')]:
    for col, cat in enumerate(CATS):
        ax = axes[row, col]
        sub = DF[DF.category == cat]
        for mi, m in enumerate(METHODS):
            vals = sub[sub.method == m][metric].values
            ax.bar(mi, vals.mean(), width=0.62, color=COLORS[m],
                   edgecolor='#666666', linewidth=0.7, zorder=2,
                   alpha=0.95)
            xj = mi + rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(xj, vals, s=13, color='#333333', alpha=0.75,
                       zorder=3, linewidths=0)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=38, ha='right')
        ax.set_title(CAT_LABEL[cat], fontsize=9.5, pad=4)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', color='#DDDDDD', linewidth=0.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        if col == 0:
            ax.set_ylabel(mlabel, fontsize=10)
        if row == 0 and col == 0:
            ax.text(-0.45, 1.10, 'A', transform=ax.transAxes,
                    fontsize=13, fontweight='bold', va='top')
        if row == 1 and col == 0:
            ax.text(-0.45, 1.10, 'B', transform=ax.transAxes,
                    fontsize=13, fontweight='bold', va='top')
fig.suptitle('Head-to-head benchmark with five mainstream methods '
             '(official implementations, 25 datasets)', fontsize=11,
             y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=300)
print('saved', OUT)
# quick numeric sanity print
print(DF.groupby(['category', 'method']).ari.mean()
        .round(3).unstack().to_string())

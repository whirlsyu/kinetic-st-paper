# -*- coding: utf-8 -*-
"""Export every panel of Figures 1-7 as standalone small PNGs into ROOT/fig/.

Reuses the panel-drawing functions from make_otmode_figs.py (Figs 1-6) and
make_extra_figs.py (Fig 7) without internal panel letters; panels that are
embedded bitmaps in the main figures are copied directly.
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = Path(__file__).parent.resolve()
ROOT = SCR.parent
for _p in (str(ROOT), str(SCR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_otmode_figs as mf
import make_extra_figs as mx

FIG_DIR = ROOT / "fig"
FIG_DIR.mkdir(exist_ok=True)


def save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", out.name)


def copy_png(src_name, dst_name):
    shutil.copy2(SCR / src_name, FIG_DIR / dst_name)
    print("copied", dst_name)


def single(func, name, figsize, *args, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    func(ax, *args, **kwargs)
    save(fig, name)


def main():
    # ---- Figure 1 ----
    fig = plt.figure(figsize=(9.2, 4.6), dpi=200)
    mf.fig1_schematic(fig.add_axes([0.01, 0.01, 0.98, 0.90]), letter=None)
    save(fig, "fig1A_pipeline.png")
    copy_png("fig_kineticst_synthetic.png", "fig1B_synthetic.png")

    # ---- Figure 2 ----
    single(mf.boxpanel, "fig2A_simulation_ARI.png", (4.0, 3.0),
           "Simulation", "ARI", "", title="Simulation (n=10 seeds)")
    single(mf.boxpanel, "fig2B_simulation_NMI.png", (4.0, 3.0),
           "Simulation", "NMI", "", title="Simulation (n=10 seeds)")
    single(mf.boxpanel, "fig2C_simulation_coherence.png", (4.0, 3.0),
           "Simulation", "Coherence", "", title="Simulation (n=10 seeds)")
    copy_png("fig_spatial_simulation_seed42.png", "fig2D_simulation_maps.png")

    # ---- Figure 3 ----
    single(mf.boxpanel, "fig3A_STSP_ARI.png", (4.0, 3.0),
           "STSP", "ARI", "", title="Mouse brain STSP (n=8)")
    single(mf.boxpanel, "fig3B_STSP_NMI.png", (4.0, 3.0),
           "STSP", "NMI", "", title="Mouse brain STSP (n=8)")
    copy_png("fig_spatial_stsp0007732.png", "fig3C_STSP0007732_maps.png")
    copy_png("fig_spatial_stsp0007728.png", "fig3D_STSP0007728_maps.png")

    # ---- Figure 4 ----
    single(mf.boxpanel, "fig4A_MOSTA_ARI.png", (4.0, 3.0),
           "MOSTA", "ARI", "", title="Mouse embryo MOSTA (n=3)")
    single(mf.boxpanel, "fig4B_heart_ARI.png", (4.0, 3.0),
           "Heart", "ARI", "", title="Mouse E11.5 heart (n=4)")
    copy_png("fig_spatial_e9.5_e2s1.png", "fig4C_embryo_maps.png")
    copy_png("fig_spatial_heart_s45.png", "fig4D_heart_maps.png")

    # ---- Figure 5 ----
    single(mf.sup_hits_panel, "fig5A_marker_hits.png", (7.6, 3.0), "")
    single(mf.sup_win_panel, "fig5B_marker_wins.png", (7.6, 3.0), "")
    single(mf.sup_rank_panel, "fig5C_marker_rank.png", (7.6, 3.0), "")

    # ---- Figure 6 ----
    copy_png("fig_kineticst_detailed.png", "fig6A_mechanistic.png")
    single(mf.runtime_panel, "fig6B_runtime.png", (7.0, 3.0), "")

    # ---- Figure 7 ----
    datasets = json.loads(mx.RESULTS_PATH.read_text(encoding="utf-8"))["datasets"]
    if mx.CACHE_PATH.exists():
        cache = json.loads(mx.CACHE_PATH.read_text(encoding="utf-8"))
    else:
        cache = mx.build_cache()

    single(mx.corr_unsup_panel, "fig7A_corr_unsupervised.png", (3.8, 3.4),
           datasets, letter="")
    single(mx.corr_sup_panel, "fig7B_corr_supervised.png", (6.2, 5.2),
           datasets, letter="")
    single(mx.venn_panel, "fig7C_marker_venn.png", (3.8, 3.6), cache, letter="")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), dpi=200)
    for j in range(3):
        mx.expr_panel(axes[j], cache, j, letter=None)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "fig7D_expression_distributions.png")

    single(mx.enrich_panel, "fig7E_KEGG.png", (8.0, 2.8), cache, "kegg", "",
           "KEGG enrichment of KineticST markers (STSP0007732)")
    single(mx.enrich_panel, "fig7F_GO.png", (8.0, 2.8), cache, "go", "",
           "GO Biological Process enrichment of KineticST markers")

    print(f"\nAll individual panels saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

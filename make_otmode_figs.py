# -*- coding: utf-8 -*-
"""
Compose OTMODE-style multi-panel main figures for the KineticST article.

OTMODE (Bioinformatics) uses ~6 dense composite figures, each with lettered
panels (A, B, C, ...). All quantitative panels are re-drawn from
analysis_results.json so panel letters are unique and consistent; spatial
domain maps embed the per-section PNGs (which carry no internal letters).

Layout notes:
  - Embedded-image rows are sized from the actual image aspect ratio so
    rows are visually balanced (no stretched/squashed rows).
  - Fig 3/4 use a three-row layout (boxplots row + full-width map rows).
  - Fig 5 uses three stacked full-width horizontal-bar panels.
  - Fig 6 places the runtime legend outside the axes.

Output: fig1_main.png ... fig6_main.png in the scripts/ folder.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.resolve()
SCR = ROOT / "scripts"
DATA = json.loads((SCR / "analysis_results.json").read_text(encoding="utf-8"))
DATASETS = DATA["datasets"] if isinstance(DATA, dict) and "datasets" in DATA else DATA

METHODS = ["DEIM+NMF", "HVG+NMF", "HVG+NMF\n(no spatial)", "PCA+KMeans", "KineticST"]
TICK = {"DEIM+NMF": "DEIM+NMF", "HVG+NMF": "HVG+NMF",
        "HVG+NMF\n(no spatial)": "HVG+NMF\n(no spat.)",
        "PCA+KMeans": "PCA+KMeans", "KineticST": "KineticST"}
COLORS = {"DEIM+NMF": "#4C72B0", "HVG+NMF": "#55A868",
          "HVG+NMF\n(no spatial)": "#8172B3", "PCA+KMeans": "#C44E52",
          "KineticST": "#DD8452"}
CATS = ["Simulation", "STSP", "MOSTA", "Heart"]


def category(d):
    s = d.get("short_name", "")
    n = d.get("name", "")
    if s == "Simulation" or "Simulation" in n:
        return "Simulation"
    if s.startswith("STSP") or "STSP" in n:
        return "STSP"
    if s.startswith("Heart") or "heart" in n.lower():
        return "Heart"
    if "E12.5" in s or "E9.5" in s or "E1S3" in s or "E2S" in s or "MOSTA" in n:
        return "MOSTA"
    return "Other"


def panel_label(ax, letter, x=-0.02, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="right")


def embed(ax, png_path, letter=None):
    img = mpimg.imread(str(png_path))
    ax.imshow(img)
    ax.axis("off")
    if letter:
        panel_label(ax, letter, x=0.005, y=1.0)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _img_hw(png_path):
    """height/width aspect of a PNG."""
    img = mpimg.imread(str(png_path))
    return img.shape[0] / img.shape[1]


def stacked_fig(row_heights, ncols, width=9.2, hspace=0.12, wspace=0.25,
                top=0.955, bottom=0.04, left=0.06, right=0.98):
    """Figure whose row heights (inches) are preserved exactly, so embedded
    images keep their native aspect and all rows are visually balanced."""
    n = len(row_heights)
    S = sum(row_heights)
    gaps = hspace * (S / n) * (n - 1)
    fig_h = (S + gaps) / (top - bottom)
    fig = plt.figure(figsize=(width, fig_h), dpi=200)
    gs = fig.add_gridspec(n, ncols, height_ratios=row_heights,
                          hspace=hspace, wspace=wspace,
                          top=top, bottom=bottom, left=left, right=right)
    return fig, gs


# ---------------------------------------------------------------------------
# schematic pieces (Fig 1A)
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc, ec, icon=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                       linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    if icon is not None:
        ih = h * 0.54
        icon(ax, x + w * 0.09, y + h - ih - 0.012, w * 0.82, ih)
        ax.text(x + w / 2, y + (h - ih) / 2 - 0.004, text, ha="center",
                va="center", fontsize=7.2)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.3)


# --- mini schematic icons (drawn inside pipeline boxes, axes coordinates) ---
# unified fresh/pastel palette: soft fills, thin light lines, no heavy blocks
_P_BLUE, _P_GREEN, _P_ORANGE = "#9BBFE0", "#A9D6B8", "#F2C9A4"
_P_PURPLE, _P_CORAL, _P_GRAY = "#C5B6DD", "#E8AFAF", "#BBBBBB"


def _ic_matrix(ax, x, y, w, h):
    rng = np.random.default_rng(1)
    ax.imshow(rng.beta(2, 5, (6, 9)), extent=(x, x + w * 0.60, y, y + h),
              aspect="auto", cmap="Blues", alpha=0.65, zorder=3)
    gx, gy = np.meshgrid(np.linspace(x + w * 0.74, x + w * 0.97, 4),
                         np.linspace(y + h * 0.12, y + h * 0.88, 4))
    ax.scatter(gx, gy, s=2.5, c=_P_ORANGE, zorder=3)


def _ic_pca(ax, x, y, w, h):
    rng = np.random.default_rng(2)
    t = rng.normal(0, 1, 40)
    ax.scatter(x + w / 2 + t * w * 0.17 + rng.normal(0, w * 0.03, 40),
               y + h / 2 + t * h * 0.17 + rng.normal(0, h * 0.06, 40),
               s=2.5, c=_P_BLUE, zorder=3)
    ax.annotate("", xy=(x + w * 0.88, y + h * 0.86),
                xytext=(x + w * 0.12, y + h * 0.14),
                arrowprops=dict(arrowstyle="-|>", color=_P_GRAY, lw=0.8))


def _ic_graph(ax, x, y, w, h):
    rng = np.random.default_rng(3)
    pts = np.column_stack([x + w * rng.uniform(0.08, 0.92, 9),
                           y + h * rng.uniform(0.12, 0.88, 9)])
    for i, p in enumerate(pts):
        d = np.hypot(*(pts - p).T)
        for j in np.argsort(d)[1:4]:
            ax.plot([p[0], pts[j, 0]], [p[1], pts[j, 1]], color="#D5D5D5",
                    lw=0.5, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c=_P_BLUE, zorder=3)


def _ic_lr(ax, x, y, w, h):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x + w * 0.27, y + h * 0.5), h * 0.33, fc="#EAF1F9",
                        ec=_P_BLUE, lw=0.9, zorder=3))
    ax.add_patch(Circle((x + w * 0.73, y + h * 0.5), h * 0.33, fc="#FBF1E6",
                        ec=_P_ORANGE, lw=0.9, zorder=3))
    for yy in np.linspace(y + h * 0.32, y + h * 0.68, 3):
        ax.annotate("", xy=(x + w * 0.60, yy), xytext=(x + w * 0.40, yy),
                    arrowprops=dict(arrowstyle="-|>", color=_P_CORAL, lw=0.6))


def _ic_solver(ax, x, y, w, h):
    rng = np.random.default_rng(4)
    xs = np.linspace(x + w * 0.06, x + w * 0.94, 60)
    ys = y + h / 2 + np.cumsum(rng.normal(0, h * 0.045, 60))
    ys = np.clip(ys, y + h * 0.06, y + h * 0.94)
    for f in (0.25, 0.5, 0.75):
        ax.plot([x + w * f, x + w * f], [y + h * 0.05, y + h * 0.95],
                ls="--", color="#E2E2E2", lw=0.6, zorder=2)
    ax.plot(xs, ys, color=_P_ORANGE, lw=1.0, zorder=3)


def _ic_field(ax, x, y, w, h):
    gx, gy = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40))
    z = np.exp(-(gx ** 2 + gy ** 2) * 2) + 0.6 * np.exp(
        -((gx - 0.5) ** 2 + (gy + 0.4) ** 2) * 4)
    X = x + w * (0.06 + 0.88 * (gx + 1) / 2)
    Y = y + h * (0.06 + 0.88 * (gy + 1) / 2)
    ax.contour(X, Y, z, levels=5, colors=[_P_BLUE], linewidths=0.7, zorder=3)


def _ic_domains(ax, x, y, w, h):
    rng = np.random.default_rng(5)
    for k, c in enumerate([_P_BLUE, _P_GREEN, _P_ORANGE]):
        cx = x + w * (0.25 + 0.25 * k)
        cy = y + h * (0.32 + 0.28 * (k % 2))
        ax.scatter(cx + rng.normal(0, w * 0.05, 12),
                   cy + rng.normal(0, h * 0.09, 12), s=2.5, c=c, zorder=3)


def _ic_outputs(ax, x, y, w, h):
    xs = np.linspace(0, 1, 60)
    ys = np.exp(-2.2 * xs) * np.cos(4.5 * np.pi * xs)  # damped relaxation wave
    X = x + w * (0.10 + 0.80 * xs)
    Y = y + h * (0.50 + 0.38 * ys)
    ax.fill_between(X, y + h * 0.12, Y, color=_P_PURPLE, alpha=0.15, zorder=2)
    ax.plot(X, Y, color=_P_PURPLE, lw=1.0, zorder=3)
    ax.plot([x + w * 0.10, x + w * 0.90], [y + h * 0.12, y + h * 0.12],
            color="#DDDDDD", lw=0.6, zorder=2)


def _ic_supervised(ax, x, y, w, h):
    for k, v in enumerate([0.9, 0.65, 0.45, 0.3]):
        ax.barh(y + h * (0.14 + 0.22 * k), w * 0.68 * v, left=x + w * 0.14,
                height=h * 0.13, color=_P_GREEN, zorder=3)


def _ic_labels(ax, x, y, w, h):
    cols = [_P_BLUE, _P_GREEN, _P_ORANGE, _P_CORAL]
    gx, gy = np.meshgrid(np.linspace(x + w * 0.18, x + w * 0.82, 5),
                         np.linspace(y + h * 0.18, y + h * 0.82, 3))
    ax.scatter(gx.ravel(), gy.ravel(), s=6,
               c=[cols[i % 4] for i in range(15)], marker="s", zorder=3)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.8,
                                 color="#C8D6E5", zorder=4))


def fig1_schematic(axa, letter="A"):
    """The 7-stage pipeline diagram (snake layout, right-angle arrows only,
    no crossings). Each stage box carries a mini schematic icon above its
    label. Drawn on an axes spanning [0,1]x[0,1]."""
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis("off")
    axa.set_title("KineticST pipeline", fontsize=11, fontweight="bold", pad=4)

    H = 0.25  # box height (icon on top, label below)
    W = 0.16  # uniform box width
    G = 0.10  # uniform column gap
    P = 0.02  # arrow padding from panel edges
    X0, X1, X2, X3 = 0.02, 0.02+W+G, 0.02+2*(W+G), 0.02+3*(W+G)
    Y1, Y2, Y3 = 0.72, 0.37, 0.02  # wider row gaps
    # fresh pastel scheme: lighter fill, edge a deeper shade of same family
    C_GRAY  = ("#EBF1F7", "#A5B4C6")   # input / reference labels
    C_BLUE  = ("#DBE7F7", "#95B5D8")   # embedding / graph
    C_ORAN  = ("#FCE9D4", "#E5B882")   # collision / solver
    C_GREEN = ("#DBEEDF", "#98C5A6")   # moment closure / invariant measure
    C_PINK  = ("#F7E4E1", "#DBA9A4")   # outputs
    C_PURP  = ("#E8E0F2", "#B19FD0")   # supervised branch
    # ---- Row 1 (unsupervised core, left -> right), 4 cols ----
    _box(axa, X0, Y1, W, H, "Input:\nexpression matrix\n+ coordinates",       *C_GRAY,  _ic_matrix)
    _box(axa, X1, Y1, W, H, "1. Latent\nembedding\n(PCA to state z)",         *C_BLUE,  _ic_pca)
    _box(axa, X2, Y1, W, H, "2. Spatial\nk-NN graph\n(adjacency)",            *C_BLUE,  _ic_graph)
    _box(axa, X3, Y1, W, H, "3. Collision\noperator\n(ligand-receptor)",      *C_ORAN,  _ic_lr)
    # ---- Row 2 (snake right -> left), 4 cols ----
    _box(axa, X3, Y2, W, H, "4. Stochastic\nkinetic solver\n(time-slicing)",  *C_ORAN,  _ic_solver)
    _box(axa, X2, Y2, W, H, "5. Moment\nclosure\n(tissue fields)",      *C_GREEN, _ic_field)
    _box(axa, X1, Y2, W, H, "6. Invariant\nmeasure\n(spatial domains)", *C_GREEN, _ic_domains)
    _box(axa, X0, Y2, W, H, "Outputs:\ndomains, fields,\nrelaxation, waves", *C_PINK, _ic_outputs)
    # ---- Row 3 (supervised branch, left -> right), 2 cols ----
    _box(axa, X1, Y3, W, H, "Reference\nlabels\n(cell types/regions)",            *C_GRAY, _ic_labels)
    _box(axa, X2, Y3, W, H, "7. Supervised\nmarker discovery\n(kinetic z-scores)", *C_PURP, _ic_supervised)

    # ---- arrows: in gaps only, not touching panels ----
    Yc1 = Y1 + H/2; Yc2 = Y2 + H/2; Yc3 = Y3 + H/2
    _arrow(axa, X0+W+P,  Yc1, X1-P,    Yc1)  # Input -> Latent
    _arrow(axa, X1+W+P,  Yc1, X2-P,    Yc1)  # Latent -> kNN
    _arrow(axa, X2+W+P,  Yc1, X3-P,    Yc1)  # kNN -> Collision
    _arrow(axa, X3+W/2,  (Y1+Y2+H)/2+0.03,  X3+W/2, (Y1+Y2+H)/2-0.03)  # Collision ↓ Solver
    _arrow(axa, X3-P,     Yc2, X2+W+P,  Yc2)  # Solver -> Moment
    _arrow(axa, X2-P,     Yc2, X1+W+P,  Yc2)  # Moment -> Invariant
    _arrow(axa, X1-P,     Yc2, X0+W+P,  Yc2)  # Invariant -> Outputs
    _arrow(axa, X2+W/2,  (Y2+Y3+H)/2+0.03,  X2+W/2, (Y2+Y3+H)/2-0.03)  # Moment ↓ Supervised
    _arrow(axa, X1+W+P,  Yc3, X2-P,    Yc3)  # Reference -> Supervised
    if letter:
        panel_label(axa, letter, x=0.0, y=1.03)


def make_fig1():
    fig = plt.figure(figsize=(9.2, 8.8), dpi=200)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.14,
                          top=0.97, bottom=0.03, left=0.02, right=0.98)
    fig1_schematic(fig.add_subplot(gs[0, 0]), letter="A")
    axb = fig.add_subplot(gs[1, 0])
    embed(axb, SCR / "fig_kineticst_synthetic.png", letter="B")
    fig.savefig(SCR / "fig1_main.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig1_main.png")


# ---------------------------------------------------------------------------
# quantitative panel helpers
# ---------------------------------------------------------------------------

def _cat_data(cat, metric):
    sub = [d for d in DATASETS if category(d) == cat]
    return [[d["unsupervised"][metric][m] for d in sub if m in d["unsupervised"][metric]] for m in METHODS], len(sub)


def boxpanel(ax, cat, metric, letter, title=None):
    """Pretty boxplot + jittered per-dataset dots for one category/metric."""
    data, n = _cat_data(cat, metric)
    rng = np.random.default_rng(0)
    ax.yaxis.grid(True, color="#EBEBEB", lw=0.7)
    ax.set_axisbelow(True)
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.3),
                    whiskerprops=dict(color="#777777", lw=0.9),
                    capprops=dict(color="#777777", lw=0.9))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.80)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.8)
    for i, dd in enumerate(data):
        ax.scatter(np.full(len(dd), i + 1) + rng.normal(0, 0.05, len(dd)),
                   dd, s=10, color="#222222", alpha=0.55, zorder=3,
                   edgecolor="none")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([TICK[m] for m in METHODS], rotation=30, ha="right",
                       fontsize=6.6)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(title or f"{cat} (n={n})", fontsize=9.5)
    ax.margins(y=0.10)
    _despine(ax)
    ax.tick_params(labelsize=8)
    panel_label(ax, letter, y=1.14)


def runtime_panel(ax, letter):
    width = 0.16
    x = np.arange(len(CATS))
    for i, m in enumerate(METHODS):
        means, sems = [], []
        for cat in CATS:
            sub = [d for d in DATASETS if category(d) == cat]
            vals = [d["unsupervised"]["Time_sec"][m] for d in sub if m in d["unsupervised"]["Time_sec"]]
            means.append(np.mean(vals)); sems.append(np.std(vals) / np.sqrt(max(len(vals), 1)))
        ax.bar(x + (i - 2) * width, means, width, yerr=sems, capsize=2,
               color=COLORS[m], label=TICK[m].replace("\n", " "), edgecolor="none",
               alpha=0.9, error_kw=dict(lw=0.8))
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=8.5)
    ax.set_ylabel("Runtime (s, log)", fontsize=9)
    # legend outside the axes so it never covers the bars
    ax.legend(fontsize=7.4, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), handlelength=1.2)
    _despine(ax); ax.tick_params(labelsize=8)
    panel_label(ax, letter)


# --- supervised panels ---

def _sup_methods():
    return DATASETS[0]["supervised"]["methods"]


def _sup_vals(m):
    return np.array([d["supervised"]["mean_hits"].get(m, np.nan) for d in DATASETS], dtype=float)


def sup_hits_panel(ax, letter):
    ms = _sup_methods()
    means = np.array([np.nanmean(_sup_vals(m)) for m in ms])
    sems = np.array([np.nanstd(_sup_vals(m)) / np.sqrt(np.sum(~np.isnan(_sup_vals(m)))) for m in ms])
    order = np.argsort(means)
    ms_o = [ms[i] for i in order]; mu = means[order]; se = sems[order]
    y = np.arange(len(ms_o))
    ax.barh(y, mu, xerr=se, color="#7A9E7E", edgecolor="none", alpha=0.9,
            error_kw=dict(lw=0.8), capsize=2, height=0.62)
    pad = max(mu) * 0.015
    for yi, v, s in zip(y, mu, se):
        ax.text(v + s + pad, yi, f"{v:.2f}", va="center", fontsize=7.2)
    ax.set_yticks(y); ax.set_yticklabels(ms_o, fontsize=8)
    ax.set_xlabel("Mean marker hits / z-score", fontsize=9.5)
    ax.set_xlim(right=max(mu + se) * 1.12)
    _despine(ax); ax.tick_params(labelsize=8.5)
    panel_label(ax, letter)


def sup_win_panel(ax, letter):
    ms = _sup_methods()
    wins = []
    for m in ms:
        tot = 0
        for d in DATASETS:
            wt = d["supervised"].get("win_types", {})
            if m in wt:
                tot += wt[m]
        wins.append(tot)
    wins = np.array(wins, dtype=float)
    order = np.argsort(wins)
    ms_o = [ms[i] for i in order]; w = wins[order]
    y = np.arange(len(ms_o))
    ax.barh(y, w, color="#4C72B0", edgecolor="none", alpha=0.9, height=0.62)
    pad = max(wins) * 0.012 if max(wins) > 0 else 0.05
    for yi, v in zip(y, w):
        ax.text(v + pad, yi, f"{int(v)}", va="center", fontsize=7.2)
    ax.set_yticks(y); ax.set_yticklabels(ms_o, fontsize=8)
    ax.set_xlabel("Total best-marker cell types", fontsize=9.5)
    ax.set_xlim(right=max(wins) * 1.10 if max(wins) > 0 else 1)
    _despine(ax); ax.tick_params(labelsize=8.5)
    panel_label(ax, letter)


def sup_rank_panel(ax, letter):
    ms = _sup_methods()
    ranks = np.zeros((len(DATASETS), len(ms)))
    for i, d in enumerate(DATASETS):
        vals = np.array([d["supervised"]["mean_hits"].get(m, np.nan) for m in ms], dtype=float)
        order = np.argsort(-vals)
        r = np.empty(len(ms)); r[order] = np.arange(1, len(ms) + 1)
        ranks[i] = r
    mean_rank = np.nanmean(ranks, axis=0)
    order = np.argsort(mean_rank)  # best (1) first
    ms_o = [ms[i] for i in order]; mr = mean_rank[order]
    y = np.arange(len(ms_o))
    ax.barh(y, mr, color="#C44E52", edgecolor="none", alpha=0.9, height=0.62)
    for yi, v in zip(y, mr):
        ax.text(v + 0.08, yi, f"{v:.1f}", va="center", fontsize=7.2)
    ax.set_yticks(y); ax.set_yticklabels(ms_o, fontsize=8)
    ax.invert_yaxis()  # best at top
    ax.axvline(len(ms) / 2, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Mean rank (1 = best)", fontsize=9.5)
    ax.set_xlim(right=max(mr) * 1.08)
    _despine(ax); ax.tick_params(labelsize=8.5)
    panel_label(ax, letter)


# ---------------------------------------------------------------------------
# figure builders
# ---------------------------------------------------------------------------

def make_fig2():
    img = SCR / "fig_spatial_simulation_seed42.png"
    left, right, W = 0.06, 0.99, 9.2
    h_box = 2.7
    h_img = (right - left) * W * _img_hw(img)
    fig, gs = stacked_fig([h_box, h_img], ncols=3, width=W,
                          hspace=0.16, wspace=0.30, top=0.94, bottom=0.05,
                          left=left, right=right)
    boxpanel(fig.add_subplot(gs[0, 0]), "Simulation", "ARI", "A",
             title="Simulation (n=10 seeds)")
    boxpanel(fig.add_subplot(gs[0, 1]), "Simulation", "NMI", "B",
             title="Simulation (n=10 seeds)")
    boxpanel(fig.add_subplot(gs[0, 2]), "Simulation", "Coherence", "C",
             title="Simulation (n=10 seeds)")
    embed(fig.add_subplot(gs[1, :]), img, letter="D")
    fig.savefig(SCR / "fig2_main.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig2_main.png")


def make_fig3():
    img1 = SCR / "fig_spatial_stsp0007732.png"
    img2 = SCR / "fig_spatial_stsp0007728.png"
    left, right, W = 0.07, 0.98, 9.2
    w_full = (right - left) * W
    h_box = 2.6
    rows = [h_box, w_full * _img_hw(img1), w_full * _img_hw(img2)]
    fig, gs = stacked_fig(rows, ncols=2, width=W, hspace=0.20, wspace=0.22,
                          top=0.955, bottom=0.035, left=left, right=right)
    boxpanel(fig.add_subplot(gs[0, 0]), "STSP", "ARI", "A",
             title="Mouse brain STSP (n=8)")
    boxpanel(fig.add_subplot(gs[0, 1]), "STSP", "NMI", "B",
             title="Mouse brain STSP (n=8)")
    embed(fig.add_subplot(gs[1, :]), img1, letter="C")
    embed(fig.add_subplot(gs[2, :]), img2, letter="D")
    fig.savefig(SCR / "fig3_main.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig3_main.png")


def make_fig4():
    img1 = SCR / "fig_spatial_e9.5_e2s1.png"
    img2 = SCR / "fig_spatial_heart_s45.png"
    left, right, W = 0.07, 0.98, 9.2
    w_full = (right - left) * W
    h_box = 2.6
    rows = [h_box, w_full * _img_hw(img1), w_full * _img_hw(img2)]
    fig, gs = stacked_fig(rows, ncols=2, width=W, hspace=0.20, wspace=0.22,
                          top=0.955, bottom=0.035, left=left, right=right)
    boxpanel(fig.add_subplot(gs[0, 0]), "MOSTA", "ARI", "A",
             title="Mouse embryo MOSTA (n=3)")
    boxpanel(fig.add_subplot(gs[0, 1]), "Heart", "ARI", "B",
             title="Mouse E11.5 heart (n=4)")
    embed(fig.add_subplot(gs[1, :]), img1, letter="C")
    embed(fig.add_subplot(gs[2, :]), img2, letter="D")
    fig.savefig(SCR / "fig4_main.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig4_main.png")


def make_fig5():
    W = 9.6
    fig, gs = stacked_fig([2.75, 2.75, 2.75], ncols=1, width=W,
                          hspace=0.30, top=0.965, bottom=0.05,
                          left=0.135, right=0.97)
    sup_hits_panel(fig.add_subplot(gs[0, 0]), "A")
    sup_win_panel(fig.add_subplot(gs[1, 0]), "B")
    sup_rank_panel(fig.add_subplot(gs[2, 0]), "C")
    fig.savefig(SCR / "fig5_main.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig5_main.png")


def make_fig6():
    img = SCR / "fig_kineticst_detailed.png"
    left, right, W = 0.05, 0.86, 9.2   # right margin reserved for legend
    h_img = (right - left) * W * _img_hw(img)
    h_rt = 2.7
    fig, gs = stacked_fig([h_img, h_rt], ncols=1, width=W,
                          hspace=0.10, top=0.965, bottom=0.045,
                          left=left, right=right)
    embed(fig.add_subplot(gs[0, 0]), img, letter="A")
    runtime_panel(fig.add_subplot(gs[1, 0]), "B")
    fig.savefig(SCR / "fig6_main.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig6_main.png")


if __name__ == "__main__":
    make_fig1(); make_fig2(); make_fig3(); make_fig4(); make_fig5(); make_fig6()
    print("ALL DONE")

# -*- coding: utf-8 -*-
"""
Extended analysis figure (Fig 7) for the KineticST article.

Panels
------
  A  Unsupervised-method correlation heatmap (performance vectors across the
     25 benchmark datasets; ARI/NMI/Coherence z-scored per dataset).
  B  Supervised-method correlation heatmap (mean-hits vectors, z-scored per
     dataset, across the 25 datasets).
  C  Venn overlap of the union of top-20 markers per type for the three
     best-ranked methods on STSP0007732 (equal-area circles).
  D  Raw vs dynamics-smoothed expression distributions of top KineticST-marker
     genes on STSP0007732 (target domain vs all other spots).
  E  KEGG enrichment of KineticST-marker genes (Enrichr, mouse libraries).
  F  GO Biological Process enrichment of the same genes.

Data strategy
-------------
  - Correlation panels use the existing analysis_results.json (25 datasets).
  - Marker-dependent panels re-run the 15 supervised methods on three
    representative datasets (Simulation seed 42, STSP0007732, Heart_S48) and
    cache markers / AUC / F1 / expression vectors in extra_marker_cache.json,
    so iterating on the figure does not re-run the expensive fits.

Output: fig7_extra.png (+ extra_marker_cache.json) in the scripts/ folder.
"""

import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCR = Path(__file__).parent.resolve()
ROOT = SCR.parent
for _p in (str(ROOT), str(SCR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE_PATH = SCR / "extra_marker_cache.json"
RESULTS_PATH = SCR / "analysis_results.json"
DATA_DIR = Path('dataset')  # TODO: set your dataset path

UNSUP_METHODS = ["DEIM+NMF", "HVG+NMF", "HVG+NMF\n(no spatial)",
                 "PCA+KMeans", "KineticST"]
UNSUP_TICK = {"DEIM+NMF": "DEIM+NMF", "HVG+NMF": "HVG+NMF",
              "HVG+NMF\n(no spatial)": "HVG+NMF\n(no spat.)",
              "PCA+KMeans": "PCA+KMeans", "KineticST": "KineticST"}
SUP_METHODS = [
    "DEIM-c(100)", "DEIM-c(200)", "DEIM-c(500)", "DEIM-c(1000)", "DEIM-c(2000)",
    "mRMR(500)", "Laplacian(500)", "HVG(500)", "DEIM-full(500)",
    "LogFC", "Wilcoxon", "T-test", "RF-importance", "LR-L1", "KineticST-marker",
]
KIN = "KineticST-marker"
VENN3_METHODS = ["KineticST-marker", "DEIM-full(500)", "LogFC"]
C_KIN = "#DD8452"
C_OTH = "#8DA0CB"


def panel_label(ax, letter, x=-0.02, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="right")


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# cache: expensive marker re-runs
# ---------------------------------------------------------------------------

def _classification_scores(Y, labels, markers, gene_names, n_splits=5, seed=0):
    """Macro AUC / F1 of a logistic classifier using each method's markers."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score
    from comprehensive_analysis import logcpm

    g2i = {str(g): i for i, g in enumerate(gene_names)}
    Yn = logcpm(Y)
    aucs, f1s = {}, {}
    for m, per_type in markers.items():
        feats = sorted({g for gs in per_type.values() for g in gs if g in g2i})
        if len(feats) < 5:
            aucs[m], f1s[m] = float("nan"), float("nan")
            continue
        idx = [g2i[g] for g in feats]
        X = np.asarray(Yn[:, idx], dtype=np.float64)
        cls, cnt = np.unique(labels, return_counts=True)
        keep = cls[cnt >= n_splits]
        mask = np.isin(labels, keep)
        X2, y2 = X[mask], labels[mask]
        keep_idx = {c: j for j, c in enumerate(keep)}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        proba = np.zeros((len(y2), len(keep)))
        pred = np.zeros(len(y2), dtype=int)
        for tr, te in skf.split(X2, y2):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X2[tr], y2[tr])
            pr = clf.predict_proba(X2[te])
            for j, c in enumerate(clf.classes_):
                proba[te, keep_idx[c]] = pr[:, j]
            pred[te] = clf.predict(X2[te])
        try:
            aucs[m] = float(roc_auc_score(y2, proba, multi_class="ovr",
                                          average="macro", labels=keep))
        except Exception:
            aucs[m] = float("nan")
        f1s[m] = float(f1_score(y2, pred, average="macro"))
    return aucs, f1s


def _run_one_dataset(name, Y, labels_raw, coords, genes, n_domains):
    from comprehensive_analysis import run_all_supervised, encode_labels
    from scipy import sparse as _sp

    if _sp.issparse(Y):
        Y = Y.toarray()
    Y = np.asarray(Y, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    labels_int, _ = encode_labels(labels_raw)

    t0 = time.time()
    res = run_all_supervised(Y, labels_int, np.asarray(genes), n_top=20,
                             true_markers=None, coords=coords,
                             n_domains=n_domains)
    markers = res["_markers"]
    print(f"  [{name}] supervised done in {time.time() - t0:.1f}s")
    auc, f1 = _classification_scores(Y, labels_int, markers, np.asarray(genes))
    return {"markers": markers, "auc": auc, "f1": f1,
            "labels": labels_int.tolist()}


def _kinetic_smooth_field(Y, coords, n_domains):
    """Fit KineticST with the exact benchmark configuration and return the
    dynamics-smoothed expression field (log-CPM scale)."""
    from kinetic_st import KineticST, KineticSTConfig

    n_spots = Y.shape[0]
    n_steps = 150 if n_spots <= 3000 else 80
    latent_dim = min(12, Y.shape[1], n_spots - 1)
    model = KineticST(KineticSTConfig(
        latent_dim=latent_dim, solver="stochastic", n_steps=n_steps,
        diffusion=0.03, alpha=0.1, beta=0.2, gamma=0.05,
        n_domains=n_domains, k_neighbors=6, run_wave_analysis=False,
        verbose=False, seed=0,
    ))
    model.fit(np.asarray(Y, dtype=np.float64), np.asarray(coords, dtype=np.float64))
    return model.decode_to_genes(model.fields_[-1].mean_state)


def _expression_panel_data(Y, labels_int, genes, smooth, kin_markers, n_genes=3):
    """Pick the top KineticST-marker gene of the three largest domains and
    store raw (log-CPM) and smoothed values per spot."""
    from comprehensive_analysis import logcpm

    cls, cnt = np.unique(labels_int, return_counts=True)
    big = cls[np.argsort(-cnt)][:n_genes]
    g2i = {str(g): i for i, g in enumerate(genes)}
    Yn = logcpm(Y)
    genes_sel, types_sel, raw, smo = [], [], {}, {}
    for c in big:
        tops = [g for g in kin_markers.get(str(int(c)), []) if g in g2i]
        if not tops:
            continue
        g = tops[0]
        i = g2i[g]
        genes_sel.append(g)
        types_sel.append(int(c))
        raw[g] = Yn[:, i].tolist()
        smo[g] = np.asarray(smooth[:, i], dtype=float).tolist()
    return {"genes": genes_sel, "types": types_sel, "raw": raw, "smooth": smo}


def _enrichment(kin_markers, top_per_type=10):
    """KEGG / GO-BP enrichment via Enrichr (mouse libraries; STSP genes are
    mouse symbols). Uses the union of each domain's top markers, restricted
    to well-annotated gene symbols (predicted Gm*/Rik* genes cannot map to
    pathways and only dilute the signal)."""
    import gseapy as gp

    pool = set()
    for genes_ in kin_markers.values():
        pool.update(g for g in genes_[:top_per_type] if _is_clean(g))
    if len(pool) < 20:  # fallback: all clean genes
        pool = {g for genes_ in kin_markers.values() for g in genes_ if _is_clean(g)}
    print(f"  enrichment pool: {len(pool)} clean genes")

    libs = gp.get_library_name(organism="mouse")
    kegg_lib = sorted([l for l in libs if "KEGG" in l])[-1]
    go_lib = sorted([l for l in libs if l.startswith("GO_Biological_Process")])[-1]
    enr = gp.enrichr(gene_list=sorted(pool),
                     gene_sets=[kegg_lib, go_lib],
                     organism="mouse", outdir=None)
    df = enr.results
    out = {}
    for lib, key in [(kegg_lib, "kegg"), (go_lib, "go")]:
        sub = df[df["Gene_set"] == lib].copy()
        sub["Adjusted P-value"] = sub["Adjusted P-value"].astype(float)
        sub = sub.sort_values("Adjusted P-value").head(10)
        with np.errstate(divide="ignore"):
            nlp = -np.log10(sub["Adjusted P-value"].values)
        finite = np.isfinite(nlp)
        if finite.any():
            nlp = np.where(finite, nlp, nlp[finite].max() + 1)
        else:
            nlp = np.full(len(nlp), 10.0)
        out[key] = {"lib": lib,
                    "terms": [t.split(" (")[0] for t in sub["Term"].tolist()],
                    "neglog10p": nlp.tolist(),
                    "overlap": sub["Overlap"].tolist(),
                    "combined": sub["Combined Score"].astype(float).tolist()}
    return out


# ---------------------------------------------------------------------------
# per-marker quality scores (single-gene AUC / best-threshold F1 on raw data)
# ---------------------------------------------------------------------------

_JUNK = re.compile(r"^(Gm\d+|AC\d+|.*Rik\d*$|.*-ps\d*$|RP\d+-\d+)")


def _is_clean(gene):
    """Skip predicted/RIKEN/pseudogene names for display purposes."""
    return not _JUNK.match(str(gene))


def _fast_auc(x, yb):
    """Mann-Whitney AUC of expression vector x for binary indicator yb."""
    from scipy.stats import rankdata

    n1 = int(yb.sum())
    n0 = len(x) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(x)
    return float((r[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _best_f1(x, yb):
    """Best F1 over expression thresholds (type-vs-rest)."""
    qs = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 99)))
    P = int(yb.sum())
    best = 0.0
    for t in qs:
        pred = x > t
        tp = int((pred & yb).sum())
        if tp == 0:
            continue
        fp = int((pred & ~yb).sum())
        fn = P - tp
        best = max(best, 2 * tp / (2 * tp + fp + fn))
    return best


def _per_gene_scores(Y, labels, markers, gene_names, n_top=20):
    """Mean per-marker AUC / max-F1 of each method's top markers on raw
    log-CPM expression (single-gene, type-vs-rest)."""
    from comprehensive_analysis import logcpm

    g2i = {str(g): i for i, g in enumerate(gene_names)}
    Yn = logcpm(Y)
    aucs, f1s = {}, {}
    for m, per_type in markers.items():
        a_all, f_all = [], []
        for l_str, genes_ in per_type.items():
            yb = labels == int(l_str)
            if yb.sum() < 3 or (~yb).sum() < 3:
                continue
            for g in genes_[:n_top]:
                i = g2i.get(g)
                if i is None:
                    continue
                x = np.asarray(Yn[:, i], dtype=float)
                a_all.append(_fast_auc(x, yb))
                f_all.append(_best_f1(x, yb))
        aucs[m] = float(np.nanmean(a_all)) if a_all else float("nan")
        f1s[m] = float(np.nanmean(f_all)) if f_all else float("nan")
    return aucs, f1s


def _load_dataset_arrays():
    """Reload (Y, labels_int, genes, coords, n_domains) for the three
    representative datasets without re-running any method."""
    from comprehensive_analysis import (load_h5ad_dataset, load_heart_slices,
                                        encode_labels)
    from spatial_nmf_deim import generate_simulation
    from scipy import sparse as _sp

    def _dense(Y):
        return Y.toarray() if _sp.issparse(Y) else np.asarray(Y, dtype=np.float64)

    sim = generate_simulation(K=10, G=5000, fold_change=8.0, dropout_rate=0.3,
                              N_per_type=300, delta=1.0, seed=42)
    li, _ = encode_labels(sim["labels"])
    out = {"Simulation": (_dense(sim["Y"]), li, np.asarray(sim["gene_names"]),
                          np.asarray(sim["coords"], dtype=np.float64),
                          len(np.unique(li)))}
    del sim

    ds = load_h5ad_dataset(DATA_DIR / "downloaded" / "Spatial_processed_STSP0007732.h5ad")
    li, _ = encode_labels(ds["labels"])
    out["STSP0007732"] = (_dense(ds["Y"]), li, np.asarray(ds["genes"]),
                          np.asarray(ds["coords"], dtype=np.float64),
                          ds["n_modules"])
    del ds

    heart = load_heart_slices(DATA_DIR / "mosta_embryo" / "mouse_E11.5_heart.h5ad",
                              slice_names=["slices_48"])[0]
    li, _ = encode_labels(heart["labels"])
    out["Heart_S48"] = (_dense(heart["Y"]), li, np.asarray(heart["genes"]),
                        np.asarray(heart["coords"], dtype=np.float64),
                        heart["n_modules"])
    return out


def augment_cache(cache):
    """Add per-marker AUC/F1, refresh the expression-panel genes (preferring
    well-annotated gene symbols) and re-run enrichment. Uses cached marker
    lists; the only refit is one KineticST run for the STSP smooth field."""
    arrays = _load_dataset_arrays()
    for name, (Y, labels, genes, _coords, _nd) in arrays.items():
        markers = cache["datasets"][name]["markers"]
        ga, gf = _per_gene_scores(Y, labels, markers, genes)
        cache["datasets"][name]["gene_auc"] = ga
        cache["datasets"][name]["gene_f1"] = gf
        print(f"  [{name}] per-gene AUC/F1 done")

    # ---- expression panel: prefer clean gene symbols ----
    Y, labels, genes, coords, n_dom = arrays["STSP0007732"]
    print("  [STSP0007732] refitting KineticST for smoothed field...")
    smooth = _kinetic_smooth_field(Y, coords, n_dom)
    kin = cache["datasets"]["STSP0007732"]["markers"][KIN]
    cls, cnt = np.unique(labels, return_counts=True)
    big = cls[np.argsort(-cnt)][:6]  # scan a few extra domains if needed
    g2i = {str(g): i for i, g in enumerate(genes)}
    from comprehensive_analysis import logcpm
    Yn = logcpm(Y)
    genes_sel, types_sel, raw, smo = [], [], {}, {}
    for c in big:
        if len(genes_sel) >= 3:
            break
        tops = [g for g in kin.get(str(int(c)), []) if g in g2i]
        if not tops:
            continue
        clean = [g for g in tops if _is_clean(g)]
        g = clean[0] if clean else tops[0]
        i = g2i[g]
        genes_sel.append(g)
        types_sel.append(int(c))
        raw[g] = Yn[:, i].tolist()
        smo[g] = np.asarray(smooth[:, i], dtype=float).tolist()
    cache["datasets"]["STSP0007732"]["expr"] = {
        "genes": genes_sel, "types": types_sel, "raw": raw, "smooth": smo}
    print(f"  expr panel genes: {genes_sel}")

    # ---- enrichment (fixed organism) ----
    try:
        cache["enrich"] = _enrichment(kin)
        print("  enrichment done")
    except Exception as e:
        print(f"  enrichment FAILED: {e}")
        cache["enrich"] = None

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"Cache updated: {CACHE_PATH}")
    return cache


def build_cache():
    from comprehensive_analysis import load_h5ad_dataset, load_heart_slices
    from spatial_nmf_deim import generate_simulation

    cache = {"datasets": {}, "enrich": None}

    # ---- 1. Simulation seed 42 ----
    print(">>> Simulation (seed=42)")
    sim = generate_simulation(K=10, G=5000, fold_change=8.0, dropout_rate=0.3,
                              N_per_type=300, delta=1.0, seed=42)
    cache["datasets"]["Simulation"] = _run_one_dataset(
        "Simulation", sim["Y"], sim["labels"], sim["coords"],
        sim["gene_names"], n_domains=len(np.unique(sim["labels"])))
    del sim

    # ---- 2. STSP0007732 (mouse brain section) ----
    print(">>> STSP0007732")
    ds = load_h5ad_dataset(DATA_DIR / "downloaded" / "Spatial_processed_STSP0007732.h5ad")
    entry = _run_one_dataset("STSP0007732", ds["Y"], ds["labels"], ds["coords"],
                             ds["genes"], n_domains=ds["n_modules"])
    # dynamics-smoothed field for the expression panel
    from scipy import sparse as _sp
    Y = ds["Y"].toarray() if _sp.issparse(ds["Y"]) else np.asarray(ds["Y"], dtype=np.float64)
    print("  [STSP0007732] fitting KineticST for smoothed field...")
    smooth = _kinetic_smooth_field(Y, ds["coords"], ds["n_modules"])
    entry["expr"] = _expression_panel_data(
        Y, np.asarray(entry["labels"]), np.asarray(ds["genes"]), smooth,
        entry["markers"][KIN])
    cache["datasets"]["STSP0007732"] = entry

    # ---- 3. Heart slice 48 ----
    print(">>> Heart_S48")
    heart = load_heart_slices(DATA_DIR / "mosta_embryo" / "mouse_E11.5_heart.h5ad",
                              slice_names=["slices_48"])[0]
    cache["datasets"]["Heart_S48"] = _run_one_dataset(
        "Heart_S48", heart["Y"], heart["labels"], heart["coords"],
        heart["genes"], n_domains=heart["n_modules"])

    # ---- 4. Enrichment on KineticST-marker genes from STSP0007732 ----
    print(">>> Enrichment (Enrichr, mouse)")
    try:
        kin = cache["datasets"]["STSP0007732"]["markers"][KIN]
        cache["enrich"] = _enrichment(kin)
        print("  enrichment done")
    except Exception as e:
        print(f"  enrichment FAILED: {e}")
        cache["enrich"] = None

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"Cache written: {CACHE_PATH}")
    return cache


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def _zscore_per_dataset(mat):
    """mat: (n_datasets, n_methods); z-score each dataset row."""
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd


def corr_heatmap(ax, M, labels, letter, title):
    v = np.nanmax(np.abs(M[np.isfinite(M)])) if np.isfinite(M).any() else 1.0
    v = max(v, 0.5)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal")
    n = len(labels)
    if n <= 8:  # annotate only small matrices; dense ones stay color-only
        for i in range(n):
            for j in range(n):
                val = M[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(val) > 0.55 * v else "black")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=5.6 if n > 8 else 6.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=5.6 if n > 8 else 6.5)
    ax.set_title(title, fontsize=9)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Pearson r", fontsize=6.5)
    panel_label(ax, letter, x=-0.02, y=1.22)


def corr_unsup_panel(ax, datasets, letter="A"):
    metric_mats = []
    for metric in ("ARI", "NMI", "Coherence"):
        mat = np.array([[d["unsupervised"][metric].get(m, np.nan)
                         for m in UNSUP_METHODS] for d in datasets], dtype=float)
        metric_mats.append(_zscore_per_dataset(mat))
    V = np.concatenate(metric_mats, axis=0)  # (75, 5)
    M = np.corrcoef(V.T)
    corr_heatmap(ax, M, [UNSUP_TICK[m] for m in UNSUP_METHODS], letter,
                 "Unsupervised methods")


def corr_sup_panel(ax, datasets, letter="B"):
    mat = np.array([[d["supervised"]["mean_hits"].get(m, np.nan)
                     for m in SUP_METHODS] for d in datasets], dtype=float)
    V = _zscore_per_dataset(mat)
    M = np.corrcoef(V.T)
    corr_heatmap(ax, M, SUP_METHODS, letter,
                 "Supervised methods (25 datasets)")


def venn_panel(ax, cache, letter="C"):
    from matplotlib_venn import venn3
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

    kin = cache["datasets"]["STSP0007732"]["markers"]
    sets = []
    for m in VENN3_METHODS:
        sets.append({g for gs in kin.get(m, {}).values() for g in gs})
    # unweighted layout: equal circles, positions independent of set sizes
    v = venn3(sets, set_labels=("KineticST-marker", "DEIM-full(500)", "LogFC"),
              ax=ax, set_colors=(C_KIN, "#4C72B0", "#55A868"), alpha=0.65,
              layout_algorithm=DefaultLayoutAlgorithm(
                  fixed_subset_sizes=(1, 1, 1, 1, 1, 1, 1)))
    if v is not None:
        for t in v.set_labels:
            if t is not None:
                t.set_fontsize(6.8)
        for t in v.subset_labels:
            if t is not None:
                t.set_fontsize(7)
    ax.set_title("Marker overlap (STSP0007732)", fontsize=9, pad=8)
    panel_label(ax, letter, x=-0.02, y=1.22)


def metric_panel(ax, cache, key, letter, xlabel):
    names = list(cache["datasets"].keys())
    means, sems = [], []
    for m in SUP_METHODS:
        vals = np.array([cache["datasets"][d][key].get(m, np.nan)
                         for d in names], dtype=float)
        means.append(np.nanmean(vals))
        sems.append(np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals))))
    means = np.array(means)
    sems = np.array(sems)
    order = np.argsort(means)
    ms = [SUP_METHODS[i] for i in order]
    mu, se = means[order], sems[order]
    colors = [C_KIN if m == KIN else C_OTH for m in ms]
    y = np.arange(len(ms))
    ax.barh(y, mu, xerr=se, color=colors, edgecolor="none", alpha=0.92,
            height=0.62, error_kw=dict(lw=0.8), capsize=2)
    pad = np.nanmax(mu) * 0.015
    for yi, v, s in zip(y, mu, se):
        ax.text(v + s + pad, yi, f"{v:.2f}", va="center", fontsize=6.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ms, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(right=np.nanmax(mu + se) * 1.18)
    _despine(ax)
    ax.tick_params(labelsize=8.5)
    panel_label(ax, letter, x=-0.05, y=1.07)


def expr_panel(ax, cache, j, letter=None):
    expr = cache["datasets"]["STSP0007732"]["expr"]
    genes, types = expr["genes"], expr["types"]
    if j >= len(genes):
        ax.axis("off")
        return
    g = genes[j]
    t = types[j]
    labels = np.asarray(cache["datasets"]["STSP0007732"]["labels"])
    raw = np.asarray(expr["raw"][g])
    smo = np.asarray(expr["smooth"][g])
    in_t = labels == t
    data = [raw[~in_t], raw[in_t], smo[~in_t], smo[in_t]]
    pos = [1, 2, 3.6, 4.6]
    colors = ["#BBBBBB", "#4C72B0", "#BBBBBB", C_KIN]
    vp = ax.violinplot(data, positions=pos, widths=0.85, showmeans=True,
                       showextrema=False)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.75)
        body.set_edgecolor("#444444")
        body.set_linewidth(0.6)
    vp["cmeans"].set_color("black")
    vp["cmeans"].set_linewidth(1.0)
    # clip the y-axis so raw outliers do not flatten the violins
    ymax = max(np.quantile(np.concatenate([raw, smo]), 0.995), 1e-6)
    ax.set_ylim(bottom=min(0.0, np.quantile(np.concatenate([raw, smo]), 0.001)),
                top=ymax * 1.08)
    ax.set_xticks(pos)
    ax.set_xticklabels(["rest", f"dom. {t}", "rest", f"dom. {t}"], fontsize=6.8)
    ax.text(0.28, -0.17, "Raw", transform=ax.transAxes, ha="center",
            fontsize=8, color="#333333")
    ax.text(0.78, -0.17, "KineticST-smoothed", transform=ax.transAxes,
            ha="center", fontsize=8, color="#333333")
    ax.set_title(f"{g} (domain {t} marker)", fontsize=9)
    ax.set_ylabel("Expression (log-CPM)" if j == 0 else "", fontsize=8.5)
    _despine(ax)
    ax.tick_params(labelsize=7.5)
    if letter:
        panel_label(ax, letter, x=-0.02, y=1.14)


def enrich_panel(ax, cache, key, letter, title):
    enr = cache.get("enrich")
    if not enr or key not in enr:
        ax.text(0.5, 0.5, "Enrichment unavailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.axis("off")
        panel_label(ax, letter)
        return
    e = enr[key]
    terms = e["terms"][::-1]
    nlp = np.asarray(e["neglog10p"][::-1])
    sizes = np.array([int(o.split("/")[0]) for o in e["overlap"][::-1]])
    y = np.arange(len(terms))
    ax.scatter(nlp, y, s=20 + 90 * sizes / max(sizes), color="#4C72B0",
               alpha=0.85, edgecolor="#333333", linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([t if len(t) <= 60 else t[:57] + "..." for t in terms],
                       fontsize=7.5)
    ax.set_xlabel(r"$-\log_{10}$ adjusted P", fontsize=9)
    xlo = min(nlp) * 0.8 if len(nlp) and min(nlp) > 0.15 else 0
    ax.set_xlim(left=xlo, right=max(nlp) * 1.18 if len(nlp) else 1)
    ax.set_title(title, fontsize=9)
    ax.yaxis.grid(True, color="#EFEFEF", lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.tick_params(labelsize=8)
    # size legend (overlap counts)
    smin, smax = int(sizes.min()), int(sizes.max())
    for k, s in [(smin, 20 + 90 * smin / max(sizes)),
                 (smax, 20 + 90 * smax / max(sizes))]:
        ax.scatter([], [], s=s, color="#4C72B0", alpha=0.85,
                   edgecolor="#333333", linewidth=0.5, label=f"{k} genes")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right",
              handletextpad=0.4, borderaxespad=0.2)
    panel_label(ax, letter, x=-0.02, y=1.13)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_fig7(cache):
    datasets = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["datasets"]

    fig = plt.figure(figsize=(9.2, 13.2), dpi=200)
    gs = fig.add_gridspec(4, 12, height_ratios=[2.4, 2.5, 2.2, 2.2],
                          hspace=0.42, wspace=1.4,
                          top=0.97, bottom=0.04, left=0.04, right=0.99)

    # Row 0: manually placed axes so A/B/C share exactly the same top and
    # height, with wide gaps that rule out any mutual occlusion
    corr_unsup_panel(fig.add_axes([0.065, 0.800, 0.190, 0.130]), datasets)
    corr_sup_panel(fig.add_axes([0.390, 0.800, 0.190, 0.130]), datasets)
    venn_panel(fig.add_axes([0.710, 0.800, 0.190, 0.130]), cache)

    for j in range(3):
        expr_panel(fig.add_subplot(gs[1, 4 * j:4 * j + 4]), cache, j,
                   letter="D" if j == 0 else None)

    # E and F on their own rows so long term labels never overlap
    enrich_panel(fig.add_subplot(gs[2, 0:9]), cache, "kegg", "E",
                 "KEGG enrichment of KineticST markers (STSP0007732)")
    enrich_panel(fig.add_subplot(gs[3, 0:9]), cache, "go", "F",
                 "GO Biological Process enrichment of KineticST markers")

    out = SCR / "fig7_extra.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    t0 = time.time()
    if CACHE_PATH.exists():
        print(f"Loading cache: {CACHE_PATH}")
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    else:
        cache = build_cache()
    if any("gene_auc" not in d for d in cache["datasets"].values()):
        print("Augmenting cache (per-marker scores / expression / enrichment)...")
        cache = augment_cache(cache)
    elif "--enrich" in sys.argv or cache.get("enrich") is None:
        # fast path: redo only the enrichment from cached marker lists
        try:
            kin = cache["datasets"]["STSP0007732"]["markers"][KIN]
            cache["enrich"] = _enrichment(kin)
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False),
                                  encoding="utf-8")
            print("enrichment refreshed")
        except Exception as e:
            print(f"enrichment FAILED: {e}")
            cache["enrich"] = None
    make_fig7(cache)

    # ---- console summary for article text ----
    print("\n==== summary ====")
    for key, lab in [("gene_auc", "per-marker AUC"), ("gene_f1", "per-marker F1")]:
        means = {m: np.nanmean([cache["datasets"][d][key].get(m, np.nan)
                                for d in cache["datasets"]]) for m in SUP_METHODS}
        top = sorted(means.items(), key=lambda kv: -kv[1])[:3]
        print(f"{lab}: " + ", ".join(f"{m}={v:.3f}" for m, v in top))
    if cache.get("enrich"):
        print("KEGG top:", cache["enrich"]["kegg"]["terms"][:3])
        print("GO top:", cache["enrich"]["go"]["terms"][:3])
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

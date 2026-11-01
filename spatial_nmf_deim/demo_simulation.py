"""
Demo: Supervised + Unsupervised analysis on simulated spatial transcriptomics data.

Demonstrates the full SpatialNMF-DEIM package:
  1. Generate simulated ST data with known ground truth
  2. Supervised DEIM: discover marker genes given labels
  3. Unsupervised DEIM: discover spatial gene modules without labels
  4. Evaluate and compare both approaches

Usage:
    python demo_simulation.py
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from spatial_nmf_deim import (
    SpatialNMFDEIM,
    SupervisedDEIM,
    generate_simulation,
    coords_to_adjacency,
    compute_module_genes,
)


def plot_spatial(coords, labels, title, save_path, K):
    """Plot spatial layout colored by labels."""
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    cmap = plt.colormaps.get_cmap('tab20')
    n_tissue = K - 1
    for k in range(K):
        mask = labels == k
        if k == K - 1:
            axes.scatter(coords[mask, 0], coords[mask, 1], c='lightgray',
                        s=3, marker='s', label='Background', alpha=0.4)
        else:
            axes.scatter(coords[mask, 0], coords[mask, 1],
                        c=[cmap(k / K)], s=8, marker='s',
                        label=f'Type {k+1}', alpha=0.7, edgecolors='none')
    axes.set_aspect('equal')
    axes.set_title(title, fontsize=13)
    axes.legend(loc='upper right', fontsize=5.5, ncol=2, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_nmf_domains(coords, W_raw, K, save_path):
    """Plot NMF-discovered spatial domains."""
    domains = np.argmax(W_raw, axis=1)
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    cmap = plt.colormaps.get_cmap('tab20')
    for k in range(K):
        mask = domains == k
        axes.scatter(coords[mask, 0], coords[mask, 1],
                    c=[cmap(k / K)], s=8, marker='s',
                    label=f'Module {k+1}', alpha=0.7, edgecolors='none')
    axes.set_aspect('equal')
    axes.set_title('Unsupervised: NMF-discovered domains', fontsize=13)
    axes.legend(loc='upper right', fontsize=5.5, ncol=2, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def evaluate_supervised(sd, true_markers, K):
    """Evaluate supervised DEIM against ground truth."""
    results = sd.evaluate(true_markers)
    n_tissue = K - 1
    print(f"\n  Supervised DEIM (pool={sd.pool_size}, top-{sd.n_top}):")
    print(f"    Mean hits/type: {results['mean_hits']:.2f}")
    print(f"    Total hits: {results['total_hits']}/{n_tissue * sd.n_top}")
    print(f"    Per type: {results['per_type']}")
    return results


def evaluate_unsupervised(W_raw, true_labels, K):
    """Evaluate unsupervised NMF domains against true labels."""
    pred = np.argmax(W_raw, axis=1)
    # Only evaluate tissue spots (not background)
    tissue_mask = true_labels < (K - 1)
    if tissue_mask.sum() == 0:
        return {}

    ari = adjusted_rand_score(true_labels[tissue_mask], pred[tissue_mask])
    nmi = normalized_mutual_info_score(true_labels[tissue_mask], pred[tissue_mask])

    print(f"\n  Unsupervised NMF (n_modules={K-1}):")
    print(f"    ARI:  {ari:.4f}")
    print(f"    NMI:  {nmi:.4f}")
    return {'ARI': ari, 'NMI': nmi, 'pred': pred}


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # ===== Parameters =====
    K = 10
    G = 5000
    FC = 8.0
    dropout = 0.3
    seed = 42

    print("=" * 60)
    print("SpatialNMF-DEIM Demo: Simulated Data")
    print("=" * 60)

    # ===== 1. Generate simulated data =====
    print(f"\n[1] Generating simulation: K={K}, G={G}, FC={FC}, dropout={dropout}")
    sim = generate_simulation(
        K=K, G=G, fold_change=FC, dropout_rate=dropout,
        N_per_type=300, delta=1.0, seed=seed
    )
    Y = sim['Y']
    labels = sim['labels']
    coords = sim['coords']
    gene_names = sim['gene_names']
    true_markers = sim['true_markers']

    print(f"    N={Y.shape[0]} spots, G={Y.shape[1]} genes, {K} types (incl. background)")

    # Plot ground truth spatial layout
    plot_spatial(coords, labels, f'Ground Truth (K={K})',
                 os.path.join(output_dir, 'demo_01_ground_truth.png'), K)

    # ===== 2. Supervised DEIM =====
    print(f"\n[2] Supervised DEIM (marker gene discovery)")
    sd = SupervisedDEIM(pool_size=2000, n_top=20)
    sd.fit(Y, labels, gene_names)

    # Show some discovered markers
    n_tissue = K - 1
    print(f"\n  Discovered markers (top 5 per type):")
    for k in range(min(5, n_tissue)):
        markers = sd.markers_[str(k)][:5]
        known = true_markers.get(str(k), set())
        hits = [g for g in markers if g in known]
        print(f"    Type {k+1}: {markers}  (hits: {hits})")

    sup_results = evaluate_supervised(sd, true_markers, K)

    # ===== 3. Unsupervised SpatialNMF-DEIM =====
    print(f"\n[3] Unsupervised SpatialNMF-DEIM (spatial module discovery)")
    n_modules = K - 1  # discover tissue modules
    model = SpatialNMFDEIM(
        n_modules=n_modules,
        n_genes=500,
        deim_method='deim_leverage',
        lambda_spatial=0.1,
        spatial_method='grid',
        max_iter=200,
        verbose=True,
    )
    model.fit(Y, coords, gene_names)

    # Evaluate spatial domains
    unsup_results = evaluate_unsupervised(model.W_, labels, K)

    # Show discovered module genes
    print(f"\n  Module genes (top 5 per module):")
    modules = model.get_module_genes(gene_names, n_top=5)
    for k in range(min(5, n_modules)):
        genes = modules[k]['genes'][:5]
        scores = modules[k]['scores'][:5]
        print(f"    Module {k+1}: {list(zip(genes, [f'{s:.3f}' for s in scores]))}")

    # Plot NMF-discovered domains
    plot_nmf_domains(coords, model.W_, K,
                    os.path.join(output_dir, 'demo_02_nmf_domains.png'))

    # ===== 4. Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Supervised DEIM:  {sup_results['mean_hits']:.2f} mean hits/type (top-20)")
    if 'ARI' in unsup_results:
        print(f"  Unsupervised NMF: ARI={unsup_results['ARI']:.4f}, NMI={unsup_results['NMI']:.4f}")
    print(f"\n  Files saved to: {output_dir}")
    print(f"    demo_01_ground_truth.png")
    print(f"    demo_02_nmf_domains.png")


if __name__ == '__main__':
    main()
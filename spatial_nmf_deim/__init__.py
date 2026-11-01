"""
SpatialNMF-DEIM: Spatial Gene Module Discovery
via DEIM-based Gene Selection and Graph-regularized NMF.

Supports two modes:
  - Unsupervised: discover spatial gene modules (SpatialNMFDEIM)
  - Supervised: discover marker genes given type labels (SupervisedDEIM)
"""

__version__ = "0.2.0"
__author__ = "SpatialNMF-DEIM"

from spatial_nmf_deim.core.deim import deim_select, compute_deim_scores
from spatial_nmf_deim.core.nmf_solver import GraphRegularizedNMF, compute_module_genes
from spatial_nmf_deim.core.pipeline import SpatialNMFDEIM
from spatial_nmf_deim.core.supervised import SupervisedDEIM
from spatial_nmf_deim.core.simulation import generate_simulation
from spatial_nmf_deim.core.graph import coords_to_adjacency

__all__ = [
    "SpatialNMFDEIM",
    "SupervisedDEIM",
    "GraphRegularizedNMF",
    "deim_select",
    "compute_deim_scores",
    "coords_to_adjacency",
    "generate_simulation",
    "compute_module_genes",
    "__version__",
]
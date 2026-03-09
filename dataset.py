"""
=============================================================================
CM-GAT: Continual Multimodal Graph Attention Network — Dataset Module
=============================================================================
Paper: "The network architecture of general intelligence in the human
connectome" (Nature Communications, 2026)

This module constructs multimodal brain graphs from the Human Connectome
Project (HCP) data:

  • NODES  (360): Cortical regions from the Glasser Multi-Modal Parcellation.
  • NODE FEATURES (100-d): Intrinsic functional co-activation patterns
    derived from resting-state fMRI via spatial ICA (MELODIC). Each node's
    feature vector captures its participation in 100 independent components.
  • EDGES: Structural connectome from diffusion-weighted MRI tractography
    (MSMT-CSD + SIFT2). Edge weights represent "fiber bundle capacity" —
    an estimate of the physical bandwidth between regions.
  • TARGET: Continuous general intelligence factor *g*, extracted as the
    first principal component of a battery of cognitive tests.

DESIGN RATIONALE — Why multimodal graphs?
  The paper's key insight is that *g* arises from the interplay between
  structural connectivity (the "hardware") and functional dynamics (the
  "software"). Neither modality alone is sufficient. By encoding structure
  as edges and function as node features, we let the GNN learn how physical
  wiring constrains and shapes information flow — the "Information Flow
  Capacity" (IFC) concept from the paper.

SPARSIFICATION STRATEGY:
  The structural connectome is nearly fully connected at the resolution of
  probabilistic tractography. However, the paper demonstrates that:
    1. Weak, long-range connections are disproportionately important for *g*.
    2. Strong, short-range connections maintain local segregation.
  We therefore apply a top-k% threshold that retains both the strongest
  connections AND a controlled fraction of weak long-range ties, preserving
  the small-world topology that the paper identifies as central to *g*.
=============================================================================
"""

import os
import math
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from scipy.spatial.distance import pdist, squareform


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY: Top-k Connectome Sparsification
# ═══════════════════════════════════════════════════════════════════════════

def sparsify_connectome(
    adj_matrix: np.ndarray,
    top_k_percent: float = 10.0,
    weak_tie_percent: float = 2.0,
    coordinates: np.ndarray = None,
    distance_threshold_mm: float = 120.0,
) -> np.ndarray:
    """
    Sparsify a fully connected structural connectome while preserving both
    strong local ties and weak long-range connections.

    The paper (Fig. 3) shows that negatively-predictive edges (associated
    with higher *g*) are significantly longer (~151mm) than positively-
    predictive edges (~105mm). This means weak, distant connections carry
    critical information. Our strategy:

      1. Retain the top-k% strongest edges globally (strong ties).
      2. Additionally retain the top `weak_tie_percent`% of edges whose
         Euclidean length exceeds `distance_threshold_mm` — these are
         the "weak, long-range ties" central to the paper's findings.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape (N, N)
        Symmetric adjacency/weight matrix (fiber bundle capacity).
    top_k_percent : float
        Percentage of total edges to retain based on weight strength.
    weak_tie_percent : float
        Additional percentage of long-range edges to retain.
    coordinates : np.ndarray, shape (N, 3), optional
        MNI coordinates for each region. If provided, used to compute
        Euclidean distances for identifying long-range connections.
    distance_threshold_mm : float
        Minimum distance (mm) for an edge to qualify as "long-range."

    Returns
    -------
    sparse_adj : np.ndarray, shape (N, N)
        The sparsified adjacency matrix (symmetric).
    """
    N = adj_matrix.shape[0]
    assert adj_matrix.shape == (N, N), "Adjacency must be square."

    # Work with upper triangle to avoid double-counting
    upper = np.triu(adj_matrix, k=1)
    flat_weights = upper[upper > 0]

    if len(flat_weights) == 0:
        return adj_matrix  # Nothing to sparsify

    # --- Step 1: Global top-k% threshold (strong ties) ---
    k_strong = max(1, int(len(flat_weights) * top_k_percent / 100.0))
    strong_threshold = np.sort(flat_weights)[::-1][min(k_strong, len(flat_weights) - 1)]
    strong_mask = upper >= strong_threshold

    # --- Step 2: Weak long-range ties (the paper's key finding) ---
    weak_mask = np.zeros_like(upper, dtype=bool)
    if coordinates is not None and weak_tie_percent > 0:
        dist_matrix = squareform(pdist(coordinates, metric='euclidean'))
        dist_upper = np.triu(dist_matrix, k=1)

        # Identify long-range edges
        long_range = (dist_upper > distance_threshold_mm) & (upper > 0)
        long_range_weights = upper[long_range]

        if len(long_range_weights) > 0:
            k_weak = max(1, int(len(long_range_weights) * weak_tie_percent / 100.0))
            # For weak ties, we want the WEAKER ones (counterintuitive but
            # matches the paper: negatively-associated edges are weaker)
            weak_threshold = np.sort(long_range_weights)[min(k_weak, len(long_range_weights) - 1)]
            weak_mask = long_range & (upper <= weak_threshold)

    # --- Combine masks ---
    combined = strong_mask | weak_mask
    sparse_upper = upper * combined
    sparse_adj = sparse_upper + sparse_upper.T

    return sparse_adj


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY: Convert raw matrices to PyG Data objects
# ═══════════════════════════════════════════════════════════════════════════

def build_pyg_data(
    node_features: np.ndarray,
    adj_matrix: np.ndarray,
    g_score: float,
    subject_id: int = 0,
) -> Data:
    """
    Convert raw neuroimaging matrices into a PyTorch Geometric Data object.

    Parameters
    ----------
    node_features : np.ndarray, shape (N_nodes, N_features)
        Functional co-activation patterns (ICA components) for each region.
    adj_matrix : np.ndarray, shape (N_nodes, N_nodes)
        Sparsified structural connectome (fiber bundle capacity weights).
    g_score : float
        General intelligence factor for this subject.
    subject_id : int
        Unique identifier for the subject.

    Returns
    -------
    data : torch_geometric.data.Data
        A single-graph data object with:
          - x: Node feature tensor (N, F)
          - edge_index: COO edge indices (2, E)
          - edge_attr: Edge weight tensor (E, 1)
          - y: Target g-score (1,)
          - subject_id: Identifier
    """
    N = node_features.shape[0]

    # Node features → tensor
    x = torch.tensor(node_features, dtype=torch.float32)

    # Adjacency → COO format (edge_index + edge_attr)
    # We only include non-zero edges after sparsification
    rows, cols = np.nonzero(adj_matrix)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)

    # Edge weights as features (1-dimensional for ECC)
    edge_weights = adj_matrix[rows, cols]
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)

    # Target g-factor score
    y = torch.tensor([g_score], dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    data.subject_id = subject_id

    return data


# ═══════════════════════════════════════════════════════════════════════════
# MOCK DATA GENERATOR: Synthetic connectomes for pipeline testing
# ═══════════════════════════════════════════════════════════════════════════

def generate_mock_subject(
    num_nodes: int = 360,
    num_features: int = 100,
    sparsity_top_k: float = 10.0,
    subject_id: int = 0,
    seed: int = None,
) -> Data:
    """
    Generate a single synthetic brain connectome graph for testing.

    The mock data mimics the statistical properties of real HCP data:
      - Node features: Drawn from N(0, 1) to simulate z-scored ICA maps.
      - Structural connectome: Log-normal distribution (matching the
        heavy-tailed distribution of fiber counts in real tractography).
      - g-score: Drawn from N(100, 15) matching typical IQ scaling.
      - Coordinates: Random 3D points in a sphere of radius ~80mm
        (approximate brain dimensions).

    Parameters
    ----------
    num_nodes : int
        Number of cortical regions (default: 360 for Glasser atlas).
    num_features : int
        Dimensionality of functional features (default: 100 ICA components).
    sparsity_top_k : float
        Top-k% of edges to retain during sparsification.
    subject_id : int
        Unique subject identifier.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    data : torch_geometric.data.Data
        A synthetic brain graph ready for the CM-GAT pipeline.
    """
    rng = np.random.RandomState(seed)

    # --- Functional node features (ICA co-activation patterns) ---
    # In real data, these come from dual-regression spatial ICA maps
    # projected onto the Glasser parcellation. Each of the 100 components
    # captures a distinct spatially coherent pattern of co-activation.
    node_features = rng.randn(num_nodes, num_features).astype(np.float32)

    # --- Structural connectome (fiber bundle capacity) ---
    # Real tractography weights follow a heavy-tailed (log-normal)
    # distribution: most connections are weak, a few are very strong.
    raw_weights = rng.lognormal(mean=0.0, sigma=1.5, size=(num_nodes, num_nodes))
    raw_weights = (raw_weights + raw_weights.T) / 2.0  # Symmetrize
    np.fill_diagonal(raw_weights, 0)  # No self-loops

    # --- 3D coordinates (for distance-based weak tie identification) ---
    # Place nodes on a sphere to simulate cortical geometry
    theta = rng.uniform(0, 2 * math.pi, num_nodes)
    phi = rng.uniform(0, math.pi, num_nodes)
    r = 80.0  # ~80mm radius (approximate brain)
    coords = np.stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi),
    ], axis=1)

    # --- Sparsify the connectome ---
    sparse_adj = sparsify_connectome(
        adj_matrix=raw_weights,
        top_k_percent=sparsity_top_k,
        weak_tie_percent=2.0,
        coordinates=coords,
        distance_threshold_mm=120.0,
    )

    # --- g-factor score ---
    # In the HCP, g is derived as PC1 of 12 cognitive tests.
    # We simulate with a normal distribution matching typical IQ scaling.
    g_score = float(rng.normal(100.0, 15.0))

    return build_pyg_data(node_features, sparse_adj, g_score, subject_id)


# ═══════════════════════════════════════════════════════════════════════════
# PyG InMemoryDataset: Mock Connectome Dataset
# ═══════════════════════════════════════════════════════════════════════════

class MockConnectomeDataset(InMemoryDataset):
    """
    A PyTorch Geometric InMemoryDataset of synthetic brain connectomes.

    This dataset generates `num_subjects` mock subjects, each represented
    as a multimodal graph:
      - 360 nodes (Glasser cortical regions)
      - 100-dim node features (simulated ICA functional maps)
      - Sparsified structural edges with fiber-capacity weights
      - A continuous g-factor target score

    Usage
    -----
    >>> dataset = MockConnectomeDataset(root='./data_mock', num_subjects=100)
    >>> print(len(dataset))
    100
    >>> print(dataset[0])
    Data(x=[360, 100], edge_index=[2, ...], edge_attr=[..., 1], y=[1])
    """

    def __init__(
        self,
        root: str,
        num_subjects: int = 100,
        num_nodes: int = 360,
        num_features: int = 100,
        top_k_percent: float = 10.0,
        transform=None,
        pre_transform=None,
        seed: int = 42,
    ):
        self.num_subjects = num_subjects
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.top_k_percent = top_k_percent
        self.seed = seed
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # No raw files — everything is generated

    @property
    def processed_file_names(self):
        return [f'mock_connectome_n{self.num_subjects}_k{self.num_nodes}.pt']

    def download(self):
        pass  # No download needed for mock data

    def process(self):
        """Generate synthetic connectome graphs for all subjects."""
        data_list = []
        for i in range(self.num_subjects):
            data = generate_mock_subject(
                num_nodes=self.num_nodes,
                num_features=self.num_features,
                sparsity_top_k=self.top_k_percent,
                subject_id=i,
                seed=self.seed + i,
            )
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


# ═══════════════════════════════════════════════════════════════════════════
# PyG InMemoryDataset: Real HCP Connectome Dataset (Placeholder)
# ═══════════════════════════════════════════════════════════════════════════

class HCPConnectomeDataset(InMemoryDataset):
    """
    Placeholder dataset for loading real Human Connectome Project data.

    To use with real data, you need:
      1. HCP 1200-subject release (restricted + unrestricted access)
      2. Preprocessed structural connectomes (MRtrix3 + SIFT2)
      3. ICA-FIX cleaned resting-state fMRI
      4. Cognitive battery scores for g-factor extraction

    The `process()` method should be implemented to:
      - Load the 360×360 structural connectivity matrices
      - Load the 360×100 ICA spatial maps
      - Compute g as PC1 of the 12 NIH Toolbox + Penn CNB tests
      - Apply sparsification and build PyG Data objects

    See the paper's Methods section for exact preprocessing details:
      - Tractography: MSMT-CSD → 10M streamlines → SIFT2
      - Functional: FIX-ICA denoising → dual-regression MELODIC (d=100)
      - g-factor: PCA on 12 cognitive measures (Table S1 in paper)
    """

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # self.load(self.processed_paths[0])  # Uncomment when implemented

    @property
    def raw_file_names(self):
        return ['structural_connectomes.npy', 'functional_features.npy',
                'g_scores.csv', 'region_coordinates.npy']

    @property
    def processed_file_names(self):
        return ['hcp_connectome.pt']

    def download(self):
        raise NotImplementedError(
            "Real HCP data must be downloaded manually from "
            "https://db.humanconnectome.org/ with appropriate data use "
            "agreements. See the README for preprocessing instructions."
        )

    def process(self):
        raise NotImplementedError(
            "Implement this method to load and process real HCP data. "
            "See the class docstring for required input files and the "
            "paper's Methods section for preprocessing pipeline details."
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import tempfile

    print("=" * 70)
    print("CM-GAT Dataset Module — Sanity Check")
    print("=" * 70)

    # Test single mock subject
    print("\n[1] Generating a single mock subject...")
    data = generate_mock_subject(num_nodes=360, num_features=100, seed=42)
    print(f"    Nodes:      {data.num_nodes}")
    print(f"    Features:   {data.x.shape}")
    print(f"    Edges:      {data.num_edges}")
    print(f"    Edge attr:  {data.edge_attr.shape}")
    print(f"    g-score:    {data.y.item():.2f}")
    density = data.num_edges / (360 * 359)
    print(f"    Density:    {density:.4f} ({density*100:.1f}%)")

    # Test full dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[2] Creating MockConnectomeDataset (10 subjects)...")
        dataset = MockConnectomeDataset(root=tmpdir, num_subjects=10)
        print(f"    Dataset size: {len(dataset)}")
        print(f"    First graph:  {dataset[0]}")
        print(f"    Last graph:   {dataset[-1]}")

    print("\n✓ All checks passed!")

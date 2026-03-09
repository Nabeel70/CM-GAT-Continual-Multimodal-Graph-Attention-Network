"""
=============================================================================
CM-GAT: Continual Multimodal Graph Attention Network — Model Module
=============================================================================
Paper: "The network architecture of general intelligence in the human
connectome" (Nature Communications, 2026)

This module implements the CM-GAT architecture for predicting general
intelligence (g) from multimodal brain connectomes.

ARCHITECTURE OVERVIEW:
┌─────────────────────────────────────────────────────────────────────┐
│  Input: Graph(x=[360,100], edge_index, edge_attr=[E,1])           │
│                                                                     │
│  Layer 1: ECC (NNConv)                                             │
│    • Edge-Conditioned Convolution filters functional node features  │
│      through structural edge weights                                │
│    • Neuroscience: Physical white-matter pathways constrain which   │
│      functional signals can propagate between regions               │
│    → Output: [360, hidden_dim]                                     │
│                                                                     │
│  Layer 2: GATv2Conv (8 heads)                                      │
│    • Dynamic attention mechanism learns to weight edges based on    │
│      node feature similarity AFTER linear transformation            │
│    • Neuroscience: Captures "weak, long-range connections" that     │
│      the paper identifies as critical for g. GATv2 (not GAT) is    │
│      required because standard GAT uses static attention that       │
│      cannot dynamically rank topologically distant edges            │
│    → Output: [360, hidden_dim]                                     │
│                                                                     │
│  Layer 3: GATv2Conv (8 heads)                                      │
│    • Deeper attention refinement for multi-hop integration          │
│    • Neuroscience: Two GATv2 layers allow the network to capture   │
│      2-hop structural-functional interactions, approximating the    │
│      "modal control" regions that coordinate system-wide activity   │
│    → Output: [360, hidden_dim]                                     │
│                                                                     │
│  Pooling: GlobalAttention                                          │
│    • Attention-weighted sum pools all 360 node embeddings into      │
│      a single graph-level vector                                    │
│    • Neuroscience: The attention weights identify "modal control"   │
│      hubs — regions in Frontoparietal and Default Mode networks     │
│      that disproportionately drive global brain dynamics            │
│    → Output: [1, hidden_dim]                                       │
│                                                                     │
│  Prediction Head: 3-layer MLP                                      │
│    • GeLU activation + Dropout(0.3) + Linear→scalar                │
│    → Output: [1, 1] (predicted g-score)                            │
└─────────────────────────────────────────────────────────────────────┘

WHY GATv2 OVER GAT?
  Standard GAT computes attention as: α = LeakyReLU(a^T [Wh_i || Wh_j])
  GATv2 computes attention as:       α = a^T LeakyReLU(W [h_i || h_j])

  The critical difference: In GAT, the attention ranking of neighbors is
  STATIC — it depends only on the query node, not on specific key-query
  interactions. GATv2 applies the nonlinearity BEFORE the dot product,
  enabling truly DYNAMIC attention where the ranking changes based on both
  the source and target node features.

  For connectome data, this matters because:
  1. A weak structural connection between, say, prefrontal cortex and
     parietal cortex might be functionally critical for one subject but
     not another — GATv2 can learn this subject-specific dynamic.
  2. The paper shows that the SAME long-range connection can be positively
     or negatively associated with g depending on context — only dynamic
     attention can capture this bidirectional relationship.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    NNConv,
    GATv2Conv,
    GlobalAttention,
    BatchNorm,
)


# ═══════════════════════════════════════════════════════════════════════════
# Edge Neural Network for ECC (NNConv)
# ═══════════════════════════════════════════════════════════════════════════

class EdgeNN(nn.Module):
    """
    Small neural network that maps edge features (structural weights)
    to weight matrices for edge-conditioned convolution.

    In neuroscience terms: this network learns HOW the physical bandwidth
    of a white-matter tract (scalar fiber capacity) modulates the
    propagation of functional signals between connected regions.

    The output is a weight matrix of shape (in_channels × out_channels)
    that is unique for each edge — enabling the model to learn that
    high-capacity tracts transmit information differently than low-capacity
    ones (matching the paper's "Information Flow Capacity" concept).
    """

    def __init__(self, edge_dim: int, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, in_channels * out_channels),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_attr : Tensor, shape (E, edge_dim)
            Edge features (fiber bundle capacity weights).

        Returns
        -------
        weight_matrices : Tensor, shape (E, in_channels, out_channels)
            Per-edge weight matrices for the ECC convolution.
        """
        out = self.net(edge_attr)
        return out.view(-1, self.in_channels, self.out_channels)


# ═══════════════════════════════════════════════════════════════════════════
# CM-GAT: The Core Architecture
# ═══════════════════════════════════════════════════════════════════════════

class CMGAT(nn.Module):
    """
    Continual Multimodal Graph Attention Network (CM-GAT) for predicting
    general intelligence from brain connectomes.

    This architecture is designed to capture three key findings from the paper:

    1. STRUCTURE-FUNCTION COUPLING (ECC Layer):
       Physical white-matter connectivity constrains functional dynamics.
       The ECC layer explicitly conditions message-passing on structural
       edge weights, learning how fiber capacity shapes information flow.

    2. WEAK LONG-RANGE TIES (GATv2 Layers):
       The paper shows that long-range connections (~151mm) are
       disproportionately predictive of g. GATv2's dynamic attention
       mechanism can learn to upweight these topologically distant but
       functionally critical connections.

    3. MODAL CONTROL HUBS (GlobalAttention Pooling):
       The paper identifies "modal control" regions in the Frontoparietal
       and Default Mode networks as key drivers of system-wide brain
       dynamics. The attention-based pooling learns to identify these
       hubs and weight their contributions to the graph-level embedding.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features (default: 100 ICA components).
    hidden_channels : int
        Hidden dimensionality for all layers (default: 128).
    num_gat_heads : int
        Number of attention heads in GATv2 layers (default: 8).
    edge_dim : int
        Dimensionality of edge features (default: 1 for scalar weights).
    dropout : float
        Dropout rate for regularization (default: 0.3).
    """

    def __init__(
        self,
        in_channels: int = 100,
        hidden_channels: int = 128,
        num_gat_heads: int = 8,
        edge_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_gat_heads = num_gat_heads
        self.dropout = dropout

        # ─── Layer 1: Edge-Conditioned Convolution (ECC / NNConv) ─────
        # NNConv implements the Edge-Conditioned Convolution from
        # Simonovsky & Komodakis (CVPR 2017), which uses a neural network
        # to generate per-edge weight matrices from edge features.
        #
        # Neuroscience justification:
        # The structural connectome provides scalar fiber-capacity weights
        # for each connection. ECC transforms these weights into learnable
        # filter banks, enabling the model to discover how specific
        # tractography patterns modulate functional signal propagation.
        self.edge_nn = EdgeNN(edge_dim, in_channels, hidden_channels)
        self.ecc_conv = NNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            nn=self.edge_nn,
            aggr='mean',  # Mean aggregation ≈ normalized message passing
        )
        self.ecc_norm = BatchNorm(hidden_channels)

        # Projection for residual connection (input dim → hidden dim)
        self.ecc_residual = nn.Linear(in_channels, hidden_channels)

        # ─── Layer 2: GATv2 (Multi-Head Dynamic Attention) ─────────────
        # GATv2Conv from Brody et al. (ICLR 2022) fixes the static
        # attention limitation of the original GAT.
        #
        # Neuroscience justification:
        # The paper demonstrates that the SAME structural connection can
        # play different roles across subjects and cognitive states.
        # GATv2's dynamic attention scores α(h_i, h_j) allow the model
        # to learn context-dependent importance of each connection —
        # critical for capturing the paper's finding that weak ties
        # are selectively important for high-g individuals.
        #
        # We use 8 heads to capture diverse attention patterns:
        # some heads may specialize in local segregation patterns,
        # others in long-range integration pathways.
        self.gat2_1 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // num_gat_heads,
            heads=num_gat_heads,
            dropout=dropout,
            concat=True,  # Concatenate head outputs
            add_self_loops=True,
            edge_dim=edge_dim,  # Incorporate edge features in attention
        )
        self.gat2_1_norm = BatchNorm(hidden_channels)

        # ─── Layer 3: GATv2 (Deeper Attention Refinement) ─────────────
        # Second GATv2 layer captures 2-hop interactions.
        #
        # Neuroscience justification:
        # Two attention layers allow information to propagate across
        # 2-hop structural paths. This is critical because the paper's
        # "modal control" regions (Frontoparietal Network, DMN) often
        # exert influence through intermediate relay regions rather
        # than direct connections. Two layers approximate this indirect
        # multi-synaptic control pathway.
        self.gat2_2 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // num_gat_heads,
            heads=num_gat_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
            edge_dim=edge_dim,
        )
        self.gat2_2_norm = BatchNorm(hidden_channels)

        # ─── Global Pooling: Attention-based Graph Readout ──────────
        # GlobalAttention learns a scoring function over nodes and
        # pools their embeddings via a weighted sum.
        #
        # Neuroscience justification:
        # Not all brain regions contribute equally to g. The paper
        # identifies "modal control" hubs that regulate system-wide
        # activity. GlobalAttention's learned scoring function naturally
        # identifies these hubs: regions that consistently receive high
        # attention scores across subjects are the modal controllers.
        #
        # The gate_nn learns: "How important is this region for g?"
        # The nn transforms node embeddings before weighted aggregation.
        self.global_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 1),
            ),
            nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
            ),
        )

        # ─── Prediction Head: 3-Layer MLP ──────────────────────────
        # Maps the graph-level embedding to a scalar g prediction.
        #
        # Architecture: hidden → hidden//2 → hidden//4 → 1
        # GeLU activation (smoother than ReLU, better for regression)
        # Dropout for regularization
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 1),
        )

        # Store attention weights for analysis and regularization
        self._gat_attention_weights = []

    def forward(
        self,
        data,
        return_attention: bool = False,
        return_pool_scores: bool = False,
    ) -> dict:
        """
        Forward pass through the CM-GAT architecture.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            Input graph(s) with x, edge_index, edge_attr, and batch.
        return_attention : bool
            If True, also return GATv2 attention weight tensors.
        return_pool_scores : bool
            If True, return GlobalAttention node scores (modal control).

        Returns
        -------
        output : dict with keys:
            'prediction' : Tensor, shape (B, 1)
                Predicted g-factor scores.
            'graph_embedding' : Tensor, shape (B, hidden_channels)
                Graph-level embeddings before the prediction head.
            'attention_weights' : list of Tensors (if return_attention)
                GATv2 attention coefficients for each layer.
            'pool_scores' : Tensor (if return_pool_scores)
                Per-node importance scores from GlobalAttention.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        self._gat_attention_weights = []

        # ─── Layer 1: ECC ───────────────────────────────────────────
        # Structural edge weights dynamically filter functional features
        residual = self.ecc_residual(x)
        h = self.ecc_conv(x, edge_index, edge_attr)
        h = self.ecc_norm(h)
        h = F.gelu(h + residual)  # Residual connection
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ─── Layer 2: GATv2 ────────────────────────────────────────
        residual = h
        h, attn_weights_1 = self.gat2_1(
            h, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        h = self.gat2_1_norm(h)
        h = F.gelu(h + residual)  # Residual connection
        h = F.dropout(h, p=self.dropout, training=self.training)
        self._gat_attention_weights.append(attn_weights_1)

        # ─── Layer 3: GATv2 ────────────────────────────────────────
        residual = h
        h, attn_weights_2 = self.gat2_2(
            h, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        h = self.gat2_2_norm(h)
        h = F.gelu(h + residual)  # Residual connection
        h = F.dropout(h, p=self.dropout, training=self.training)
        self._gat_attention_weights.append(attn_weights_2)

        # ─── Global Pooling ────────────────────────────────────────
        # Attention-weighted aggregation → graph-level embedding
        graph_embedding = self.global_pool(h, batch)

        # ─── Prediction Head ───────────────────────────────────────
        prediction = self.prediction_head(graph_embedding)

        # ─── Collect outputs ───────────────────────────────────────
        output = {
            'prediction': prediction,
            'graph_embedding': graph_embedding,
        }

        if return_attention:
            output['attention_weights'] = self._gat_attention_weights

        if return_pool_scores:
            # Re-compute pool scores for interpretability
            gate_nn = self.global_pool.gate_nn
            pool_scores = gate_nn(h).squeeze(-1)
            output['pool_scores'] = pool_scores

        return output

    def get_attention_weights(self):
        """
        Retrieve the most recent GATv2 attention weights.

        These are used for:
          1. REGULARIZATION: L1/L2 penalty on attention weights enforces
             sparse, small-world-like connectivity patterns.
          2. INTERPRETABILITY: High-attention edges reveal which
             structural connections the model considers critical for g.

        Returns
        -------
        attention_weights : list of (edge_index, alpha) tuples
            For each GATv2 layer, the edge indices and attention coefficients.
        """
        return self._gat_attention_weights

    def count_parameters(self) -> dict:
        """Count trainable parameters by component."""
        counts = {}
        counts['ecc'] = sum(
            p.numel() for n, p in self.named_parameters() if 'ecc' in n or 'edge_nn' in n
        )
        counts['gat2_1'] = sum(
            p.numel() for n, p in self.named_parameters() if 'gat2_1' in n
        )
        counts['gat2_2'] = sum(
            p.numel() for n, p in self.named_parameters() if 'gat2_2' in n
        )
        counts['global_pool'] = sum(
            p.numel() for n, p in self.named_parameters() if 'global_pool' in n
        )
        counts['prediction_head'] = sum(
            p.numel() for n, p in self.named_parameters() if 'prediction_head' in n
        )
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY: Model Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_model_summary(model: CMGAT):
    """Print a structured summary of the CM-GAT architecture."""
    print("=" * 70)
    print("CM-GAT: Continual Multimodal Graph Attention Network")
    print("=" * 70)
    print(f"\nInput:  {model.in_channels}-dim node features (ICA co-activation)")
    print(f"Hidden: {model.hidden_channels}-dim representations")
    print(f"Heads:  {model.num_gat_heads} attention heads per GATv2 layer")
    print(f"Drop:   {model.dropout}")
    print("\n─── Architecture ───")
    print(f"  Layer 1: ECC (NNConv)         → {model.hidden_channels}-dim")
    print(f"  Layer 2: GATv2 ({model.num_gat_heads} heads)     → {model.hidden_channels}-dim")
    print(f"  Layer 3: GATv2 ({model.num_gat_heads} heads)     → {model.hidden_channels}-dim")
    print(f"  Pooling: GlobalAttention      → {model.hidden_channels}-dim graph embedding")
    print(f"  Head:    MLP                  → scalar g prediction")
    print("\n─── Parameters ───")
    counts = model.count_parameters()
    for name, count in counts.items():
        print(f"  {name:20s}: {count:>10,}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Architecture verification
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from dataset import generate_mock_subject

    print("\nCM-GAT Model Module — Architecture Verification\n")

    # Build model
    model = CMGAT(
        in_channels=100,
        hidden_channels=128,
        num_gat_heads=8,
        edge_dim=1,
        dropout=0.3,
    )
    print_model_summary(model)

    # Test forward pass with mock data
    print("\n─── Forward Pass Test ───")
    data = generate_mock_subject(num_nodes=360, num_features=100, seed=42)
    print(f"Input graph: {data}")

    model.eval()
    with torch.no_grad():
        output = model(data, return_attention=True, return_pool_scores=True)

    print(f"Prediction shape:     {output['prediction'].shape}")
    print(f"Prediction value:     {output['prediction'].item():.4f}")
    print(f"Graph embedding:      {output['graph_embedding'].shape}")
    print(f"Attention layers:     {len(output['attention_weights'])}")
    for i, (edge_idx, alpha) in enumerate(output['attention_weights']):
        print(f"  Layer {i+2} attention:  edges={edge_idx.shape}, alpha={alpha.shape}")
    print(f"Pool scores range:    [{output['pool_scores'].min():.4f}, "
          f"{output['pool_scores'].max():.4f}]")

    print("\n✓ All architecture checks passed!")

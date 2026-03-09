# CM-GAT: Continual Multimodal Graph Attention Network

**Predicting General Intelligence (*g*) from Brain Connectomes**

A state-of-the-art non-linear reimplementation of the methodology from:

> *"The network architecture of general intelligence in the human connectome"*
> Nature Communications, 2026

This project replaces the paper's linear CPM + Elastic Net approach with a **Graph Neural Network** pipeline that directly models the multimodal (structure + function) brain connectivity architecture.

---

## Architecture

```
Input: Multimodal Brain Graph (360 nodes × 100 features, structural edges)
  │
  ├─ Layer 1: Edge-Conditioned Conv (ECC / NNConv)
  │   └─ Structural edge weights filter functional node features
  │
  ├─ Layer 2: GATv2 (8-head dynamic attention)
  │   └─ Captures weak, long-range connections critical for g
  │
  ├─ Layer 3: GATv2 (8-head dynamic attention)
  │   └─ Multi-hop integration through relay regions
  │
  ├─ Pooling: GlobalAttention
  │   └─ Identifies "modal control" hub regions
  │
  └─ Head: 3-layer MLP (GeLU + Dropout) → scalar g prediction
```

## Modules

| File | Description |
|---|---|
| `dataset.py` | Multimodal graph construction, sparsification, mock data generator |
| `model.py` | CM-GAT architecture (ECC → GATv2 → GlobalAttention → MLP) |
| `continual_memory.py` | EWC + Episodic Replay for continual learning |
| `train.py` | Training pipeline with stratified 5-fold CV |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with mock data (quick test)
python train.py --num_subjects 20 --epochs 10 --folds 2

# Full 5-fold CV
python train.py --num_subjects 100 --epochs 200 --folds 5

# With continual learning
python train.py --num_subjects 100 --epochs 100 --use_continual --lambda_ewc 2000

# GPU
python train.py --num_subjects 100 --epochs 200 --device cuda
```

## Metrics

Matches the paper's evaluation:
- **R²** — Coefficient of Determination
- **Pearson r** — Linear correlation
- **nRMSD** — Normalized Root Mean Square Deviation

Paper baseline (CPM + Elastic Net): R²=0.12, r=0.35, nRMSD=0.94

## Key Design Choices

1. **GATv2 over GAT**: Dynamic attention allows the model to learn subject-specific importance of connections — essential because the same structural connection can be positively or negatively associated with *g*.

2. **Edge-Conditioned Convolution**: Explicitly models how white-matter tract capacity constrains functional signal propagation.

3. **Elastic Net on Attention Weights**: L1+L2 regularization enforces sparse, small-world-like learned connectivity patterns.

4. **Continual Learning**: EWC + episodic replay enables sequential training on adult → infant → adolescent brain data without catastrophic forgetting.

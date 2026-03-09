"""
=============================================================================
CM-GAT: Continual Multimodal Graph Attention Network — Continual Memory
=============================================================================
Paper: "The network architecture of general intelligence in the human
connectome" (Nature Communications, 2026)

This module implements continual learning mechanisms that allow the CM-GAT
model to:
  1. Train on ADULT connectomes (HCP, N=1151, ages 22-36)
  2. Subsequently train on NEW populations (e.g., infant BCP, adolescent
     ABCD) WITHOUT catastrophic forgetting of adult brain architectures

WHY CONTINUAL LEARNING FOR CONNECTOMICS?
  Brain network topology changes dramatically across the lifespan:
    - Adults: Mature small-world architecture with established weak ties
    - Infants: Hyper-connected, immature topology with few long-range links
    - Adolescents: Pruning phase, increasing modular structure

  A model trained on adult data captures mature small-world topology
  principles. When fine-tuning on infant data (drastically different
  topology), standard training would overwrite the adult knowledge.
  Continual learning preserves the learned adult architectural principles
  while adapting to new populations.

TWO COMPLEMENTARY STRATEGIES:

  1. ELASTIC WEIGHT CONSOLIDATION (EWC):
     Computes the Fisher Information Matrix after learning the initial task
     (adults). Parameters important for adult topology are penalized during
     subsequent training, preventing catastrophic forgetting.

  2. EPISODIC REPLAY BUFFER:
     Stores representative samples from previous tasks. During new-task
     training, old samples are replayed alongside new data. The buffer is
     "distributionally robust" — it stratifies stored samples by g-score
     quantiles to maintain a balanced representation across the full range
     of intelligence scores.
=============================================================================
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from collections import defaultdict
from typing import Optional, List, Dict


# ═══════════════════════════════════════════════════════════════════════════
# ELASTIC WEIGHT CONSOLIDATION (EWC)
# ═══════════════════════════════════════════════════════════════════════════

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017).

    After training on a task (e.g., adult HCP connectomes), EWC computes
    the Fisher Information Matrix (FIM) — a measure of how important each
    model parameter is for the learned task. During subsequent training on
    a new task (e.g., infant BCP connectomes), EWC adds a quadratic penalty
    that discourages changes to important parameters:

      L_ewc = λ/2 * Σ_i F_i * (θ_i - θ*_i)²

    where:
      - F_i: Fisher Information for parameter i (importance)
      - θ*_i: Optimal parameter value after the previous task
      - θ_i: Current parameter value during new-task training
      - λ: EWC strength coefficient

    NEUROSCIENCE JUSTIFICATION:
      The adult brain's small-world topology is the "ground truth" baseline.
      EWC preserves model parameters that encode these fundamental
      architectural principles (segregation + long-range integration),
      while allowing parameters to adapt to population-specific differences
      in topology (e.g., lower modularity in infants).

    Parameters
    ----------
    model : nn.Module
        The CM-GAT model after training on the initial task.
    lambda_ewc : float
        Strength of the EWC penalty. Higher values = stronger memory
        retention but less plasticity for new tasks.
        Recommended: 1000–5000 for connectome data.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc

        # Store a snapshot of the trained parameters (θ*)
        self.optimal_params: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

        # Fisher Information Matrix (diagonal approximation)
        # Initialized to zeros — must call compute_fisher() to populate
        self.fisher: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)

        self._fisher_computed = False

    def compute_fisher(
        self,
        model: nn.Module,
        data_loader,
        device: torch.device,
        num_samples: int = None,
    ):
        """
        Compute the diagonal Fisher Information Matrix using the
        empirical Fisher approximation.

        The FIM measures the curvature of the loss landscape around the
        optimal parameters. High curvature (high Fisher) means the
        parameter is critical for the task — changing it would
        significantly degrade performance.

        Parameters
        ----------
        model : nn.Module
            The model after training on the current task.
        data_loader : DataLoader
            DataLoader for the current task's training data.
        device : torch.device
            Device for computation.
        num_samples : int, optional
            Number of samples to use for FIM estimation. If None, uses
            all samples in the data_loader. More samples = more accurate
            but slower estimation.
        """
        model.eval()

        # Reset Fisher to zeros
        for name in self.fisher:
            self.fisher[name] = torch.zeros_like(self.fisher[name])

        count = 0
        for batch in data_loader:
            batch = batch.to(device)

            model.zero_grad()
            output = model(batch)
            prediction = output['prediction']

            # For regression: use squared loss gradient
            # The empirical Fisher is E[∇log p(y|x,θ) ∇log p(y|x,θ)^T]
            # For MSE loss with Gaussian likelihood, this simplifies to
            # the gradient of the squared prediction
            loss = (prediction ** 2).mean()
            loss.backward()

            # Accumulate squared gradients (diagonal FIM approximation)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data.clone() ** 2

            count += 1
            if num_samples is not None and count >= num_samples:
                break

        # Normalize by number of samples
        for name in self.fisher:
            self.fisher[name] /= max(count, 1)

        self._fisher_computed = True

        # Update optimal params snapshot
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty term for the total loss.

        L_ewc = λ/2 * Σ_i F_i * (θ_i - θ*_i)²

        This penalty is ADDED to the task-specific loss during training
        on a new dataset, preventing catastrophic forgetting.

        Parameters
        ----------
        model : nn.Module
            The model currently being trained on the new task.

        Returns
        -------
        ewc_loss : torch.Tensor
            Scalar EWC penalty to add to the total loss.
        """
        if not self._fisher_computed:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                # Quadratic penalty weighted by Fisher importance
                fisher_diag = self.fisher[name].to(param.device)
                optimal = self.optimal_params[name].to(param.device)
                ewc_loss += (fisher_diag * (param - optimal) ** 2).sum()

        return (self.lambda_ewc / 2.0) * ewc_loss

    def state_dict(self) -> dict:
        """Serialize EWC state for checkpointing."""
        return {
            'lambda_ewc': self.lambda_ewc,
            'optimal_params': self.optimal_params,
            'fisher': self.fisher,
            'fisher_computed': self._fisher_computed,
        }

    def load_state_dict(self, state: dict):
        """Restore EWC state from checkpoint."""
        self.lambda_ewc = state['lambda_ewc']
        self.optimal_params = state['optimal_params']
        self.fisher = state['fisher']
        self._fisher_computed = state['fisher_computed']


# ═══════════════════════════════════════════════════════════════════════════
# EPISODIC REPLAY BUFFER (Distributionally Robust)
# ═══════════════════════════════════════════════════════════════════════════

class EpisodicReplayBuffer:
    """
    Distributionally Robust Episodic Replay Buffer for continual learning.

    Stores a fixed-size memory of representative brain graphs from previous
    tasks. During new-task training, old graphs are replayed to maintain
    performance on previous tasks.

    DISTRIBUTIONAL ROBUSTNESS:
      Naive replay buffers may oversample subjects near the mean g-score
      and undersample extreme values (very high or very low g). Since
      the paper shows that the relationship between topology and g is
      non-linear (modal control regions show threshold effects), it is
      critical to maintain balanced representation across the full g range.

      We achieve this by stratifying stored samples into quantile bins
      by g-score, ensuring equal representation of all intelligence levels.

    Parameters
    ----------
    max_size : int
        Maximum number of graphs to store in the buffer.
    num_quantile_bins : int
        Number of g-score bins for stratified storage. Graphs are evenly
        distributed across bins to maintain distributional coverage.
    """

    def __init__(self, max_size: int = 500, num_quantile_bins: int = 5):
        self.max_size = max_size
        self.num_quantile_bins = num_quantile_bins

        # Storage: maps quantile bin index → list of Data objects
        self.bins: Dict[int, List[Data]] = defaultdict(list)
        self.max_per_bin = max(1, max_size // num_quantile_bins)

        # Track g-score statistics for bin assignment
        self._g_scores: List[float] = []
        self._quantile_edges: Optional[np.ndarray] = None

    def _compute_bin_edges(self):
        """Recompute quantile edges from observed g-scores."""
        if len(self._g_scores) < self.num_quantile_bins:
            # Not enough data for quantile binning; use uniform spacing
            g_min = min(self._g_scores) if self._g_scores else 70.0
            g_max = max(self._g_scores) if self._g_scores else 130.0
            self._quantile_edges = np.linspace(
                g_min, g_max, self.num_quantile_bins + 1
            )
        else:
            quantiles = np.linspace(0, 100, self.num_quantile_bins + 1)
            self._quantile_edges = np.percentile(self._g_scores, quantiles)

    def _assign_bin(self, g_score: float) -> int:
        """Assign a g-score to a quantile bin."""
        if self._quantile_edges is None:
            return 0
        bin_idx = np.searchsorted(self._quantile_edges[1:], g_score)
        return min(int(bin_idx), self.num_quantile_bins - 1)

    def add(self, data_list: List[Data]):
        """
        Add a batch of brain graphs to the replay buffer.

        Graphs are distributed across quantile bins based on their g-score.
        When a bin is full, the oldest sample is replaced (FIFO within bin).

        Parameters
        ----------
        data_list : list of Data
            PyG Data objects representing brain connectome graphs.
        """
        for data in data_list:
            g = data.y.item() if isinstance(data.y, torch.Tensor) else float(data.y)
            self._g_scores.append(g)

        # Recompute bin edges with updated g-score distribution
        self._compute_bin_edges()

        for data in data_list:
            g = data.y.item() if isinstance(data.y, torch.Tensor) else float(data.y)
            bin_idx = self._assign_bin(g)

            # Move data to CPU for storage efficiency
            cpu_data = data.clone().cpu()

            if len(self.bins[bin_idx]) < self.max_per_bin:
                self.bins[bin_idx].append(cpu_data)
            else:
                # FIFO replacement: remove oldest, add newest
                self.bins[bin_idx].pop(0)
                self.bins[bin_idx].append(cpu_data)

    def sample(self, batch_size: int, device: torch.device = None) -> List[Data]:
        """
        Sample a balanced batch from the replay buffer.

        Sampling is stratified: we draw approximately equal numbers
        from each quantile bin to maintain distributional robustness.

        Parameters
        ----------
        batch_size : int
            Number of graphs to sample.
        device : torch.device, optional
            Device to move sampled graphs to.

        Returns
        -------
        samples : list of Data
            Balanced batch of replayed brain graphs.
        """
        # Collect all available samples with their bin indices
        all_bins = [k for k in self.bins if len(self.bins[k]) > 0]
        if not all_bins:
            return []

        # Stratified sampling: equal from each non-empty bin
        per_bin = max(1, batch_size // len(all_bins))
        remainder = batch_size - per_bin * len(all_bins)

        samples = []
        for i, bin_idx in enumerate(all_bins):
            bin_data = self.bins[bin_idx]
            n = per_bin + (1 if i < remainder else 0)
            n = min(n, len(bin_data))

            # Random sample without replacement from this bin
            indices = np.random.choice(len(bin_data), size=n, replace=False)
            for idx in indices:
                data = bin_data[idx].clone()
                if device is not None:
                    data = data.to(device)
                samples.append(data)

        return samples[:batch_size]

    @property
    def size(self) -> int:
        """Total number of graphs currently in the buffer."""
        return sum(len(v) for v in self.bins.values())

    def stats(self) -> dict:
        """Return diagnostic statistics about the buffer."""
        return {
            'total_stored': self.size,
            'max_size': self.max_size,
            'num_bins': self.num_quantile_bins,
            'bin_counts': {k: len(v) for k, v in sorted(self.bins.items())},
            'bin_edges': self._quantile_edges.tolist() if self._quantile_edges is not None else None,
        }

    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing."""
        return {
            'bins': {k: [d.to('cpu') for d in v] for k, v in self.bins.items()},
            'g_scores': self._g_scores,
            'quantile_edges': self._quantile_edges,
            'max_size': self.max_size,
            'num_quantile_bins': self.num_quantile_bins,
        }

    def load_state_dict(self, state: dict):
        """Restore buffer state from checkpoint."""
        self.bins = defaultdict(list, state['bins'])
        self._g_scores = state['g_scores']
        self._quantile_edges = state['quantile_edges']
        self.max_size = state['max_size']
        self.num_quantile_bins = state['num_quantile_bins']
        self.max_per_bin = max(1, self.max_size // self.num_quantile_bins)


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED CONTINUAL LEARNING MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ContinualLearningManager:
    """
    Orchestrates both EWC and Episodic Replay for the CM-GAT pipeline.

    This manager:
      1. Maintains the EWC module for weight importance tracking
      2. Maintains the replay buffer for experience replay
      3. Provides a unified interface for computing the continual loss
      4. Handles task transitions (consolidate → buffer → train new)

    Workflow
    --------
    >>> manager = ContinualLearningManager(model, lambda_ewc=1000)
    >>>
    >>> # After training on Task 1 (e.g., adult HCP):
    >>> manager.consolidate_task(model, train_loader, device, task_data_list)
    >>>
    >>> # During Task 2 training (e.g., infant BCP):
    >>> ewc_loss = manager.ewc_penalty(model)
    >>> replay_batch = manager.sample_replay(batch_size=8, device=device)
    >>> # Add ewc_loss to total loss; train on replay_batch alongside new data

    Parameters
    ----------
    model : nn.Module
        The CM-GAT model.
    lambda_ewc : float
        EWC penalty strength.
    buffer_size : int
        Maximum replay buffer capacity.
    num_quantile_bins : int
        Number of g-score bins for stratified replay.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        buffer_size: int = 500,
        num_quantile_bins: int = 5,
    ):
        self.ewc = EWC(model, lambda_ewc)
        self.buffer = EpisodicReplayBuffer(buffer_size, num_quantile_bins)
        self.num_tasks_consolidated = 0

    def consolidate_task(
        self,
        model: nn.Module,
        data_loader,
        device: torch.device,
        task_data_list: List[Data] = None,
        fisher_samples: int = None,
    ):
        """
        Consolidate knowledge after completing training on a task.

        This method should be called AFTER training on each task:
          1. Computes the Fisher Information Matrix for EWC
          2. Stores representative samples in the replay buffer

        Parameters
        ----------
        model : nn.Module
            The model after training on the completed task.
        data_loader : DataLoader
            Training DataLoader for the completed task.
        device : torch.device
            Computation device.
        task_data_list : list of Data, optional
            Full dataset for buffer storage. If None, samples from loader.
        fisher_samples : int, optional
            Number of samples for FIM estimation.
        """
        print(f"\n[ContinualLearning] Consolidating Task {self.num_tasks_consolidated + 1}...")

        # Step 1: Compute Fisher Information Matrix
        print("  → Computing Fisher Information Matrix...")
        self.ewc.compute_fisher(model, data_loader, device, fisher_samples)

        # Step 2: Fill replay buffer
        if task_data_list is not None:
            print(f"  → Storing {len(task_data_list)} samples in replay buffer...")
            self.buffer.add(task_data_list)
        else:
            print("  → Storing samples from data loader in replay buffer...")
            stored = []
            for batch in data_loader:
                # Handle batched data — extract individual graphs
                if hasattr(batch, 'batch') and batch.batch is not None:
                    for i in range(batch.num_graphs):
                        mask = batch.batch == i
                        single = Data(
                            x=batch.x[mask],
                            edge_index=batch.edge_index[:, mask[batch.edge_index[0]]],
                            edge_attr=batch.edge_attr[mask[batch.edge_index[0]]],
                            y=batch.y[i:i+1] if len(batch.y.shape) > 0 else batch.y,
                        )
                        stored.append(single)
                else:
                    stored.append(batch)
            self.buffer.add(stored)

        self.num_tasks_consolidated += 1
        stats = self.buffer.stats()
        print(f"  → Buffer: {stats['total_stored']}/{stats['max_size']} samples "
              f"across {len(stats['bin_counts'])} bins")
        print(f"  ✓ Task {self.num_tasks_consolidated} consolidated.\n")

    def ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty for the current model state."""
        return self.ewc.penalty(model)

    def sample_replay(
        self, batch_size: int, device: torch.device = None
    ) -> List[Data]:
        """Sample from the replay buffer."""
        return self.buffer.sample(batch_size, device)

    def state_dict(self) -> dict:
        """Serialize manager state."""
        return {
            'ewc': self.ewc.state_dict(),
            'buffer': self.buffer.state_dict(),
            'num_tasks': self.num_tasks_consolidated,
        }

    def load_state_dict(self, state: dict):
        """Restore manager state."""
        self.ewc.load_state_dict(state['ewc'])
        self.buffer.load_state_dict(state['buffer'])
        self.num_tasks_consolidated = state['num_tasks']


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Verification
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from model import CMGAT
    from dataset import generate_mock_subject
    from torch_geometric.loader import DataLoader

    print("=" * 70)
    print("Continual Learning Module — Verification")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model and mock data
    model = CMGAT().to(device)
    data_list = [generate_mock_subject(seed=i) for i in range(20)]
    loader = DataLoader(data_list, batch_size=4, shuffle=False)

    # Initialize ContinualLearningManager
    manager = ContinualLearningManager(
        model,
        lambda_ewc=1000.0,
        buffer_size=50,
        num_quantile_bins=5,
    )

    # Simulate training on Task 1 (just forward passes)
    print("\n[Test 1] Simulating Task 1 training...")
    model.train()
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        loss = output['prediction'].mean()  # Dummy loss
        loss.backward()

    # Consolidate Task 1
    print("\n[Test 2] Consolidating Task 1...")
    manager.consolidate_task(model, loader, device, data_list)

    # Compute EWC penalty
    print("[Test 3] Computing EWC penalty...")
    ewc_loss = manager.ewc_penalty(model)
    print(f"  EWC penalty: {ewc_loss.item():.6f}")

    # Sample replay
    print("\n[Test 4] Sampling from replay buffer...")
    replay = manager.sample_replay(batch_size=5, device=device)
    print(f"  Replayed {len(replay)} graphs")
    for i, d in enumerate(replay):
        print(f"    Graph {i}: g={d.y.item():.1f}, nodes={d.num_nodes}")

    # Test serialization
    print("\n[Test 5] Testing serialization...")
    state = manager.state_dict()
    new_manager = ContinualLearningManager(model)
    new_manager.load_state_dict(state)
    print(f"  Restored: {new_manager.num_tasks_consolidated} tasks consolidated")
    print(f"  Buffer:   {new_manager.buffer.size} samples")

    print("\n✓ All continual learning checks passed!")

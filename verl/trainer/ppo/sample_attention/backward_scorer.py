# Copyright 2024 Sample Attention Authors
# SPDX-License-Identifier: Apache-2.0
"""
Backward Scorer: Energy-based sample importance scoring.

This module implements the Learning-Zone Energy Score formula:
    E = D_init × 4·p·(1-p) × (1 + α · momentum)

Where:
- D_init: Initial difficulty anchor, D_init = max(1 - p₀, d_init_min),
  set once when a sample is first observed. Biases selection toward
  harder samples: a sample initially at p=0 gets D_init=1.0 while one
  initially at p=0.75 gets D_init=0.25.
- p: Current pass rate (fraction of correct rollouts)
- 4·p·(1-p): Learning Zone score (normalized Bernoulli variance),
  proportional to GRPO gradient signal magnitude.
  Peaked at p=0.5 (model's "learning frontier"), zero at p=0 and p=1.
- momentum: min(2·|p - ema_μ|, 1.0), measures how much the sample's
  pass rate is changing (active learning signal).
- α: Momentum weight (default 0.3)

Key behavior:
- p=0 or p=1 (dead groups): E=0 regardless of D_init (correctly skipped)
- Two active samples at the same p, but one was harder initially:
  the harder one gets proportionally higher energy via D_init
- Selection uses energy-ranked Top-K with Gumbel noise for exploration
- Gate values (Z-score + Sigmoid) are computed for logging only
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SampleState:
    """Per-sample state for tracking metrics across training.
    
    Attributes:
        sample_id: Unique identifier for the sample
        d_init: Initial difficulty D_i^(0) = max(1 - p_i^(0), 0.5)
        ema_mu: Exponential moving average of pass rate
        welford_mean: Welford mean for variance calculation
        welford_m2: Welford M2 for variance calculation  
        welford_count: Number of observations for Welford
        last_pass_rate: Most recent pass rate
        last_energy: Most recent energy score
        last_gate: Most recent gate value
        observation_count: Total number of times this sample was observed
        epoch_first_seen: Epoch when sample was first observed
    """
    sample_id: str
    d_init: float = -1.0  # -1 indicates not yet initialized
    ema_mu: float = 0.5
    welford_mean: float = 0.0
    welford_m2: float = 0.0
    welford_count: int = 0
    last_pass_rate: float = 0.0
    last_energy: float = 0.0
    last_gate: float = 0.5
    last_momentum: float = 0.0
    observation_count: int = 0
    epoch_first_seen: int = -1

    def get_variance(self, epsilon: float = 1e-6) -> float:
        """Compute variance from Welford state with numerical stability."""
        if self.welford_count < 2:
            return epsilon
        return max(self.welford_m2 / self.welford_count, epsilon)

    def update_welford(self, value: float) -> None:
        """Update Welford online variance statistics."""
        self.welford_count += 1
        delta = value - self.welford_mean
        self.welford_mean += delta / self.welford_count
        delta2 = value - self.welford_mean
        self.welford_m2 += delta * delta2

    def update_ema(self, value: float, decay: float = 0.9) -> None:
        """Update exponential moving average."""
        self.ema_mu = decay * self.ema_mu + (1 - decay) * value

    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        return {
            "sample_id": self.sample_id,
            "d_init": self.d_init,
            "ema_mu": self.ema_mu,
            "welford_mean": self.welford_mean,
            "welford_m2": self.welford_m2,
            "welford_count": self.welford_count,
            "last_pass_rate": self.last_pass_rate,
            "last_energy": self.last_energy,
            "last_gate": self.last_gate,
            "last_momentum": self.last_momentum,
            "observation_count": self.observation_count,
            "epoch_first_seen": self.epoch_first_seen,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SampleState":
        """Deserialize state from dictionary."""
        return cls(**data)


class BackwardScorer:
    """Backward scoring module using Learning-Zone Energy for sample selection.
    
    This class computes energy scores for each sample group based on:
    1. Learning Zone: 4·p·(1-p), the normalized Bernoulli variance,
       proportional to expected GRPO gradient magnitude.
    2. Momentum: |p - ema_μ|, how much the pass rate is changing.
    
    Selection uses energy-ranked Top-K with Gumbel noise, ensuring the
    most informative samples (near the model's learning frontier) are
    prioritized for backpropagation.
    
    Args:
        ema_decay: Decay factor for EMA (default: 0.9)
        d_init_min: Minimum value for D_init (default: 0.05, kept for state tracking)
        epsilon: Small constant for numerical stability (default: 1e-6)
        momentum_weight: Weight for momentum term in energy (default: 0.3)
        exploration_noise: Scale of Gumbel noise for Top-K exploration (default: 0.05)
        sigma_e_min: Minimum std for Z-score in gate computation (default: 0.1)
        selection_ratio: Target fraction of samples to select (default: 0.7)
        min_selection_ratio: Minimum fraction to select (default: 0.3)
        warmup_transition_epochs: Epochs to ramp from full selection to target (default: 3)
    """

    def __init__(
        self,
        ema_decay: float = 0.9,
        d_init_min: float = 0.05,
        epsilon: float = 1e-6,
        inv_sqrt_var_max: float = 10.0,  # deprecated, kept for config compat
        sigma_e_min: float = 0.1,
        z_threshold: float = 1.5,  # deprecated, kept for config compat
        gate_temperature: float = 0.2,  # deprecated, kept for config compat
        selection_ratio: float = 0.7,
        min_selection_ratio: float = 0.3,
        warmup_epochs: int = 0,
        warmup_transition_epochs: int = 3,
        momentum_weight: float = 0.3,
        exploration_noise: float = 0.05,
    ):
        self.ema_decay = ema_decay
        self.d_init_min = d_init_min
        self.epsilon = epsilon
        self.sigma_e_min = sigma_e_min
        self.selection_ratio = selection_ratio
        self.min_selection_ratio = min_selection_ratio
        self.warmup_epochs = warmup_epochs
        self.warmup_transition_epochs = warmup_transition_epochs
        self.momentum_weight = momentum_weight
        self.exploration_noise = exploration_noise
        
        # Deprecated params kept for backward config compatibility
        self.inv_sqrt_var_max = inv_sqrt_var_max
        self.z_threshold = z_threshold
        self.gate_temperature = gate_temperature
        
        # Sample states: sample_id -> SampleState
        self.sample_states: Dict[str, SampleState] = {}
        
        # Tracking for logging
        self.current_epoch = 0
        self.is_warmup = (warmup_epochs > 0)  # Default: no warmup, start selection immediately
        
    def reset(self) -> None:
        """Reset all sample states."""
        self.sample_states.clear()
        self.current_epoch = 0
        self.is_warmup = (self.warmup_epochs > 0)
        
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch and update warmup status."""
        self.current_epoch = epoch
        self.is_warmup = (epoch < self.warmup_epochs)
        
    def get_or_create_state(self, sample_id: str) -> SampleState:
        """Get existing state or create new one for a sample."""
        if sample_id not in self.sample_states:
            self.sample_states[sample_id] = SampleState(sample_id=sample_id)
        return self.sample_states[sample_id]
    
    def compute_pass_rate(
        self, 
        rewards: torch.Tensor,
        n_rollout: int = 8
    ) -> float:
        """Compute pass rate from rewards tensor.
        
        Args:
            rewards: Tensor of shape [n_rollout] or scalar
            n_rollout: Number of rollouts (for normalization)
            
        Returns:
            Pass rate in [0, 1]
        """
        if rewards.dim() == 0:
            return float(rewards.item())
        return float(rewards.float().mean().item())
    
    def compute_energy(
        self,
        state: SampleState,
        pass_rate: float,
    ) -> float:
        """Compute Learning-Zone Energy Score for a single sample.
        
        E = D_init × 4·p·(1-p) × (1 + α · momentum)
        
        Components:
        - D_init: Difficulty anchor from first observation, ∈ [d_init_min, 1.0].
          Biases toward harder samples: initially-hard samples (low p₀)
          get higher D_init, so when they later become active, they
          are prioritized over initially-easy samples at the same p.
        - 4·p·(1-p): Learning Zone (normalized Bernoulli variance),
          proportional to GRPO gradient magnitude.
          p=0 → 0, p=0.5 → 1.0 (peak), p=1 → 0.
        - momentum: How much the pass rate is changing → active learning.
        
        Key: D_init does NOT break dead group filtering because
        p=0 or p=1 → 4·p·(1-p)=0 → E=0 regardless of D_init.
        
        Args:
            state: Sample state containing history (d_init, ema_mu, etc.)
            pass_rate: Current pass rate
            
        Returns:
            Energy score (non-negative)
        """
        p = pass_rate
        
        # Difficulty anchor: set once at first observation
        # D_init = max(1 - p₀, d_init_min), higher for harder samples
        d_init = state.d_init
        if d_init < 0:  # Not initialized yet (warmup epoch 0)
            d_init = self.d_init_min
        
        # Learning Zone: normalized Bernoulli variance ∈ [0, 1]
        # Proportional to GRPO gradient signal: total_grad ∝ n × p × (1-p)
        # Peaks at p=0.5, zero at p=0 and p=1
        learning_zone = 4.0 * p * (1.0 - p)
        
        # Momentum: how much pass rate is changing from historical average
        # |p - ema_μ| normalized to [0, 1] with saturation at 0.5 absolute change
        raw_momentum = abs(p - state.ema_mu)
        momentum = min(raw_momentum * 2.0, 1.0)
        
        # Combined energy: difficulty × learning zone × momentum boost
        energy = d_init * learning_zone * (1.0 + self.momentum_weight * momentum)
        
        return energy
    
    def compute_gate(
        self,
        energies: List[float],
    ) -> List[float]:
        """Compute gate values via Z-score normalization + Sigmoid.
        
        g_i = σ((E_i - μ_E) / σ_E)
        
        NOTE: Gates are computed for logging/visualization only.
        Selection is now done by energy-ranked Top-K, not by gate values.
        
        Args:
            energies: List of energy scores
            
        Returns:
            List of gate values in (0, 1)
        """
        if not energies:
            return []
            
        energies_arr = np.array(energies, dtype=np.float64)
        mu_e = energies_arr.mean()
        sigma_e = max(energies_arr.std(), self.sigma_e_min)
        
        z_scores = (energies_arr - mu_e) / sigma_e
        gates = 1.0 / (1.0 + np.exp(-z_scores))  # Sigmoid
        
        return gates.tolist()
    
    def update_states(
        self,
        sample_ids: List[str],
        pass_rates: List[float],
        epoch: int,
    ) -> None:
        """Update sample states with new observations.
        
        Args:
            sample_ids: List of sample IDs
            pass_rates: List of pass rates
            epoch: Current epoch number
        """
        for sample_id, p in zip(sample_ids, pass_rates):
            state = self.get_or_create_state(sample_id)
            
            # Initialize D_init on first observation (any epoch, not just epoch 0)
            # This ensures samples first seen in later epochs get proper initialization
            # instead of falling back to d_init_min (0.05).
            if state.d_init < 0:
                state.d_init = max(1.0 - p, self.d_init_min)
                state.epoch_first_seen = epoch
                state.ema_mu = p  # Initialize EMA with first observation
                
            # Update tracking
            state.update_ema(p, self.ema_decay)
            state.update_welford(p)
            state.last_pass_rate = p
            state.observation_count += 1
    
    def score_batch(
        self,
        sample_ids: List[str],
        rewards_per_sample: Dict[str, torch.Tensor],
        epoch: int,
        n_rollout: int = 8,
    ) -> Tuple[List[float], List[float], List[bool]]:
        """Score a batch of samples and determine selection.
        
        Args:
            sample_ids: List of unique sample IDs in the batch
            rewards_per_sample: Dict mapping sample_id to rewards tensor
            epoch: Current epoch number
            n_rollout: Number of rollouts per sample
            
        Returns:
            energies: List of energy scores
            gates: List of gate values
            selected: List of boolean selection decisions
        """
        self.set_epoch(epoch)
        
        # Compute pass rates
        pass_rates = []
        for sid in sample_ids:
            rewards = rewards_per_sample.get(sid)
            if rewards is not None:
                p = self.compute_pass_rate(rewards, n_rollout)
            else:
                p = 0.0
            pass_rates.append(p)
        
        # Initialize D_init for newly seen samples (must happen before energy computation)
        for sample_id, p in zip(sample_ids, pass_rates):
            state = self.get_or_create_state(sample_id)
            if state.d_init < 0:
                state.d_init = max(1.0 - p, self.d_init_min)
                state.epoch_first_seen = epoch
                state.ema_mu = p  # Initialize EMA with first observation
        
        # Compute energies BEFORE updating EMA/Welford,
        # so that momentum = |p_new - ema_old| reflects actual change.
        energies = []
        for sid, p in zip(sample_ids, pass_rates):
            state = self.sample_states[sid]
            # Record momentum BEFORE EMA update so logging reflects actual change.
            raw_momentum = abs(p - state.ema_mu)
            state.last_momentum = min(raw_momentum * 2.0, 1.0)
            e = self.compute_energy(state, p)
            state.last_energy = e
            energies.append(e)
        
        # NOW update states (EMA, Welford, last_pass_rate) with current observation
        self.update_states(sample_ids, pass_rates, epoch)
        
        # Compute gates
        gates = self.compute_gate(energies)
        
        # Store gates in states
        for sid, g in zip(sample_ids, gates):
            self.sample_states[sid].last_gate = g
        
        # Selection: During warmup, select all. After warmup, gradually ramp
        # down selection ratio over warmup_transition_epochs.
        # warmup_epochs=0 (default): start selection from epoch 0
        # warmup_epochs=1: epoch 0 records only, selection starts at epoch 1
        if self.is_warmup:
            selected = [True] * len(sample_ids)
        else:
            # Gradual transition: epoch warmup_epochs → epoch (warmup_epochs + transition)
            # selection_ratio ramps from 1.0 → target
            epochs_since_warmup = epoch - self.warmup_epochs
            if self.warmup_transition_epochs <= 0:
                # No transition: jump to target ratio immediately
                progress = 1.0
            else:
                progress = min(epochs_since_warmup / self.warmup_transition_epochs, 1.0)
            current_ratio = 1.0 - progress * (1.0 - self.selection_ratio)
            current_min = 1.0 - progress * (1.0 - self.min_selection_ratio)
            
            # Select by energy ranking (Top-K with Gumbel noise)
            selected = self._topk_select(energies, current_ratio, current_min)
        
        return energies, gates, selected
    
    def _topk_select(
        self,
        energies: List[float],
        target_selection_ratio: float = 0.7,
        min_selection_ratio: float = 0.3,
    ) -> List[bool]:
        """Select samples by energy ranking with Gumbel noise for exploration.
        
        Ranks samples by energy score (with small Gumbel noise added for
        stochastic exploration), then selects the Top-K samples.
        
        This replaces the old gate-based importance sampling, providing:
        - Deterministic discrimination: high-energy samples are always preferred
        - Stochastic exploration: Gumbel noise gives low-energy samples a chance
        - Principled selection: top-K by rank, not probability
        
        Args:
            energies: List of energy scores
            target_selection_ratio: Target fraction of samples to select
            min_selection_ratio: Minimum fraction of samples to select
            
        Returns:
            List of boolean selection decisions
        """
        n = len(energies)
        if n == 0:
            return []
            
        energies_arr = np.array(energies, dtype=np.float64)
        
        # Determine number of samples to select
        target_k = max(1, int(n * target_selection_ratio))
        min_k = max(1, int(n * min_selection_ratio))
        k = max(min_k, target_k)
        k = min(k, n)
        
        if k >= n:
            return [True] * n
        
        # Dead groups (energy=0) should not occupy selection quota.
        # Only select dead samples if there aren't enough active samples to fill k.
        active_mask = energies_arr > 0
        n_active = int(active_mask.sum())
        
        if n_active == 0:
            # All dead: select none (dead group filter will zero them anyway)
            return [False] * n
        
        # Add Gumbel noise for exploration (among active samples only)
        # Gumbel(0, scale) has mean ≈ 0.577*scale, so noise scale is relative to energy spread
        active_energies = energies_arr[active_mask]
        energy_spread = max(active_energies.max() - active_energies.min(), 0.01)
        noise_scale = self.exploration_noise * energy_spread
        gumbel_noise = np.random.gumbel(0, max(noise_scale, 1e-8), size=n)
        noisy_energies = energies_arr + gumbel_noise
        
        # Suppress dead samples: set their noisy energy to -inf so they're never selected
        # unless we need more samples than available active ones
        noisy_energies[~active_mask] = -np.inf
        
        # Select top-k by noisy energy (dead samples only fill in if n_active < k)
        effective_k = min(k, n_active)  # Don't select more than available active samples
        topk_indices = set(np.argsort(noisy_energies)[-effective_k:])
        
        return [i in topk_indices for i in range(n)]
    
    # Keep old name as alias for backward compatibility
    _importance_sample = _topk_select
    
    def get_selection_mask(
        self,
        batch_sample_ids: List[str],
        batch_indices: List[int],
        rewards_per_uid: Dict[str, torch.Tensor],
        epoch: int,
        n_rollout: int = 8,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """Get selection mask for a batch.
        
        This is the main interface called by ray_trainer.
        
        Args:
            batch_sample_ids: Sample IDs for each item in batch (may have duplicates)
            batch_indices: Original dataset indices
            rewards_per_uid: Dict mapping unique sample_id to rewards tensor
            epoch: Current epoch
            n_rollout: Number of rollouts
            
        Returns:
            mask: Boolean tensor of shape [batch_size]
            energy_dict: Dict mapping sample_id to energy score
            gate_dict: Dict mapping sample_id to gate value
        """
        # Get unique sample IDs
        unique_ids = list(rewards_per_uid.keys())
        
        # Score unique samples
        energies, gates, selected = self.score_batch(
            unique_ids, rewards_per_uid, epoch, n_rollout
        )
        
        # Build lookup dicts
        energy_dict = dict(zip(unique_ids, energies))
        gate_dict = dict(zip(unique_ids, gates))
        selected_dict = dict(zip(unique_ids, selected))
        
        # Build mask for full batch
        mask = torch.tensor(
            [selected_dict.get(sid, True) for sid in batch_sample_ids],
            dtype=torch.bool
        )
        
        return mask, energy_dict, gate_dict
    
    def get_all_states(self) -> Dict[str, Dict]:
        """Get all sample states as dictionaries for serialization."""
        return {sid: state.to_dict() for sid, state in self.sample_states.items()}
    
    def load_states(self, states_dict: Dict[str, Dict]) -> None:
        """Load sample states from dictionaries."""
        self.sample_states = {
            sid: SampleState.from_dict(data) 
            for sid, data in states_dict.items()
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics for logging."""
        if not self.sample_states:
            return {}
            
        energies = [s.last_energy for s in self.sample_states.values()]
        gates = [s.last_gate for s in self.sample_states.values()]
        pass_rates = [s.last_pass_rate for s in self.sample_states.values()]
        d_inits = [s.d_init for s in self.sample_states.values() if s.d_init >= 0]
        # Use momentum captured at scoring time (before EMA update).
        momentums = [s.last_momentum for s in self.sample_states.values() if s.observation_count > 0]
        
        return {
            "energy_mean": float(np.mean(energies)) if energies else 0.0,
            "energy_std": float(np.std(energies)) if energies else 0.0,
            "energy_max": float(np.max(energies)) if energies else 0.0,
            "energy_min": float(np.min(energies)) if energies else 0.0,
            "gate_mean": float(np.mean(gates)) if gates else 0.5,
            "gate_std": float(np.std(gates)) if gates else 0.0,
            "pass_rate_mean": float(np.mean(pass_rates)) if pass_rates else 0.0,
            "d_init_mean": float(np.mean(d_inits)) if d_inits else 0.5,
            "momentum_mean": float(np.mean(momentums)) if momentums else 0.0,
            "num_samples_tracked": len(self.sample_states),
        }

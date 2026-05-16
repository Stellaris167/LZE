# SPDX-License-Identifier: Apache-2.0
"""Sample-attention helpers shared by trainer variants."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .backward_scorer import BackwardScorer
from .logging_utils import SampleAttentionLogger


MATH_CATEGORIES = {
    "algebra": 0.20,
    "counting_and_probability": 0.10,
    "geometry": 0.10,
    "intermediate_algebra": 0.15,
    "number_theory": 0.15,
    "prealgebra": 0.15,
    "precalculus": 0.15,
}


class SampleAttentionMixin:
    """Mixin that adds backward sample selection and related metrics."""

    def _init_sample_attention(self) -> None:
        """Initialize sample-attention configuration and helpers."""
        try:
            from omegaconf import OmegaConf

            sa_cfg = (
                OmegaConf.to_container(self.config.sample_attention, resolve=True)
                if hasattr(self.config, "sample_attention")
                else {}
            )
        except Exception:
            sa_cfg = {}
        if not isinstance(sa_cfg, dict):
            sa_cfg = {}

        backward_cfg = sa_cfg.get("backward", {}) or {}
        self.sa_backward_enabled = backward_cfg.get("enabled", False)
        self.sa_ema_decay = float(backward_cfg.get("ema_decay", 0.9))
        # Keep a small positive floor so initially solved samples can still receive gradients later.
        self.sa_d_init_min = float(backward_cfg.get("d_init_min", 0.05))
        self.sa_z_threshold = float(backward_cfg.get("z_threshold", 1.5))
        self.sa_gate_temperature = float(backward_cfg.get("gate_temperature", 0.2))
        self.sa_selection_ratio = float(backward_cfg.get("selection_ratio", 0.7))
        self.sa_min_selection_ratio = float(backward_cfg.get("min_selection_ratio", 0.3))

        log_cfg = sa_cfg.get("logging", {}) or {}
        self.sa_log_dir = log_cfg.get("log_dir", None)
        self.sa_use_kde = log_cfg.get("use_kde", False)
        self.sa_log_math_categories = log_cfg.get("log_math_categories", True)

        self.sa_enabled = self.sa_backward_enabled
        self.backward_scorer: Optional[BackwardScorer] = None
        self.sa_logger: Optional[SampleAttentionLogger] = None
        self.sa_current_epoch = 0

        if not self.sa_enabled:
            print("[SampleAttention] Disabled")
            return

        self.backward_scorer = BackwardScorer(
            ema_decay=self.sa_ema_decay,
            d_init_min=self.sa_d_init_min,
            z_threshold=self.sa_z_threshold,
            gate_temperature=self.sa_gate_temperature,
            selection_ratio=self.sa_selection_ratio,
            min_selection_ratio=self.sa_min_selection_ratio,
        )
        print(
            "[SampleAttention] Backward scorer initialized: "
            f"selection_ratio={self.sa_selection_ratio}, min={self.sa_min_selection_ratio}"
        )

        if self.sa_log_dir:
            os.makedirs(self.sa_log_dir, exist_ok=True)
            self.sa_logger = SampleAttentionLogger(
                log_dir=self.sa_log_dir,
                use_wandb=True,
                use_kde=self.sa_use_kde,
            )
            print(f"[SampleAttention] Logger initialized: {self.sa_log_dir}")

    def _sa_start_epoch(self, epoch: int) -> None:
        """Initialize sample attention for a new epoch."""
        if not self.sa_enabled:
            return

        self.sa_current_epoch = epoch
        if self.backward_scorer:
            self.backward_scorer.set_epoch(epoch)

        is_warmup = epoch == 0
        print(f"[SampleAttention] Epoch {epoch} started. Warmup={is_warmup}, Backward={self.sa_backward_enabled}")

    def _sa_finalize_epoch(self, epoch: int) -> None:
        """Finalize sample attention for the epoch and log statistics."""
        if not self.sa_enabled:
            return

        if self.backward_scorer and self.sa_logger:
            stats = self.backward_scorer.get_statistics()
            self.sa_logger.log_epoch_stats(epoch, stats)
            self.sa_logger.save_scorer_state(self.backward_scorer, epoch)

        print(f"[SampleAttention] Epoch {epoch} finalized.")

    def _sa_apply_selection(
        self,
        batch,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Dict[str, Any],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply backward sample selection to a batch."""
        metrics: Dict[str, float] = {}

        if not self.sa_enabled:
            return torch.ones(len(batch), dtype=torch.bool), metrics

        sample_ids = batch.non_tensor_batch.get("uid")
        batch_indices = batch.non_tensor_batch.get("index")

        if sample_ids is not None:
            sample_ids = [str(sample_id) for sample_id in sample_ids]
        elif batch_indices is not None:
            sample_ids = [str(index) for index in batch_indices]
        else:
            return torch.ones(len(batch), dtype=torch.bool), metrics

        if batch_indices is None:
            batch_indices = np.array([hash(sample_id) % (2**31) for sample_id in sample_ids])

        n_rollout = self.config.actor_rollout_ref.rollout.get("n", 8)

        rewards_tensor = reward_tensor.detach().float().cpu()
        rewards = rewards_tensor.sum(dim=-1) if rewards_tensor.dim() > 1 else rewards_tensor
        rewards = rewards.view(-1)

        uid_to_positions: Dict[str, List[int]] = defaultdict(list)
        for position, sample_id in enumerate(sample_ids):
            uid_to_positions[sample_id].append(position)

        rewards_per_uid: Dict[str, torch.Tensor] = {}
        for sample_id, positions in uid_to_positions.items():
            rewards_per_uid[sample_id] = rewards[positions]

        if self.backward_scorer is None:
            select_mask = torch.ones(len(batch), dtype=torch.bool)
        else:
            select_mask, energy_dict, gate_dict = self.backward_scorer.get_selection_mask(
                batch_sample_ids=sample_ids,
                batch_indices=list(batch_indices),
                rewards_per_uid=rewards_per_uid,
                epoch=epoch,
                n_rollout=n_rollout,
            )

            energies = list(energy_dict.values())
            gates = list(gate_dict.values())
            pass_rates = [float(rewards_per_uid[sample_id].mean().item()) for sample_id in energy_dict.keys()]

            metrics["sa/energy_mean"] = float(np.mean(energies)) if energies else 0.0
            metrics["sa/energy_std"] = float(np.std(energies)) if energies else 0.0
            metrics["sa/gate_mean"] = float(np.mean(gates)) if gates else 0.5
            metrics["sa/pass_rate_mean"] = float(np.mean(pass_rates)) if pass_rates else 0.0
            metrics["sa/selected_ratio"] = float(select_mask.float().mean().item())
            metrics["sa/selected_count"] = int(select_mask.sum().item())
            metrics["sa/total_count"] = len(select_mask)

            if self.sa_logger and epoch > 0:
                backward_stats = self.backward_scorer.get_statistics()
                self.sa_logger.log_step_stats(
                    step=getattr(self, "global_steps", 0),
                    epoch=epoch,
                    stats=backward_stats,
                    energies=energies,
                    gates=gates,
                    pass_rates=pass_rates,
                )

        if self.sa_log_math_categories:
            metrics.update(self._compute_math_category_metrics(batch, rewards, reward_extra_infos_dict))

        return select_mask, metrics

    @staticmethod
    def _extract_math_category(data_source: str = None, type_field: str = None):
        """Extract a normalized MATH category name."""
        if data_source and "MATH" in str(data_source):
            parts = str(data_source).rstrip("/").split("/")
            if len(parts) >= 1:
                return parts[-1].lower().replace(" ", "_").replace("&", "and")
        if type_field and str(type_field).strip():
            category = str(type_field).lower().strip().replace(" ", "_").replace("&", "and")
            aliases = {
                "counting_&_probability": "counting_and_probability",
                "counting_probability": "counting_and_probability",
            }
            return aliases.get(category, category)
        return None

    def _compute_math_category_metrics(
        self,
        batch,
        rewards: torch.Tensor,
        reward_extra_infos_dict: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute MATH category-weighted metrics."""
        metrics: Dict[str, float] = {}

        data_sources = batch.non_tensor_batch.get("data_source")
        type_fields = batch.non_tensor_batch.get("type")
        extra_categories = None
        if reward_extra_infos_dict:
            extra_categories = reward_extra_infos_dict.get("category") or reward_extra_infos_dict.get("type")

        if data_sources is None and type_fields is None and extra_categories is None:
            return metrics

        category_rewards: Dict[str, List[float]] = defaultdict(list)
        for index in range(min(len(rewards), len(batch))):
            data_source = str(data_sources[index]) if data_sources is not None and index < len(data_sources) else None
            type_field = str(type_fields[index]) if type_fields is not None and index < len(type_fields) else None
            extra_category = (
                str(extra_categories[index]) if extra_categories is not None and index < len(extra_categories) else None
            )

            category = SampleAttentionMixin._extract_math_category(data_source=data_source, type_field=type_field)
            if category is None and extra_category:
                category = extra_category.lower().replace(" ", "_").replace("&", "and")

            if category is not None:
                category_rewards[category].append(float(rewards[index].item()))

        category_accs: Dict[str, float] = {}
        for category, category_values in category_rewards.items():
            if category_values:
                accuracy = float(np.mean(category_values))
                category_accs[category] = accuracy
                metrics[f"math/{category}_acc"] = accuracy
                metrics[f"math/{category}_count"] = len(category_values)

        if category_accs:
            weighted_sum = 0.0
            total_weight = 0.0
            for category, accuracy in category_accs.items():
                weight = MATH_CATEGORIES.get(category, 0.1)
                weighted_sum += accuracy * weight
                total_weight += weight

            if total_weight > 0:
                metrics["math/weighted_avg_acc"] = weighted_sum / total_weight
            metrics["math/unweighted_avg_acc"] = float(np.mean(list(category_accs.values())))

        return metrics
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass
class PruneReplayState:
    full_correct_streak: int = 0
    last_pass_rate: float = 0.0
    is_pruned: bool = False


class PruneReplayTracker:
    def __init__(
        self,
        consecutive_full_correct_epochs: int = 2,
        use_replay: bool = True,
        replay_ratio: float = 0.1,
        seed: int = 0,
    ):
        self.consecutive_full_correct_epochs = max(1, int(consecutive_full_correct_epochs))
        self.use_replay = bool(use_replay)
        self.replay_ratio = min(max(float(replay_ratio), 0.0), 1.0)
        self._rng = random.Random(seed)

        self.states: Dict[str, PruneReplayState] = {}
        self.pruned_pool: Set[str] = set()
        self.replay_pool: Set[str] = set()
        self.current_epoch = -1
        self.epoch_stats: Dict[str, float] = {}

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.replay_pool = set()
        if self.use_replay and self.pruned_pool and self.replay_ratio > 0:
            replay_count = int(len(self.pruned_pool) * self.replay_ratio)
            if replay_count == 0:
                replay_count = 1
            replay_count = min(replay_count, len(self.pruned_pool))
            self.replay_pool = set(self._rng.sample(list(self.pruned_pool), replay_count))

        self.epoch_stats = {
            "prune_pool_size": float(len(self.pruned_pool)),
            "replay_pool_size": float(len(self.replay_pool)),
            "newly_pruned": 0.0,
            "reactivated": 0.0,
            "replayed_correct": 0.0,
            "replayed_incorrect": 0.0,
            "forward_pruned_samples": 0.0,
            "forward_kept_samples": 0.0,
        }

    def should_skip(self, sample_key: str) -> bool:
        return sample_key in self.pruned_pool and sample_key not in self.replay_pool

    def filter_active(self, sample_keys: Iterable[str]) -> List[bool]:
        mask = []
        pruned = 0
        kept = 0
        for sample_key in sample_keys:
            keep = not self.should_skip(sample_key)
            mask.append(keep)
            if keep:
                kept += 1
            else:
                pruned += 1
        self.epoch_stats["forward_pruned_samples"] += float(pruned)
        self.epoch_stats["forward_kept_samples"] += float(kept)
        return mask

    def update_batch(self, pass_rate_by_key: Dict[str, float]) -> None:
        for sample_key, pass_rate in pass_rate_by_key.items():
            state = self.states.setdefault(sample_key, PruneReplayState())
            state.last_pass_rate = pass_rate

            if pass_rate >= 1.0 - 1e-8:
                state.full_correct_streak += 1
            else:
                state.full_correct_streak = 0

            if sample_key in self.replay_pool:
                if pass_rate >= 1.0 - 1e-8:
                    self.epoch_stats["replayed_correct"] += 1.0
                    state.is_pruned = True
                    self.pruned_pool.add(sample_key)
                else:
                    self.epoch_stats["replayed_incorrect"] += 1.0
                    if state.is_pruned:
                        self.epoch_stats["reactivated"] += 1.0
                    state.is_pruned = False
                    self.pruned_pool.discard(sample_key)
                continue

            if state.full_correct_streak >= self.consecutive_full_correct_epochs:
                if not state.is_pruned:
                    self.epoch_stats["newly_pruned"] += 1.0
                state.is_pruned = True
                self.pruned_pool.add(sample_key)
            else:
                if state.is_pruned and pass_rate < 1.0 - 1e-8:
                    self.epoch_stats["reactivated"] += 1.0
                state.is_pruned = False
                self.pruned_pool.discard(sample_key)

    def get_epoch_stats(self) -> Dict[str, float]:
        stats = dict(self.epoch_stats)
        stats["prune_pool_size"] = float(len(self.pruned_pool))
        stats["replay_pool_size"] = float(len(self.replay_pool))
        total = stats.get("forward_pruned_samples", 0.0) + stats.get("forward_kept_samples", 0.0)
        stats["forward_pruned_ratio"] = stats.get("forward_pruned_samples", 0.0) / max(total, 1.0)
        return stats

# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper exposing a sample-attention-enabled trainer."""

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from .trainer_mixin import SampleAttentionMixin


class SampleAttentionTrainer(SampleAttentionMixin, RayPPOTrainer):
    """RayPPOTrainer variant with backward sample selection helpers.

    Configuration example:
    ```yaml
    sample_attention:
      backward:
        enabled: true
        ema_decay: 0.9
        d_init_min: 0.05
        selection_ratio: 0.7
        min_selection_ratio: 0.3
      prune:
        enabled: false
        consecutive_full_correct_epochs: 2
        use_replay: true
        replay_ratio: 0.1
      logging:
        log_dir: "/path/to/logs"
        log_math_categories: true
    ```
    """

    pass

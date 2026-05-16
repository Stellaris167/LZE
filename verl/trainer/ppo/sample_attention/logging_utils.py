# SPDX-License-Identifier: Apache-2.0
"""Logging helpers for sample attention."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class SampleLogEntry:
    """Log entry for a single sample observation."""
    sample_id: str
    batch_idx: int
    epoch: int
    step: int
    pass_rate: float
    energy: float
    gate: float
    d_init: float
    ema_mu: float
    variance: float
    selected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "batch_idx": self.batch_idx,
            "epoch": self.epoch,
            "step": self.step,
            "pass_rate": self.pass_rate,
            "energy": self.energy,
            "gate": self.gate,
            "d_init": self.d_init,
            "ema_mu": self.ema_mu,
            "variance": self.variance,
            "selected": self.selected,
        }


class SampleAttentionLogger:
    """Comprehensive logger for Sample Attention metrics.
    
    Supports:
    - Per-sample JSONL logs
    - Step-level statistics
    - Epoch-level summaries
    - Distribution plots (histograms with optional KDE)
    - WandB integration
    
    Args:
        log_dir: Base directory for logs
        use_wandb: Whether to log to WandB
        use_kde: Whether to use KDE for distribution plots
        plot_dpi: DPI for saved plots
    """
    
    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = True,
        use_kde: bool = False,
        plot_dpi: int = 100,
        plot_every_n_steps: int = 5,
    ):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_kde = use_kde
        self.plot_dpi = plot_dpi
        self.plot_every_n_steps = max(1, plot_every_n_steps)
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        self.sample_log_dir = os.path.join(log_dir, "samples")
        self.step_log_dir = os.path.join(log_dir, "steps")
        self.epoch_log_dir = os.path.join(log_dir, "epochs")
        self.plot_dir = os.path.join(log_dir, "plots")
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        
        for d in [self.sample_log_dir, self.step_log_dir, self.epoch_log_dir, 
                  self.plot_dir, self.checkpoint_dir]:
            os.makedirs(d, exist_ok=True)
            
        # File paths
        self.sample_log_path = os.path.join(self.sample_log_dir, "samples.jsonl")
        self.step_stats_path = os.path.join(self.step_log_dir, "step_stats.jsonl")
        self.epoch_stats_path = os.path.join(self.epoch_log_dir, "epoch_stats.jsonl")
        
        # Buffers
        self.step_buffer: List[SampleLogEntry] = []
        self.epoch_buffer: List[Dict[str, Any]] = []
        
        # WandB handle
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                print("[SampleAttentionLogger] WandB not available, disabling")
                self.use_wandb = False
                
    def log_sample(self, entry: SampleLogEntry) -> None:
        """Log a single sample observation."""
        self.step_buffer.append(entry)
        
        # Write to file
        with open(self.sample_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            
    def log_batch(
        self,
        sample_ids: List[str],
        batch_indices: List[int],
        epoch: int,
        step: int,
        pass_rates: List[float],
        energies: List[float],
        gates: List[float],
        d_inits: List[float],
        ema_mus: List[float],
        variances: List[float],
        selected: List[bool],
    ) -> None:
        """Log an entire batch of samples."""
        n = len(sample_ids)
        
        for i in range(n):
            entry = SampleLogEntry(
                sample_id=sample_ids[i],
                batch_idx=batch_indices[i],
                epoch=epoch,
                step=step,
                pass_rate=pass_rates[i],
                energy=energies[i],
                gate=gates[i],
                d_init=d_inits[i],
                ema_mu=ema_mus[i],
                variance=variances[i],
                selected=selected[i],
            )
            self.log_sample(entry)
            
    def log_step_stats(
        self,
        step: int,
        epoch: int,
        stats: Dict[str, float],
        energies: Optional[List[float]] = None,
        gates: Optional[List[float]] = None,
        pass_rates: Optional[List[float]] = None,
    ) -> None:
        """Log step-level statistics and optionally create distribution plots."""
        # Compute histogram statistics
        hist_stats = {}
        if energies:
            hist_stats["energy"] = self._compute_hist_stats(energies)
        if gates:
            hist_stats["gate"] = self._compute_hist_stats(gates)
        if pass_rates:
            hist_stats["pass_rate"] = self._compute_hist_stats(pass_rates)
            
        # Combine stats
        full_stats = {
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **stats,
            **hist_stats,
        }
        
        # Write to file
        with open(self.step_stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(full_stats, ensure_ascii=False) + "\n")
            
        # Create and log distribution plot (only every N steps to avoid matplotlib overhead)
        should_plot = (step % self.plot_every_n_steps == 0)
        if should_plot and HAS_MATPLOTLIB and (energies or gates or pass_rates):
            plot_path = self._create_distribution_plot(
                step, energies, gates, pass_rates
            )
            
            # Log to WandB
            if self.use_wandb and self._wandb and self._wandb.run:
                try:
                    wandb_stats = {f"sample_attention/{k}": v for k, v in stats.items()}
                    wandb_stats["sample_attention/step_plot"] = self._wandb.Image(plot_path)
                    self._wandb.log(wandb_stats, step=step)
                except Exception as e:
                    print(f"[SampleAttentionLogger] WandB log failed: {e}")
                    
        # Clear step buffer
        self.step_buffer.clear()
        
    def log_epoch_stats(
        self,
        epoch: int,
        stats: Dict[str, float],
    ) -> None:
        """Log epoch-level summary statistics."""
        full_stats = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **stats,
        }
        
        with open(self.epoch_stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(full_stats, ensure_ascii=False) + "\n")
            
        # Log to WandB
        if self.use_wandb and self._wandb and self._wandb.run:
            try:
                wandb_stats = {f"sample_attention/epoch_{k}": v for k, v in stats.items()}
                self._wandb.log(wandb_stats, step=epoch)
            except Exception as e:
                print(f"[SampleAttentionLogger] WandB epoch log failed: {e}")
                
    def _compute_hist_stats(
        self,
        values: List[float],
        bins: int = 10,
    ) -> Dict[str, Any]:
        """Compute histogram statistics."""
        arr = np.array(values, dtype=np.float64)
        counts, bin_edges = np.histogram(arr, bins=bins)
        
        return {
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }
        
    def _create_distribution_plot(
        self,
        step: int,
        energies: Optional[List[float]],
        gates: Optional[List[float]],
        pass_rates: Optional[List[float]],
    ) -> str:
        """Create distribution plot with optional KDE."""
        n_plots = sum([energies is not None, gates is not None, pass_rates is not None])
        if n_plots == 0:
            return ""
            
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
            
        plot_idx = 0
        
        def plot_distribution(ax, data, title, color, range_limits=None):
            arr = np.array(data, dtype=np.float64)
            
            # Histogram
            ax.hist(arr, bins=20, density=True, alpha=0.6, color=color, 
                   range=range_limits, label="Histogram")
            
            # Optional KDE
            if self.use_kde and HAS_SCIPY and len(arr) > 10:
                try:
                    kde = stats.gaussian_kde(arr)
                    x_range = np.linspace(arr.min(), arr.max(), 100)
                    ax.plot(x_range, kde(x_range), color=color, linewidth=2, 
                           label="KDE")
                except Exception:
                    pass  # KDE may fail for degenerate data
                    
            ax.set_title(f"{title} (step {step})")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            
            # Add statistics annotation
            stats_text = f"μ={arr.mean():.3f}\nσ={arr.std():.3f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        if energies is not None:
            plot_distribution(axes[plot_idx], energies, "Energy Score", "blue")
            plot_idx += 1
            
        if gates is not None:
            plot_distribution(axes[plot_idx], gates, "Gate Value", "green", 
                            range_limits=(0, 1))
            plot_idx += 1
            
        if pass_rates is not None:
            plot_distribution(axes[plot_idx], pass_rates, "Pass Rate", "orange",
                            range_limits=(0, 1))
            plot_idx += 1
            
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plot_dir, f"step_{step}.png")
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close(fig)
        
        return plot_path
        
    def save_scorer_state(
        self,
        scorer: Any,
        epoch: int,
    ) -> str:
        """Save backward scorer state."""
        state_path = os.path.join(
            self.checkpoint_dir,
            f"scorer_state_epoch{epoch}.json"
        )
        
        if hasattr(scorer, 'get_all_states'):
            states = scorer.get_all_states()
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(states, f, ensure_ascii=False, indent=2)
                
        return state_path
        
    def get_log_paths(self) -> Dict[str, str]:
        """Get all log file paths."""
        return {
            "sample_log": self.sample_log_path,
            "step_stats": self.step_stats_path,
            "epoch_stats": self.epoch_stats_path,
            "plot_dir": self.plot_dir,
            "checkpoint_dir": self.checkpoint_dir,
        }


def create_summary_table(
    energies: List[float],
    gates: List[float],
    pass_rates: List[float],
    selected: List[bool],
) -> str:
    """Create a summary table string for console output."""
    n = len(energies)
    n_selected = sum(selected)
    
    e_arr = np.array(energies)
    g_arr = np.array(gates)
    p_arr = np.array(pass_rates)
    
    lines = [
        "┌" + "─" * 50 + "┐",
        "│ Sample Attention Summary" + " " * 25 + "│",
        "├" + "─" * 50 + "┤",
        f"│ Total samples: {n:>10} │ Selected: {n_selected:>10} ({100*n_selected/n:.1f}%) │",
        "├" + "─" * 50 + "┤",
        f"│ Energy  │ mean={e_arr.mean():>6.3f} std={e_arr.std():>6.3f} range=[{e_arr.min():.3f},{e_arr.max():.3f}]│",
        f"│ Gate    │ mean={g_arr.mean():>6.3f} std={g_arr.std():>6.3f} range=[{g_arr.min():.3f},{g_arr.max():.3f}]│",
        f"│ PassRate│ mean={p_arr.mean():>6.3f} std={p_arr.std():>6.3f} range=[{p_arr.min():.3f},{p_arr.max():.3f}]│",
        "└" + "─" * 50 + "┘",
    ]
    
    return "\n".join(lines)

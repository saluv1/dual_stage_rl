"""Training-history plots for Phase-I.

These read the evaluation CSV the trainer appends at every evaluation
(``training_evaluations.csv``) and render curves versus environment timestep.
They are intentionally cheap: no rollouts happen here, so they can run every
evaluation without slowing training. Heavy per-trajectory plotting lives in
``evaluation_report.py`` and runs only at the end.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_REGION_ORDER = ("general_trace", "near_ceiling", "bridge", "base_shell")
_REGION_COLORS = {
    "general_trace": "#1f77b4",
    "near_ceiling": "#d62728",
    "bridge": "#2ca02c",
    "base_shell": "#9467bd",
}


def _read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _col(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for row in rows:
        try:
            out.append(float(row.get(key, "")))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _discovered_regions(rows: list[dict]) -> list[str]:
    present = []
    for region in _REGION_ORDER:
        if f"{region}_success_horizon_rate" in rows[0]:
            present.append(region)
    return present


def plot_training_history(csv_path: str | Path, output_dir: str | Path) -> None:
    """Render the full set of training-history curves into ``output_dir``."""
    csv_path = Path(csv_path)
    rows = _read_rows(csv_path)
    if not rows:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = _col(rows, "timestep")
    mu_sa = _col(rows, "mu_sa")
    difficulty = _col(rows, "difficulty")
    success = _col(rows, "success_rate")
    success_h = _col(rows, "success_horizon_rate")
    failure = _col(rows, "failure_rate")
    timeout = _col(rows, "timeout_rate")
    arrival = _col(rows, "mean_arrival_time_s")
    worst = _col(rows, "worst_region_success_horizon_rate")
    regions = _discovered_regions(rows)

    # 1. mu_SA + difficulty (kept for continuity)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, mu_sa, color="#1f77b4", label=r"weighted $\mu_{SA}$")
    ax1.set_xlabel("Environment timestep")
    ax1.set_ylabel(r"Weighted $\mu_{SA}$")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(steps, difficulty, linestyle="--", color="#7f7f7f", label="curriculum s")
    ax2.set_ylabel("Curriculum difficulty s")
    ax2.set_ylim(-0.02, 1.02)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "training_progress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. overall outcome rates
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, success, color="#2ca02c", label="strict success")
    ax.plot(steps, success_h, color="#1f77b4", label="within-horizon arrival")
    ax.plot(steps, failure, color="#d62728", label="failure")
    ax.plot(steps, timeout, color="#ff7f0e", label="timeout")
    ax.plot(steps, worst, color="#9467bd", linestyle=":", label="worst-region success")
    ax.set_xlabel("Environment timestep")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize=9)
    ax.set_title("Overall outcome rates")
    fig.tight_layout()
    fig.savefig(output_dir / "success_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3 + 4. per-region success / failure
    if regions:
        fig, ax = plt.subplots(figsize=(9, 5))
        for region in regions:
            ax.plot(steps, _col(rows, f"{region}_success_horizon_rate"),
                    color=_REGION_COLORS.get(region), label=region)
        ax.set_xlabel("Environment timestep")
        ax.set_ylabel("Within-horizon arrival rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_title("Per-region arrival rate")
        fig.tight_layout()
        fig.savefig(output_dir / "per_region_success.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for region in regions:
            ax.plot(steps, _col(rows, f"{region}_failure_rate"),
                    color=_REGION_COLORS.get(region), label=region)
        ax.set_xlabel("Environment timestep")
        ax.set_ylabel("Failure rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_title("Per-region failure rate")
        fig.tight_layout()
        fig.savefig(output_dir / "per_region_failure.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 5. arrival time
    if np.any(np.isfinite(arrival)):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, arrival, color="#8c564b", label="mean arrival time (success)")
        ax.set_xlabel("Environment timestep")
        ax.set_ylabel("Arrival time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_title("Mean safe-arrival time")
        fig.tight_layout()
        fig.savefig(output_dir / "arrival_time.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 6. combined summary panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(steps, mu_sa, color="#1f77b4")
    axb = ax.twinx()
    axb.plot(steps, difficulty, linestyle="--", color="#7f7f7f")
    axb.set_ylim(-0.02, 1.02)
    axb.set_ylabel("difficulty s")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(r"$\mu_{SA}$")
    ax.set_title(r"Weighted $\mu_{SA}$ and curriculum")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, success, color="#2ca02c", label="success")
    ax.plot(steps, success_h, color="#1f77b4", label="within-horizon")
    ax.plot(steps, failure, color="#d62728", label="failure")
    ax.plot(steps, timeout, color="#ff7f0e", label="timeout")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Overall outcome rates")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    for region in regions:
        ax.plot(steps, _col(rows, f"{region}_success_horizon_rate"),
                color=_REGION_COLORS.get(region), label=region)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("timestep")
    ax.set_title("Per-region arrival rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if np.any(np.isfinite(arrival)):
        ax.plot(steps, arrival, color="#8c564b")
        ax.set_ylabel("arrival time (s)")
    ax.set_xlabel("timestep")
    ax.set_title("Mean safe-arrival time (success only)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Phase-I training summary", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "summary_panel.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
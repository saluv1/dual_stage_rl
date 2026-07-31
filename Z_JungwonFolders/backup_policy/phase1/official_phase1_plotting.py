"""Plots for the official-style Phase-I evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .official_phase1_eval import OfficialEpisodeRecord, OfficialTrajectory
from .official_reset_library import OFFICIAL_REGIONS


def _save(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_split_summary(
    split: str,
    summary: dict[str, Any],
    records: list[OfficialEpisodeRecord],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    subset = summary["subset_metrics"]
    labels = [region for region in OFFICIAL_REGIONS if subset[region]["count"] > 0]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width,
        [subset[r]["entered_terminal_rate"] for r in labels],
        width,
        label="entered base set",
    )
    ax.bar(
        x,
        [subset[r]["terminal_at_horizon_rate"] for r in labels],
        width,
        label="in base at horizon",
    )
    ax.bar(
        x + width,
        [subset[r]["success_rate"] for r in labels],
        width,
        label="strict success",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(f"{split}: official Phase-I success criteria")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _save(fig, output / f"{split}_success_by_region.png")

    success_data = []
    success_labels = []
    for region in labels:
        values = [
            record.entry_time_sec
            for record in records
            if record.region == region and np.isfinite(record.entry_time_sec)
        ]
        if values:
            success_data.append(values)
            success_labels.append(region)
    if success_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        # ``labels`` is compatible with older Matplotlib versions.
        ax.boxplot(success_data, labels=success_labels, showfliers=True)
        ax.set_ylabel("First base-set entry time [s]")
        ax.set_title(f"{split}: entry-time distribution")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        _save(fig, output / f"{split}_entry_time_boxplot.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for region in labels:
        values = [
            record.min_hard_deck_margin
            for record in records
            if record.region == region
        ]
        ax.hist(values, bins=20, alpha=0.45, label=region)
    ax.axvline(0.0, linestyle="--", linewidth=1.2, label="safety boundary")
    ax.set_xlabel("Minimum ceiling margin 3 - z [m]")
    ax.set_ylabel("Episodes")
    ax.set_title(f"{split}: minimum safe-set margin")
    ax.legend()
    _save(fig, output / f"{split}_minimum_safety_margin.png")


def plot_trajectory(
    trajectory: OfficialTrajectory,
    *,
    dt: float,
    z_max: float,
    z_des: float,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    stem = (
        f"{trajectory.split}_{trajectory.region}_"
        f"{trajectory.index:04d}_"
        f"{'success' if trajectory.strict_success else 'failure'}"
    )
    states = trajectory.states
    times_state = np.arange(states.shape[0]) * dt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=2)
    ax.scatter(states[0, 0], states[0, 1], states[0, 2], marker="o", label="start")
    ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2], marker="x", label="end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(
        f"{trajectory.split}/{trajectory.region}: "
        f"{'strict success' if trajectory.strict_success else 'not strict success'}"
    )
    ax.legend()
    _save(fig, output / f"{stem}_3d.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_state, states[:, 2], label="altitude")
    ax.axhline(z_max, linestyle="--", label="safety ceiling")
    ax.axhline(z_des, linestyle=":", label="hover target")
    if trajectory.entry_step >= 0:
        ax.axvline(trajectory.entry_step * dt, linestyle="-.", label="first entry into B")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"{trajectory.split}/{trajectory.region}: altitude")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, output / f"{stem}_altitude.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_state, trajectory.base_margin, label="base margin h_B")
    ax.plot(times_state, trajectory.safety_margin, label="safe margin h_S")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Margin")
    ax.set_title(f"{trajectory.split}/{trajectory.region}: set margins")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, output / f"{stem}_margins.png")

    if trajectory.applied_actions.shape[0] > 0:
        times_action = np.arange(trajectory.applied_actions.shape[0]) * dt
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        action_names = ("a_cmd", "omega_x", "omega_y", "omega_z")
        for idx, axis in enumerate(axes):
            axis.plot(times_action, trajectory.raw_actions[:, idx], label="actor raw")
            axis.plot(times_action, trajectory.applied_actions[:, idx], label="applied")
            axis.set_ylabel(action_names[idx])
            axis.grid(alpha=0.3)
        axes[0].legend()
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"{trajectory.split}/{trajectory.region}: actor/LQR handoff")
        _save(fig, output / f"{stem}_actions.png")

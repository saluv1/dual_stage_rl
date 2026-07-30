"""Plotting utilities for Phase-I training and final evaluation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .evaluation_detailed import EpisodeRecord, TrajectoryRecord, REGION_ORDER


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_training_history(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_training_history(history_csv: Path, output_dir: Path) -> None:
    rows = load_training_history(history_csv)
    if not rows:
        return
    steps = np.asarray([float(r["timestep"]) for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, [float(r["mu_sa"]) for r in rows], label="weighted mu_SA")
    ax.plot(steps, [float(r["success_rate"]) for r in rows], label="total safe arrival")
    ax.plot(steps, [float(r["success_horizon_rate"]) for r in rows], label="within horizon")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, output_dir / "training_safe_arrival_rates.png")

    region_columns = [c for c in rows[0] if c.endswith("_success_horizon_rate")]
    if region_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for column in region_columns:
            region = column.replace("_success_horizon_rate", "")
            values = [float(r[column]) if r[column] else np.nan for r in rows]
            ax.plot(steps, values, label=region)
        ax.set_xlabel("Training timestep")
        ax.set_ylabel("Safe arrival within horizon")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, output_dir / "training_per_curriculum_success.png")

    time_columns = [c for c in rows[0] if c.endswith("_mean_arrival_time_s")]
    if time_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for column in time_columns:
            label = column.replace("_mean_arrival_time_s", "")
            values = [float(r[column]) if r[column] not in ("", "None") else np.nan for r in rows]
            ax.plot(steps, values, label=label)
        ax.set_xlabel("Training timestep")
        ax.set_ylabel("Mean successful arrival time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, output_dir / "training_arrival_time.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(steps, [float(r["difficulty"]) for r in rows], where="post")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Curriculum difficulty s")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "training_curriculum_difficulty.png")


def plot_final_summary(summary: dict, records: Sequence[EpisodeRecord], output_dir: Path) -> None:
    per_region = summary["per_region"]
    names = list(per_region.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, [per_region[n]["safe_arrival_rate"] for n in names], width, label="Any horizon")
    ax.bar(x + width / 2, [per_region[n]["safe_arrival_within_horizon_rate"] for n in names], width, label="Within horizon")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, output_dir / "final_per_curriculum_success.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    labels = []
    for name in names:
        times = [r.arrival_time_s for r in records if r.region == name and r.arrival_time_s is not None]
        if times:
            data.append(times)
            labels.append(name)
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Successful arrival time (s)")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, output_dir / "final_arrival_time_boxplot.png")
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        margins = [r.min_hs for r in records if r.region == name]
        ax.hist(margins, bins=30, alpha=0.45, label=name)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Minimum h_S during episode")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, output_dir / "final_safety_margin_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        hb = [r.final_hb for r in records if r.region == name]
        ax.hist(hb, bins=30, alpha=0.45, label=name)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Final h_B")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, output_dir / "final_base_barrier_histogram.png")


def plot_trajectory(record: TrajectoryRecord, output_dir: Path, dt: float) -> None:
    states = record.states
    t = np.arange(len(states)) * dt
    stem = f"{record.region}_trajectory_{record.episode + 1:02d}_{record.outcome}"

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=2)
    ax.scatter(states[0, 0], states[0, 1], states[0, 2], marker="o", s=45, label="start")
    ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2], marker="x", s=55, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{record.region}: {record.outcome}, {record.terminal_step} steps")
    ax.legend()
    _save(fig, output_dir / f"{stem}_3d.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, states[:, 2], label="altitude z")
    ax.axhline(3.0, linestyle="--", linewidth=1, label="ceiling")
    ax.axhline(2.0, linestyle=":", linewidth=1, label="base equilibrium z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, output_dir / f"{stem}_altitude.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, record.hb, label="h_B")
    ax.plot(t, record.hs, label="h_S")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Barrier value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, output_dir / f"{stem}_barriers.png")


def animate_trajectory_gif(record: TrajectoryRecord, output_path: Path, dt: float, fps: int = 20) -> None:
    states = record.states
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    mins = states[:, :3].min(axis=0)
    maxs = states[:, :3].max(axis=0)
    spans = np.maximum(maxs - mins, 0.5)
    centers = 0.5 * (mins + maxs)
    radius = 0.6 * float(np.max(spans))
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    line, = ax.plot([], [], [], linewidth=2)
    point, = ax.plot([], [], [], marker="o")
    title = ax.set_title("")

    def update(frame):
        xyz = states[:frame + 1, :3]
        line.set_data(xyz[:, 0], xyz[:, 1])
        line.set_3d_properties(xyz[:, 2])
        point.set_data([xyz[-1, 0]], [xyz[-1, 1]])
        point.set_3d_properties([xyz[-1, 2]])
        title.set_text(f"{record.region} | t={frame * dt:.2f}s | {record.outcome}")
        return line, point, title

    animation = FuncAnimation(fig, update, frames=len(states), interval=1000 / fps, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

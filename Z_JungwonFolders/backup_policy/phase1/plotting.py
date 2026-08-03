"""Minimal plots for Phase-I training history."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(csv_path: str | Path, output_dir: str | Path) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    steps = np.asarray([float(row["timestep"]) for row in rows])
    mu_sa = np.asarray([float(row["mu_sa"]) for row in rows])
    difficulty = np.asarray([float(row["difficulty"]) for row in rows])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, mu_sa, label=r"weighted $\mu_{SA}$")
    ax1.set_xlabel("Environment timestep")
    ax1.set_ylabel(r"Weighted $\mu_{SA}$")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, difficulty, linestyle="--", label="curriculum difficulty")
    ax2.set_ylabel("Curriculum difficulty s")
    ax2.set_ylim(-0.02, 1.02)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "training_progress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

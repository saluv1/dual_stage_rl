"""Small, dependency-free helpers for indexed Phase-I experiments."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _next_index(root: Path) -> int:
    indices = []
    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and child.name.isdigit():
                indices.append(int(child.name))
    return max(indices, default=0) + 1


def create_run_directory(
    experiment_root: str | Path,
    run_index: int | None = None,
    evaluation_root: str | Path = "evaluation",
) -> dict[str, Path]:
    """Create ``Trained Models/001`` style output folders."""

    experiment_root = Path(experiment_root)
    evaluation_root = Path(evaluation_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    index = _next_index(experiment_root) if run_index is None else int(run_index)
    if index <= 0:
        raise ValueError("run_index must be positive")

    run_root = experiment_root / f"{index:03d}"
    if run_root.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_root}. "
            "Omit --run-index to create the next index."
        )

    paths = {
        "root": run_root,
        "checkpoints": run_root / "checkpoints",
        "configs": run_root / "config",
        "logs": run_root / "logs",
        "training_plots": evaluation_root / "Training Progress" / f"{index:03d}",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    save_json(
        run_root / "metadata.json",
        {
            "run_index": index,
            "status": "created",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return paths


def checkpoint_prefix(checkpoint_dir: str | Path, label: str) -> str:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / label)


def update_metadata(run_root: str | Path, **updates: Any) -> None:
    path = Path(run_root) / "metadata.json"
    payload: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    payload.update(updates)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(path, payload)

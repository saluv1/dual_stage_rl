"""Experiment directory, checkpoint, and training-history utilities."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

RUN_PATTERN = re.compile(r"^safe_arrival_(\d+)$")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return value


def next_run_index(root: os.PathLike[str] | str) -> int:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for child in root.iterdir():
        match = RUN_PATTERN.match(child.name)
        if child.is_dir() and match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def create_run_directory(
    root: os.PathLike[str] | str = "experiments/safe_arrival",
    run_index: Optional[int] = None,
) -> Dict[str, Path]:
    root = Path(root)
    index = next_run_index(root) if run_index is None else int(run_index)
    run_dir = root / f"safe_arrival_{index:03d}"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    paths = {
        "root": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "training_plots": run_dir / "training_plots",
        "evaluation": run_dir / "evaluation",
        "evaluation_logs": run_dir / "evaluation" / "logs",
        "evaluation_plots": run_dir / "evaluation" / "plots",
        "trajectories": run_dir / "evaluation" / "trajectories",
        "animations": run_dir / "evaluation" / "animations",
        "configs": run_dir / "configs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_index": index,
        "run_name": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "created",
    }
    save_json(paths["root"] / "run_metadata.json", metadata)
    return paths


def open_existing_run(run_dir: os.PathLike[str] | str) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    paths = {
        "root": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "training_plots": run_dir / "training_plots",
        "evaluation": run_dir / "evaluation",
        "evaluation_logs": run_dir / "evaluation" / "logs",
        "evaluation_plots": run_dir / "evaluation" / "plots",
        "trajectories": run_dir / "evaluation" / "trajectories",
        "animations": run_dir / "evaluation" / "animations",
        "configs": run_dir / "configs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path: os.PathLike[str] | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, sort_keys=True)


def update_metadata(run_dir: os.PathLike[str] | str, **updates: Any) -> None:
    path = Path(run_dir) / "run_metadata.json"
    metadata: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    metadata.update(_jsonable(updates))
    metadata["updated_at"] = datetime.now().astimezone().isoformat()
    save_json(path, metadata)


def append_csv(path: os.PathLike[str] | str, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {key: _jsonable(value) for key, value in row.items()}
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def checkpoint_prefix(checkpoint_dir: os.PathLike[str] | str, label: str) -> str:
    return str(Path(checkpoint_dir) / label)


def copy_checkpoint(policy, source_prefix: str, destination_prefix: str) -> None:
    """Copy a four-file TD3 checkpoint to a stable alias such as `best`."""
    suffixes = ["_critic", "_critic_optimizer", "_actor", "_actor_optimizer"]
    for suffix in suffixes:
        source = Path(source_prefix + suffix)
        destination = Path(destination_prefix + suffix)
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

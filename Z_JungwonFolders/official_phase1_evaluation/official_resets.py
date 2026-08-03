"""Portable loader and sampler for the released PS2-RL reset library.

Only ``reset_library.pkl`` is needed from the official repository. It contains
the four state pools and the official perturbation/configuration dictionaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np

OFFICIAL_REGIONS = (
    "general_trace",
    "near_ceiling",
    "bridge",
    "base_shell",
)

PAPER_COUNTS = {
    "general_trace": 1024,
    "near_ceiling": 1024,
    "bridge": 1024,
    "base_shell": 512,
}

DEFAULT_LIBRARY_CONFIG = {
    "position_perturb_min": 0.05,
    "position_perturb_max": 0.40,
    "velocity_perturb_min": 0.10,
    "velocity_perturb_max": 2.50,
    "tilt_perturb_deg_min": 2.0,
    "tilt_perturb_deg_max": 35.0,
    "yaw_perturb_deg_min": 1.0,
    "yaw_perturb_deg_max": 12.0,
    "general_region_multiplier": 1.0,
    "near_ceiling_region_multiplier": 1.5,
    "bridge_region_multiplier": 1.8,
    "base_shell_region_multiplier": 0.8,
    "max_resample_tries": 80,
}


@dataclass(frozen=True)
class ResetLibrary:
    path: Path
    cbf_config: dict[str, Any]
    library_config: dict[str, Any]
    trace_source_config: dict[str, Any]
    metadata: dict[str, Any]
    all_pools: dict[str, np.ndarray]
    split_pools: dict[str, dict[str, np.ndarray]]
    heldout_reset_sets: dict[str, dict[str, np.ndarray]]


def normalize_region_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    aliases = {
        "capture_shell": "base_shell",
        "synthetic_capture": "base_shell",
        "trace_general": "general_trace",
    }
    return aliases.get(text, text)


def _validate_pool(name: str, states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64).copy()
    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(f"Pool {name!r} must have shape (N, 10), got {states.shape}")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"Pool {name!r} contains non-finite values")
    q = states[:, 6:10]
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm[:, 0] < 1e-12):
        raise ValueError(f"Pool {name!r} contains a zero quaternion")
    states[:, 6:10] = q / norm
    return states


def load_reset_library(path: str | Path) -> ResetLibrary:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Official reset library not found: {source}")

    with source.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("reset_library.pkl must contain a dictionary")

    all_pools_raw = dict(raw.get("all_pools", {}))
    if not all_pools_raw:
        raise KeyError("reset_library.pkl does not contain non-empty 'all_pools'")

    all_pools: dict[str, np.ndarray] = {}
    for key, value in all_pools_raw.items():
        name = normalize_region_name(key)
        all_pools[name] = _validate_pool(name, value)

    missing = [name for name in OFFICIAL_REGIONS if name not in all_pools]
    if missing:
        raise KeyError(f"Official reset library is missing pools: {missing}")

    split_pools: dict[str, dict[str, np.ndarray]] = {}
    for split, pools in dict(raw.get("split_pools", {})).items():
        split_pools[str(split)] = {
            normalize_region_name(key): _validate_pool(
                f"{split}/{normalize_region_name(key)}", value
            )
            for key, value in dict(pools).items()
        }

    heldout_reset_sets: dict[str, dict[str, np.ndarray]] = {}
    for split, payload in dict(raw.get("heldout_reset_sets", {})).items():
        payload = dict(payload)
        states = _validate_pool(f"heldout/{split}", payload.get("states", np.zeros((0, 10))))
        regions = np.asarray(
            [normalize_region_name(value) for value in np.asarray(payload.get("region", []), dtype=object).reshape(-1)],
            dtype=str,
        )
        if len(states) != len(regions):
            raise ValueError(
                f"Held-out split {split!r} has {len(states)} states but {len(regions)} labels"
            )
        heldout_reset_sets[str(split)] = {"states": states, "region": regions}

    return ResetLibrary(
        path=source,
        cbf_config=dict(raw.get("cbf_cfg", {})),
        library_config=dict(raw.get("library_cfg", {})),
        trace_source_config=dict(raw.get("trace_source_cfg", {})),
        metadata=dict(raw.get("metadata", {})),
        all_pools=all_pools,
        split_pools=split_pools,
        heldout_reset_sets=heldout_reset_sets,
    )


def _cfg_float(library: ResetLibrary, key: str) -> float:
    return float(library.library_config.get(key, DEFAULT_LIBRARY_CONFIG[key]))


def _region_multiplier(library: ResetLibrary, region: str) -> float:
    key = {
        "general_trace": "general_region_multiplier",
        "near_ceiling": "near_ceiling_region_multiplier",
        "bridge": "bridge_region_multiplier",
        "base_shell": "base_shell_region_multiplier",
    }[region]
    return _cfg_float(library, key)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return normalize_quaternion(
        np.array(
            [
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                cy * sp * cr + sy * cp * sr,
                sy * cp * cr - cy * sp * sr,
            ],
            dtype=np.float64,
        )
    )


def perturb_ranges(
    library: ResetLibrary, region: str, curriculum_scale: float
) -> dict[str, float]:
    s = float(np.clip(curriculum_scale, 0.0, 1.0))
    multiplier = _region_multiplier(library, region)

    def lerp(low_key: str, high_key: str) -> float:
        return (1.0 - s) * _cfg_float(library, low_key) + s * _cfg_float(
            library, high_key
        )

    return {
        "position": multiplier * lerp("position_perturb_min", "position_perturb_max"),
        "velocity": multiplier * lerp("velocity_perturb_min", "velocity_perturb_max"),
        "tilt": np.deg2rad(
            multiplier * lerp("tilt_perturb_deg_min", "tilt_perturb_deg_max")
        ),
        "yaw": np.deg2rad(
            multiplier * lerp("yaw_perturb_deg_min", "yaw_perturb_deg_max")
        ),
    }


def perturb_state(
    library: ResetLibrary,
    base_state: np.ndarray,
    *,
    rng: np.random.Generator,
    curriculum_scale: float,
    region: str,
) -> np.ndarray:
    """Port of official ``QuadrotorResetLibrary.perturb_state``."""
    base = np.asarray(base_state, dtype=np.float64).reshape(10)
    ranges = perturb_ranges(library, region, curriculum_scale)
    max_tries = int(
        library.library_config.get(
            "max_resample_tries", DEFAULT_LIBRARY_CONFIG["max_resample_tries"]
        )
    )
    z_max = float(library.cbf_config.get("z_max", 3.0))

    for _ in range(max(1, max_tries)):
        state = base.copy()
        state[0:3] += rng.uniform(
            -ranges["position"], ranges["position"], size=3
        )
        state[3:6] += rng.uniform(
            -ranges["velocity"], ranges["velocity"], size=3
        )
        delta_q = quaternion_from_euler_zyx(
            float(rng.uniform(-ranges["tilt"], ranges["tilt"])),
            float(rng.uniform(-ranges["tilt"], ranges["tilt"])),
            float(rng.uniform(-ranges["yaw"], ranges["yaw"])),
        )
        state[6:10] = normalize_quaternion(
            quaternion_multiply(delta_q, base[6:10])
        )
        if state[2] <= z_max:
            return state

    fallback = base.copy()
    fallback[6:10] = normalize_quaternion(fallback[6:10])
    if fallback[2] > z_max:
        raise RuntimeError(f"Could not create a safe reset for region {region!r}")
    return fallback


def sample_region_states(
    library: ResetLibrary,
    region: str,
    *,
    count: int,
    seed: int,
    curriculum_scale: float = 1.0,
) -> np.ndarray:
    """Port of official ``sample_perturbed_region_states``."""
    region = normalize_region_name(region)
    states = library.all_pools.get(region)
    if states is None or len(states) == 0:
        raise ValueError(f"No official states are available for region {region!r}")
    rng = np.random.default_rng(int(seed))
    output = np.empty((int(count), 10), dtype=np.float64)
    for index in range(int(count)):
        base_index = int(rng.integers(0, len(states)))
        output[index] = perturb_state(
            library,
            states[base_index],
            rng=rng,
            curriculum_scale=curriculum_scale,
            region=region,
        )
    return output


def generate_benchmark(
    library: ResetLibrary,
    *,
    seed: int,
    counts: dict[str, int],
    curriculum_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Create and label the paper-sized regional benchmark.

    A distinct deterministic stream is used for every region. The released
    paper command supplies the common benchmark seed (0); the explicit stream
    mapping is saved in the benchmark metadata so every local checkpoint uses
    exactly the same frozen states.
    """
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    seeds: dict[str, int] = {}
    for region_index, region in enumerate(OFFICIAL_REGIONS):
        count = int(counts.get(region, 0))
        if count <= 0:
            continue
        region_seed = int(seed + 10_000 * region_index)
        seeds[region] = region_seed
        block = sample_region_states(
            library,
            region,
            count=count,
            seed=region_seed,
            curriculum_scale=curriculum_scale,
        )
        blocks.append(block)
        labels.extend([region] * count)
    if not blocks:
        raise ValueError("At least one benchmark region count must be positive")
    return (
        np.concatenate(blocks, axis=0),
        np.asarray(labels, dtype=str),
        seeds,
    )


def save_benchmark(
    path: str | Path,
    *,
    states: np.ndarray,
    regions: np.ndarray,
    source_library: str | Path,
    seed: int,
    region_seeds: dict[str, int],
    curriculum_scale: float,
    counts: dict[str, int],
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=np.asarray(states, dtype=np.float64),
        regions=np.asarray(regions, dtype=str),
        source_library=np.asarray(str(Path(source_library).resolve())),
        seed=np.asarray(int(seed), dtype=np.int64),
        curriculum_scale=np.asarray(float(curriculum_scale), dtype=np.float64),
        region_names=np.asarray(list(OFFICIAL_REGIONS), dtype=str),
        region_counts=np.asarray([counts.get(r, 0) for r in OFFICIAL_REGIONS]),
        region_seeds=np.asarray([region_seeds.get(r, -1) for r in OFFICIAL_REGIONS]),
    )
    return output


def load_benchmark(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float64)
        regions = np.asarray(data["regions"], dtype=str)
    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(f"Benchmark states must have shape (N, 10), got {states.shape}")
    if len(regions) != len(states):
        raise ValueError("Benchmark state and region counts do not match")
    return states, regions

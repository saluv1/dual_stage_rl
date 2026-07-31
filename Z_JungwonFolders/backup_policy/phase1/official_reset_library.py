"""Load the official PS2-RL quadrotor Phase-I held-out reset sets.

The released ``reset_library.pkl`` is a plain pickle payload containing numpy
arrays and dictionaries. Loading it does not require JAX or the official
``ps2rl`` package.
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


@dataclass(frozen=True)
class OfficialHeldoutSplit:
    """Fixed initial states and region labels for one official split."""

    name: str
    states: np.ndarray
    regions: np.ndarray

    def trimmed(self, max_resets: int = 0) -> "OfficialHeldoutSplit":
        if max_resets <= 0:
            return self
        # Match the official entry script: cap the full split arrays, not each
        # region independently.
        n = min(int(max_resets), int(self.states.shape[0]))
        return OfficialHeldoutSplit(
            name=self.name,
            states=self.states[:n].copy(),
            regions=self.regions[:n].copy(),
        )


@dataclass(frozen=True)
class OfficialResetLibraryPayload:
    """Official reset-library arrays and metadata needed by the evaluator."""

    path: Path
    cbf_config: dict[str, Any]
    library_config: dict[str, Any]
    trace_source_config: dict[str, Any]
    metadata: dict[str, Any]
    all_pools: dict[str, np.ndarray]
    split_pools: dict[str, dict[str, np.ndarray]]
    splits: dict[str, OfficialHeldoutSplit]


def _normalise_region_name(value: Any) -> str:
    text = str(value)
    # Older payloads used ``capture_shell``. The current official loader maps
    # that legacy name to ``base_shell``.
    return "base_shell" if text == "capture_shell" else text


def _validate_split(name: str, raw: dict[str, Any]) -> OfficialHeldoutSplit:
    if "states" not in raw or "region" not in raw:
        raise KeyError(
            f"Held-out split '{name}' must contain 'states' and 'region'. "
            f"Available keys: {sorted(raw.keys())}"
        )

    states = np.asarray(raw["states"], dtype=np.float64)
    regions = np.asarray(
        [_normalise_region_name(v) for v in np.asarray(raw["region"]).reshape(-1)],
        dtype=str,
    )

    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(
            f"Expected official split '{name}' states with shape (N, 10), "
            f"got {states.shape}."
        )
    if states.shape[0] != regions.shape[0]:
        raise ValueError(
            f"Split '{name}' has {states.shape[0]} states but "
            f"{regions.shape[0]} region labels."
        )
    if not np.all(np.isfinite(states)):
        bad = np.argwhere(~np.isfinite(states))[:10].tolist()
        raise ValueError(f"Split '{name}' contains non-finite states at {bad}.")

    # Normalise all quaternions. The official builder also normalises them.
    q = states[:, 6:10]
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    bad_q = q_norm[:, 0] < 1e-12
    q_norm[bad_q] = 1.0
    states[:, 6:10] = q / q_norm
    states[bad_q, 6:10] = np.asarray([1.0, 0.0, 0.0, 0.0])

    unknown = sorted(set(regions.tolist()) - set(OFFICIAL_REGIONS))
    if unknown:
        raise ValueError(
            f"Split '{name}' contains unknown region labels {unknown}. "
            f"Expected labels: {list(OFFICIAL_REGIONS)}"
        )

    return OfficialHeldoutSplit(name=name, states=states, regions=regions)


def load_official_reset_library(path: str | Path) -> OfficialResetLibraryPayload:
    """Load the released official ``reset_library.pkl`` without JAX."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Official reset library not found: {source}")

    with source.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(
            "Expected the official reset library pickle to contain a dict, "
            f"got {type(payload)!r}."
        )

    heldout = payload.get("heldout_reset_sets")
    if not isinstance(heldout, dict):
        raise KeyError("reset_library.pkl does not contain 'heldout_reset_sets'.")

    splits: dict[str, OfficialHeldoutSplit] = {}
    for split_name in ("val", "test"):
        if split_name not in heldout:
            raise KeyError(
                f"reset_library.pkl is missing held-out split '{split_name}'."
            )
        splits[split_name] = _validate_split(split_name, heldout[split_name])

    cbf_config = dict(payload.get("cbf_cfg", {}))
    library_config = dict(payload.get("library_cfg", {}))
    trace_source_config = dict(payload.get("trace_source_cfg", {}))
    metadata = dict(payload.get("metadata", {}))
    all_pools = {
        _normalise_region_name(key): np.asarray(value, dtype=np.float64)
        for key, value in dict(payload.get("all_pools", {})).items()
    }
    split_pools = {
        str(split): {
            _normalise_region_name(key): np.asarray(value, dtype=np.float64)
            for key, value in dict(pools).items()
        }
        for split, pools in dict(payload.get("split_pools", {})).items()
    }

    return OfficialResetLibraryPayload(
        path=source,
        cbf_config=cbf_config,
        library_config=library_config,
        trace_source_config=trace_source_config,
        metadata=metadata,
        all_pools=all_pools,
        split_pools=split_pools,
        splits=splits,
    )


_DEFAULT_LIBRARY_CONFIG = {
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

PAPER_SIZE_COUNTS = {
    "general_trace": 1024,
    "near_ceiling": 1024,
    "bridge": 1024,
    "base_shell": 512,
}


def _cfg_float(payload: OfficialResetLibraryPayload, key: str) -> float:
    return float(payload.library_config.get(key, _DEFAULT_LIBRARY_CONFIG[key]))


def _region_multiplier(payload: OfficialResetLibraryPayload, region: str) -> float:
    key = {
        "general_trace": "general_region_multiplier",
        "near_ceiling": "near_ceiling_region_multiplier",
        "bridge": "bridge_region_multiplier",
        "base_shell": "base_shell_region_multiplier",
    }[region]
    return _cfg_float(payload, key)


def _normalise_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quaternion_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return _normalise_quaternion(
        np.asarray(
            [
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                cy * sp * cr + sy * cp * sr,
                sy * cp * cr - cy * sp * sr,
            ],
            dtype=np.float64,
        )
    )


def _perturb_ranges(
    payload: OfficialResetLibraryPayload,
    region: str,
    curriculum_scale: float,
) -> dict[str, float]:
    s = float(np.clip(curriculum_scale, 0.0, 1.0))
    multiplier = _region_multiplier(payload, region)
    return {
        "pos": multiplier
        * ((1.0 - s) * _cfg_float(payload, "position_perturb_min")
           + s * _cfg_float(payload, "position_perturb_max")),
        "vel": multiplier
        * ((1.0 - s) * _cfg_float(payload, "velocity_perturb_min")
           + s * _cfg_float(payload, "velocity_perturb_max")),
        "tilt_rad": np.deg2rad(
            multiplier
            * ((1.0 - s) * _cfg_float(payload, "tilt_perturb_deg_min")
               + s * _cfg_float(payload, "tilt_perturb_deg_max"))
        ),
        "yaw_rad": np.deg2rad(
            multiplier
            * ((1.0 - s) * _cfg_float(payload, "yaw_perturb_deg_min")
               + s * _cfg_float(payload, "yaw_perturb_deg_max"))
        ),
    }


def perturb_official_state(
    payload: OfficialResetLibraryPayload,
    base_state: np.ndarray,
    *,
    rng: np.random.Generator,
    curriculum_scale: float,
    region: str,
) -> np.ndarray:
    """Port of the official reset library's ``perturb_state`` method."""

    x0 = np.asarray(base_state, dtype=np.float64).reshape(10)
    ranges = _perturb_ranges(payload, region, curriculum_scale)
    max_tries = int(payload.library_config.get(
        "max_resample_tries", _DEFAULT_LIBRARY_CONFIG["max_resample_tries"]
    ))
    z_max = float(payload.cbf_config.get("z_max", 3.0))

    for _ in range(max(1, max_tries)):
        x = x0.copy()
        x[0:3] += rng.uniform(-ranges["pos"], ranges["pos"], size=3)
        x[3:6] += rng.uniform(-ranges["vel"], ranges["vel"], size=3)
        q_delta = _quaternion_from_euler_zyx(
            float(rng.uniform(-ranges["tilt_rad"], ranges["tilt_rad"])),
            float(rng.uniform(-ranges["tilt_rad"], ranges["tilt_rad"])),
            float(rng.uniform(-ranges["yaw_rad"], ranges["yaw_rad"])),
        )
        x[6:10] = _normalise_quaternion(
            _quaternion_multiply(q_delta, x0[6:10])
        )
        if x[2] <= z_max:
            return x

    x_safe = x0.copy()
    x_safe[6:10] = _normalise_quaternion(x_safe[6:10])
    if x_safe[2] > z_max:
        raise RuntimeError(
            f"Could not sample a safe perturbed reset from region={region!r}."
        )
    return x_safe


def sample_official_region_states(
    payload: OfficialResetLibraryPayload,
    region: str,
    *,
    count: int,
    seed: int,
    curriculum_scale: float = 1.0,
    split: str | None = None,
) -> np.ndarray:
    """Match official ``sample_perturbed_region_states`` without JAX."""

    region = _normalise_region_name(region)
    pools = payload.all_pools if split is None else payload.split_pools.get(split, {})
    states = pools.get(region)
    if states is None or len(states) == 0:
        raise ValueError(
            f"No states available for region={region!r}, split={split!r}."
        )
    rng = np.random.default_rng(int(seed))
    output = []
    for _ in range(int(count)):
        index = int(rng.integers(0, len(states)))
        output.append(
            perturb_official_state(
                payload,
                states[index],
                rng=rng,
                curriculum_scale=curriculum_scale,
                region=region,
            )
        )
    return np.asarray(output, dtype=np.float64)


def generate_paper_size_evaluation_set(
    payload: OfficialResetLibraryPayload,
    *,
    seed: int = 1234,
    curriculum_scale: float = 1.0,
    counts: dict[str, int] | None = None,
) -> OfficialHeldoutSplit:
    """Generate the paper-sized 3,584-state Monte Carlo benchmark.

    The per-region seed is ``seed + 10_000 * region_index``. Saving this set
    once ensures all later policy comparisons use exactly the same states.
    """

    requested = PAPER_SIZE_COUNTS if counts is None else counts
    state_blocks: list[np.ndarray] = []
    labels: list[str] = []
    for region_index, region in enumerate(OFFICIAL_REGIONS):
        count = int(requested.get(region, 0))
        if count <= 0:
            continue
        region_states = sample_official_region_states(
            payload,
            region,
            count=count,
            seed=int(seed + 10_000 * region_index),
            curriculum_scale=curriculum_scale,
            split=None,
        )
        state_blocks.append(region_states)
        labels.extend([region] * region_states.shape[0])
    if not state_blocks:
        raise ValueError("No benchmark states were requested.")
    return OfficialHeldoutSplit(
        name="paper_size",
        states=np.concatenate(state_blocks, axis=0),
        regions=np.asarray(labels, dtype=str),
    )


def save_evaluation_state_file(
    path: str | Path,
    evaluation_set: OfficialHeldoutSplit,
    *,
    source_reset_library: str | Path,
    seed: int,
    curriculum_scale: float,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=np.asarray(evaluation_set.states, dtype=np.float64),
        region=np.asarray(evaluation_set.regions, dtype=str),
        split=np.asarray(evaluation_set.name),
        seed=np.asarray(int(seed), dtype=np.int64),
        curriculum_scale=np.asarray(float(curriculum_scale), dtype=np.float64),
        source_reset_library=np.asarray(str(Path(source_reset_library).resolve())),
    )
    return output


def load_evaluation_state_file(path: str | Path) -> OfficialHeldoutSplit:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Evaluation state file not found: {source}")
    with np.load(source, allow_pickle=False) as data:
        if "states" not in data or "region" not in data:
            raise KeyError("State file must contain 'states' and 'region'.")
        raw = {
            "states": np.asarray(data["states"], dtype=np.float64),
            "region": np.asarray(data["region"]),
        }
        name = "paper_size"
        if "split" in data and np.asarray(data["split"]).size:
            name = str(np.asarray(data["split"]).reshape(-1)[0])
    return _validate_split(name, raw)

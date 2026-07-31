"""PyTorch-compatible reproduction of the official Phase-I held-out evaluator."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .official_quadrotor import (
    OfficialDLQR,
    OfficialQuadrotorConfig,
    clip_physical_action,
    is_safe,
    safety_margin,
)
from .official_reset_library import OFFICIAL_REGIONS, OfficialHeldoutSplit

OFFICIAL_RECOVERABILITY_WEIGHTS = {
    "general_trace": 1.0,
    "near_ceiling": 2.0,
    "bridge": 2.5,
    "base_shell": 1.0,
}


@dataclass
class OfficialEpisodeRecord:
    split: str
    index: int
    region: str
    success: float
    crash: float
    safe_rate: float
    capture_rate: float
    terminal_rate: float
    safe_rollout: float
    terminal_at_horizon: float
    entered_terminal: float
    invariance_after_entry: float
    discounted_ra_score: float
    entry_step: int
    entry_time_sec: float
    min_hard_deck_margin: float
    post_entry_terminal_steps: int
    post_entry_total_steps: int
    initial_z: float
    final_z: float
    max_z: float


@dataclass
class OfficialTrajectory:
    split: str
    index: int
    region: str
    states: np.ndarray
    raw_actions: np.ndarray
    applied_actions: np.ndarray
    safe: np.ndarray
    terminal: np.ndarray
    base_margin: np.ndarray
    safety_margin: np.ndarray
    used_lqr: np.ndarray
    crashed: bool
    strict_success: bool
    entry_step: int


def _summary_stats(values: Iterable[float]) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p10": float(np.quantile(data, 0.10)),
        "p90": float(np.quantile(data, 0.90)),
    }


def _mean(records: list[OfficialEpisodeRecord], field: str) -> float:
    if not records:
        return 0.0
    return float(np.mean([float(getattr(record, field)) for record in records]))


def aggregate_episode_records(
    records: list[OfficialEpisodeRecord],
) -> dict[str, Any]:
    if not records:
        empty = _summary_stats([])
        return {
            "count": 0,
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "safe_rate": 0.0,
            "capture_rate": 0.0,
            "terminal_rate": 0.0,
            "safe_rollout_rate": 0.0,
            "terminal_at_horizon_rate": 0.0,
            "entered_terminal_rate": 0.0,
            "invariance_after_entry_rate": 0.0,
            "post_entry_terminal_step_rate": 0.0,
            "post_entry_terminal_steps": 0,
            "post_entry_total_steps": 0,
            "mean_discounted_ra_score": 0.0,
            "entry_time_sec": empty,
            "minimum_hard_deck_margin": empty,
            "maximum_altitude": empty,
        }

    post_terminal = int(sum(r.post_entry_terminal_steps for r in records))
    post_total = int(sum(r.post_entry_total_steps for r in records))
    return {
        "count": int(len(records)),
        "success_rate": _mean(records, "success"),
        "crash_rate": _mean(records, "crash"),
        "safe_rate": _mean(records, "safe_rate"),
        "capture_rate": _mean(records, "capture_rate"),
        "terminal_rate": _mean(records, "terminal_rate"),
        "safe_rollout_rate": _mean(records, "safe_rollout"),
        "terminal_at_horizon_rate": _mean(records, "terminal_at_horizon"),
        "entered_terminal_rate": _mean(records, "entered_terminal"),
        "invariance_after_entry_rate": _mean(records, "invariance_after_entry"),
        "post_entry_terminal_step_rate": (
            float(post_terminal / post_total) if post_total > 0 else 0.0
        ),
        "post_entry_terminal_steps": post_terminal,
        "post_entry_total_steps": post_total,
        "mean_discounted_ra_score": _mean(records, "discounted_ra_score"),
        "entry_time_sec": _summary_stats(r.entry_time_sec for r in records),
        "minimum_hard_deck_margin": _summary_stats(
            r.min_hard_deck_margin for r in records
        ),
        "maximum_altitude": _summary_stats(r.max_z for r in records),
    }


def weighted_recoverability_score(
    subset_metrics: dict[str, dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> float:
    weight_map = OFFICIAL_RECOVERABILITY_WEIGHTS if weights is None else weights
    numerator = 0.0
    denominator = 0.0
    for region, weight in weight_map.items():
        metrics = subset_metrics.get(region)
        if metrics is None or int(metrics.get("count", 0)) <= 0:
            continue
        numerator += float(weight) * float(metrics.get("success_rate", 0.0))
        denominator += float(weight)
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def rollout_official_episode(
    *,
    split: str,
    index: int,
    region: str,
    initial_state: np.ndarray,
    actor_action_phys: Callable[[np.ndarray], np.ndarray],
    integrator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    cfg: OfficialQuadrotorConfig,
    base_controller: OfficialDLQR,
    horizon_steps: int,
    beta: float,
) -> tuple[OfficialEpisodeRecord, OfficialTrajectory]:
    """Run one episode with the same metric semantics as the official JAX code."""

    x = np.asarray(initial_state, dtype=np.float64).copy()
    x[6:10] /= max(float(np.linalg.norm(x[6:10])), 1e-12)

    safe0 = is_safe(x, cfg)
    terminal0 = base_controller.contains(x)
    goal0 = safe0 and terminal0

    step_count = 0
    safe_sum = 0.0
    capture_sum = 0.0
    terminal_sum = 0.0
    safe_rollout = True
    entered_terminal = bool(goal0)
    left_after_entry = False
    entry_step = 0 if goal0 else -1
    crash_flag = False
    min_margin = safety_margin(x, cfg)
    post_entry_terminal_steps = 0
    post_entry_total_steps = 0

    states = [x.copy()]
    raw_actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    safe_trace: list[float] = []
    terminal_trace: list[float] = []
    base_margin_trace = [base_controller.base_margin(x)]
    safety_margin_trace = [min_margin]
    used_lqr_trace: list[float] = []

    for k in range(int(horizon_steps)):
        raw = clip_physical_action(actor_action_phys(x), cfg)
        use_lqr = base_controller.contains(x)
        applied = base_controller.action(x) if use_lqr else raw
        applied = clip_physical_action(applied, cfg)

        x_next = integrator(x, applied)
        safe = is_safe(x_next, cfg)
        terminal = base_controller.contains(x_next)
        capture = terminal
        goal = safe and terminal

        step_count += 1
        safe_sum += float(safe)
        capture_sum += float(capture)
        terminal_sum += float(terminal)
        safe_rollout = bool(safe_rollout and safe)

        post_entry_active = bool(entered_terminal or goal)
        if post_entry_active:
            post_entry_total_steps += 1
            post_entry_terminal_steps += int(terminal)

        if (not entered_terminal) and goal:
            entry_step = k + 1
        if entered_terminal and (not goal):
            left_after_entry = True
        entered_terminal = bool(entered_terminal or goal)

        margin = safety_margin(x_next, cfg)
        min_margin = min(min_margin, margin)

        raw_actions.append(raw.copy())
        applied_actions.append(applied.copy())
        safe_trace.append(float(safe))
        terminal_trace.append(float(terminal))
        used_lqr_trace.append(float(use_lqr))
        states.append(x_next.copy())
        base_margin_trace.append(base_controller.base_margin(x_next))
        safety_margin_trace.append(margin)

        x = x_next
        if not safe:
            crash_flag = True
            break

    eval_len = max(step_count, 1)
    terminal_at_horizon = bool(
        (not crash_flag) and is_safe(x, cfg) and base_controller.contains(x)
    )
    invariance_after_entry = bool(entered_terminal and (not left_after_entry))
    strict_success = bool(
        safe_rollout
        and entered_terminal
        and (not left_after_entry)
        and terminal_at_horizon
    )
    discounted_score = (
        float(beta ** entry_step) if strict_success and entry_step >= 0 else 0.0
    )
    entry_time_sec = float(entry_step * cfg.dt) if entry_step >= 0 else float("nan")

    states_arr = np.asarray(states, dtype=np.float64)
    episode = OfficialEpisodeRecord(
        split=split,
        index=int(index),
        region=str(region),
        success=float(strict_success),
        crash=float(crash_flag),
        safe_rate=float(safe_sum / eval_len),
        capture_rate=float(capture_sum / eval_len),
        terminal_rate=float(terminal_sum / eval_len),
        safe_rollout=float(safe_rollout),
        terminal_at_horizon=float(terminal_at_horizon),
        entered_terminal=float(entered_terminal),
        invariance_after_entry=float(invariance_after_entry),
        discounted_ra_score=discounted_score,
        entry_step=int(entry_step),
        entry_time_sec=entry_time_sec,
        min_hard_deck_margin=float(min_margin),
        post_entry_terminal_steps=int(post_entry_terminal_steps),
        post_entry_total_steps=int(post_entry_total_steps),
        initial_z=float(states_arr[0, 2]),
        final_z=float(states_arr[-1, 2]),
        max_z=float(np.max(states_arr[:, 2])),
    )
    trajectory = OfficialTrajectory(
        split=split,
        index=int(index),
        region=str(region),
        states=states_arr,
        raw_actions=np.asarray(raw_actions, dtype=np.float64).reshape(-1, 4),
        applied_actions=np.asarray(applied_actions, dtype=np.float64).reshape(-1, 4),
        safe=np.asarray(safe_trace, dtype=np.float64),
        terminal=np.asarray(terminal_trace, dtype=np.float64),
        base_margin=np.asarray(base_margin_trace, dtype=np.float64),
        safety_margin=np.asarray(safety_margin_trace, dtype=np.float64),
        used_lqr=np.asarray(used_lqr_trace, dtype=np.float64),
        crashed=bool(crash_flag),
        strict_success=bool(strict_success),
        entry_step=int(entry_step),
    )
    return episode, trajectory


def evaluate_official_split(
    *,
    heldout: OfficialHeldoutSplit,
    actor_action_phys: Callable[[np.ndarray], np.ndarray],
    integrator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    cfg: OfficialQuadrotorConfig,
    base_controller: OfficialDLQR,
    horizon_steps: int = 100,
    beta: float = 0.984035,
    trajectories_per_region: int = 3,
) -> tuple[dict[str, Any], list[OfficialEpisodeRecord], list[OfficialTrajectory]]:
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must lie in (0, 1), got {beta}")

    episodes: list[OfficialEpisodeRecord] = []
    chosen_trajectories: list[OfficialTrajectory] = []
    region_trajectory_counts = {region: 0 for region in OFFICIAL_REGIONS}

    for idx, (state, region) in enumerate(zip(heldout.states, heldout.regions)):
        episode, trajectory = rollout_official_episode(
            split=heldout.name,
            index=idx,
            region=str(region),
            initial_state=state,
            actor_action_phys=actor_action_phys,
            integrator=integrator,
            cfg=cfg,
            base_controller=base_controller,
            horizon_steps=horizon_steps,
            beta=beta,
        )
        episodes.append(episode)
        if region_trajectory_counts[str(region)] < int(trajectories_per_region):
            chosen_trajectories.append(trajectory)
            region_trajectory_counts[str(region)] += 1

    overall = aggregate_episode_records(episodes)
    subset_metrics = {
        region: aggregate_episode_records(
            [episode for episode in episodes if episode.region == region]
        )
        for region in OFFICIAL_REGIONS
    }
    overall["weighted_recoverability_score"] = weighted_recoverability_score(
        subset_metrics
    )
    overall["subset_metrics"] = subset_metrics
    return overall, episodes, chosen_trajectories


def save_episode_csv(path: str | Path, records: list[OfficialEpisodeRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else [
        field.name for field in OfficialEpisodeRecord.__dataclass_fields__.values()
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            # CSVs are easier to process when missing entry times are empty.
            if not np.isfinite(float(row["entry_time_sec"])):
                row["entry_time_sec"] = ""
            writer.writerow(row)


def save_subset_csv(path: str | Path, split_summaries: dict[str, dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split",
        "region",
        "count",
        "success_rate",
        "crash_rate",
        "safe_rate",
        "capture_rate",
        "terminal_rate",
        "safe_rollout_rate",
        "terminal_at_horizon_rate",
        "entered_terminal_rate",
        "invariance_after_entry_rate",
        "post_entry_terminal_step_rate",
        "mean_discounted_ra_score",
        "entry_time_mean_sec",
        "entry_time_median_sec",
        "min_hard_deck_margin_mean",
        "min_hard_deck_margin_min",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for split, summary in split_summaries.items():
            for region, metrics in summary["subset_metrics"].items():
                writer.writerow(
                    {
                        "split": split,
                        "region": region,
                        "count": metrics["count"],
                        "success_rate": metrics["success_rate"],
                        "crash_rate": metrics["crash_rate"],
                        "safe_rate": metrics["safe_rate"],
                        "capture_rate": metrics["capture_rate"],
                        "terminal_rate": metrics["terminal_rate"],
                        "safe_rollout_rate": metrics["safe_rollout_rate"],
                        "terminal_at_horizon_rate": metrics[
                            "terminal_at_horizon_rate"
                        ],
                        "entered_terminal_rate": metrics["entered_terminal_rate"],
                        "invariance_after_entry_rate": metrics[
                            "invariance_after_entry_rate"
                        ],
                        "post_entry_terminal_step_rate": metrics[
                            "post_entry_terminal_step_rate"
                        ],
                        "mean_discounted_ra_score": metrics[
                            "mean_discounted_ra_score"
                        ],
                        "entry_time_mean_sec": metrics["entry_time_sec"]["mean"],
                        "entry_time_median_sec": metrics["entry_time_sec"]["median"],
                        "min_hard_deck_margin_mean": metrics[
                            "minimum_hard_deck_margin"
                        ]["mean"],
                        "min_hard_deck_margin_min": metrics[
                            "minimum_hard_deck_margin"
                        ]["min"],
                    }
                )


def save_trajectory_npz(path: str | Path, trajectory: OfficialTrajectory) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        split=np.asarray(trajectory.split),
        index=np.asarray(trajectory.index),
        region=np.asarray(trajectory.region),
        states=trajectory.states,
        raw_actions=trajectory.raw_actions,
        applied_actions=trajectory.applied_actions,
        safe=trajectory.safe,
        terminal=trajectory.terminal,
        base_margin=trajectory.base_margin,
        safety_margin=trajectory.safety_margin,
        used_lqr=trajectory.used_lqr,
        crashed=np.asarray(trajectory.crashed),
        strict_success=np.asarray(trajectory.strict_success),
        entry_step=np.asarray(trajectory.entry_step),
    )


def save_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2)

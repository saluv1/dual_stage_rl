"""Detailed safe-arrival evaluation, trajectory collection, plots, and animations."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from env.dynamics import Dynamics
from .state_action import compute_bfc, compute_reduced_state, reset_dynamics_state, scale_action
from .sampling import MU_SA_WEIGHTS, sample_initial_state

REGION_ORDER = [
    "synthetic_capture",
    "synthetic_mid",
    "trace_general",
    "near_ceiling",
    "bridge",
]


@dataclass
class EpisodeRecord:
    region: str
    episode: int
    outcome: str
    terminal_step: int
    arrival_time_s: Optional[float]
    arrived_within_horizon: bool
    min_hs: float
    final_hb: float
    max_altitude: float
    initial_hb: float
    initial_hs: float


@dataclass
class TrajectoryRecord:
    region: str
    episode: int
    outcome: str
    terminal_step: int
    states: np.ndarray
    actions_norm: np.ndarray
    actions_physical: np.ndarray
    hb: np.ndarray
    hs: np.ndarray


def _available_regions(regions: Dict[str, np.ndarray]) -> List[str]:
    return [
        name for name in REGION_ORDER
        if name in ("synthetic_capture", "synthetic_mid")
        or (name in regions and len(regions[name]) > 0)
    ]


def rollout_episode(policy, sets, initial_state, max_episode_steps: int) -> TrajectoryRecord:
    dyn = Dynamics()
    state = np.asarray(initial_state, dtype=float).copy()
    reset_dynamics_state(dyn, state)

    states = [state.copy()]
    actions_norm = []
    actions_physical = []
    hb_values = []
    hs_values = []

    compute_bfc(sets, state)
    hb_values.append(float(sets.hb))
    hs_values.append(float(sets.hs))

    outcome = "timeout"
    terminal_step = max_episode_steps

    for step in range(max_episode_steps):
        action_norm = np.clip(policy.select_action(state), -1.0, 1.0)
        action = scale_action(action_norm, dyn.g)
        next_state = dyn.step(action).copy()
        b_next, f_next, _ = compute_bfc(sets, next_state)

        actions_norm.append(action_norm.copy())
        actions_physical.append(action.copy())
        states.append(next_state.copy())
        hb_values.append(float(sets.hb))
        hs_values.append(float(sets.hs))
        state = next_state

        if b_next == 1.0:
            outcome = "success"
            terminal_step = step + 1
            break
        if f_next == 1.0:
            outcome = "failure"
            terminal_step = step + 1
            break

    return TrajectoryRecord(
        region="",
        episode=-1,
        outcome=outcome,
        terminal_step=terminal_step,
        states=np.asarray(states, dtype=float),
        actions_norm=np.asarray(actions_norm, dtype=float),
        actions_physical=np.asarray(actions_physical, dtype=float),
        hb=np.asarray(hb_values, dtype=float),
        hs=np.asarray(hs_values, dtype=float),
    )


def evaluate_detailed(
    policy,
    sets,
    regions,
    rng,
    difficulty: float = 1.0,
    episodes_per_region: int = 100,
    trajectories_per_region: int = 5,
    max_episode_steps: int = 300,
    success_horizon_steps: int = 100,
) -> Tuple[dict, List[EpisodeRecord], List[TrajectoryRecord]]:
    dt = Dynamics().del_t
    available = _available_regions(regions)
    episode_records: List[EpisodeRecord] = []
    trajectory_records: List[TrajectoryRecord] = []

    for region in available:
        for episode in range(episodes_per_region):
            initial_state, region_name = sample_initial_state(
                sets=sets,
                regions=regions,
                s=difficulty,
                rng=rng,
                return_region=True,
                force_region=region,
            )
            compute_bfc(sets, initial_state)
            initial_hb = float(sets.hb)
            initial_hs = float(sets.hs)

            trajectory = rollout_episode(policy, sets, initial_state, max_episode_steps)
            trajectory.region = region_name
            trajectory.episode = episode

            arrived_within = (
                trajectory.outcome == "success"
                and trajectory.terminal_step <= success_horizon_steps
            )
            arrival_time = (
                trajectory.terminal_step * dt
                if trajectory.outcome == "success"
                else None
            )
            episode_records.append(EpisodeRecord(
                region=region_name,
                episode=episode,
                outcome=trajectory.outcome,
                terminal_step=trajectory.terminal_step,
                arrival_time_s=arrival_time,
                arrived_within_horizon=arrived_within,
                min_hs=float(np.min(trajectory.hs)),
                final_hb=float(trajectory.hb[-1]),
                max_altitude=float(np.max(trajectory.states[:, 2])),
                initial_hb=initial_hb,
                initial_hs=initial_hs,
            ))

            if episode < trajectories_per_region:
                trajectory_records.append(trajectory)

    summary = summarize_records(
        episode_records,
        success_horizon_steps=success_horizon_steps,
        dt=dt,
    )
    return summary, episode_records, trajectory_records


def summarize_records(records: Sequence[EpisodeRecord], success_horizon_steps: int, dt: float) -> dict:
    total = len(records)
    if total == 0:
        raise ValueError("No evaluation episodes were collected.")

    def summarize_subset(subset: Sequence[EpisodeRecord]) -> dict:
        n = len(subset)
        successes = [r for r in subset if r.outcome == "success"]
        within = [r for r in subset if r.arrived_within_horizon]
        failures = [r for r in subset if r.outcome == "failure"]
        timeouts = [r for r in subset if r.outcome == "timeout"]
        arrival_times = [r.arrival_time_s for r in successes if r.arrival_time_s is not None]
        return {
            "episodes": n,
            "safe_arrival_rate": len(successes) / n,
            "safe_arrival_within_horizon_rate": len(within) / n,
            "failure_rate": len(failures) / n,
            "timeout_rate": len(timeouts) / n,
            "mean_arrival_time_s_success_only": float(np.mean(arrival_times)) if arrival_times else None,
            "median_arrival_time_s_success_only": float(np.median(arrival_times)) if arrival_times else None,
            "p90_arrival_time_s_success_only": float(np.percentile(arrival_times, 90)) if arrival_times else None,
            "mean_terminal_time_s_all": float(np.mean([r.terminal_step * dt for r in subset])),
            "mean_min_hs": float(np.mean([r.min_hs for r in subset])),
            "worst_min_hs": float(np.min([r.min_hs for r in subset])),
            "mean_final_hb": float(np.mean([r.final_hb for r in subset])),
            "median_final_hb": float(np.median([r.final_hb for r in subset])),
            "max_altitude_m": float(np.max([r.max_altitude for r in subset])),
        }

    per_region = {}
    for region in REGION_ORDER:
        subset = [r for r in records if r.region == region]
        if subset:
            per_region[region] = summarize_subset(subset)

    mu_sa = 0.0
    weight_total = 0.0
    for region, metrics in per_region.items():
        weight = MU_SA_WEIGHTS.get(region, 0.0)
        if weight > 0:
            mu_sa += weight * metrics["safe_arrival_within_horizon_rate"]
            weight_total += weight
    if weight_total > 0:
        mu_sa /= weight_total

    overall = summarize_subset(records)
    overall["weighted_mu_sa"] = float(mu_sa)
    overall["worst_region_within_horizon_rate"] = float(min(
        metrics["safe_arrival_within_horizon_rate"] for metrics in per_region.values()
    ))
    overall["success_horizon_steps"] = success_horizon_steps
    overall["success_horizon_s"] = success_horizon_steps * dt

    return {"overall": overall, "per_region": per_region}


def save_episode_csv(path: Path, records: Sequence[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EpisodeRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def save_trajectory_npz(directory: Path, records: Sequence[TrajectoryRecord]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        filename = directory / f"{record.region}_trajectory_{record.episode + 1:02d}.npz"
        np.savez_compressed(
            filename,
            region=record.region,
            episode=record.episode,
            outcome=record.outcome,
            terminal_step=record.terminal_step,
            states=record.states,
            actions_norm=record.actions_norm,
            actions_physical=record.actions_physical,
            hb=record.hb,
            hs=record.hs,
        )


def write_text_report(path: Path, summary: dict, checkpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = summary["overall"]
    lines = [
        "SAFE-ARRIVAL POLICY EVALUATION REPORT",
        "=" * 44,
        f"Checkpoint: {checkpoint}",
        "",
        "OVERALL RESULTS",
        f"Episodes: {overall['episodes']}",
        f"Weighted mu_SA: {overall['weighted_mu_sa']:.4f}",
        f"Safe-arrival rate: {overall['safe_arrival_rate']:.4f}",
        f"Safe arrival within {overall['success_horizon_s']:.2f} s: "
        f"{overall['safe_arrival_within_horizon_rate']:.4f}",
        f"Failure rate: {overall['failure_rate']:.4f}",
        f"Timeout rate: {overall['timeout_rate']:.4f}",
        f"Worst-region within-horizon rate: {overall['worst_region_within_horizon_rate']:.4f}",
        f"Mean successful arrival time: {overall['mean_arrival_time_s_success_only']}",
        f"Median successful arrival time: {overall['median_arrival_time_s_success_only']}",
        f"90th-percentile successful arrival time: {overall['p90_arrival_time_s_success_only']}",
        f"Worst observed h_S margin: {overall['worst_min_hs']:.6f}",
        f"Maximum observed altitude: {overall['max_altitude_m']:.6f} m",
        "",
        "PER-CURRICULUM RESULTS",
    ]
    for region, metrics in summary["per_region"].items():
        lines.extend([
            f"[{region}]",
            f"  Episodes: {metrics['episodes']}",
            f"  Safe-arrival rate: {metrics['safe_arrival_rate']:.4f}",
            f"  Within-horizon rate: {metrics['safe_arrival_within_horizon_rate']:.4f}",
            f"  Failure / timeout: {metrics['failure_rate']:.4f} / {metrics['timeout_rate']:.4f}",
            f"  Mean arrival time: {metrics['mean_arrival_time_s_success_only']}",
            f"  Worst h_S margin: {metrics['worst_min_hs']:.6f}",
        ])
    lines.extend([
        "",
        "REPORT INTERPRETATION NOTES",
        "- weighted mu_SA is the fixed-weight estimate across curriculum regions.",
        "- within-horizon success is the Phase-I quantity relevant to the backup horizon.",
        "- h_S < 0 indicates a safety-set violation.",
        "- h_B >= 0 indicates entry into the certified base set.",
        "- Compare overall and worst-region values; a high average can hide a weak curriculum region.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

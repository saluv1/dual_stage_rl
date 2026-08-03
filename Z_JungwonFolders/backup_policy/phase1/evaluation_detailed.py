"""Fixed, region-stratified Phase-I evaluation utilities.

The rollout follows the released Phase-I evaluator: the actor is used outside
B, the certified DLQR is used inside B, and evaluation continues to the full
100-step horizon unless the state becomes unsafe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from bcbf.lqrgain import LQRGain
from env.dynamics import Dynamics
from .sampling import MU_SA_WEIGHTS, OfficialPhase1Sampler
from .state_action import compute_bfc, lqr_action, reset_dynamics_state, scale_action


@dataclass
class EpisodeRecord:
    region: str
    episode: int
    outcome: str
    success: bool
    entered_base: bool
    safe_arrival_within_horizon: bool
    failure: bool
    timeout: bool
    safe_rollout: bool
    terminal_at_horizon: bool
    invariance_after_entry: bool
    arrival_step: int | None
    arrival_time_s: float | None
    discounted_safe_arrival_score: float
    min_hs: float
    initial_z: float
    final_z: float
    max_z: float


def build_fixed_evaluation_set(sets, regions, seed, episodes_per_region, difficulty=1.0):
    if not isinstance(regions, OfficialPhase1Sampler):
        raise TypeError("regions must be OfficialPhase1Sampler")
    return regions.fixed_set(
        sets,
        seed=int(seed),
        episodes_per_region=int(episodes_per_region),
        difficulty=float(difficulty),
    )


def save_evaluation_set(path, evaluation_set):
    np.savez_compressed(path, **evaluation_set)


def load_evaluation_set(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}


def _aggregate(records: list[EpisodeRecord]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "safe_arrival_rate": 0.0,
            "safe_arrival_within_horizon_rate": 0.0,
            "entered_base_rate": 0.0,
            "failure_rate": 0.0,
            "timeout_rate": 0.0,
            "safe_rollout_rate": 0.0,
            "terminal_at_horizon_rate": 0.0,
            "invariance_after_entry_rate": 0.0,
            "mean_discounted_safe_arrival_score": 0.0,
            "mean_arrival_time_s_success_only": None,
            "median_arrival_time_s_success_only": None,
            "mean_min_hs": None,
            "worst_min_hs": None,
        }
    times = [r.arrival_time_s for r in records if r.arrival_time_s is not None]
    return {
        "count": len(records),
        # Strict official-style success: safe, entered B, remained under the
        # handoff policy, and finished the horizon inside B.
        "safe_arrival_rate": float(np.mean([r.success for r in records])),
        # Diagnostic reachability before the horizon, even if invariance later
        # fails.  weighted_mu_sa below deliberately uses strict success.
        "safe_arrival_within_horizon_rate": float(
            np.mean([r.safe_arrival_within_horizon for r in records])
        ),
        "entered_base_rate": float(np.mean([r.entered_base for r in records])),
        "failure_rate": float(np.mean([r.failure for r in records])),
        "timeout_rate": float(np.mean([r.timeout for r in records])),
        "safe_rollout_rate": float(np.mean([r.safe_rollout for r in records])),
        "terminal_at_horizon_rate": float(
            np.mean([r.terminal_at_horizon for r in records])
        ),
        "invariance_after_entry_rate": float(
            np.mean([r.invariance_after_entry for r in records])
        ),
        "mean_discounted_safe_arrival_score": float(
            np.mean([r.discounted_safe_arrival_score for r in records])
        ),
        "mean_arrival_time_s_success_only": float(np.mean(times)) if times else None,
        "median_arrival_time_s_success_only": float(np.median(times)) if times else None,
        "mean_min_hs": float(np.mean([r.min_hs for r in records])),
        "worst_min_hs": float(np.min([r.min_hs for r in records])),
    }


def evaluate_detailed(
    policy,
    sets,
    regions,
    rng=None,
    difficulty=1.0,
    episodes_per_region=64,
    trajectories_per_region=1,
    max_episode_steps=100,
    success_horizon_steps=100,
    evaluation_set=None,
    integrator="euler",
    beta=0.99,
):
    if evaluation_set is None:
        if rng is None:
            raise ValueError("rng is required when evaluation_set is not supplied")
        evaluation_set = build_fixed_evaluation_set(
            sets,
            regions,
            seed=int(rng.integers(0, 2**31 - 1)),
            episodes_per_region=episodes_per_region,
            difficulty=difficulty,
        )

    names = [name for name in regions.region_names if name in evaluation_set]
    records: list[EpisodeRecord] = []
    trajectories: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in names}
    K, _ = LQRGain(dt=0.02, g=9.81).gain()

    for region in names:
        for episode, initial_state in enumerate(np.asarray(evaluation_set[region], dtype=float)):
            dyn = Dynamics(integrator=integrator, dt=0.02, gravity=9.81)
            state = np.asarray(initial_state, dtype=float).copy()
            reset_dynamics_state(dyn, state)
            states = [state.copy()]
            actions = []
            raw_actions = []

            b0, f0, _ = compute_bfc(sets, state)
            safe_rollout = f0 == 0.0
            entered_base = bool(safe_rollout and b0 > 0.5)
            left_after_entry = False
            arrival_step = 0 if entered_base else None
            failure = not safe_rollout
            steps_simulated = 0
            min_hs = float(sets.hs)
            max_z = float(state[2])

            for step in range(1, int(max_episode_steps) + 1):
                if failure:
                    break
                b_cur, _, _ = compute_bfc(sets, state)
                action_norm = np.clip(policy.select_action(state), -1.0, 1.0)
                raw_action = scale_action(action_norm, dyn.g)
                applied_action = (
                    lqr_action(state, K, dyn.g) if b_cur > 0.5 else raw_action
                )
                next_state = dyn.step(applied_action).copy()
                steps_simulated = step
                if not np.all(np.isfinite(next_state)):
                    failure = True
                    safe_rollout = False
                    break

                b_next, f_next, _ = compute_bfc(sets, next_state)
                safe_next = f_next == 0.0
                goal_next = safe_next and b_next > 0.5
                previously_entered = entered_base
                if goal_next and not entered_base:
                    entered_base = True
                    arrival_step = step
                if previously_entered and not goal_next:
                    left_after_entry = True
                safe_rollout = safe_rollout and safe_next
                failure = not safe_next

                min_hs = min(min_hs, float(sets.hs))
                max_z = max(max_z, float(next_state[2]))
                state = next_state
                states.append(state.copy())
                raw_actions.append(raw_action.copy())
                actions.append(applied_action.copy())

            b_final, f_final, _ = compute_bfc(sets, state)
            terminal_at_horizon = bool(
                not failure
                and steps_simulated == int(max_episode_steps)
                and f_final == 0.0
                and b_final > 0.5
            )
            invariance_after_entry = bool(entered_base and not left_after_entry)
            strict_success = bool(
                safe_rollout
                and entered_base
                and invariance_after_entry
                and terminal_at_horizon
            )
            reached_within_horizon = bool(
                entered_base
                and arrival_step is not None
                and arrival_step <= int(success_horizon_steps)
            )
            timeout = not strict_success and not failure
            discounted = (
                float(beta ** arrival_step)
                if strict_success and arrival_step is not None
                else 0.0
            )
            record = EpisodeRecord(
                region=region,
                episode=episode,
                outcome="success" if strict_success else "failure" if failure else "timeout",
                success=strict_success,
                entered_base=entered_base,
                safe_arrival_within_horizon=reached_within_horizon,
                failure=failure,
                timeout=timeout,
                safe_rollout=safe_rollout,
                terminal_at_horizon=terminal_at_horizon,
                invariance_after_entry=invariance_after_entry,
                arrival_step=arrival_step,
                arrival_time_s=(
                    float(arrival_step * dyn.del_t)
                    if arrival_step is not None
                    else None
                ),
                discounted_safe_arrival_score=discounted,
                min_hs=min_hs,
                initial_z=float(states[0][2]),
                final_z=float(state[2]),
                max_z=max_z,
            )
            records.append(record)
            if len(trajectories[region]) < int(trajectories_per_region):
                trajectories[region].append({
                    "states": np.asarray(states),
                    "actions": np.asarray(actions),
                    "raw_actions": np.asarray(raw_actions),
                    "success": np.asarray([strict_success]),
                })

    per_region = {
        name: _aggregate([r for r in records if r.region == name]) for name in names
    }
    overall = _aggregate(records)
    weights = {name: MU_SA_WEIGHTS[name] for name in names}
    total_weight = sum(weights.values())
    overall["weighted_mu_sa"] = float(
        sum(weights[name] * per_region[name]["safe_arrival_rate"] for name in names)
        / total_weight
    )
    overall["worst_region_within_horizon_rate"] = float(
        min(per_region[name]["safe_arrival_rate"] for name in names)
    )
    summary = {
        "protocol": f"official_reset_regions_{integrator}",
        "difficulty": float(difficulty),
        "episodes_per_region": {
            name: int(len(evaluation_set[name])) for name in names
        },
        "max_episode_steps": int(max_episode_steps),
        "success_horizon_steps": int(success_horizon_steps),
        "beta": float(beta),
        "valid_regions": names,
        "weights": weights,
        "overall": overall,
        "per_region": per_region,
    }
    return summary, [asdict(r) for r in records], trajectories

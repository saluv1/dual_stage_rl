#!/usr/bin/env python3
"""Evaluate a local PyTorch TD3 actor with the paper-sized Phase-I protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backup_policy.td3 import Actor, device, safe_arrival_obs, infer_actor_obs_dim
from official_phase1_evaluation.official_model import (
    OfficialDLQR,
    OfficialQuadrotorConfig,
    clip_physical_action,
    is_safe,
    safety_margin,
    step_official_euler,
)
from official_phase1_evaluation.official_resets import (
    OFFICIAL_REGIONS,
    PAPER_COUNTS,
    generate_benchmark,
    load_benchmark,
    load_reset_library,
    save_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained TD3 actor on 3,584 official-region Phase-I "
            "initial states using official forward-Euler dynamics and DLQR handoff."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reset-library", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--general-count", type=int, default=1024)
    parser.add_argument("--near-ceiling-count", type=int, default=1024)
    parser.add_argument("--bridge-count", type=int, default=1024)
    parser.add_argument("--base-shell-count", type=int, default=512)
    parser.add_argument("--max-per-region", type=int, default=0)
    parser.add_argument("--curriculum-scale", type=float, default=1.0)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--training-gravity", type=float, default=9.81)
    parser.add_argument("--benchmark-file", default="")
    parser.add_argument("--regenerate-benchmark", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PyTorchActorPolicy:
    def __init__(self, checkpoint: Path, training_gravity: float):
        self.checkpoint = checkpoint
        self.training_gravity = float(training_gravity)
        if checkpoint.suffix == ".pt":
            payload = torch.load(checkpoint, map_location=device)
            if not isinstance(payload, dict) or "actor" not in payload:
                raise KeyError(
                    f"Compact checkpoint {checkpoint} does not contain an 'actor' state dict"
                )
            state_dict = payload["actor"]
        else:
            state_dict = torch.load(checkpoint, map_location=device)

        self.obs_dim = infer_actor_obs_dim(state_dict)
        self.actor = Actor(self.obs_dim, 4, 1.0).to(device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def normalized_action(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float64)
        single = state_array.ndim == 1
        observation = safe_arrival_obs(state_array, obs_dim=self.obs_dim)
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(tensor).detach().cpu().numpy()
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return action[0] if single else action

    def physical_action(
        self, state: np.ndarray, cfg: OfficialQuadrotorConfig
    ) -> np.ndarray:
        action = self.normalized_action(state)
        single = action.ndim == 1
        batch = action[None, :] if single else action
        physical = np.empty_like(batch, dtype=np.float64)
        physical[:, 0] = 2.0 * self.training_gravity * (batch[:, 0] + 1.0)
        physical[:, 1:4] = cfg.omega_max * batch[:, 1:4]
        physical = clip_physical_action(physical, cfg)
        return physical[0] if single else physical


def resolve_checkpoint(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.exists():
        return path
    compact = path.with_suffix(".pt")
    if compact.exists():
        return compact
    legacy = Path(str(path) + "_actor")
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"Checkpoint not found. Tried {path}, {compact}, and {legacy}"
    )


def benchmark_name(seed: int, counts: dict[str, int]) -> str:
    return (
        f"paper_seed_{seed}_"
        f"g{counts['general_trace']}_"
        f"n{counts['near_ceiling']}_"
        f"b{counts['bridge']}_"
        f"s{counts['base_shell']}.npz"
    )


def prepare_benchmark(
    args: argparse.Namespace,
    output_root: Path,
    reset_library,
    counts: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, Path, dict[str, int]]:
    if args.benchmark_file:
        benchmark_path = Path(args.benchmark_file).expanduser().resolve()
    else:
        benchmark_path = (
            output_root / "benchmark_states" / benchmark_name(args.seed, counts)
        )

    if benchmark_path.exists() and not args.regenerate_benchmark:
        states, regions = load_benchmark(benchmark_path)
        actual_counts = {
            region: int(np.sum(regions == region)) for region in OFFICIAL_REGIONS
        }
        if actual_counts != counts:
            raise ValueError(
                f"Existing benchmark counts {actual_counts} do not match requested {counts}. "
                "Use --regenerate-benchmark or a different benchmark file."
            )
        return states, regions, benchmark_path, {}

    states, regions, region_seeds = generate_benchmark(
        reset_library,
        seed=args.seed,
        counts=counts,
        curriculum_scale=args.curriculum_scale,
    )
    save_benchmark(
        benchmark_path,
        states=states,
        regions=regions,
        source_library=reset_library.path,
        seed=args.seed,
        region_seeds=region_seeds,
        curriculum_scale=args.curriculum_scale,
        counts=counts,
    )
    return states, regions, benchmark_path, region_seeds


def evaluate_batch(
    initial_states: np.ndarray,
    regions: np.ndarray,
    policy: PyTorchActorPolicy,
    cfg: OfficialQuadrotorConfig,
    dlqr: OfficialDLQR,
    horizon_steps: int,
    beta: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    state = np.asarray(initial_states, dtype=np.float64).copy()
    n_episodes = len(state)
    if n_episodes == 0:
        raise ValueError("No initial states were supplied")

    safe_initial = np.asarray(is_safe(state, cfg), dtype=bool)
    terminal_initial = np.asarray(dlqr.contains(state), dtype=bool)
    active = safe_initial.copy()
    crashed = ~safe_initial
    entered = safe_initial & terminal_initial
    left_after_entry = np.zeros(n_episodes, dtype=bool)
    entry_step = np.where(entered, 0, -1).astype(np.int32)

    step_count = np.zeros(n_episodes, dtype=np.int32)
    safe_steps = np.zeros(n_episodes, dtype=np.int32)
    terminal_steps = np.zeros(n_episodes, dtype=np.int32)
    post_entry_steps = np.zeros(n_episodes, dtype=np.int32)
    post_entry_terminal_steps = np.zeros(n_episodes, dtype=np.int32)
    min_margin = np.asarray(safety_margin(state, cfg), dtype=np.float64)
    max_z = state[:, 2].copy()

    example_indices = {
        region: int(np.flatnonzero(regions == region)[0])
        for region in OFFICIAL_REGIONS
        if np.any(regions == region)
    }
    examples: dict[str, dict[str, list[np.ndarray]]] = {
        region: {
            "states": [state[index].copy()],
            "raw_actions": [],
            "applied_actions": [],
            "used_lqr": [],
            "base_margin": [np.asarray(dlqr.base_margin(state[index])).copy()],
            "safety_margin": [np.asarray(safety_margin(state[index], cfg)).copy()],
        }
        for region, index in example_indices.items()
    }

    for step in range(1, int(horizon_steps) + 1):
        previous_active = active.copy()
        if not np.any(previous_active):
            break

        raw_action = policy.physical_action(state, cfg)
        in_base = np.asarray(dlqr.contains(state), dtype=bool)
        lqr_action = dlqr.action(state)
        applied_action = np.where(in_base[:, None], lqr_action, raw_action)

        next_state = step_official_euler(state, applied_action, cfg)
        finite = np.all(np.isfinite(next_state), axis=1)
        safe_next = finite & np.asarray(is_safe(next_state, cfg), dtype=bool)
        terminal_next = finite & np.asarray(dlqr.contains(next_state), dtype=bool)
        goal_next = safe_next & terminal_next

        step_count[previous_active] += 1
        safe_steps[previous_active] += safe_next[previous_active].astype(np.int32)
        terminal_steps[previous_active] += terminal_next[previous_active].astype(np.int32)

        post_mask = previous_active & (entered | goal_next)
        post_entry_steps[post_mask] += 1
        post_entry_terminal_steps[post_mask] += terminal_next[post_mask].astype(
            np.int32
        )

        newly_entered = previous_active & (~entered) & goal_next
        entry_step[newly_entered] = step
        left_after_entry |= previous_active & entered & (~goal_next)
        entered |= previous_active & goal_next

        next_margin = np.asarray(safety_margin(next_state, cfg), dtype=np.float64)
        min_margin[previous_active] = np.minimum(
            min_margin[previous_active], next_margin[previous_active]
        )
        max_z[previous_active] = np.maximum(
            max_z[previous_active], next_state[previous_active, 2]
        )

        crash_now = previous_active & (~safe_next)
        crashed |= crash_now
        state[previous_active] = next_state[previous_active]
        active = previous_active & safe_next

        for region, index in example_indices.items():
            example = examples[region]
            example["states"].append(state[index].copy())
            example["raw_actions"].append(raw_action[index].copy())
            example["applied_actions"].append(applied_action[index].copy())
            example["used_lqr"].append(np.asarray(in_base[index]).copy())
            example["base_margin"].append(np.asarray(dlqr.base_margin(state[index])).copy())
            example["safety_margin"].append(
                np.asarray(safety_margin(state[index], cfg)).copy()
            )

    safe_rollout = ~crashed
    terminal_at_horizon = (
        safe_rollout
        & (step_count == int(horizon_steps))
        & np.asarray(dlqr.contains(state), dtype=bool)
    )
    invariance_after_entry = entered & (~left_after_entry)
    success = (
        safe_rollout
        & entered
        & terminal_at_horizon
        & invariance_after_entry
    )
    denominator = np.maximum(step_count, 1)
    discounted_score = np.where(
        success & (entry_step >= 0), np.power(beta, entry_step), 0.0
    )

    result = {
        "region": np.asarray(regions, dtype=str),
        "success": success.astype(np.int8),
        "crash": crashed.astype(np.int8),
        "safe_rollout": safe_rollout.astype(np.int8),
        "entered_base": entered.astype(np.int8),
        "terminal_at_horizon": terminal_at_horizon.astype(np.int8),
        "invariance_after_entry": invariance_after_entry.astype(np.int8),
        "entry_step": entry_step,
        "entry_time_s": np.where(entry_step >= 0, entry_step * cfg.dt, np.nan),
        "discounted_safe_arrival_score": discounted_score,
        "safe_step_rate": safe_steps / denominator,
        "terminal_step_rate": terminal_steps / denominator,
        "post_entry_terminal_step_rate": np.divide(
            post_entry_terminal_steps,
            np.maximum(post_entry_steps, 1),
        ),
        "minimum_ceiling_margin_m": min_margin,
        "initial_z_m": initial_states[:, 2],
        "final_z_m": state[:, 2],
        "maximum_z_m": max_z,
        "steps_simulated": step_count,
    }

    converted_examples: dict[str, dict[str, np.ndarray]] = {}
    for region, values in examples.items():
        converted_examples[region] = {
            key: np.asarray(value) for key, value in values.items()
        }
    return result, converted_examples


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def summarize_subset(result: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}

    def rate(key: str) -> float:
        return float(np.mean(result[key][mask]))

    strict_success_rate = rate("success")
    safe_arrival_rate = rate("entered_base")
    return {
        "count": count,
        "success_rate": strict_success_rate,
        "strict_success_rate": strict_success_rate,
        "empirical_mu_sa": safe_arrival_rate,
        "crash_rate": rate("crash"),
        "safe_rollout_rate": rate("safe_rollout"),
        "entered_base_rate": safe_arrival_rate,
        "terminal_at_horizon_rate": rate("terminal_at_horizon"),
        "invariance_after_entry_rate": rate("invariance_after_entry"),
        "mean_discounted_safe_arrival_score": float(
            np.mean(result["discounted_safe_arrival_score"][mask])
        ),
        "entry_time_s": finite_stats(result["entry_time_s"][mask]),
        "minimum_ceiling_margin_m": finite_stats(
            result["minimum_ceiling_margin_m"][mask]
        ),
        "maximum_z_m": finite_stats(result["maximum_z_m"][mask]),
    }


def summarize(result: dict[str, np.ndarray]) -> dict[str, Any]:
    regions = result["region"]
    per_region = {
        region: summarize_subset(result, regions == region)
        for region in OFFICIAL_REGIONS
        if np.any(regions == region)
    }
    overall = summarize_subset(result, np.ones(len(regions), dtype=bool))
    region_success = [metrics["success_rate"] for metrics in per_region.values()]
    region_arrival = [metrics["entered_base_rate"] for metrics in per_region.values()]
    overall["strict_success_rate"] = overall["success_rate"]
    overall["region_balanced_strict_success_rate"] = float(np.mean(region_success))
    overall["region_balanced_safe_arrival_rate"] = float(np.mean(region_arrival))
    # mu_SA counts states that reach B safely within the horizon. The stricter
    # success metric additionally checks final-horizon membership and that the
    # rollout never leaves B after the LQR handoff.
    overall["empirical_mu_sa"] = overall["entered_base_rate"]
    return {"overall": overall, "per_region": per_region}


def write_episode_csv(path: Path, result: dict[str, np.ndarray]) -> None:
    keys = list(result.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_index", *keys])
        writer.writeheader()
        for index in range(len(result["region"])):
            row: dict[str, Any] = {"episode_index": index}
            for key in keys:
                value = result[key][index]
                row[key] = value.item() if hasattr(value, "item") else value
            writer.writerow(row)


def plot_rates(summary: dict[str, Any], output: Path) -> None:
    names = list(summary["per_region"].keys())
    success = [summary["per_region"][name]["success_rate"] for name in names]
    entered = [summary["per_region"][name]["entered_base_rate"] for name in names]
    safe = [summary["per_region"][name]["safe_rollout_rate"] for name in names]

    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, success, width, label="strict success")
    ax.bar(x, entered, width, label="entered base")
    ax.bar(x + width, safe, width, label="safe rollout")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(
        "TD3 Phase-I Evaluation on Official Regions\n"
        f"empirical mu_SA = {summary['overall']['empirical_mu_sa']:.3f}"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_examples(
    examples: dict[str, dict[str, np.ndarray]],
    cfg: OfficialQuadrotorConfig,
    output: Path,
) -> None:
    names = [region for region in OFFICIAL_REGIONS if region in examples]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.5 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, region in zip(axes, names):
        states = examples[region]["states"]
        time = np.arange(len(states)) * cfg.dt
        ax.plot(time, states[:, 2], label="altitude")
        ax.axhline(cfg.z_max, linestyle="--", label="ceiling")
        ax.axhline(cfg.z_des, linestyle=":", label="hover target")
        used_lqr = examples[region]["used_lqr"].astype(bool)
        if np.any(used_lqr):
            ax.axvline(int(np.argmax(used_lqr)) * cfg.dt, linestyle="-.", label="LQR active")
        ax.set_ylabel("z [m]")
        ax.set_title(region)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("One Deterministic TD3 Rollout per Official Region")
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_examples(path: Path, examples: dict[str, dict[str, np.ndarray]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for region, values in examples.items():
        for key, value in values.items():
            arrays[f"{region}__{key}"] = value
    np.savez_compressed(path, **arrays)


def main() -> None:
    args = parse_args()
    if args.horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if not 0.0 < args.beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    if not 0.0 <= args.curriculum_scale <= 1.0:
        raise ValueError("curriculum_scale must lie in [0, 1]")

    counts = {
        "general_trace": args.general_count,
        "near_ceiling": args.near_ceiling_count,
        "bridge": args.bridge_count,
        "base_shell": args.base_shell_count,
    }
    if args.max_per_region > 0:
        counts = {
            region: min(count, args.max_per_region) for region, count in counts.items()
        }
    if any(count < 0 for count in counts.values()):
        raise ValueError("Region counts cannot be negative")

    checkpoint = resolve_checkpoint(args.checkpoint)
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / args.run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {run_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    data_dir = run_dir / "data"
    plots_dir = run_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    reset_library = load_reset_library(args.reset_library)
    cfg = OfficialQuadrotorConfig.from_reset_payload(reset_library.cbf_config)
    dlqr = OfficialDLQR(cfg)
    policy = PyTorchActorPolicy(checkpoint, args.training_gravity)

    states, regions, benchmark_path, generated_region_seeds = prepare_benchmark(
        args, output_root, reset_library, counts
    )
    result, examples = evaluate_batch(
        states,
        regions,
        policy,
        cfg,
        dlqr,
        horizon_steps=args.horizon_steps,
        beta=args.beta,
    )
    metrics = summarize(result)

    summary = {
        "protocol": "portable_official_ps2rl_phase1_paper_size",
        "backend_note": (
            "Official state pools, perturbation rules, forward-Euler equations, "
            "DLQR/base set, horizon, and regional counts are reproduced locally. "
            "The actor is the repository's PyTorch TD3 policy rather than the "
            "official JAX policy implementation."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "reset_library": str(reset_library.path),
        "reset_library_sha256": file_sha256(reset_library.path),
        "benchmark_file": str(benchmark_path),
        "benchmark_seed": int(args.seed),
        "generated_region_seeds": generated_region_seeds,
        "curriculum_scale": float(args.curriculum_scale),
        "region_counts": counts,
        "total_states": int(len(states)),
        "horizon_steps": int(args.horizon_steps),
        "horizon_seconds": float(args.horizon_steps * cfg.dt),
        "beta": float(args.beta),
        "training_gravity_for_action_scaling": float(args.training_gravity),
        "official_config": cfg.to_dict(),
        **metrics,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    write_episode_csv(data_dir / "episode_metrics.csv", result)
    save_examples(data_dir / "example_trajectories.npz", examples)
    plot_rates(metrics, plots_dir / "rates_by_region.png")
    plot_examples(examples, cfg, plots_dir / "example_altitude_trajectories.png")

    print(json.dumps(metrics["overall"], indent=2))
    print(f"Benchmark states: {benchmark_path}")
    print(f"Evaluation saved to: {run_dir}")


if __name__ == "__main__":
    main()

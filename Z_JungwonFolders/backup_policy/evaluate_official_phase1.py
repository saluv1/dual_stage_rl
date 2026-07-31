#!/usr/bin/env python3
"""Evaluate a PyTorch Phase-I actor with the official PS2-RL held-out protocol."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from backup_policy.td3 import Actor, SAFE_ARRIVAL_OBS_DIM, safe_arrival_obs, device
from backup_policy.phase1.official_phase1_eval import (
    evaluate_official_split,
    save_episode_csv,
    save_json,
    save_subset_csv,
    save_trajectory_npz,
)
from backup_policy.phase1.official_phase1_plotting import (
    plot_split_summary,
    plot_trajectory,
)
from backup_policy.phase1.official_quadrotor import (
    OfficialDLQR,
    OfficialQuadrotorConfig,
    clip_physical_action,
    make_integrator,
)
from backup_policy.phase1.official_reset_library import (
    load_evaluation_state_file,
    load_official_reset_library,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a PyTorch safe-arrival actor on the official PS2-RL "
            "quadrotor validation/test reset sets."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Example: experiments/safe_arrival/safe_arrival_001",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint prefix under run-dir/checkpoints, e.g. best or interrupted.",
    )
    parser.add_argument(
        "--reset-library",
        required=True,
        help="Path to official checkpoints/quadrotor_vanilla/reset_library.pkl",
    )
    parser.add_argument(
        "--split", choices=("val", "test", "both"), default="both"
    )
    parser.add_argument(
        "--state-file",
        default="",
        help=(
            "Optional frozen .npz with states and region arrays. When set, "
            "the evaluator uses this set instead of val/test held-out splits."
        ),
    )
    parser.add_argument(
        "--max-resets",
        type=int,
        default=0,
        help="Optional cap for a quick smoke test; 0 evaluates the full split.",
    )
    parser.add_argument(
        "--integrator",
        choices=("official_euler", "rk45"),
        default="official_euler",
    )
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.984035,
        help="Official Phase-I default discount.",
    )
    parser.add_argument(
        "--training-gravity",
        type=float,
        default=9.807,
        help=(
            "Gravity used by your actor's normalized-to-physical action map. "
            "Keep 9.807 for checkpoints trained by the current repository."
        ),
    )
    parser.add_argument("--trajectories-per-region", type=int, default=3)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


class PyTorchSafeArrivalActor:
    """Load only the actor weights and return deterministic physical actions."""

    def __init__(self, actor_path: Path, training_gravity: float):
        if not actor_path.exists():
            raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
        self.training_gravity = float(training_gravity)
        self.actor = Actor(SAFE_ARRIVAL_OBS_DIM, 4, 1.0).to(device)
        state_dict = torch.load(actor_path, map_location=device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def action_normalized(self, state: np.ndarray) -> np.ndarray:
        observation = safe_arrival_obs(np.asarray(state, dtype=np.float64))
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)[None, :]
        with torch.no_grad():
            action = self.actor(tensor)[0].detach().cpu().numpy()
        return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

    def action_physical(
        self, state: np.ndarray, cfg: OfficialQuadrotorConfig
    ) -> np.ndarray:
        action = self.action_normalized(state)
        physical = np.asarray(
            [
                2.0 * self.training_gravity * (action[0] + 1.0),
                cfg.omega_max * action[1],
                cfg.omega_max * action[2],
                cfg.omega_max * action[3],
            ],
            dtype=np.float64,
        )
        return clip_physical_action(physical, cfg)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    actor_path = run_dir / "checkpoints" / f"{args.checkpoint}_actor"
    reset_payload = load_official_reset_library(args.reset_library)
    cfg = OfficialQuadrotorConfig.from_reset_payload(reset_payload.cbf_config)
    base_controller = OfficialDLQR(cfg)
    integrator = make_integrator(args.integrator, cfg)
    actor = PyTorchSafeArrivalActor(actor_path, args.training_gravity)

    actor_action: Callable[[np.ndarray], np.ndarray] = (
        lambda state: actor.action_physical(state, cfg)
    )

    custom_state_set = (
        load_evaluation_state_file(args.state_file) if args.state_file else None
    )
    split_names = (
        (custom_state_set.name,)
        if custom_state_set is not None
        else (("val", "test") if args.split == "both" else (args.split,))
    )
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = run_dir / "evaluation" / "official_ps2rl"
    suffix = args.output_suffix.strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [
        "quadBackup_eval",
        timestamp,
        args.checkpoint,
        args.integrator,
    ]
    if suffix:
        name_parts.append(suffix)
    output_dir = output_root / "-".join(name_parts)
    logs_dir = output_dir / "logs"
    plots_dir = output_dir / "plots"
    trajectory_dir = output_dir / "trajectories"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "protocol": (
            "official_ps2rl_phase1_frozen_paper_size"
            if custom_state_set is not None
            else "official_ps2rl_phase1_heldout"
        ),
        "run_dir": str(run_dir),
        "actor_path": str(actor_path),
        "reset_library_path": str(reset_payload.path),
        "state_file": (
            str(Path(args.state_file).expanduser().resolve())
            if args.state_file
            else None
        ),
        "integrator": args.integrator,
        "horizon_steps": int(args.horizon_steps),
        "horizon_seconds": float(args.horizon_steps * cfg.dt),
        "beta": float(args.beta),
        "training_gravity_for_actor_scaling": float(args.training_gravity),
        "benchmark_config": cfg.to_dict(),
        "splits": {},
    }

    all_episode_records = []
    split_summaries = {}
    for split_name in split_names:
        heldout = (
            custom_state_set.trimmed(args.max_resets)
            if custom_state_set is not None
            else reset_payload.splits[split_name].trimmed(args.max_resets)
        )
        split_summary, records, trajectories = evaluate_official_split(
            heldout=heldout,
            actor_action_phys=actor_action,
            integrator=integrator,
            cfg=cfg,
            base_controller=base_controller,
            horizon_steps=args.horizon_steps,
            beta=args.beta,
            trajectories_per_region=args.trajectories_per_region,
        )
        split_summaries[split_name] = split_summary
        summary["splits"][split_name] = split_summary
        all_episode_records.extend(records)

        save_episode_csv(logs_dir / f"{split_name}_episode_metrics.csv", records)
        plot_split_summary(split_name, split_summary, records, plots_dir)
        for trajectory in trajectories:
            stem = (
                f"{trajectory.split}_{trajectory.region}_"
                f"{trajectory.index:04d}_"
                f"{'success' if trajectory.strict_success else 'failure'}"
            )
            save_trajectory_npz(trajectory_dir / f"{stem}.npz", trajectory)
            plot_trajectory(
                trajectory,
                dt=cfg.dt,
                z_max=cfg.z_max,
                z_des=cfg.z_des,
                output_dir=trajectory_dir,
            )

    save_episode_csv(logs_dir / "all_episode_metrics.csv", all_episode_records)
    save_subset_csv(logs_dir / "subset_metrics.csv", split_summaries)
    save_json(logs_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()

"""Standalone report-ready evaluation for a trained Phase-I safe-arrival policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env.dynamics import Dynamics
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from backup_policy.td3 import TD3
from backup_policy.phase1.experiment_io import open_existing_run, save_json, update_metadata
from backup_policy.phase1.sampling import classify_trace_states, generate_reference_trace
from backup_policy.phase1.evaluation_detailed import (
    evaluate_detailed,
    save_episode_csv,
    save_trajectory_npz,
    write_text_report,
)
from backup_policy.phase1.plotting import (
    animate_trajectory_gif,
    plot_final_summary,
    plot_training_history,
    plot_trajectory,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Example: experiments/safe_arrival/safe_arrival_001")
    parser.add_argument("--checkpoint", default="best", help="best, latest, final, or eval_0001000")
    parser.add_argument("--episodes-per-region", type=int, default=100)
    parser.add_argument("--trajectories-per-region", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--success-horizon-steps", type=int, default=100)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--animations", action="store_true")
    parser.add_argument("--animation-count-per-region", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    paths = open_existing_run(args.run_dir)
    checkpoint_prefix = str(paths["checkpoints"] / args.checkpoint)
    actor_file = Path(checkpoint_prefix + "_actor")
    if not actor_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {actor_file}")

    dyn = Dynamics()
    K, P = LQRGain(dt=dyn.del_t, g=dyn.g).gain()
    sets = SetIndicator(P=P, c_b=8.0, zceil=3.0)
    sets.P = P
    trace_states = generate_reference_trace(n_points=200, n_variants=20, seed=0)
    regions = classify_trace_states(sets, trace_states, P, 8.0, near_ceiling_margin=0.25)

    policy = TD3(state_dim=10, action_dim=4, max_action=1.0)
    policy.load(checkpoint_prefix)

    summary, episode_records, trajectories = evaluate_detailed(
        policy=policy,
        sets=sets,
        regions=regions,
        rng=rng,
        difficulty=args.difficulty,
        episodes_per_region=args.episodes_per_region,
        trajectories_per_region=args.trajectories_per_region,
        max_episode_steps=args.max_episode_steps,
        success_horizon_steps=args.success_horizon_steps,
    )

    save_json(paths["evaluation_logs"] / "evaluation_summary.json", summary)
    save_episode_csv(paths["evaluation_logs"] / "evaluation_episodes.csv", episode_records)
    write_text_report(paths["evaluation_logs"] / "evaluation_report.txt", summary, checkpoint_prefix)
    save_trajectory_npz(paths["trajectories"], trajectories)

    plot_final_summary(summary, episode_records, paths["evaluation_plots"])
    plot_training_history(paths["logs"] / "training_evaluations.csv", paths["training_plots"])
    for trajectory in trajectories:
        plot_trajectory(trajectory, paths["trajectories"], dyn.del_t)

    if args.animations:
        used = {}
        for trajectory in trajectories:
            count = used.get(trajectory.region, 0)
            if count >= args.animation_count_per_region:
                continue
            filename = paths["animations"] / (
                f"{trajectory.region}_trajectory_{trajectory.episode + 1:02d}_{trajectory.outcome}.gif"
            )
            animate_trajectory_gif(trajectory, filename, dyn.del_t)
            used[trajectory.region] = count + 1

    update_metadata(
        paths["root"],
        last_evaluated_checkpoint=args.checkpoint,
        evaluation_summary=summary["overall"],
    )
    print(json.dumps(summary, indent=2))
    print(f"\nEvaluation outputs saved in: {paths['evaluation']}")


if __name__ == "__main__":
    main()

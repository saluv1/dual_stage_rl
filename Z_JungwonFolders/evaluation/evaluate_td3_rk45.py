"""Evaluate a trained Phase-I TD3 policy in the local RK45 environment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backup_policy.td3 import TD3
from backup_policy.phase1.evaluation_detailed import evaluate_detailed
from backup_policy.phase1.sampling import load_official_sampler
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from env.dynamics import Dynamics


def save_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def plot_summary(summary: dict, output_dir: Path) -> None:
    names = summary["valid_regions"]
    success = [summary["per_region"][name]["safe_arrival_rate"] for name in names]
    within = [
        summary["per_region"][name]["safe_arrival_within_horizon_rate"]
        for name in names
    ]
    failure = [summary["per_region"][name]["failure_rate"] for name in names]

    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, within, width, label="success within T")
    ax.bar(x, success, width, label="success by episode end")
    ax.bar(x + width, failure, width, label="failure")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(
        f"TD3 Safe-Arrival Evaluation: RK45 (weighted mu_SA={summary['overall']['weighted_mu_sa']:.3f})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "summary_by_region.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(trajectories: dict, output_dir: Path, dt: float, zceil: float) -> None:
    names = [name for name, values in trajectories.items() if values]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.4 * len(names)), sharex=False)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        states = trajectories[name][0]["states"]
        time = np.arange(states.shape[0]) * dt
        ax.plot(time, states[:, 2], label="z")
        ax.axhline(zceil, linestyle="--", label="ceiling")
        ax.axhline(2.0, linestyle=":", label="hover target")
        ax.set_ylabel("z [m]")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Example RK45 Evaluation Trajectories")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_examples.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default="Trained Models")
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--checkpoint", default="best", choices=("best", "last", "interrupted"))
    parser.add_argument("--episodes-per-region", type=int, default=64)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--success-horizon-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--trajectories-per-region", type=int, default=1)
    parser.add_argument("--output-root", default="evaluation/TD3 Evaluation RK45")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_name = f"{args.run_index:03d}"
    checkpoint_prefix = Path(args.model_root) / run_name / "checkpoints" / args.checkpoint
    output_dir = Path(args.output_root) / f"run_{run_name}_{args.checkpoint}"
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = TD3.from_checkpoint(checkpoint_prefix, state_dim=10, action_dim=4, max_action=1.0)

    dyn = Dynamics(integrator="rk45", dt=0.02, gravity=9.81)
    _, p_matrix = LQRGain(dt=dyn.del_t, g=dyn.g).gain()
    sets = SetIndicator(P=p_matrix, c_b=8.0, zceil=3.0)
    regions = load_official_sampler(
        PROJECT_ROOT / "official_phase1_evaluation" / "assets" / "reset_library.pkl",
        split="test",
    )

    summary, records, trajectories = evaluate_detailed(
        policy=policy,
        sets=sets,
        regions=regions,
        rng=rng,
        difficulty=args.difficulty,
        episodes_per_region=args.episodes_per_region,
        trajectories_per_region=args.trajectories_per_region,
        max_episode_steps=args.max_episode_steps,
        success_horizon_steps=args.success_horizon_steps,
        integrator="rk45",
    )
    summary["run_index"] = run_name
    summary["checkpoint"] = args.checkpoint
    summary["checkpoint_path"] = str(checkpoint_prefix)
    summary["seed"] = args.seed

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    save_records(output_dir / "episodes.csv", records)

    arrays = {}
    for region, region_trajectories in trajectories.items():
        for index, trajectory in enumerate(region_trajectories):
            arrays[f"{region}_{index}_states"] = trajectory["states"]
            arrays[f"{region}_{index}_actions"] = trajectory["actions"]
    np.savez_compressed(output_dir / "trajectory_examples.npz", **arrays)

    plot_summary(summary, output_dir)
    plot_trajectories(trajectories, output_dir, dyn.del_t, sets.zceil)

    print(json.dumps(summary["overall"], indent=2))
    print(f"Saved RK45 evaluation to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

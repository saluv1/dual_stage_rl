"""Final Phase-I evaluation report: numerical metrics + rich plots + GIFs.

This module is the shared, method-agnostic evaluator. Both the current method
and the "my methods" variant call it the SAME way, on the SAME fixed validation
states and the SAME Euler dynamics, so their numbers are directly comparable.
The only thing that differs between methods is which checkpoint is loaded; the
observation width (10-D official vs 8-D reduced) is auto-detected from the
checkpoint, so no per-method branching is needed here.

It reuses ``evaluate_detailed`` for the rollouts, which guarantees the plotted
trajectories come from exactly the same rollout logic that produces the
metrics.

Outputs (under --output-dir):
    metrics/summary.json          full nested summary
    metrics/summary.csv           flat per-region + overall table
    metrics/report.txt            human-readable one-page report
    plots_3d/<region>.png         3D safe-arrival trajectories
    plots_time/<region>.png       altitude / speed / attitude / hs-margin vs step
    plots_actuator/<region>.png   thrust + body-rate commands vs step
    gifs/<region>.gif             one rotating 3D GIF per region (if enabled)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from backup_policy.td3 import TD3
from backup_policy.phase1.sampling import load_official_sampler
from backup_policy.phase1.evaluation_detailed import (
    build_fixed_evaluation_set,
    evaluate_detailed,
    load_evaluation_set,
)
from backup_policy.phase1.state_action import compute_reduced_state


_REGION_COLORS = {
    "general_trace": "#1f77b4",
    "near_ceiling": "#d62728",
    "bridge": "#2ca02c",
    "base_shell": "#9467bd",
}
Z_CEIL = 3.0
Z_DES = 2.0


# --------------------------------------------------------------------------
# numerical
# --------------------------------------------------------------------------
def write_metrics(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    overall = summary["overall"]
    per_region = summary["per_region"]
    metric_keys = [
        "count", "safe_arrival_rate", "safe_arrival_within_horizon_rate",
        "entered_base_rate", "failure_rate", "timeout_rate", "safe_rollout_rate",
        "terminal_at_horizon_rate", "invariance_after_entry_rate",
        "mean_discounted_safe_arrival_score", "mean_arrival_time_s_success_only",
        "median_arrival_time_s_success_only", "mean_min_hs", "worst_min_hs",
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope"] + metric_keys)
        for region, values in per_region.items():
            writer.writerow([region] + [values.get(k) for k in metric_keys])
        writer.writerow(["overall"] + [overall.get(k) for k in metric_keys])

    lines = []
    lines.append("=" * 64)
    lines.append("PHASE-I SAFE-ARRIVAL EVALUATION REPORT")
    lines.append("=" * 64)
    lines.append(f"protocol: {summary.get('protocol')}")
    lines.append(f"episodes/region: {summary.get('episodes_per_region')}")
    lines.append(f"horizon steps: {summary.get('max_episode_steps')}  beta: {summary.get('beta')}")
    lines.append("")
    lines.append(f"weighted mu_SA          : {overall['weighted_mu_sa']:.4f}")
    lines.append(f"overall success (strict): {overall['safe_arrival_rate']:.4f}")
    lines.append(f"within-horizon arrival  : {overall['safe_arrival_within_horizon_rate']:.4f}")
    lines.append(f"failure rate            : {overall['failure_rate']:.4f}")
    lines.append(f"timeout rate            : {overall['timeout_rate']:.4f}")
    lines.append(f"safe-rollout rate       : {overall['safe_rollout_rate']:.4f}")
    lines.append(f"invariance-after-entry  : {overall['invariance_after_entry_rate']:.4f}")
    at = overall.get("mean_arrival_time_s_success_only")
    lines.append(f"mean arrival time (s)   : {at:.3f}" if at is not None else "mean arrival time (s)   : n/a")
    lines.append("")
    lines.append("-" * 64)
    lines.append(f"{'region':<16}{'success':>9}{'within-T':>10}{'failure':>9}{'arr.s':>8}")
    lines.append("-" * 64)
    for region, v in per_region.items():
        art = v.get("mean_arrival_time_s_success_only")
        art_s = f"{art:.2f}" if art is not None else "n/a"
        lines.append(
            f"{region:<16}{v['safe_arrival_rate']:>9.3f}"
            f"{v['safe_arrival_within_horizon_rate']:>10.3f}"
            f"{v['failure_rate']:>9.3f}{art_s:>8}"
        )
    lines.append("-" * 64)
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def _attitude_error_norm(state: np.ndarray) -> float:
    reduced = compute_reduced_state(state, z_des=Z_DES)
    return float(np.linalg.norm(reduced[4:7]))


def plot_region_3d(region: str, trajectories: list, out_dir: Path) -> None:
    if not trajectories:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for traj in trajectories:
        states = traj["states"]
        ok = bool(np.asarray(traj["success"]).ravel()[0])
        color = "#2ca02c" if ok else "#d62728"
        ax.plot(states[:, 0], states[:, 1], states[:, 2], color=color, alpha=0.8, lw=1.5)
        ax.scatter(*states[0, :3], color=color, s=30, marker="o")
        ax.scatter(*states[-1, :3], color=color, s=40, marker="^")
    # z_des plane marker
    ax.scatter([0], [0], [Z_DES], color="k", s=60, marker="*", label="hover target")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{region}: safe-arrival trajectories\n(green=success, red=fail)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{region}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_region_timeseries(region: str, trajectories: list, out_dir: Path) -> None:
    if not trajectories:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for traj in trajectories:
        states = traj["states"]
        ok = bool(np.asarray(traj["success"]).ravel()[0])
        color = "#2ca02c" if ok else "#d62728"
        steps = np.arange(len(states))
        axes[0, 0].plot(steps, states[:, 2], color=color, alpha=0.7)
        speed = np.linalg.norm(states[:, 3:6], axis=1)
        axes[0, 1].plot(steps, speed, color=color, alpha=0.7)
        att = np.asarray([_attitude_error_norm(s) for s in states])
        axes[1, 0].plot(steps, att, color=color, alpha=0.7)
        axes[1, 1].plot(steps, states[:, 2] - Z_DES, color=color, alpha=0.7)
    axes[0, 0].axhline(Z_CEIL, color="k", ls="--", lw=1, label="ceiling")
    axes[0, 0].set_title("altitude z"); axes[0, 0].set_ylabel("z (m)"); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("speed |v|"); axes[0, 1].set_ylabel("m/s")
    axes[1, 0].set_title("attitude error |theta|"); axes[1, 0].set_ylabel("rad"); axes[1, 0].set_xlabel("step")
    axes[1, 1].set_title("altitude error (z - z_des)"); axes[1, 1].set_ylabel("m"); axes[1, 1].set_xlabel("step")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{region}: state time-series (green=success, red=fail)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"{region}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_region_actuators(region: str, trajectories: list, out_dir: Path) -> None:
    if not trajectories:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    labels = ["thrust a_cmd", "omega_x", "omega_y", "omega_z"]
    limits = [(0.0, 4 * 9.81), (-18, 18), (-18, 18), (-18, 18)]
    for traj in trajectories:
        actions = traj.get("actions")
        if actions is None or len(actions) == 0:
            continue
        ok = bool(np.asarray(traj["success"]).ravel()[0])
        color = "#2ca02c" if ok else "#d62728"
        steps = np.arange(len(actions))
        for i in range(4):
            axes[i].plot(steps, actions[:, i], color=color, alpha=0.7)
    for i in range(4):
        axes[i].axhline(limits[i][0], color="k", ls=":", lw=0.8)
        axes[i].axhline(limits[i][1], color="k", ls=":", lw=0.8)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"{region}: applied actuator commands (green=success, red=fail)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"{region}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_region_gif(region: str, trajectories: list, out_dir: Path, n_frames: int = 36) -> bool:
    """One rotating 3D GIF per region. Returns False if deps missing."""
    if not trajectories:
        return True
    try:
        import imageio.v2 as imageio
    except Exception:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for azim in np.linspace(0, 360, n_frames, endpoint=False):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        for traj in trajectories:
            states = traj["states"]
            ok = bool(np.asarray(traj["success"]).ravel()[0])
            color = "#2ca02c" if ok else "#d62728"
            ax.plot(states[:, 0], states[:, 1], states[:, 2], color=color, alpha=0.8, lw=1.3)
            ax.scatter(*states[0, :3], color=color, s=20)
        ax.scatter([0], [0], [Z_DES], color="k", s=50, marker="*")
        ax.set_title(f"{region}")
        ax.view_init(elev=20, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_dir / f"{region}.gif", frames, duration=0.08, loop=0)
    return True


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_report(
    checkpoint: str,
    reset_library: str,
    output_dir: str,
    episodes_per_region: int = 128,
    trajectories_per_region: int = 12,
    validation_seed: int = 1234,
    make_gifs: bool = False,
    fixed_states_path: str | None = None,
) -> dict:
    out = Path(output_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    K, P = LQRGain(dt=0.02, g=9.81).gain()
    sets = SetIndicator(P=P, c_b=8.0, zceil=Z_CEIL)
    val_sampler = load_official_sampler(reset_library, split="val")

    if fixed_states_path and Path(fixed_states_path).exists():
        evaluation_set = load_evaluation_set(fixed_states_path)
    else:
        evaluation_set = build_fixed_evaluation_set(
            sets, val_sampler, seed=validation_seed,
            episodes_per_region=episodes_per_region, difficulty=1.0,
        )

    policy = TD3(state_dim=10, action_dim=4, max_action=1.0)
    policy.load(checkpoint, load_optimizers=False)

    summary, records, trajectories = evaluate_detailed(
        policy, sets, val_sampler,
        evaluation_set=evaluation_set,
        trajectories_per_region=trajectories_per_region,
        max_episode_steps=100, success_horizon_steps=100,
        integrator="euler", beta=0.99,
    )

    write_metrics(summary, out / "metrics")
    with (out / "metrics" / "episode_records.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    for region, trajs in trajectories.items():
        plot_region_3d(region, trajs, out / "plots_3d")
        plot_region_timeseries(region, trajs, out / "plots_time")
        plot_region_actuators(region, trajs, out / "plots_actuator")

    if make_gifs:
        ok = True
        for region, trajs in trajectories.items():
            ok = make_region_gif(region, trajs, out / "gifs") and ok
        if not ok:
            print("GIF dependency (imageio) missing; PNGs written, GIFs skipped.")

    print(f"Report written to {out}")
    print(f"  weighted mu_SA = {summary['overall']['weighted_mu_sa']:.4f}, "
          f"success = {summary['overall']['safe_arrival_rate']:.4f}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Phase-I final evaluation report.")
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint prefix, e.g. 'Trained Models/001/checkpoints/best'")
    p.add_argument("--reset-library", default="official_phase1_evaluation/assets/reset_library.pkl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes-per-region", type=int, default=128)
    p.add_argument("--trajectories-per-region", type=int, default=12)
    p.add_argument("--validation-seed", type=int, default=1234)
    p.add_argument("--fixed-states", default=None,
                   help="optional path to a shared fixed_validation_states.npz")
    p.add_argument("--make-gifs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_report(
        checkpoint=args.checkpoint,
        reset_library=args.reset_library,
        output_dir=args.output_dir,
        episodes_per_region=args.episodes_per_region,
        trajectories_per_region=args.trajectories_per_region,
        validation_seed=args.validation_seed,
        make_gifs=args.make_gifs,
        fixed_states_path=args.fixed_states,
    )


if __name__ == "__main__":
    main()
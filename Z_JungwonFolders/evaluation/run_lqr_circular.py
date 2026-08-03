"""Evaluate the official hover DLQR on the local Euler and RK45 dynamics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bcbf.lqrgain import LQRGain
from desired_trajectory.desired_trajectory import DesiredTrajectory
from env.dynamics import Dynamics


def quat_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def tracking_error(state, desired):
    # q_err = q_des * conjugate(q), consistent with the official hover error.
    q_err = quat_multiply(desired[6:10], quat_conjugate(state[6:10]))
    if q_err[0] < 0.0:
        q_err = -q_err
    return np.array([
        state[2] - desired[2],
        state[3] - desired[3],
        state[4] - desired[4],
        state[5] - desired[5],
        2.0*q_err[1], 2.0*q_err[2], 2.0*q_err[3],
    ])


def hover_sanity(integrator, seconds):
    dyn = Dynamics(integrator=integrator)
    initial = dyn.reset().copy()
    for _ in range(int(round(seconds / dyn.del_t))):
        dyn.step([dyn.g, 0.0, 0.0, 0.0])
    return {
        "position_drift_m": float(np.linalg.norm(dyn.state[:3] - initial[:3])),
        "velocity_drift_mps": float(np.linalg.norm(dyn.state[3:6] - initial[3:6])),
        "quaternion_norm_error": float(abs(np.linalg.norm(dyn.state[6:10]) - 1.0)),
    }


def run_circle(integrator, steps):
    dyn = Dynamics(integrator=integrator)
    state = dyn.reset().copy()
    desired_states, reference_inputs = DesiredTrajectory(dt=dyn.del_t, traj_type=1).compute_trajectory()
    if steps > len(desired_states):
        raise ValueError(f"Requested {steps} steps but trajectory has {len(desired_states)}")
    K, _ = LQRGain(dt=dyn.del_t, g=dyn.g).gain()
    actual, desired_used, controls = [], [], []
    for index in range(steps):
        desired = np.asarray(desired_states[index], dtype=float)
        u_ref = np.asarray(reference_inputs[index], dtype=float)
        control = u_ref - K @ tracking_error(state, desired)
        control = dyn.clipAction(control)
        state = dyn.step(control).copy()
        actual.append(state)
        desired_used.append(desired)
        controls.append(control)
    return dyn, np.asarray(actual), np.asarray(desired_used), np.asarray(controls)


def metrics(actual, desired):
    p_error = actual[:, :3] - desired[:, :3]
    v_error = actual[:, 3:6] - desired[:, 3:6]
    return {
        "position_rmse_m": float(np.sqrt(np.mean(p_error**2))),
        "xy_position_rmse_m": float(np.sqrt(np.mean(p_error[:, :2]**2))),
        "max_position_error_m": float(np.max(np.linalg.norm(p_error, axis=1))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(v_error**2))),
        "final_position_error_m": p_error[-1].tolist(),
    }


def save_plots(output_dir, desired, results, dt):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(desired[:, 0], desired[:, 1], "--", linewidth=2.5, label="desired")
    for name, (actual, _) in results.items():
        ax.plot(actual[:, 0], actual[:, 1], linewidth=2.0, label=name)
    ax.scatter(desired[0, 0], desired[0, 1], s=70, marker="o", label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("DLQR Circular Trajectory: Official Euler vs RK45")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "circular_trajectory_xy.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    time = np.arange(len(desired)) * dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis_index, label in enumerate(("x", "y", "z")):
        axes[0].plot(time, desired[:, axis_index], "--", label=f"{label} desired")
        for name, (actual, _) in results.items():
            axes[0].plot(time, actual[:, axis_index], label=f"{label} {name}")
    axes[0].set_ylabel("position [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=7)
    for name, (actual, controls) in results.items():
        axes[1].plot(time, np.linalg.norm(actual[:, :3] - desired[:, :3], axis=1), label=name)
        axes[2].plot(time, controls[:, 0], label=f"a_cmd {name}")
    axes[1].set_ylabel("position error [m]")
    axes[2].set_ylabel("thrust accel [m/s²]")
    axes[2].set_xlabel("time [s]")
    for ax in axes[1:]:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "circular_tracking_signals.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--hover-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", default="evaluation/LQR Evaluation")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    report = {"dt_s": 0.02, "steps": args.steps, "duration_s": 0.02*args.steps, "hover": {}}
    desired = None
    for integrator in ("euler", "rk45"):
        report["hover"][integrator] = hover_sanity(integrator, args.hover_seconds)
        dyn, actual, desired_now, controls = run_circle(integrator, args.steps)
        desired = desired_now
        results[integrator] = (actual, controls)
        report[integrator] = metrics(actual, desired_now)

    np.savez_compressed(
        output_dir / "trajectory_data.npz",
        desired=desired,
        euler_actual=results["euler"][0],
        euler_controls=results["euler"][1],
        rk45_actual=results["rk45"][0],
        rk45_controls=results["rk45"][1],
        dt=np.asarray([0.02]),
    )
    save_plots(output_dir, desired, results, 0.02)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    print(f"Saved LQR evaluation to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

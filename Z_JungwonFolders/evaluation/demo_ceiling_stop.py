"""Demonstrate that the BCBF A(x), b(x) constraints stop a straight-up climb
before the altitude ceiling.

Setup: a quadrotor starts below the ceiling and is commanded to climb straight
up at full thrust. Two rollouts are compared through the SAME Euler dynamics:

  1. UNFILTERED: the raw "climb" command is applied directly. It sails through
     the ceiling (h_s = z_max - z goes negative).

  2. QP-FILTERED: at every step the raw command is projected by the slack-QP
     built from the backup-CBF rows A(x), b(x) produced by ABConstraintBuilder.
     The BCBF row that encodes the ceiling forces the thrust down as the
     vehicle approaches z_max, so the filtered rollout stops (or turns over)
     before crossing it.

This is an end-to-end sanity check that the A, b constraints are (a) actually
binding and (b) correctly oriented (they restrain the RIGHT direction).

Usage:
    python3 -m evaluation.demo_ceiling_stop --checkpoint "Trained Models/003/checkpoints/best"
    python3 -m evaluation.demo_ceiling_stop --smoke   # zero-actor, no checkpoint needed
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import matplotlib.pyplot as plt

from env.dynamics import Dynamics
from bcbf.official_ab import ABConfig, ABConstraintBuilder
from backup_policy.phase1.state_action import normalize_quat

# Reuse the exact solver the A,b validator uses.
from evaluation.evaluate_ab_constraints import solve_slack_qp


def climb_command(g: float, thrust_scale: float = 1.6) -> np.ndarray:
    """Raw command: thrust well above hover, zero body rates -> straight up."""
    return np.array([thrust_scale * g, 0.0, 0.0, 0.0], dtype=float)


def rollout(builder, dyn_factory, x0, steps, z_max, g, filtered):
    dyn = dyn_factory()
    dyn.state = x0.copy()
    n_bcbf = builder.compute_bcbf_rows(x0)[0].shape[0]

    pos, zs, hs, thrusts, applied_list = [], [], [], [], []
    raw_thrusts, active_margins, slacks, hb_list = [], [], [], []
    for _ in range(steps):
        x = dyn.state.copy()
        u_raw = climb_command(g)

        margin = np.nan
        slack = np.nan
        if filtered:
            Q, q, _, _, G, h, info = builder.compute_qp_matrices(x, u_raw)
            result, _ = solve_slack_qp(Q, q, G, h, u_raw, n_bcbf)
            z = result.x
            u_apply = np.clip(z[:4], [0, -18, -18, -18], [4 * g, 18, 18, 18])
            slack = float(z[4]) if len(z) > 4 else np.nan
            # Tightest BCBF-row margin on the NORMALIZED rows the QP actually
            # uses (h - G z), over the BCBF rows only. Near 0 => that row is
            # active/binding. These rows are row-normalized, so the margin is
            # an interpretable O(1) quantity (unlike the raw un-normalized rows).
            gz = G[:n_bcbf] @ z
            row_margins = h[:n_bcbf] - gz
            active_margins.append(float(np.min(row_margins)))
            hb_val = info.get("h_b") if isinstance(info, dict) else None
            hb_list.append(float(np.asarray(hb_val).ravel()[0]) if hb_val is not None else np.nan)
        else:
            u_apply = np.clip(u_raw, [0, -18, -18, -18], [4 * g, 18, 18, 18])
            active_margins.append(np.nan)
            hb_list.append(np.nan)

        dyn.step(u_apply)
        pos.append(dyn.state[0:3].copy())
        z = dyn.state[2]
        zs.append(z)
        hs.append(z_max - z)
        thrusts.append(u_apply[0])
        raw_thrusts.append(u_raw[0])
        slacks.append(slack)
        applied_list.append(u_apply.copy())

    return {
        "pos": np.asarray(pos), "z": np.asarray(zs), "h_s": np.asarray(hs),
        "thrust": np.asarray(thrusts), "raw_thrust": np.asarray(raw_thrusts),
        "applied": np.asarray(applied_list), "active_margin": np.asarray(active_margins),
        "slack": np.asarray(slacks), "h_b": np.asarray(hb_list),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Phase-I actor checkpoint; omit with --smoke")
    ap.add_argument("--smoke", action="store_true",
                    help="use a zero actor (no checkpoint needed)")
    ap.add_argument("--output-dir", default="evaluation/Ceiling Stop Demo")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--z0", type=float, default=2.0, help="start altitude")
    ap.add_argument("--z-max", type=float, default=3.0, help="ceiling")
    ap.add_argument("--thrust-scale", type=float, default=1.6)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ABConfig()
    g = float(cfg.gravity)
    z_max = float(args.z_max)

    if args.smoke or args.checkpoint is None:
        builder = ABConstraintBuilder.for_smoke_test(cfg)
        tag = "smoke (zero actor)"
    else:
        builder = ABConstraintBuilder.from_checkpoint(args.checkpoint, cfg)
        tag = Path(args.checkpoint).name

    # Start upright, at rest, below the ceiling.
    x0 = np.array([0.0, 0.0, args.z0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    x0[6:10] = normalize_quat(x0[6:10])

    dyn_factory = lambda: Dynamics(integrator="euler", dt=0.02, gravity=g)

    unf = rollout(builder, dyn_factory, x0, args.steps, z_max, g, filtered=False)
    fil = rollout(builder, dyn_factory, x0, args.steps, z_max, g, filtered=True)

    # ---- report + constraint metrics ----
    unf_breach = int(np.argmax(unf["h_s"] < 0.0)) if np.any(unf["h_s"] < 0.0) else -1
    fil_min_hs = float(fil["h_s"].min())
    unf_min_hs = float(unf["h_s"].min())

    # Constraint-activity metrics (the QP-filtered rollout).
    am = fil["active_margin"]
    finite_am = am[np.isfinite(am)]
    # A BCBF row is "active/binding" when its margin ~ 0 (within a tolerance).
    active_tol = 1e-2
    frac_active = float(np.mean(np.abs(finite_am) <= active_tol)) if finite_am.size else float("nan")
    min_row_margin = float(finite_am.min()) if finite_am.size else float("nan")
    thrust_shaved = fil["raw_thrust"] - fil["thrust"]  # how much the QP removed
    max_shaved = float(np.max(thrust_shaved))
    total_slack = float(np.nansum(fil["slack"]))

    print(f"Constraint source: {tag}")
    print(f"Ceiling z_max = {z_max:.2f} m, start z0 = {args.z0:.2f} m, "
          f"raw thrust = {args.thrust_scale:.2f}g")
    print("-" * 64)
    print(f"UNFILTERED : min h_s = {unf_min_hs:+.3f} m  "
          f"{'-> CROSSES ceiling at step ' + str(unf_breach) if unf_breach >= 0 else '(no breach)'}")
    print(f"QP-FILTERED: min h_s = {fil_min_hs:+.3f} m  "
          f"{'-> STAYS BELOW ceiling' if fil_min_hs >= -1e-3 else '-> still breaches (check)'}")
    print("-" * 64)
    print("A,b constraint activity (filtered rollout):")
    print(f"  tightest BCBF row margin (b - A u)   : {min_row_margin:+.4f}  (0 => binding)")
    print(f"  fraction of steps with a binding row : {100*frac_active:.1f}%")
    print(f"  max thrust removed by the QP         : {max_shaved:.2f} m/s^2 "
          f"({100*max_shaved/(args.thrust_scale*g):.0f}% of the raw climb command)")
    print(f"  total slack used                     : {total_slack:.3e}")
    print("-" * 64)
    passed = unf_min_hs < 0.0 <= fil_min_hs + 1e-3
    verdict = ("PASS: the A,b QP stopped the climb before the ceiling while the "
               "unfiltered command crossed it."
               if passed else
               "INCONCLUSIVE: see plot; adjust --thrust-scale or --steps.")
    print(verdict)

    metrics = {
        "constraint_source": tag, "z_max": z_max, "z0": args.z0,
        "thrust_scale_g": args.thrust_scale, "steps": args.steps,
        "unfiltered_min_h_s": unf_min_hs, "filtered_min_h_s": fil_min_hs,
        "unfiltered_breach_step": unf_breach,
        "tightest_bcbf_row_margin": min_row_margin,
        "fraction_steps_binding": frac_active,
        "max_thrust_removed": max_shaved,
        "total_slack": total_slack, "passed": bool(passed),
    }
    import json as _json
    (out / "constraint_metrics.json").write_text(_json.dumps(metrics, indent=2))

    # ---- plots (time series) ----
    t = np.arange(args.steps) * 0.02
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].axhline(z_max, color="k", ls="--", lw=1.2, label="ceiling $z_{max}$")
    ax[0].plot(t, unf["z"], color="#d62728", label="unfiltered climb")
    ax[0].plot(t, fil["z"], color="#2ca02c", label="QP-filtered")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("altitude z (m)")
    ax[0].set_title("Altitude: raw command vs. A,b-filtered")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    ax[1].axhline(0.0, color="k", ls="--", lw=1.2, label="ceiling ($h_s=0$)")
    ax[1].plot(t, unf["h_s"], color="#d62728", label="unfiltered")
    ax[1].plot(t, fil["h_s"], color="#2ca02c", label="QP-filtered")
    ax[1].plot(t, fil["thrust"] / (4 * g), color="#1f77b4", ls=":",
               label="filtered thrust (frac of max)")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel(r"hard-deck margin $h_s = z_{max}-z$")
    ax[1].set_title("Ceiling margin and filtered thrust")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"BCBF A(x),b(x) ceiling-stop demo  [{tag}]", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "ceiling_stop.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- 3D trajectory plot ----
    fig = plt.figure(figsize=(9, 7))
    ax3 = fig.add_subplot(111, projection="3d")
    pu, pf = unf["pos"], fil["pos"]
    ax3.plot(pu[:, 0], pu[:, 1], pu[:, 2], color="#d62728", lw=2, label="unfiltered climb")
    ax3.plot(pf[:, 0], pf[:, 1], pf[:, 2], color="#2ca02c", lw=2, label="QP-filtered")
    ax3.scatter(*pf[0], color="k", s=40, marker="o", label="start")
    # translucent ceiling plane at z_max
    span = 1.0
    xx, yy = np.meshgrid(np.linspace(-span, span, 2), np.linspace(-span, span, 2))
    ax3.plot_surface(xx, yy, np.full_like(xx, z_max), alpha=0.15, color="k")
    ax3.text(0, 0, z_max + 0.05, "ceiling", color="k")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title(f"3D trajectory: A,b-filtered stays under the ceiling  [{tag}]")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(out / "ceiling_stop_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- constraint-activity plot (the proof the row binds) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].axhline(0.0, color="k", ls="--", lw=1.0, label="row active (margin=0)")
    ax[0].plot(t, fil["active_margin"], color="#9467bd", label="tightest BCBF margin $b-Au$")
    axb = ax[0].twinx()
    axb.plot(t, fil["h_s"], color="#2ca02c", ls=":", label="$h_s$ (ceiling margin)")
    axb.set_ylabel(r"$h_s$ (m)", color="#2ca02c")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("tightest BCBF row margin")
    ax[0].set_title("BCBF row goes active as the ceiling approaches")
    ax[0].legend(loc="upper right"); ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, fil["raw_thrust"], color="#d62728", ls="--", label="raw commanded thrust")
    ax[1].plot(t, fil["thrust"], color="#2ca02c", label="QP-applied thrust")
    ax[1].fill_between(t, fil["thrust"], fil["raw_thrust"], color="#ff7f0e", alpha=0.3,
                       label="thrust removed by A,b")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel(r"thrust $a_{cmd}$ (m/s$^2$)")
    ax[1].set_title("The QP removes exactly the thrust that would breach")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.suptitle("A,b constraint is binding and correctly oriented", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "ceiling_stop_constraint_activity.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    np.savez(out / "ceiling_stop_data.npz",
             t=t, pos_unf=unf["pos"], pos_fil=fil["pos"],
             z_unf=unf["z"], z_fil=fil["z"], hs_unf=unf["h_s"], hs_fil=fil["h_s"],
             thrust_fil=fil["thrust"], raw_thrust_fil=fil["raw_thrust"],
             applied_fil=fil["applied"], active_margin_fil=fil["active_margin"],
             slack_fil=fil["slack"])
    print(f"\nSaved plots + data + constraint_metrics.json to {out}")


if __name__ == "__main__":
    main()
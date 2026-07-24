"""
Evaluation for the PS2-RL Phase I safe-arrival policy.

Produces, in one run:

  1. TRAINING CURVES  (from results/*_eval.npy, logged during training)
       - mu_SA and near-ceiling rate vs training progress
       - success / failure / timeout rates
       - success-horizon rate and curriculum difficulty s
       - steps-to-success and final h_B

  2. TRAJECTORY PLOTS (roll out a saved checkpoint)
       - 3D powerloop-frame trajectories, several per curriculum region
       - altitude p_z vs time against the ceiling
       - attitude error and speed vs time
       - a per-region numeric summary table

Usage
-----
    # both, using the best checkpoint:
    python evaluate.py

    # only the training curves (no rollouts):
    python evaluate.py --curves_only

    # only trajectories, from a specific checkpoint:
    python evaluate.py --traj_only --checkpoint ./models/td3_safe_arrival_v4_best

    # more rollouts per region:
    python evaluate.py --episodes_per_region 8

Outputs go to ./eval_plots/ as PNGs, plus a printed numeric summary.
"""

import os
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch

# Import the training module and its siblings regardless of how this script is
# launched. Two layouts are supported:
#   (a) run from the project root as a package:  python3 -m backup_policy.evaluate
#       -> train/td3/replay_buffer live in the backup_policy package
#   (b) run from inside backup_policy/:          python3 evaluate.py
#       -> they are top-level modules on sys.path
# env/ and bcbf/ sit at the project root in both cases, so the project root is
# added to sys.path explicitly.
import os as _os
import sys as _sys

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.dirname(_THIS_DIR)

for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

try:
    # Layout (a): package-qualified imports.
    from backup_policy import train as T
    from backup_policy.td3 import TD3
    from backup_policy.replay_buffer import ReplayBuffer  # noqa: F401
except ImportError:
    # Layout (b): flat imports.
    import train as T
    from td3 import TD3
    from replay_buffer import ReplayBuffer  # noqa: F401

from env.dynamics import Dynamics
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator


# Column layout of each row saved in results/*_eval.npy:
#   (s, success_rate, success_horizon_rate, failure_rate, timeout_rate,
#    avg_steps, avg_min_hs, avg_final_hb, median_final_hb,
#    worst_region_success, mu_sa)
COLS = {
    "s": 0,
    "success_rate": 1,
    "success_horizon_rate": 2,
    "failure_rate": 3,
    "timeout_rate": 4,
    "avg_steps": 5,
    "avg_min_hs": 6,
    "avg_final_hb": 7,
    "median_final_hb": 8,
    "worst_region_success": 9,
    "mu_sa": 10,
}

REGIONS = [
    "synthetic_capture",
    "synthetic_mid",
    "trace_general",
    "near_ceiling",
    "bridge",
]

REGION_COLORS = {
    "synthetic_capture": "#4C72B0",
    "synthetic_mid": "#55A868",
    "trace_general": "#8172B3",
    "near_ceiling": "#C44E52",
    "bridge": "#CCB974",
}

# Paper reference numbers (Sec. 5.2), for horizontal guide lines.
PAPER_MU_SA = 0.859
PAPER_NEAR_CEILING = 0.693


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_env(seed=0):
    """Reconstruct the exact objects training used."""

    dyn = Dynamics()

    gains = LQRGain(dt=dyn.del_t, g=dyn.g)
    K, P = gains.gain()

    c_b = 8.0

    sets = SetIndicator(P=P, c_b=c_b, zceil=3.0)
    sets.P = P

    trace_states = T.generate_reference_trace(
        n_points=200, n_variants=20, seed=0
    )

    regions = T.classify_trace_states(
        sets=sets,
        trace_states=trace_states,
        P=P,
        c_b=c_b,
        near_ceiling_margin=0.25,
    )

    return dyn, sets, regions, K, P


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(eval_path, eval_freq, out_dir):

    if not os.path.exists(eval_path):
        print(f"[curves] no eval log at {eval_path} -- skipping training curves.")
        print("         (it is written once the first evaluation runs.)")
        return

    data = np.load(eval_path)

    # Prefer the eval_freq saved by training over the CLI value, so the x-axis
    # is correct even if --eval_freq was left at its default.
    freq_path = eval_path.replace("_eval.npy", "_eval_freq.npy")
    if os.path.exists(freq_path):
        saved_freq = int(np.load(freq_path)[0])
        if saved_freq != eval_freq:
            print(f"[curves] using saved eval_freq={saved_freq} "
                  f"(CLI passed {eval_freq})")
        eval_freq = saved_freq

    if data.ndim != 2 or data.shape[0] == 0:
        print(f"[curves] eval log at {eval_path} is empty -- skipping.")
        return

    # x-axis: environment steps at each evaluation.
    x = (np.arange(data.shape[0]) + 1) * eval_freq

    def col(name):
        return data[:, COLS[name]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) mu_SA and near-ceiling proxy (worst-region as a floor indicator)
    ax = axes[0, 0]
    ax.plot(x, col("mu_sa"), color="#C44E52", lw=2, label="mu_SA (weighted)")
    ax.plot(x, col("worst_region_success"), color="#4C72B0", lw=1.5,
            alpha=0.8, label="worst-region rate")
    ax.axhline(PAPER_MU_SA, color="#C44E52", ls="--", alpha=0.5,
               label=f"paper mu_SA = {PAPER_MU_SA}")
    ax.axhline(PAPER_NEAR_CEILING, color="gray", ls=":", alpha=0.6,
               label=f"paper near-ceiling = {PAPER_NEAR_CEILING}")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("rate")
    ax.set_title("Safe-arrival measure mu_SA")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) success / failure / timeout
    ax = axes[0, 1]
    ax.plot(x, col("success_rate"), color="#55A868", lw=2, label="success (any)")
    ax.plot(x, col("success_horizon_rate"), color="#55A868", lw=1.5, ls="--",
            label="success <= T")
    ax.plot(x, col("failure_rate"), color="#C44E52", lw=2, label="failure")
    ax.plot(x, col("timeout_rate"), color="#CCB974", lw=2, label="timeout")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("rate")
    ax.set_title("Episode outcomes")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) curriculum difficulty s
    ax = axes[1, 0]
    ax.plot(x, col("s"), color="#8172B3", lw=2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("curriculum difficulty s")
    ax.set_title("Curriculum progression")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)

    # (d) steps-to-success and final h_B (twin axis)
    ax = axes[1, 1]
    l1, = ax.plot(x, col("avg_steps"), color="#4C72B0", lw=2,
                  label="avg steps (to termination)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("avg steps", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    l2, = ax2.plot(x, col("median_final_hb"), color="#C44E52", lw=2,
                   label="median final h_B")
    ax2.set_ylabel("median final h_B (clipped)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    ax.set_title("Arrival speed and terminal proximity to B")
    ax.legend(handles=[l1, l2], fontsize=8, loc="best")

    fig.suptitle("PS2-RL Phase I -- training curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[curves] saved {path}")

    # Numeric tail summary.
    last = data[-1]
    print("\n[curves] latest evaluation:")
    print(f"    env steps          : {int(x[-1]):,}")
    print(f"    curriculum s        : {last[COLS['s']]:.3f}")
    print(f"    mu_SA (weighted)    : {last[COLS['mu_sa']]:.3f}   (paper {PAPER_MU_SA})")
    print(f"    worst-region rate   : {last[COLS['worst_region_success']]:.3f}")
    print(f"    success (any)       : {last[COLS['success_rate']]:.3f}")
    print(f"    success <= T        : {last[COLS['success_horizon_rate']]:.3f}")
    print(f"    failure             : {last[COLS['failure_rate']]:.3f}")
    print(f"    timeout             : {last[COLS['timeout_rate']]:.3f}")
    print(f"    avg steps           : {last[COLS['avg_steps']]:.1f}")


# ---------------------------------------------------------------------------
# 2. Trajectory rollouts
# ---------------------------------------------------------------------------

def rollout(policy, dyn, sets, state, max_steps, success_horizon):
    """
    Roll the deterministic policy from `state`. Returns a dict of logged
    quantities and the outcome.
    """

    T.reset_dynamics_state(dyn, state)

    xs = [state.copy()]
    hb_log = []
    hs_log = []
    att_log = []
    spd_log = []

    outcome = "timeout"
    term_step = max_steps

    s = state.copy()

    for step in range(max_steps):

        a_norm = policy.select_action(np.array(s))
        a_norm = np.clip(a_norm, -1.0, 1.0)

        a = T.scale_action(a_norm, dyn.g)
        nxt = dyn.step(a).copy()

        b_next, f_next, c_next = T.compute_bfc(sets, nxt)
        hs_log.append(sets.hs)

        reduced = T.compute_reduced_state(nxt)
        sets.compute_hb(reduced)
        hb_log.append(sets.hb)

        _, _, vnorm, att = T.state_difficulty(nxt, sets.P, sets.c_b, sets.zceil)
        att_log.append(att)
        spd_log.append(vnorm)

        xs.append(nxt.copy())
        s = nxt.copy()

        if b_next == 1.0:
            outcome = "success"
            term_step = step + 1
            break

        if f_next == 1.0:
            outcome = "failure"
            term_step = step + 1
            break

    xs = np.array(xs)

    return {
        "xs": xs,
        "hb": np.array(hb_log),
        "hs": np.array(hs_log),
        "att": np.array(att_log),
        "spd": np.array(spd_log),
        "outcome": outcome,
        "term_step": term_step,
        "within_horizon": (outcome == "success" and term_step <= success_horizon),
    }


def plot_trajectories(policy, dyn, sets, regions, out_dir,
                      episodes_per_region=5, max_steps=300,
                      success_horizon=100, dt=0.02, seed=1):

    rng = np.random.default_rng(seed)

    zceil = sets.zceil

    # Roll out everything first, grouped by region.
    all_traj = {}
    summary = {}

    for region in REGIONS:

        if region in ["trace_general", "near_ceiling", "bridge"]:
            if region not in regions or len(regions[region]) == 0:
                continue

        trajs = []
        for _ in range(episodes_per_region):
            try:
                state, rname = T.sample_initial_state(
                    sets=sets, regions=regions, s=1.0, rng=rng,
                    return_region=True, force_region=region,
                )
            except RuntimeError:
                continue

            trajs.append(rollout(policy, dyn, sets, state,
                                 max_steps, success_horizon))

        if not trajs:
            continue

        all_traj[region] = trajs

        n = len(trajs)
        n_succ = sum(t["outcome"] == "success" for t in trajs)
        n_horizon = sum(t["within_horizon"] for t in trajs)
        n_fail = sum(t["outcome"] == "failure" for t in trajs)
        n_to = sum(t["outcome"] == "timeout" for t in trajs)
        succ_steps = [t["term_step"] for t in trajs if t["outcome"] == "success"]

        summary[region] = {
            "n": n,
            "success": n_succ / n,
            "within_horizon": n_horizon / n,
            "failure": n_fail / n,
            "timeout": n_to / n,
            "avg_success_steps": np.mean(succ_steps) if succ_steps else float("nan"),
        }

    if not all_traj:
        print("[traj] no trajectories rolled out -- skipping.")
        return

    # ---- 3D trajectories, one subplot per region ----
    regions_present = list(all_traj.keys())
    ncol = min(3, len(regions_present))
    nrow = int(np.ceil(len(regions_present) / ncol))

    fig = plt.figure(figsize=(6 * ncol, 5 * nrow))

    # Ceiling plane extent.
    for i, region in enumerate(regions_present):

        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")

        for t in all_traj[region]:
            xs = t["xs"]
            c = {"success": "#55A868", "failure": "#C44E52",
                 "timeout": "#CCB974"}[t["outcome"]]
            ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color=c, lw=1.3, alpha=0.8)
            ax.scatter(xs[0, 0], xs[0, 1], xs[0, 2], color="black",
                       s=18, marker="o")
            ax.scatter(xs[-1, 0], xs[-1, 1], xs[-1, 2], color=c,
                       s=28, marker="*")

        # Ceiling plane.
        xr = ax.get_xlim()
        yr = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(xr[0], xr[1], 2),
                             np.linspace(yr[0], yr[1], 2))
        ax.plot_surface(xx, yy, np.full_like(xx, zceil),
                        color="red", alpha=0.12)

        ax.set_xlabel("p_x")
        ax.set_ylabel("p_y")
        ax.set_zlabel("p_z")
        sm = summary[region]
        ax.set_title(f"{region}\nsucc={sm['success']:.2f} "
                     f"(<=T {sm['within_horizon']:.2f})")

    fig.suptitle("Safe-arrival rollouts -- 3D  "
                 "(green=success, red=failure, gold=timeout; "
                 "o=start, *=end; red plane=ceiling)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "trajectories_3d.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[traj] saved {path}")

    # ---- Time-series: altitude, attitude error, speed ----
    fig, axes = plt.subplots(len(regions_present), 3,
                             figsize=(15, 3.2 * len(regions_present)),
                             squeeze=False)

    for i, region in enumerate(regions_present):

        ax_z, ax_a, ax_v = axes[i]

        for t in all_traj[region]:
            c = {"success": "#55A868", "failure": "#C44E52",
                 "timeout": "#CCB974"}[t["outcome"]]
            steps = t["xs"].shape[0]
            tt = np.arange(steps) * dt

            ax_z.plot(tt, t["xs"][:, 2], color=c, lw=1.1, alpha=0.8)
            if len(t["att"]):
                ta = np.arange(len(t["att"])) * dt
                ax_a.plot(ta, t["att"], color=c, lw=1.1, alpha=0.8)
                ax_v.plot(ta, t["spd"], color=c, lw=1.1, alpha=0.8)

        ax_z.axhline(zceil, color="red", ls="--", alpha=0.6)
        ax_z.axhline(2.0, color="gray", ls=":", alpha=0.5)  # z_des
        ax_z.set_ylabel(f"{region}\np_z [m]")
        ax_z.grid(alpha=0.3)

        ax_a.axhline(np.sqrt(sets.c_b), color="gray", ls=":", alpha=0.0)
        ax_a.set_ylabel("|att err|")
        ax_a.grid(alpha=0.3)

        ax_v.set_ylabel("|v| [m/s]")
        ax_v.grid(alpha=0.3)

        if i == 0:
            ax_z.set_title("altitude p_z  (red=ceiling, gray=z_des)")
            ax_a.set_title("attitude error")
            ax_v.set_title("speed")

        if i == len(regions_present) - 1:
            for ax in (ax_z, ax_a, ax_v):
                ax.set_xlabel("time [s]")

    fig.suptitle("Safe-arrival rollouts -- time series", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "trajectories_timeseries.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[traj] saved {path}")

    # ---- Numeric summary ----
    print("\n[traj] per-region rollout summary "
          f"({episodes_per_region} episodes each, s=1.0):")
    print(f"    {'region':<20} {'n':>4} {'succ':>6} {'<=T':>6} "
          f"{'fail':>6} {'t/o':>6} {'avg_steps':>10}")
    mu = 0.0
    wtot = 0.0
    for region in regions_present:
        sm = summary[region]
        w = T.MU_SA_WEIGHTS.get(region, 0.0)
        mu += w * sm["within_horizon"]
        wtot += w
        avg = sm["avg_success_steps"]
        avg_s = f"{avg:.1f}" if not np.isnan(avg) else "  -  "
        print(f"    {region:<20} {sm['n']:>4} {sm['success']:>6.2f} "
              f"{sm['within_horizon']:>6.2f} {sm['failure']:>6.2f} "
              f"{sm['timeout']:>6.2f} {avg_s:>10}")
    if wtot > 0:
        print(f"\n    weighted mu_SA over these rollouts: {mu / wtot:.3f}   "
              f"(paper {PAPER_MU_SA}; near-ceiling target {PAPER_NEAR_CEILING})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                    default="./models/td3_safe_arrival_v4_best",
                    help="policy checkpoint prefix (without _actor/_critic)")
    ap.add_argument("--eval_log", type=str,
                    default="./results/td3_safe_arrival_v4_eval.npy")
    ap.add_argument("--eval_freq", type=int, default=10000,
                    help="must match training eval_freq for a correct x-axis")
    ap.add_argument("--episodes_per_region", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--success_horizon", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="./eval_plots")
    ap.add_argument("--curves_only", action="store_true")
    ap.add_argument("--traj_only", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.traj_only:
        plot_training_curves(args.eval_log, args.eval_freq, args.out_dir)

    if args.curves_only:
        return

    # Trajectory rollouts need a checkpoint.
    actor_path = args.checkpoint + "_actor"
    if not os.path.exists(actor_path):
        print(f"\n[traj] checkpoint not found: {actor_path}")
        print("       (train for at least one eval_freq, or pass --checkpoint)")
        return

    dyn, sets, regions, K, P = build_env(seed=args.seed)

    policy = TD3(
        state_dim=10,
        action_dim=4,
        max_action=1.0,
    )
    policy.load(args.checkpoint)
    print(f"\n[traj] loaded checkpoint {args.checkpoint}")

    plot_trajectories(
        policy=policy,
        dyn=dyn,
        sets=sets,
        regions=regions,
        out_dir=args.out_dir,
        episodes_per_region=args.episodes_per_region,
        max_steps=args.max_steps,
        success_horizon=args.success_horizon,
        dt=dyn.del_t,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
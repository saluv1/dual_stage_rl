"""
Step 1 - Backup-policy composition and convergence validation.

Composes the backup policy

    pi_b(x) = pi_B(x)   if x in B   (certified LQR base controller)
              pi_SA(x)  otherwise   (Phase-I safe-arrival TD3 actor)

and checks, by closed-loop RK4 rollout, whether trajectories launched from
random states in the design region arrive at (and stay near) the hover
equilibrium

    x* = [px, py, z_des, 0,0,0, 1,0,0,0].

Initial states come from BOTH:
  * the fixed validation set saved with the trained model
    (Trained Models/<run>/config/fixed_validation_states.npz), and
  * fresh uniform samples drawn to match each design region's per-coordinate
    range in that validation set.

Both switch modes ("hard", "blend") are evaluated side by side.

All artifacts are written under
    Z_JungwonFolders/evaluation/backup_validation/
in a dated, clearly labelled tree.  Nothing is written anywhere else.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bcbf.backup_policy import (
    ActorParams,
    BackupContext,
    make_closed_loop,
    pi_b,
    random_actor_params,
)
from bcbf.barriers import h_B, h_S, reduced_state
from bcbf.dynamics_jax import G_ACC
from bcbf.lqrgain import LQRGain

REGIONS = ["base_shell", "bridge", "general_trace", "near_ceiling"]
MODES = ["hard", "blend"]


# ----------------------------------------------------------------------
# actor loading
# ----------------------------------------------------------------------

def load_actor_from_checkpoint(path: Path, max_action: float = 1.0) -> ActorParams:
    """Load the Phase-I TD3 actor from a full training checkpoint.

    The checkpoint saved by backup_policy/td3.py is a dict with keys
    {actor, actor_target, critic, ...}; the actor sub-dict holds l1/l2/l3.
    A bare actor state_dict (l1.weight at top level) is also accepted.
    """
    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "actor" in payload:
        sd = payload["actor"]
    else:
        sd = payload

    def get(name):
        return jnp.asarray(sd[name].detach().cpu().numpy())

    return ActorParams(
        W1=get("l1.weight").T, b1=get("l1.bias"),
        W2=get("l2.weight").T, b2=get("l2.bias"),
        W3=get("l3.weight").T, b3=get("l3.bias"),
        max_action=max_action,
    )


def load_actor_from_npz(path: Path, max_action: float = 1.0) -> ActorParams:
    """Framework-independent loader from an extracted actor .npz (l*_weight/bias)."""
    d = np.load(path)
    j = lambda a: jnp.asarray(a, dtype=jnp.float32)
    return ActorParams(
        W1=j(d["l1_weight"]).T, b1=j(d["l1_bias"]),
        W2=j(d["l2_weight"]).T, b2=j(d["l2_bias"]),
        W3=j(d["l3_weight"]).T, b3=j(d["l3_bias"]),
        max_action=max_action,
    )


# ----------------------------------------------------------------------
# initial-state sources
# ----------------------------------------------------------------------

def load_validation_states(npz_path: Path) -> dict[str, np.ndarray]:
    d = np.load(npz_path)
    return {r: np.asarray(d[r], dtype=np.float64) for r in REGIONS if r in d.files}


def sample_fresh_states(val_states: dict[str, np.ndarray], n_per_region: int,
                        seed: int) -> dict[str, np.ndarray]:
    """Uniformly sample fresh states matching each region's per-coordinate range.

    Ranges are taken from the validation set so "design region" means exactly
    the region the model was validated on.  Quaternions are renormalized.
    """
    rng = np.random.default_rng(seed)
    fresh = {}
    for r, arr in val_states.items():
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1e-6)
        s = lo + rng.random((n_per_region, arr.shape[1])) * span
        q = s[:, 6:10]
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        s[:, 6:10] = np.where(qn > 1e-9, q / qn, np.array([1.0, 0, 0, 0]))
        fresh[r] = s
    return fresh


# ----------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------

def make_rk4_rollout(ctx: BackupContext, mode: str, dt: float, steps: int):
    f_cl = make_closed_loop(ctx, mode=mode)

    def step(x, _):
        k1 = f_cl(x)
        k2 = f_cl(x + 0.5 * dt * k1)
        k3 = f_cl(x + 0.5 * dt * k2)
        k4 = f_cl(x + dt * k3)
        xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q = xn[6:10]
        xn = xn.at[6:10].set(q / (jnp.linalg.norm(q) + 1e-12))
        return xn, xn

    def rollout(x0):
        _, xs = jax.lax.scan(step, x0, None, length=steps)
        return jnp.concatenate([x0[None], xs], axis=0)  # (steps+1, 10)

    return jax.jit(jax.vmap(rollout))


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def convergence_metrics(traj: np.ndarray, P: np.ndarray, z_des: float,
                        c_b: float, z_ceil: float, pos_tol: float,
                        vel_tol: float) -> dict:
    """traj: (N, T+1, 10). Return per-trajectory arrival + safety metrics."""
    xT = traj[:, -1, :]
    hover = np.array([0, 0, z_des, 0, 0, 0, 1, 0, 0, 0.0])

    pos_err = np.linalg.norm(xT[:, 0:3] - hover[0:3], axis=1)
    vel_err = np.linalg.norm(xT[:, 3:6], axis=1)

    e = np.stack([_reduced(x, z_des) for x in xT])
    hb_T = c_b - np.einsum("ni,ij,nj->n", e, P, e)

    pz = traj[:, :, 2]
    hs_min = (z_ceil - pz).min(axis=1)  # over the whole trajectory

    arrived = (pos_err < pos_tol) & (vel_err < vel_tol)
    in_base = hb_T >= 0.0
    ceiling_ok = hs_min >= 0.0

    return dict(
        pos_err=pos_err, vel_err=vel_err, hb_T=hb_T, hs_min=hs_min,
        arrived=arrived, in_base=in_base, ceiling_ok=ceiling_ok,
    )


def _reduced(x, z_des):
    pz = x[2]; v = x[3:6]; qw = x[6]; qv = x[7:10]
    s = -1.0 if qw < 0 else 1.0
    return np.concatenate([[pz - z_des], v, 2.0 * s * qv])


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------

def plot_region(traj: np.ndarray, m: dict, region: str, mode: str,
                source: str, z_des: float, z_ceil: float, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(traj.shape[1])
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Backup rollout - {source} / {region} / mode={mode}",
                 fontsize=13, fontweight="bold")

    for i in range(traj.shape[0]):
        ax[0, 0].plot(t, traj[i, :, 2], lw=0.6, alpha=0.5)
    ax[0, 0].axhline(z_des, color="g", ls="--", lw=1, label="z_des")
    ax[0, 0].axhline(z_ceil, color="r", ls="--", lw=1, label="z_ceil")
    ax[0, 0].set(title="altitude p_z", xlabel="step", ylabel="p_z [m]")
    ax[0, 0].legend(fontsize=8)

    speed = np.linalg.norm(traj[:, :, 3:6], axis=2)
    for i in range(traj.shape[0]):
        ax[0, 1].plot(t, speed[i], lw=0.6, alpha=0.5)
    ax[0, 1].set(title="speed ||v||", xlabel="step", ylabel="[m/s]")

    ax[1, 0].hist(m["pos_err"], bins=20, color="steelblue")
    ax[1, 0].set(title=f"final position error (arrived {m['arrived'].mean()*100:.0f}%)",
                 xlabel="||p - p*|| [m]", ylabel="count")

    ax[1, 1].hist(m["hb_T"], bins=20, color="indianred")
    ax[1, 1].axvline(0, color="k", ls="--", lw=1)
    ax[1, 1].set(title=f"terminal h_B (in-base {m['in_base'].mean()*100:.0f}%)",
                 xlabel="h_B(x_T)", ylabel="count")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{region}.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to Phase-I actor checkpoint (best.pt).")
    ap.add_argument("--actor-npz", type=Path, default=None,
                    help="Alternative: framework-free actor .npz.")
    ap.add_argument("--val-states", type=Path, required=True,
                    help="fixed_validation_states.npz")
    ap.add_argument("--out-root", type=Path,
                    default=Path("Z_JungwonFolders/evaluation/backup_validation"))
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--fresh-per-region", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c-b", type=float, default=8.0)
    ap.add_argument("--z-des", type=float, default=2.0)
    ap.add_argument("--z-ceil", type=float, default=3.0)
    ap.add_argument("--pos-tol", type=float, default=0.25)
    ap.add_argument("--vel-tol", type=float, default=0.25)
    ap.add_argument("--use-random-actor", action="store_true",
                    help="Testing only: random-weight actor, no checkpoint.")
    args = ap.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- backup context ---
    K, P = LQRGain(dt=args.dt, g=G_ACC).gain()
    if args.use_random_actor:
        actor = random_actor_params(jax.random.PRNGKey(args.seed))
        actor_src = "RANDOM (test only)"
    elif args.actor_npz is not None:
        actor = load_actor_from_npz(args.actor_npz)
        actor_src = str(args.actor_npz)
    elif args.checkpoint is not None:
        actor = load_actor_from_checkpoint(args.checkpoint)
        actor_src = str(args.checkpoint)
    else:
        raise SystemExit("provide --checkpoint, --actor-npz, or --use-random-actor")

    ctx = BackupContext(
        K=jnp.asarray(K), P=jnp.asarray(P),
        u_star=jnp.array([G_ACC, 0.0, 0.0, 0.0]),
        actor=actor, c_b=args.c_b, z_des=args.z_des, z_ceil=args.z_ceil,
    )

    # --- initial states ---
    val_states = load_validation_states(args.val_states)
    fresh_states = sample_fresh_states(val_states, args.fresh_per_region, args.seed)
    sources = {"validation": val_states, "fresh": fresh_states}

    summary = {"stamp": stamp, "actor_source": actor_src,
               "dt": args.dt, "steps": args.steps,
               "z_des": args.z_des, "z_ceil": args.z_ceil, "c_b": args.c_b,
               "results": {}}

    for mode in MODES:
        for source_name, states in sources.items():
            for region, x0s in states.items():
                rollout = make_rk4_rollout(ctx, mode, args.dt, args.steps)
                traj = np.asarray(rollout(jnp.asarray(x0s, dtype=jnp.float32)))
                m = convergence_metrics(traj, P, args.z_des, args.c_b,
                                        args.z_ceil, args.pos_tol, args.vel_tol)

                out_dir = run_dir / f"mode_{mode}" / source_name
                plot_region(traj, m, region, mode, source_name,
                            args.z_des, args.z_ceil, out_dir)

                key = f"{mode}/{source_name}/{region}"
                summary["results"][key] = {
                    "n": int(traj.shape[0]),
                    "arrived_frac": float(m["arrived"].mean()),
                    "in_base_frac": float(m["in_base"].mean()),
                    "ceiling_ok_frac": float(m["ceiling_ok"].mean()),
                    "pos_err_mean": float(m["pos_err"].mean()),
                    "pos_err_p95": float(np.percentile(m["pos_err"], 95)),
                    "vel_err_mean": float(m["vel_err"].mean()),
                    "hb_T_mean": float(m["hb_T"].mean()),
                    "hs_min_min": float(m["hs_min"].min()),
                }
                print(f"[{key:38s}] arrived {m['arrived'].mean()*100:5.1f}%  "
                      f"in-base {m['in_base'].mean()*100:5.1f}%  "
                      f"ceil-ok {m['ceiling_ok'].mean()*100:5.1f}%  "
                      f"pos_err {m['pos_err'].mean():.3f}")

    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved everything under: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
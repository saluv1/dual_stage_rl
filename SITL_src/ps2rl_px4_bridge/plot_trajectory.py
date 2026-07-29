#!/usr/bin/env python3
"""Plot commanded vs achieved trajectory from a PS2-RL bridge flight log.

    python3 plot_trajectory.py /tmp/run_final.csv
    python3 plot_trajectory.py run1.csv run2.csv run3.csv --out compare.png

The centrepiece is the x-z plane view. The powerloop lives in that plane, and
drawing the safety ceiling across it shows the whole argument of the method in
one picture: the reference deliberately climbs through z_max, and the CIL has
to hold the vehicle underneath it. A run where the vehicle never approaches the
ceiling is not a safe run — it is a run that failed to track.

Every panel is annotated with the control rate actually achieved, because a
trajectory flown at 22 Hz is not a measurement of the same system as one flown
at 50 Hz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Z_MAX_DEFAULT = 3.0
NOMINAL_DT = 0.02


def load(path: str) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.size == 0:
        raise SystemExit(f"{path} contains no data rows")
    return {name: np.atleast_1d(np.asarray(raw[name], dtype=np.float64))
            for name in raw.dtype.names}


def summarise(name: str, d: dict[str, np.ndarray], z_max: float) -> dict:
    t = d["t"]
    n = t.size
    dt = float(np.median(np.diff(t))) if n > 1 else NOMINAL_DT

    pos = np.column_stack([d["px"], d["py"], d["pz"]])
    ref = np.column_stack([d["ref_px"], d["ref_py"], d["ref_pz"]])
    err = np.linalg.norm(pos - ref, axis=1)

    return {
        "name": name,
        "n": n,
        "dt": dt,
        "hz": 1.0 / dt if dt > 0 else float("nan"),
        "pos": pos,
        "ref": ref,
        "err": err,
        "rms_err": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(err.max()),
        "max_z": float(pos[:, 2].max()),
        "ref_max_z": float(ref[:, 2].max()),
        "violated": float(pos[:, 2].max()) > z_max,
        "t": t,
        "d": d,
    }


def plot_xz(ax, runs, z_max: float) -> None:
    ref = runs[0]["ref"]
    ax.plot(ref[:, 0], ref[:, 2], "k--", lw=2.0, label="reference", zorder=3)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
    for run, c in zip(runs, colors):
        pos = run["pos"]
        label = f"{run['name']} ({run['hz']:.0f} Hz)"
        ax.plot(pos[:, 0], pos[:, 2], "-", color=c, lw=1.8, label=label, zorder=4)
        ax.plot(pos[0, 0], pos[0, 2], "o", color=c, ms=7, zorder=5)

    ax.axhline(z_max, color="crimson", ls="-", lw=1.6, zorder=2)
    ax.axhspan(z_max, max(z_max, ref[:, 2].max()) + 0.6,
               color="crimson", alpha=0.08, zorder=1)
    ax.text(0.99, z_max + 0.06, f"safety ceiling  z ≤ {z_max:.1f} m",
            color="crimson", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=9)

    ax.set_xlabel("x [m]  (ENU)")
    ax.set_ylabel("z [m]")
    ax.set_title("Powerloop in the x–z plane\n"
                 "reference passes through the ceiling; the CIL must not",
                 fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def plot_z_time(ax, runs, z_max: float) -> None:
    ax.plot(runs[0]["t"], runs[0]["ref"][:, 2], "k--", lw=1.8, label="reference")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
    for run, c in zip(runs, colors):
        ax.plot(run["t"], run["pos"][:, 2], "-", color=c, lw=1.5, label=run["name"])
    ax.axhline(z_max, color="crimson", lw=1.4)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("z [m]")
    ax.set_title("Altitude vs the ceiling", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_error(ax, runs) -> None:
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
    for run, c in zip(runs, colors):
        ax.plot(run["t"], run["err"], "-", color=c, lw=1.5,
                label=f"{run['name']}  RMS {run['rms_err']:.2f} m")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("‖p − p_ref‖ [m]")
    ax.set_title("Position tracking error", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_rates(ax, run) -> None:
    d, t = run["d"], run["t"]
    if not all(f"w{a}_meas" in d for a in "xyz"):
        ax.text(0.5, 0.5, "no measured rates in log", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return
    for a, c in zip("xyz", ("tab:blue", "tab:orange", "tab:green")):
        ax.plot(t, d[f"w{a}"], "-", color=c, lw=1.6, label=f"ω{a} commanded")
        ax.plot(t, d[f"w{a}_meas"], "--", color=c, lw=1.2, alpha=0.85,
                label=f"ω{a} achieved")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("body rate [rad/s]")
    ax.set_title(f"Body rates: commanded vs achieved  ({run['name']})", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)


def plot_thrust(ax, run) -> None:
    d, t = run["d"], run["t"]
    ax.plot(t, d["a_cmd"], "-", color="tab:red", lw=1.6, label="a_cmd commanded")
    if "a_cmd_meas" in d and np.any(np.isfinite(d["a_cmd_meas"])):
        ax.plot(t, d["a_cmd_meas"], "--", color="tab:red", lw=1.2, alpha=0.85,
                label="a_cmd achieved")
    if "saturated" in d:
        sat = d["saturated"] > 0.5
        if sat.any():
            ax.fill_between(t, 0, 1, where=sat, transform=ax.get_xaxis_transform(),
                            color="crimson", alpha=0.12,
                            label=f"thrust saturated ({100 * sat.mean():.0f}%)")
    ax.axhline(9.81, color="gray", ls=":", lw=1.0)
    ax.text(t[0], 9.9, "hover", color="gray", fontsize=8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("a_cmd [m/s²]")
    ax.set_title("Collective thrust", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_cil(ax, run) -> None:
    d, t = run["d"], run["t"]
    drawn = False
    if "proj_norm" in d:
        ax.plot(t, d["proj_norm"], "-", color="tab:purple", lw=1.5,
                label="‖u_safe − u_raw‖  (CIL intervention)")
        drawn = True
    if "slack" in d and np.any(d["slack"] > 0):
        ax2 = ax.twinx()
        ax2.plot(t, d["slack"], "-", color="tab:brown", lw=1.2, alpha=0.8)
        ax2.set_ylabel("QP slack", color="tab:brown")
        ax2.tick_params(axis="y", labelcolor="tab:brown")
    if not drawn:
        ax.text(0.5, 0.5, "no CIL columns in log", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.set_xlabel("t [s]")
    ax.set_ylabel("projection magnitude")
    ax.set_title("How hard the safety filter intervened", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="one or more bridge log files")
    ap.add_argument("--out", default="trajectory.png")
    ap.add_argument("--z-max", type=float, default=Z_MAX_DEFAULT,
                    help=f"safety ceiling from cbf.z_max (default {Z_MAX_DEFAULT})")
    args = ap.parse_args()

    runs = []
    for path in args.csv:
        d = load(path)
        missing = [c for c in ("t", "px", "pz", "ref_px", "ref_pz") if c not in d]
        if missing:
            raise SystemExit(f"{path} is missing {missing} — older bridge build?")
        runs.append(summarise(Path(path).stem, d, args.z_max))

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)

    plot_xz(fig.add_subplot(gs[:, 0]), runs, args.z_max)
    plot_z_time(fig.add_subplot(gs[0, 1]), runs, args.z_max)
    plot_error(fig.add_subplot(gs[1, 1]), runs)
    plot_rates(fig.add_subplot(gs[0, 2]), runs[0])
    plot_thrust(fig.add_subplot(gs[1, 2]), runs[0])

    fig.suptitle("PS2-RL powerloop in PX4/Gazebo SITL", fontsize=14, y=0.98)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    fig2, ax = plt.subplots(figsize=(9, 4))
    plot_cil(ax, runs[0])
    cil_out = str(Path(args.out).with_name(Path(args.out).stem + "_cil.png"))
    fig2.savefig(cil_out, dpi=150, bbox_inches="tight")
    print(f"wrote {cil_out}")

    print()
    for r in runs:
        flag = "CEILING VIOLATED" if r["violated"] else "within ceiling"
        print(f"{r['name']}:")
        print(f"  control rate     {r['hz']:.1f} Hz over {r['n']} steps "
              f"(nominal {1 / NOMINAL_DT:.0f} Hz)")
        print(f"  position error   RMS {r['rms_err']:.3f} m, max {r['max_err']:.3f} m")
        print(f"  peak altitude    {r['max_z']:.3f} m  ({flag}; "
              f"reference peaks at {r['ref_max_z']:.3f} m)")
        if r["hz"] < 45.0:
            print(f"  ! control rate is {100 * (1 - r['hz'] / 50):.0f}% below nominal — "
                  f"tracking error here is not purely a plant property")
        print()


if __name__ == "__main__":
    main()

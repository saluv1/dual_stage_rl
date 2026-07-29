#!/usr/bin/env python3
"""Diagnose the commanded-vs-achieved body rates in a PS2-RL bridge flight log.

    python3 analyze_flight.py /tmp/simtime.csv

A large tracking error has two very different causes, and they need different
fixes, so the point of this script is to tell them apart before anyone starts
tuning gains:

* **Frame or sign error** — the achieved rate is a mirrored or permuted version
  of the command. Correlation with the *negated* command beats correlation with
  the command itself, and the error does not shrink when the command is smooth.
  This is a bug in the bridge, not a property of the vehicle.

* **Tracking lag** — the achieved rate is a delayed copy. Correlation peaks at a
  positive lag of a few samples, and the residual is largest where the command
  changes fastest. This is the real plant dynamics that PS2-RL's model omits,
  and it is the thing worth measuring.

The script also reports the best-fit first-order lag, which is the number to
feed back into a corrected model if you decide to retrain.
"""

from __future__ import annotations

import sys

import numpy as np

AXES = ("x", "y", "z")


def load(path: str) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.size == 0:
        raise SystemExit(f"{path} has no data rows")
    return {name: np.asarray(raw[name], dtype=np.float64) for name in raw.dtype.names}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-12 else 0.0


def best_lag(cmd: np.ndarray, meas: np.ndarray, max_lag: int = 15) -> tuple[int, float]:
    """Sample shift that best aligns measurement with command."""
    best = (0, -2.0)
    for lag in range(0, max_lag + 1):
        if lag >= cmd.size:
            break
        c = corr(cmd[: cmd.size - lag], meas[lag:])
        if c > best[1]:
            best = (lag, c)
    return best


def fit_first_order(cmd: np.ndarray, meas: np.ndarray, dt: float) -> float:
    """Time constant tau of meas' = (cmd - meas)/tau, least squares."""
    d_meas = np.diff(meas) / dt
    err = (cmd[:-1] - meas[:-1])
    denom = float(err @ err)
    if denom < 1e-9:
        return float("nan")
    k = float(err @ d_meas) / denom
    return 1.0 / k if abs(k) > 1e-9 else float("nan")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    data = load(sys.argv[1])

    required = [f"w{a}" for a in AXES] + [f"w{a}_meas" for a in AXES]
    missing = [c for c in required if c not in data]
    if missing:
        raise SystemExit(f"log is missing columns {missing} — is it from an older bridge build?")

    t = data["t"]
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.02
    n = t.size
    print(f"{n} samples, median dt = {dt * 1e3:.1f} ms (nominal 20.0 ms)")
    if dt > 0.03:
        print("  ! the control loop did not run at 50 Hz — treat everything below as provisional")
    print()

    verdict = []
    for a in AXES:
        cmd = data[f"w{a}"]
        meas = data[f"w{a}_meas"]
        if not np.any(np.isfinite(meas)):
            print(f"axis {a}: no measured rate in log")
            continue

        good = np.isfinite(cmd) & np.isfinite(meas)
        cmd, meas = cmd[good], meas[good]

        c_pos = corr(cmd, meas)
        c_neg = corr(-cmd, meas)
        lag, c_lag = best_lag(cmd, meas)
        rms = float(np.sqrt(np.mean((cmd - meas) ** 2)))
        tau = fit_first_order(cmd, meas, dt)

        print(f"axis {a}:  RMS error {rms:6.2f} rad/s   "
              f"cmd range [{cmd.min():+6.2f}, {cmd.max():+6.2f}]  "
              f"meas range [{meas.min():+6.2f}, {meas.max():+6.2f}]")
        print(f"          corr(cmd, meas) = {c_pos:+.3f}    corr(-cmd, meas) = {c_neg:+.3f}")
        print(f"          best alignment at lag {lag} samples ({lag * dt * 1e3:.0f} ms), "
              f"corr {c_lag:+.3f}")
        if np.isfinite(tau):
            print(f"          first-order fit: tau = {tau * 1e3:.0f} ms")

        if c_neg > 0.5 and c_neg > c_pos + 0.3:
            verdict.append(f"axis {a}: SIGN FLIPPED — measurement tracks the negated command")
        elif c_pos < 0.3 and c_neg < 0.3:
            verdict.append(f"axis {a}: UNCORRELATED — likely a frame/axis permutation, or the "
                           f"rate setpoint is not reaching the controller")
        elif lag > 0 and c_lag > c_pos + 0.05:
            verdict.append(f"axis {a}: lag of ~{lag * dt * 1e3:.0f} ms — plant dynamics")
        else:
            verdict.append(f"axis {a}: tracks in phase, residual is gain/bandwidth")
        print()

    # cross-axis check catches a permutation (e.g. x command showing up on y)
    print("cross-axis correlation (row = commanded, col = measured):")
    header = "        " + "".join(f"{a + '_meas':>10}" for a in AXES)
    print(header)
    for a in AXES:
        row = f"  {a}_cmd "
        for b in AXES:
            row += f"{corr(data['w' + a], data['w' + b + '_meas']):>10.3f}"
        print(row)
    print("  (a clean setup has the diagonal near +1 and off-diagonals near 0)")
    print()

    if "a_cmd_meas" in data and np.any(np.isfinite(data["a_cmd_meas"])):
        cmd, meas = data["a_cmd"], data["a_cmd_meas"]
        good = np.isfinite(cmd) & np.isfinite(meas)
        cmd, meas = cmd[good], meas[good]
        ratio = float(np.median(meas / np.maximum(cmd, 1e-6)))
        print(f"thrust: RMS error {np.sqrt(np.mean((cmd - meas) ** 2)):.2f} m/s^2, "
              f"median achieved/commanded = {ratio:.2f}")
        if abs(ratio - 1.0) > 0.15:
            print(f"  ! hover_thrust is off by roughly this factor — "
                  f"try hover_thrust / {ratio:.2f}")
        print()

    if "saturated" in data:
        sat = float(np.mean(data["saturated"]))
        print(f"thrust saturated on {sat * 100:.0f}% of steps")

    print("\n=== verdict ===")
    for line in verdict:
        print("  " + line)


if __name__ == "__main__":
    main()

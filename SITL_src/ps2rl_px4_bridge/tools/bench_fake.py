#!/usr/bin/env python3
"""Fly the obstacle bridge against ``fake_px4.py`` over real ROS 2 topics and report what the bridge costs.

Starts the stand-in PX4, then ``ps2rl_cyl_online`` (from this package's source, with the YAML plus the overrides given
here), waits for the flight to end and prints one line: outcome, control steps against the expected number, the
command gaps PX4 saw, the step times, and the CPU of the bridge and controller processes over the controller phase.

    python3 tools/bench_fake.py --ps2rl_path ~/ws_shared/PS2-RL --label poll -- io:=poll
    python3 tools/bench_fake.py --ps2rl_path ~/ws_shared/PS2-RL --label exec -- io:=executor executor:=single
    python3 tools/bench_fake.py ... --offset_jumps 14:+6,18:+3 -- ...       # time-sync offset jumps, as on the Pi

``--node slow_track`` flies the slow-track node (its YAML, PD controller) instead. The defaults fly the Pi test course (one cylinder r 0.27 at (1.3, -0.1), goal (3.3, 0), straight nominal). Pin with
``--bridge_cpu`` / ``--fake_cpu`` (taskset) to mimic a core layout. Needs a sourced ROS 2 with px4_msgs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps2rl_path", required=True)
    ap.add_argument("--label", default="bench")
    ap.add_argument("--out", default="/tmp/bench_fake")
    ap.add_argument("--node", default="cyl_online", choices=("cyl_online", "slow_track"))
    ap.add_argument("--config", default="", help="default: config/<node>_x500.yaml")
    ap.add_argument("--hover_thrust", type=float, default=0.5)
    ap.add_argument("--offset_jumps", default="")
    ap.add_argument("--odom_rate", type=float, default=100.0)
    ap.add_argument("--bridge_cpu", default="")
    ap.add_argument("--fake_cpu", default="")
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--pyspy", action="store_true", help="sample the bridge with py-spy (collapsed stacks in pyspy.txt)")
    ap.add_argument("--quota", type=float, default=0.0,
                    help="emulate a slower computer: the bridge and the controller process each get this fraction of one "
                         "core (CPU cgroup v1 under /sys/fs/cgroup/cpu, needs root; 0.15 ~ a Raspberry Pi 4B core "
                         "against a recent x86 laptop core for this code)")
    ap.add_argument("--domain", default="91")
    ap.add_argument("overrides", nargs="*", help="name:=value parameters for the bridge")
    a = ap.parse_args()

    out = Path(a.out) / a.label
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PKG)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["ROS_DOMAIN_ID"] = a.domain
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    env["PYTHONUNBUFFERED"] = "1"

    fake_cmd = [sys.executable, str(PKG / "tools" / "fake_px4.py"), "--ps2rl_path", a.ps2rl_path, "--hover_thrust",
                str(a.hover_thrust), "--duration", str(a.timeout + 10), "--report", str(out / "fake.json"),
                "--offset_jumps", a.offset_jumps, "--odom_rate", str(a.odom_rate)]
    if a.fake_cpu:
        fake_cmd = ["taskset", "-c", a.fake_cpu] + fake_cmd
    params = {"ps2rl_path": a.ps2rl_path, "thrust_model": "linear", "hover_thrust": str(a.hover_thrust),
              "log_root": str(out / "flights"), "label": a.label}
    if a.node == "cyl_online":
        params.update({"course": "O3", "nominal": "straight", "obstacles": "1.3,-0.1,0.27", "goal": "3.3,0"})
    config = a.config or str(PKG / "config" / f"{a.node}_x500.yaml")
    for kv in a.overrides:
        k, v = kv.split(":=", 1)
        params[k] = v
    bridge_cmd = [sys.executable, "-m", f"ps2rl_px4_bridge.{a.node}_node", "--ros-args", "--params-file", config]
    for k, v in params.items():
        bridge_cmd += ["-p", f"{k}:={v}"]
    if a.pyspy:
        bridge_cmd = ["py-spy", "record", "--format", "raw", "--rate", "250", "-o", str(out / "pyspy.txt"), "--"] + bridge_cmd
    if a.bridge_cpu:
        bridge_cmd = ["taskset", "-c", a.bridge_cpu] + bridge_cmd

    groups = {}
    if a.quota > 0:
        for name in ("bench_bridge", "bench_worker"):
            g = Path("/sys/fs/cgroup/cpu") / name
            g.mkdir(exist_ok=True)
            (g / "cpu.cfs_period_us").write_text("10000")
            (g / "cpu.cfs_quota_us").write_text(str(int(10000 * a.quota)))
            groups[name] = g
        bridge_cmd = ["sh", "-c", f'echo $$ > {groups["bench_bridge"] / "cgroup.procs"}; exec "$@"', "sh"] + bridge_cmd

    fake = subprocess.Popen(fake_cmd, env=env, stdout=open(out / "fake.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(1.0)
    t0 = time.monotonic()
    bridge = subprocess.Popen(bridge_cmd, env=env, stdout=open(out / "bridge.log", "w"), stderr=subprocess.STDOUT)
    if groups:   # the controller process starts inside the bridge's group: give it its own "core"
        moved = False
        while not moved and bridge.poll() is None and time.monotonic() - t0 < 60:
            time.sleep(0.5)
            for pid in os.listdir("/proc"):
                if pid.isdigit():
                    try:
                        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
                    except OSError:
                        continue
                    if b"cyl_online_worker" in cmd:
                        (groups["bench_worker"] / "cgroup.procs").write_text(pid)
                        moved = True
    try:
        bridge.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        bridge.send_signal(signal.SIGINT)
        try:
            bridge.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bridge.kill()
    wall = time.monotonic() - t0
    fake.send_signal(signal.SIGINT)
    try:
        fake.wait(timeout=10)
    except subprocess.TimeoutExpired:
        fake.kill()

    log = (out / "bridge.log").read_text(errors="replace")
    rep = {"label": a.label, "overrides": a.overrides, "quota": a.quota, "wall_s": round(wall, 1)}
    for key, pat in (("bridge_cpu_pct", r"bridge process CPU during the controller phase: (\d+) %"),
                     ("worker_cpu_pct", r"controller process CPU during the controller phase: (\d+) %"),
                     ("loop", r"(bridge loop: [^\n]*)"),
                     ("outcome", r"flight complete \(([^)]*)\)"),
                     ("compute", r"compute: ([^\n]*)")):
        m = re.search(pat, log)
        rep[key] = m.group(1) if m else None
    flights = sorted(glob.glob(str(out / "flights" / "*")))
    if flights:
        s = Path(flights[-1]) / "summary.json"
        if s.exists():
            summ = json.loads(s.read_text())
            st = summ.get("controller", summ)
            rep["steps"] = f"{st.get('steps')}/{st.get('expected_steps')}"
            rep["outcome"] = rep["outcome"] or summ.get("outcome") or st.get("outcome")
    if (out / "fake.json").exists():
        fk = json.loads((out / "fake.json").read_text())
        rep["px4_rate_gap_ms"] = f"{fk['rate_gap_ms_median']:.1f} / p95 {fk['rate_gap_ms_p95']:.1f} / max {fk['rate_gap_ms_max']:.1f}"
        rep["px4_failsafe"] = fk["failsafe"]
        rep["px4_events"] = fk["events"]
    (out / "bench.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

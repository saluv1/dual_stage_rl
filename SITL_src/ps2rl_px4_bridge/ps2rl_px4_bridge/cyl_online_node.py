#!/usr/bin/env python3
"""Obstacle-only demo (online backup filter, virtual cylinders) <-> PX4 offboard bridge (Gazebo SITL, then hardware).

Flight sequence (decided by ``ps2rl.deploy.cyl_online_flight``, not here):

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> RECOVER (hover) -> LAND -> DONE
                                              \\-> RECOVER (hover where it is) -> LAND   (abort)

PX4's position controller takes off and holds the hand-over point (``flight_height`` above the take-off point, yaw 0 =
east). In POLICY the go-to-goal nominal passed through the online backup filter (PS2-RL ``ps2rl/online``, the exported
Phase-I backup) commands body rates and collective thrust at 50 Hz, east past the course's virtual cylinders to its
goal; inside the goal circle and slow, or on any abort (contact with a virtual cylinder, tilt, speed, height, stale
odometry, an exception), the vehicle goes back to PX4 position hold where it is, hovers, and PX4's land mode lands it.

Everything else (frames, clock, quality of service, arming, setpoint stream, fail-safes, the PX4 clock-jump guard, the
CPU record) is the slow-track bridge's (``slow_track_node.SlowTrackBridge``), which this node subclasses: only the
runtime parameters and the task hooks differ.

The filter runs in numpy, or compiled with numba when it is installed for the system python3 (``backend: auto``):
about 0.4 ms per decision against 4-30 ms in numpy on one core. The runtime compiles it (a few seconds, cached
afterwards) before the take-off.

Compute on a small computer (Raspberry Pi 4B: a control step of 8 ms on its own took 23-32 ms inside this node):
  process: true          the controller runs in its own process (``ps2rl/deploy/cyl_online_worker.py``): it no longer
                         shares Python's GIL with the executor's threads; ``worker_cpu`` pins it to a core
  executor: single       one executor thread for the PX4 messages and timers instead of three (the control step has
                         its own thread anyway); the environment variable PS2RL_EXECUTOR (single / multi), if set,
                         takes precedence
  control_period: 0.04   one command every 40 ms (25 Hz), held by PX4 in between; the filter's rollouts hold every
                         command as long. policy_rate follows it (set here, whatever the YAML says)
"""

from __future__ import annotations

from pathlib import Path

import os

import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

from ps2rl_px4_bridge.slow_track_node import SlowTrackBridge

# parameters of the runtime: name -> default (see PS2-RL ps2rl/deploy/cyl_online_runtime.py, CylOnlineRuntimeConfig)
CYL_RUNTIME_PARAMS = {
    "backup": "checkpoints/cyl_online/td3_w128_seed0",
    "course": "S1",                       # O1..O4, S1..S4 (docker_batch/configs/cyl_online_candidates.json) or custom
    "candidates_file": "docker_batch/configs/cyl_online_candidates.json",
    "obstacles": "",                      # "x,y,r;x,y,r" (task frame: x east of the take-off point) - replaces the course's
    "goal": "",                           # "x,y" - replaces the course's
    "nominal": "planner",                 # planner (heading search) / straight (the filter alone goes round)
    "search_margin": 1.0,
    "cruise": 1.0,
    "blends": 2,
    "horizon": 1.0,
    "period": 1,
    "backend": "auto",                    # numpy / numba / auto
    "sensing_range": 6.0,
    "flight_height": 1.5,
    "observer_gain": 2.0,
    "observer_max": 1.0,
    "goal_radius": 0.5,
    "control_period": 0.02,               # [s], a multiple of 0.02; 0.04 = 25 Hz (policy_rate follows)
    "process": False,                     # the controller in its own process (no GIL shared with the executor)
    "worker_cpu": -1,                     # process: pin it to this CPU core (-1: no pinning)
}


class CylOnlineBridge(SlowTrackBridge):
    NODE_NAME = "ps2rl_cyl_online"
    RUNTIME_PARAMS = CYL_RUNTIME_PARAMS
    DEFAULT_LABEL = "cyl_online"

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("executor", "multi")
        env = os.environ.get("PS2RL_EXECUTOR", "").strip().lower()
        self.executor_kind = env or str(self.get_parameter("executor").value).strip().lower()
        if self.executor_kind not in ("single", "multi"):
            raise RuntimeError(f"executor must be single or multi, got '{self.executor_kind}'")

    def _watch_cpu(self) -> None:
        """The slow-track record of this process's CPU over the controller phase, plus the controller process's (process:
        true - that work no longer shows in this process)."""
        try:
            in_policy = self.flight.phase is self._Phase.POLICY
        except AttributeError:
            return
        if in_policy and getattr(self, "_worker_mark", None) is None:
            self._worker_mark = (getattr(self.runtime, "cpu_s", None), self.wall_s())
        elif not in_policy and getattr(self, "_worker_mark", None) is not None and not getattr(self, "_worker_reported", False):
            self._worker_reported = True
            cpu0, wall0 = self._worker_mark
            cpu1, span = getattr(self.runtime, "cpu_s", None), self.wall_s() - wall0
            if cpu0 is not None and cpu1 is not None and span > 1.0:
                text = f"controller process CPU during the controller phase: {100.0 * (cpu1 - cpu0) / span:.0f} % of one core"
                self.get_logger().info(text)
                try:
                    with self._flight_lock:
                        self.flight.add_event(self.now_s(), text)
                except Exception:
                    pass
        super()._watch_cpu()

    def _check_task_params(self) -> None:
        backup = self._path(self.param("backup"))
        if not backup or not (Path(backup) / "backup.json").exists():
            raise RuntimeError(f"Parameter 'backup' must be an exported backup folder with backup.json; got '{backup}'.")
        if self.param("nominal") not in ("planner", "straight"):
            raise RuntimeError(f"Parameter 'nominal' must be planner or straight, got '{self.param('nominal')}'.")

    def _task_imports(self):
        from ps2rl.deploy.cyl_online_flight import CylFlightConfig, CylOnlineFlight, Phase
        return CylFlightConfig, CylOnlineFlight, Phase

    def _make_runtime(self):
        from ps2rl.deploy.cyl_online_runtime import CylOnlineRuntimeConfig, make_runtime
        values = {k: self.param(k) for k in self.RUNTIME_PARAMS}
        for k in ("backup", "candidates_file"):
            values[k] = self._path(values[k])
        runtime = make_runtime(CylOnlineRuntimeConfig(**values),
                               ps2rl_path=str(self._ps2rl_root) if self._ps2rl_root else None, logger=self.get_logger())
        rate = 1.0 / runtime.dt
        if abs(self.policy_rate - rate) > 1e-6:
            self.get_logger().info(f"policy_rate {self.policy_rate:g} Hz -> {rate:g} Hz (control_period {runtime.dt:g} s)")
        self.policy_rate = rate              # the control loop and the logic timer run at the control period
        return runtime


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CylOnlineBridge()
    executor = SingleThreadedExecutor() if node.executor_kind == "single" else MultiThreadedExecutor(num_threads=3)
    node.get_logger().info(f"executor: {type(executor).__name__}")
    executor.add_node(node)
    try:
        while rclpy.ok() and not (node.exit_when_done and node.done_event.is_set()):
            executor.spin_once(timeout_sec=0.1)
        if node.done_event.is_set():
            node.get_logger().info("Flight complete - shutting the bridge down.")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

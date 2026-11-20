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
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

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
}


class CylOnlineBridge(SlowTrackBridge):
    NODE_NAME = "ps2rl_cyl_online"
    RUNTIME_PARAMS = CYL_RUNTIME_PARAMS
    DEFAULT_LABEL = "cyl_online"

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
        from ps2rl.deploy.cyl_online_runtime import CylOnlineRuntime, CylOnlineRuntimeConfig
        values = {k: self.param(k) for k in self.RUNTIME_PARAMS}
        for k in ("backup", "candidates_file"):
            values[k] = self._path(values[k])
        return CylOnlineRuntime(CylOnlineRuntimeConfig(**values),
                                ps2rl_path=str(self._ps2rl_root) if self._ps2rl_root else None, logger=self.get_logger())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CylOnlineBridge()
    executor = MultiThreadedExecutor(num_threads=3)
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

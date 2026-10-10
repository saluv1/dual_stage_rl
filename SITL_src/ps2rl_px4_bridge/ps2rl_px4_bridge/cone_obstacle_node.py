#!/usr/bin/env python3
"""Outside-cone task <-> PX4 offboard bridge for Gazebo SITL (light x500_powerloop airframe).

Flight sequence (decided by ``ps2rl.deploy.cone_obstacle_flight``, not here):

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> RECOVER (hold) -> LAND -> DONE

Up to GOTO_START PX4's position controller flies the vehicle. In POLICY a trained Phase-II policy,
passed through the cone-obstacle safety filter (UE-bCBF by default) with its disturbance observer,
commands body rates and collective thrust every control step for one training episode. The funnel
and the target cylinder are virtual (task frame anchored at the take-off point).

Everything but the task (frames, clock, quality of service, arming, setpoint stream, fail-safes) is
the landing bridge's (``landing_node.LandingBridge``), which this node subclasses.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

from ps2rl_px4_bridge.landing_node import LandingBridge

# parameters of the cone-obstacle runtime: name -> default (see ps2rl/deploy/cone_obstacle_runtime.py)
CONE_RUNTIME_PARAMS = {
    "run": "",                 # Phase-II run directory (the policy)
    "weights": "best",
    "filter": "ue",            # ue / nominal / none
    "backup_from": "",         # chosen_backup.json of the Phase-I comparison (default: the run's own backup)
    "observer": "measured",    # measured / commanded / off
    "qp_solve_dtype": "float64",
    "jax_platform": "cpu",
    "predict_ahead": 0.0,      # delay compensation [s]; 0 = off
}


class ConeObstacleBridge(LandingBridge):
    NODE_NAME = "ps2rl_cone_obstacle"
    RUNTIME_PARAMS = CONE_RUNTIME_PARAMS
    VECTOR_PARAMS = ("start_position", "pad_offset")
    DEFAULT_LABEL = "cone_obstacle"

    def _check_task_params(self) -> None:
        run = self._path(self.param("run"))
        if not run:
            raise RuntimeError("Parameter 'run' is required (a Phase-II run directory of train_phase2_cone_obstacle_ue.py).")
        if not (Path(run) / "configs.json").exists():
            raise RuntimeError(f"Parameter 'run': '{run}/configs.json' does not exist.")

    def _task_imports(self):
        from ps2rl.deploy.cone_obstacle_flight import ConeFlightConfig, ConeObstacleFlight, Phase
        return ConeFlightConfig, ConeObstacleFlight, Phase

    def _make_runtime(self):
        from ps2rl.deploy.cone_obstacle_runtime import ConeObstacleRuntime, ConeObstacleRuntimeConfig
        cfg = ConeObstacleRuntimeConfig(
            run=self._path(self.param("run")), weights=self.param("weights"), filter=self.param("filter"),
            backup_from=self._path(self.param("backup_from")), observer=self.param("observer"),
            qp_solve_dtype=self.param("qp_solve_dtype"), jax_platform=self.param("jax_platform"),
            single_thread=self.param("jax_single_thread"), predict_ahead=float(self.param("predict_ahead")),
        )
        return ConeObstacleRuntime(cfg, ps2rl_path=str(self._ps2rl_root) if self._ps2rl_root else None, logger=self.get_logger())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConeObstacleBridge()
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

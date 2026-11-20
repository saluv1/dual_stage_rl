#!/usr/bin/env python3
"""Slow forward + loop tracking <-> PX4 offboard bridge (Gazebo SITL, then hardware).

Flight sequence (decided by ``ps2rl.deploy.slow_track_flight``, not here):

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> RECOVER (hover) -> LAND -> DONE
                                              \\-> RECOVER (hover where it is) -> LAND   (abort)

PX4's position controller takes off and holds the hand-over point (``handover_height`` above the
take-off point, yaw 0 = east). In POLICY the PD controller or a trained policy commands body rates and
collective thrust at 50 Hz along the slow reference; at its end, or on any abort (tracking error,
tilt, speed, height, stale odometry, an exception), the vehicle goes back to PX4 position hold where
it is, hovers, and PX4's land mode lands it.

Everything but the task (frames, clock, quality of service, arming, setpoint stream, fail-safes) is
the landing bridge's (``landing_node.LandingBridge``), which this node subclasses, with one change:
the stamps of odometry and vehicle status pass through ``px4_stamp_guard.StampGuard`` first. During a
uXRCE-DDS time-sync reset PX4 stamps a few messages in the computer's clock instead of its own; the
landing bridge read the step back as an autopilot restart and aborted (three SITL policy flights,
2026-11-18). The guard keeps the stamps on one continuous clock and records each absorbed jump as a
flight event. A real PX4 restart still ends the flight: arming and offboard mode are lost.

The node also records its own CPU use over the controller phase as a flight event. A bridge that
burns several cores (spinning BLAS threads, see ``ps2rl_px4_bridge/__init__.py``) starves Gazebo and
PX4 in SITL: IMU samples arrive late, EKF2's vertical velocity jumps up and the height sags.

Outgoing stamps (parameter ``outgoing_stamp``, 2026-11-20). PX4 v1.16 / v1.17 convert the stamp of every message they
receive (Tools/msg/templates/ucdr/msg.h.em): 0 becomes PX4's time of receipt, anything else min(stamp - time-sync
offset, now). Commander counts the offboard signal as lost when now >= offboard_control_mode.timestamp + COM_OF_LOSS_T.
The landing bridge stamped with the odometry clock, which here the stamp guard keeps continuous: after a time-sync
jump of +6 s every outgoing stamp was 6 s old in PX4's domain, PX4 declared the offboard signal lost and landed (all
five Pi SITL flights, 2026-11-20).
  receipt   (default) stamp 0: PX4 stamps on receipt, so the offboard check measures only whether messages arrive
  node      this computer's clock (the px4_ros_com examples): right once the time sync has converged, stale while a
            jump re-converges
  px4       the latest raw odometry stamp (the Pi-side fix): stale for one odometry period after a jump

PX4 topic names (2026-11-20): PX4 appends ``_v<N>`` to a topic whose message has MESSAGE_VERSION N > 0 (v1.17:
vehicle_local_position_v1, vehicle_status_v1; v1.16: vehicle_status_v1 only). A ``topic_*`` parameter left at its
default is named from the installed px4_msgs this way; one set explicitly is used as given.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import resource

import rclpy
from rclpy.executors import MultiThreadedExecutor

from ps2rl_px4_bridge import landing_node as _landing
from ps2rl_px4_bridge.landing_node import LandingBridge
from ps2rl_px4_bridge.px4_stamp_guard import StampGuard

OUTGOING_STAMPS = ("receipt", "node", "px4")

# the px4_msgs class carried on each topic of LandingBridge's TOPICS (for the version suffix)
_TOPIC_CLASS = {
    "odometry": "VehicleOdometry",
    "local_position": "VehicleLocalPosition",
    "status": "VehicleStatus",
    "land_detected": "VehicleLandDetected",
    "offboard_control_mode": "OffboardControlMode",
    "trajectory_setpoint": "TrajectorySetpoint",
    "rates_setpoint": "VehicleRatesSetpoint",
    "vehicle_command": "VehicleCommand",
}


def versioned_topic(name: str, msg_class) -> str:
    """PX4's DDS name of a topic: the base name, plus _v<N> when the message has MESSAGE_VERSION N > 0."""
    base = re.sub(r"_v\d+$", "", str(name))
    version = int(getattr(msg_class, "MESSAGE_VERSION", 0) or 0)
    return f"{base}_v{version}" if version > 0 else base

# parameters of the slow-track runtime: name -> default (see ps2rl/deploy/slow_track_runtime.py)
SLOW_RUNTIME_PARAMS = {
    "controller": "pd",          # pd / policy
    "run": "",                   # policy: run directory of scripts/train_slow_track.py
    "weights": "best",
    "reference_path": "",        # pd: reference .npz (empty: the slow-loop reference); policy: the run's
    "observer": "commanded",     # commanded (as in training) / measured / off
    "handover_height": 2.0,      # hand-over point above the take-off point [m]
    "extra_time": 2.0,           # keep tracking the reference's last point this long [s]
}


def _process_cpu_s() -> float:
    """CPU time used so far by this process, all threads [s]."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _blas_threads() -> str:
    try:
        from threadpoolctl import threadpool_info
        pools = [f"{p.get('internal_api')} {p.get('num_threads')} thread(s)" for p in threadpool_info()]
        if pools:
            return ", ".join(pools)
    except Exception:
        pass
    return f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}"


class SlowTrackBridge(LandingBridge):
    NODE_NAME = "ps2rl_slow_track"
    RUNTIME_PARAMS = SLOW_RUNTIME_PARAMS
    VECTOR_PARAMS = ("start_position", "pad_offset")
    DEFAULT_LABEL = "slow_track"

    def __init__(self, *args, **kwargs) -> None:
        # set before the subscriptions exist (LandingBridge.__init__ creates them)
        self._odom_guard = StampGuard()
        self._status_guard = StampGuard()
        self._cpu_mark: tuple[float, float] | None = None
        self._cpu_reported = False
        self._stamp_mode = "receipt"     # until the parameter is read; nothing is published before that
        self._raw_px4_us = 0
        super().__init__(*args, **kwargs)
        self.declare_parameter("outgoing_stamp", "receipt")
        self._stamp_mode = str(self.get_parameter("outgoing_stamp").value).strip().lower()
        if self._stamp_mode not in OUTGOING_STAMPS:
            raise RuntimeError(f"outgoing_stamp must be one of {', '.join(OUTGOING_STAMPS)}, got '{self._stamp_mode}'")
        self.get_logger().info(f"BLAS threading: {_blas_threads()}")
        self.get_logger().info("PX4 topics: " + ", ".join(f"{k} {v}" for k, v in self._topic.items() if v)
                               + f"; outgoing stamps: {self._stamp_mode}")

    def param(self, name: str):
        """A topic left at LandingBridge's default is named from the installed px4_msgs (version suffix)."""
        value = super().param(name)
        if name.startswith("topic_") and value and value == self._defaults.get(name):
            cls = getattr(_landing, _TOPIC_CLASS.get(name[len("topic_"):], ""), None)
            if cls is not None:
                value = versioned_topic(value, cls)
        return value

    def now_us(self) -> int:
        """Stamp of the messages sent to PX4 (see the module docstring, outgoing stamps)."""
        if self._stamp_mode == "receipt":
            return 0
        if self._stamp_mode == "node":
            return int(self.get_clock().now().nanoseconds // 1000)
        return self._raw_px4_us if self._raw_px4_us > 0 else super().now_us()

    def _logic_tick(self) -> None:
        self._watch_cpu()
        super()._logic_tick()

    def _watch_cpu(self) -> None:
        """Record this process's CPU use over the controller phase as a flight event."""
        try:
            in_policy = self.flight.phase is self._Phase.POLICY
        except AttributeError:
            return
        if in_policy and self._cpu_mark is None:
            self._cpu_mark = (_process_cpu_s(), self.wall_s())
        elif not in_policy and self._cpu_mark is not None and not self._cpu_reported:
            self._cpu_reported = True
            cpu0, wall0 = self._cpu_mark
            span = self.wall_s() - wall0
            if span > 1.0:
                pct = 100.0 * (_process_cpu_s() - cpu0) / span
                text = f"bridge process CPU during the controller phase: {pct:.0f} % of one core ({_blas_threads()})"
                self.get_logger().info(text)
                try:
                    with self._flight_lock:
                        self.flight.add_event(self.now_s(), text)
                except Exception:
                    pass

    def _guard(self, guard: StampGuard, msg, topic: str) -> None:
        raw = int(getattr(msg, "timestamp", 0))
        mapped, jump = guard(raw, self.wall_s())
        if mapped != raw:
            msg.timestamp = int(mapped)
        if jump is not None:
            text = (f"PX4 {topic} stamps jumped by {jump.raw_step_s:+.3f} s (uXRCE-DDS time-sync reset?) - absorbed, "
                    f"the flight clock continues ({len(guard.jumps)} so far)")
            self.get_logger().warn(text)
            try:
                with self._flight_lock:
                    self.flight.add_event(self.now_s(), text)
            except Exception:
                pass

    def _on_odom(self, msg) -> None:
        self._raw_px4_us = int(getattr(msg, "timestamp", 0))   # PX4's own stamp, before the guard maps it
        self._guard(self._odom_guard, msg, "odometry")
        super()._on_odom(msg)

    def _on_status(self, msg) -> None:
        self._guard(self._status_guard, msg, "vehicle status")
        super()._on_status(msg)

    def _check_task_params(self) -> None:
        if self.param("controller") not in ("pd", "policy"):
            raise RuntimeError(f"Parameter 'controller' must be pd or policy, got '{self.param('controller')}'.")
        if self.param("controller") == "policy":
            run = self._path(self.param("run"))
            if not run or not (Path(run) / "configs.json").exists():
                raise RuntimeError(f"controller policy needs 'run' (a train_slow_track.py run directory with configs.json); got '{run}'.")

    def _task_imports(self):
        from ps2rl.deploy.slow_track_flight import Phase, SlowFlightConfig, SlowTrackFlight
        return SlowFlightConfig, SlowTrackFlight, Phase

    def _make_runtime(self):
        from ps2rl.deploy.slow_track_runtime import SlowTrackRuntime, SlowTrackRuntimeConfig
        cfg = SlowTrackRuntimeConfig(
            controller=self.param("controller"), run=self._path(self.param("run")), weights=self.param("weights"),
            reference_path=self._path(self.param("reference_path")), observer=self.param("observer"),
            handover_height=float(self.param("handover_height")), extra_time=float(self.param("extra_time")),
        )
        return SlowTrackRuntime(cfg, ps2rl_path=str(self._ps2rl_root) if self._ps2rl_root else None, logger=self.get_logger())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlowTrackBridge()
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

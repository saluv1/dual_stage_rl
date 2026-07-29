#!/usr/bin/env python3
"""PS2-RL <-> PX4 offboard bridge for Gazebo SITL.

Flight sequence
---------------
``BOOT`` -> ``ARM`` -> ``TAKEOFF`` -> ``LINEUP`` -> ``DASH`` -> ``POLICY``
-> ``RECOVER`` -> ``LAND``

Everything up to ``POLICY`` uses PX4's own position controller via
``TrajectorySetpoint``.  Only the ``POLICY`` phase hands control to PS2-RL,
which publishes ``VehicleRatesSetpoint`` (body rates + collective thrust) —
the closest PX4 interface to the environment's action space
``u = [a_cmd, wx, wy, wz]``.

The handover matters: the powerloop reference starts at 4.5 m/s, so the
vehicle has to already be flying that trajectory's entry condition when the
policy clock starts at t = 0.  ``LINEUP`` positions the vehicle upstream of the
entry point and ``DASH`` accelerates it through; the policy engages the moment
the vehicle crosses the entry plane at the right speed.

Two timers run concurrently:

* ``policy_timer`` at ``policy_rate`` (default 50 Hz, matching ``env_cfg.dt``)
  computes the action.
* ``stream_timer`` at ``setpoint_rate`` (default 100 Hz) republishes the latest
  setpoint plus ``OffboardControlMode``.  PX4 drops out of offboard if the
  stream stops for ~0.5 s, and a rate setpoint held at only 50 Hz is coarser
  than the inner loop wants, so the two are decoupled with a zero-order hold.
"""

from __future__ import annotations

import csv
from enum import Enum, auto
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleOdometry,
    VehicleRatesSetpoint,
    VehicleStatus,
)

try:
    from px4_msgs.msg import FailsafeFlags

    _HAS_FAILSAFE_FLAGS = True
except ImportError:  # older px4_msgs
    FailsafeFlags = None
    _HAS_FAILSAFE_FLAGS = False

from ps2rl_px4_bridge import frame_transforms as ft
from ps2rl_px4_bridge.policy_runner import PS2RLPolicy, PolicyOutput
from ps2rl_px4_bridge.thrust_model import ThrustModel

NAN = float("nan")

VEHICLE_CMD_NAV_LAND = 21
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400


class Phase(Enum):
    BOOT = auto()
    ARM = auto()
    TAKEOFF = auto()
    LINEUP = auto()
    DASH = auto()
    POLICY = auto()
    RECOVER = auto()
    LAND = auto()
    ABORT = auto()


class PS2RLBridge(Node):
    def __init__(self) -> None:
        super().__init__("ps2rl_px4_bridge")

        # ----------------------------- parameters ----------------------- #
        p = self.declare_parameter
        p("ps2rl_path", "")
        p("run_dir", "")
        p("checkpoint", "best")
        p("learned_backup_policy_path", "")
        p("reference_path", "")
        p("jax_platform", "cpu")
        p("jax_single_thread", True)

        p("policy_rate", 50.0)
        p("setpoint_rate", 50.0)
        p("policy_timer_oversample", 1.0)

        p("thrust_model", "linear")
        p("hover_thrust", 0.60)
        p("thrust_k0", 0.0)
        p("thrust_k1", 0.0)
        p("thrust_k2", 0.0)
        p("thrust_min", 0.02)
        p("thrust_max", 1.0)

        p("origin_offset_enu", [0.0, 0.0, 0.0])
        p("takeoff_altitude", 2.5)
        p("lineup_distance", 10.0)
        p("lineup_position_tol", 0.35)
        p("lineup_settle_time", 3.0)
        p("dash_speed_tol", 0.15)
        p("dash_tilt_tol_deg", 4.0)
        p("dash_pos_tol", 0.10)
        p("dash_accel", 3.0)
        p("align_origin_at_handover", True)
        p("dash_timeout", 15.0)
        p("recover_altitude", 3.0)
        p("recover_time", 4.0)

        p("max_prearm_tilt_deg", 30.0)
        p("z_abort_margin", 1.0)
        p("arena_radius", 40.0)
        p("odom_timeout", 0.30)
        p("auto_start", True)
        p("dry_run_policy", False)
        p("log_csv", "")

        self.policy_rate = float(self.get_parameter("policy_rate").value)
        self.setpoint_rate = float(self.get_parameter("setpoint_rate").value)
        self.policy_oversample = float(self.get_parameter("policy_timer_oversample").value)
        self.origin_offset = np.asarray(
            self.get_parameter("origin_offset_enu").value, dtype=np.float64
        )
        self.takeoff_altitude = float(self.get_parameter("takeoff_altitude").value)
        self.lineup_distance = float(self.get_parameter("lineup_distance").value)
        self.lineup_position_tol = float(self.get_parameter("lineup_position_tol").value)
        self.lineup_settle_time = float(self.get_parameter("lineup_settle_time").value)
        self.dash_speed_tol = float(self.get_parameter("dash_speed_tol").value)
        self.dash_tilt_tol_deg = float(self.get_parameter("dash_tilt_tol_deg").value)
        self.dash_pos_tol = float(self.get_parameter("dash_pos_tol").value)
        self.dash_accel = float(self.get_parameter("dash_accel").value)
        self.align_origin = bool(self.get_parameter("align_origin_at_handover").value)
        self.dash_timeout = float(self.get_parameter("dash_timeout").value)
        self.recover_altitude = float(self.get_parameter("recover_altitude").value)
        self.recover_time = float(self.get_parameter("recover_time").value)
        self.max_prearm_tilt_deg = float(self.get_parameter("max_prearm_tilt_deg").value)
        self.z_abort_margin = float(self.get_parameter("z_abort_margin").value)
        self.arena_radius = float(self.get_parameter("arena_radius").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.dry_run_policy = bool(self.get_parameter("dry_run_policy").value)

        run_dir = str(self.get_parameter("run_dir").value).strip()
        if not run_dir:
            raise RuntimeError("Parameter 'run_dir' is required (a Phase-2 output directory).")

        # ----------------------------- policy --------------------------- #
        ft.test_frame_transforms()
        self.get_logger().info("Frame transforms self-check passed.")

        self.policy = PS2RLPolicy(
            run_dir=run_dir,
            checkpoint=str(self.get_parameter("checkpoint").value),
            ps2rl_path=str(self.get_parameter("ps2rl_path").value) or None,
            learned_backup_policy_path=str(
                self.get_parameter("learned_backup_policy_path").value
            )
            or None,
            reference_path=str(self.get_parameter("reference_path").value) or None,
            jax_platform=str(self.get_parameter("jax_platform").value),
            single_thread=bool(self.get_parameter("jax_single_thread").value),
            logger=self.get_logger(),
        )

        self.thrust = ThrustModel(
            model=str(self.get_parameter("thrust_model").value),
            hover_thrust=float(self.get_parameter("hover_thrust").value),
            k0=float(self.get_parameter("thrust_k0").value),
            k1=float(self.get_parameter("thrust_k1").value),
            k2=float(self.get_parameter("thrust_k2").value),
            thrust_min=float(self.get_parameter("thrust_min").value),
            thrust_max=float(self.get_parameter("thrust_max").value),
        )
        self.get_logger().info(f"Thrust model: {self.thrust.describe()}")
        if self.dry_run_policy:
            self.get_logger().warn(
                "dry_run_policy is ON — publishing a fixed hover action, NOT the policy. "
                "Use this only to measure the loop's overhead."
            )

        # The powerloop reference peaks around 23.3 m/s^2 (~2.4 g). If the
        # airframe cannot deliver that, thrust saturates and the CIL's
        # assumption that the commanded action was applied stops holding.
        peak_accel_required = 23.4
        if self.thrust.max_accel < peak_accel_required:
            self.get_logger().warn(
                f"Airframe delivers only {self.thrust.max_accel / 9.81:.2f} g at full thrust, "
                f"but the reference peaks near {peak_accel_required / 9.81:.2f} g. Expect thrust "
                f"saturation and degraded tracking — raise the motor constants in the model SDF."
            )

        # entry / exit conditions of the reference, in the policy's ENU frame
        self.entry_pos = self.policy.entry_state[0:3].copy()
        entry_vel = self.policy.entry_state[3:6].copy()
        speed = float(np.linalg.norm(entry_vel))
        if speed < 1e-6:
            self.entry_dir = np.array([1.0, 0.0, 0.0])
            self.entry_speed = 0.0
        else:
            self.entry_dir = entry_vel / speed
            self.entry_speed = speed
        self.entry_vel = entry_vel
        self.approach_pos = self.entry_pos - self.entry_dir * self.lineup_distance

        self.get_logger().info(
            f"Reference entry: p={np.round(self.entry_pos, 3)} v={np.round(entry_vel, 3)} "
            f"({self.entry_speed:.2f} m/s), approach from {np.round(self.approach_pos, 2)}"
        )

        # ----------------------------- ROS I/O -------------------------- #
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        cb_sub = ReentrantCallbackGroup()

        self.pub_offboard = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_pub
        )
        self.pub_traj = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_pub
        )
        self.pub_rates = self.create_publisher(
            VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", qos_pub
        )
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_pub)

        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odom, qos_sub, callback_group=cb_sub
        )
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_local_pos, qos_sub, callback_group=cb_sub
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, qos_sub, callback_group=cb_sub
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status", self._on_status, qos_sub, callback_group=cb_sub
        )
        if _HAS_FAILSAFE_FLAGS:
            self.create_subscription(
                FailsafeFlags, "/fmu/out/failsafe_flags", self._on_failsafe_flags, qos_sub, callback_group=cb_sub
            )

        # ----------------------------- state ---------------------------- #
        # The clock fields come first: now_s() reads them, and phase_started
        # below is the first thing that calls it.
        self.px4_time_s = 0.0            # PX4/simulation clock, drives control
        self.px4_time_valid = False
        # Ratio of simulation time to wall time. Gazebo is configured slower
        # than real time on purpose here, so every wall-clock deadline has to
        # be stretched by the same factor or it fires spuriously.
        self.rtf = 1.0
        self._rtf_px4_ref = 0.0
        self._rtf_wall_ref = 0.0
        self._px4_stamp_wall = 0.0      # wall time at which px4_time_s arrived
        self._odom_intervals: list[float] = []

        self._lock = threading.Lock()
        self._phase_lock = threading.Lock()
        # Signalled on every odometry message; the policy loop blocks on this
        # instead of polling. Kept separate from _lock so the two never nest.
        self._odom_cv = threading.Condition()
        self.phase = Phase.BOOT
        self.phase_started = self.now_s()
        self.stream_ticks = 0

        self.state_x: np.ndarray | None = None       # 10D PS2-RL state
        self.odom_stamp = 0.0            # wall clock, for staleness only
        # Measured actuation, for quantifying the plant's deviation from the
        # model PS2-RL was trained against (which assumes body rates and
        # collective thrust are applied instantaneously).
        self.omega_meas_flu = np.zeros(3)
        self.omega_meas_valid = False
        self.accel_ned = np.zeros(3)
        self.accel_valid = False
        self.rate_err: list[np.ndarray] = []
        self.accel_err: list[float] = []
        self._last_policy_px4_t = -1e9
        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self.failsafe_flags = None
        self._last_cmd_t = 0.0
        self._arm_wait_logged = 0.0

        self.policy_t0 = 0.0
        self.rate_sp = np.array([0.0, 0.0, 0.0], dtype=np.float64)   # FRD rad/s
        self.thrust_sp = 0.0
        self.traj_sp = self._make_traj_setpoint(position_enu=self.approach_pos)
        self.lineup_in_tol_since: float | None = None

        self.entry_dev_pos = float("nan")
        self.entry_dev_speed = float("nan")
        self.entry_dev_tilt = float("nan")
        self.max_z_seen = -1e9
        self.violations = 0
        self.saturations = 0
        self.max_latency = 0.0
        self.slack_max = 0.0
        self.latencies: list[float] = []
        self.overruns = 0
        self._policy_busy = threading.Lock()

        self._csv_file = None
        self._csv = None
        log_csv = str(self.get_parameter("log_csv").value).strip()
        if log_csv:
            self._csv_file = open(log_csv, "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(
                [
                    "t", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz",
                    "ref_px", "ref_py", "ref_pz",
                    "a_cmd", "wx", "wy", "wz",
                    "a_raw", "wx_raw", "wy_raw", "wz_raw",
                    "wx_meas", "wy_meas", "wz_meas", "a_cmd_meas",
                    "thrust_norm", "saturated", "slack", "proj_norm", "latency_ms",
                ]
            )

        cb_stream = ReentrantCallbackGroup()
        cb_policy = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / self.setpoint_rate, self._stream_tick, callback_group=cb_stream)
        self.create_timer(
            1.0 / (self.policy_rate * self.policy_oversample),
            self._policy_tick,
            callback_group=cb_policy,
        )
        self.create_timer(1.0, self._supervise, callback_group=cb_stream)

        self._stop_policy = threading.Event()
        self._policy_thread = threading.Thread(
            target=self._policy_loop, name="ps2rl_policy", daemon=True
        )
        self._policy_thread.start()

        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.get_logger().info("Bridge up. Streaming setpoints before requesting offboard.")

    # ------------------------------------------------------------------ #
    # utilities                                                           #
    # ------------------------------------------------------------------ #

    def wall_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def now_s(self) -> float:
        """Simulation time when available, otherwise wall clock.

        Gazebo does not promise real time. On a machine that cannot keep up it
        silently runs slow, and PX4's clock follows the simulation, not the
        wall. Driving the reference trajectory off the wall clock therefore
        feeds the vehicle a trajectory that advances faster than the world it
        is flying in — the aggressive part of the powerloop arrives early and
        the policy is asked to track a state it never reaches.
        """
        if getattr(self, "px4_time_valid", False):
            # Raw PX4 timestamp, deliberately not extrapolated.
            #
            # This used to add (wall elapsed) * rtf to bridge the gaps between
            # messages, from a time when odometry appeared to arrive at 25 Hz.
            # It is now measured at 125 Hz — 8 ms granularity against a 20 ms
            # control period — so the raw stamp is sampled finely enough on its
            # own, and multiplying by an estimated real-time factor put a
            # feedback loop in the control clock: one bad estimate and the
            # reference advanced at the wrong rate, which is exactly what a
            # 2.51x "simulation faster than real time" reading means. The
            # estimate is still computed, but only for wall-clock deadlines
            # and reporting, never for the trajectory clock.
            return self.px4_time_s
        return self.wall_s()

    def now_us(self) -> int:
        if getattr(self, "px4_time_valid", False):
            return int(self.px4_time_s * 1e6)
        return int(self.get_clock().now().nanoseconds / 1000)

    def _set_phase(self, phase: Phase) -> None:
        # Called from both timers; without the lock two threads can pass the
        # equality check and log the same transition twice.
        with self._phase_lock:
            if phase is self.phase:
                return
            self.get_logger().info(f"phase: {self.phase.name} -> {phase.name}")
            self.phase = phase
            self.phase_started = self.now_s()

    def _policy_enu_to_ned(self, p_policy_enu: np.ndarray) -> np.ndarray:
        return ft.enu_to_ned(np.asarray(p_policy_enu, dtype=np.float64) + self.origin_offset)

    # ------------------------------------------------------------------ #
    # subscriptions                                                       #
    # ------------------------------------------------------------------ #

    def _on_odom(self, msg: VehicleOdometry) -> None:
        pos_ned = np.asarray(msg.position, dtype=np.float64)
        q = np.asarray(msg.q, dtype=np.float64)
        vel = np.asarray(msg.velocity, dtype=np.float64)

        # velocity may be published in NED or in the body FRD frame
        vel_frame = int(getattr(msg, "velocity_frame", 1))
        body_frames = {
            int(getattr(VehicleOdometry, "VELOCITY_FRAME_FRD", 2)),
            int(getattr(VehicleOdometry, "VELOCITY_FRAME_BODY_FRD", 3)),
        }
        if vel_frame in body_frames:
            vel = ft.rotation_matrix(q) @ vel

        if not (np.all(np.isfinite(pos_ned)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(q))):
            return

        omega_frd = np.asarray(msg.angular_velocity, dtype=np.float64)
        stamp_us = int(getattr(msg, "timestamp", 0))
        x = ft.odometry_to_ps2rl_state(pos_ned, vel, q, origin_offset_enu=self.origin_offset)
        with self._lock:
            if stamp_us > 0:
                new_sim = stamp_us * 1e-6
                if self.px4_time_valid and new_sim > self.px4_time_s:
                    self._odom_intervals.append(new_sim - self.px4_time_s)
                self.px4_time_s = new_sim
                self._px4_stamp_wall = self.wall_s()
                if not self.px4_time_valid:
                    self.px4_time_valid = True
                    self._rtf_px4_ref = self.px4_time_s
                    self._rtf_wall_ref = self.wall_s()
                else:
                    d_sim = self.px4_time_s - self._rtf_px4_ref
                    d_wall = self.wall_s() - self._rtf_wall_ref
                    if d_wall > 1.0:
                        if d_sim > 0.0:
                            self.rtf = float(np.clip(d_sim / d_wall, 0.02, 5.0))
                        self._rtf_px4_ref = self.px4_time_s
                        self._rtf_wall_ref = self.wall_s()
            if np.all(np.isfinite(omega_frd)):
                self.omega_meas_flu = ft.frd_to_flu(omega_frd)
                self.omega_meas_valid = True
            self.state_x = x
            self.odom_stamp = self.wall_s()
        with self._odom_cv:
            self._odom_cv.notify_all()
            self.max_z_seen = max(self.max_z_seen, float(x[2]))

    def _on_local_pos(self, msg: VehicleLocalPosition) -> None:
        a = np.array([msg.ax, msg.ay, msg.az], dtype=np.float64)
        if np.all(np.isfinite(a)):
            self.accel_ned = a
            self.accel_valid = True

    def _measured_a_cmd(self, x: np.ndarray) -> float:
        """Thrust acceleration actually achieved, along the body up axis.

        The EKF reports acceleration in NED; adding gravity back recovers the
        specific force, and projecting that onto the thrust axis gives the same
        quantity the policy commands as ``a_cmd``.
        """
        if not self.accel_valid:
            return float("nan")
        a_enu = ft.ned_to_enu(self.accel_ned) + np.array([0.0, 0.0, 9.81])
        up = ft.rotation_matrix(x[6:10]) @ np.array([0.0, 0.0, 1.0])
        return float(np.dot(a_enu, up))

    def _on_status(self, msg: VehicleStatus) -> None:
        armed_value = int(getattr(VehicleStatus, "ARMING_STATE_ARMED", 2))
        offboard_value = int(getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14))
        self.armed = int(msg.arming_state) == armed_value
        self.offboard = int(msg.nav_state) == offboard_value
        # PX4 >= v1.14 exposes the aggregate preflight verdict. Without it we
        # would just hammer the FMU with arm requests it is going to refuse.
        self.preflight_ok = bool(getattr(msg, "pre_flight_checks_pass", True))

    def _on_failsafe_flags(self, msg) -> None:
        self.failsafe_flags = msg

    def _blocking_flags(self) -> list[str]:
        """Names of the failsafe flags currently preventing arming."""
        msg = self.failsafe_flags
        if msg is None:
            return []
        interesting = (
            "angular_velocity_invalid",
            "attitude_invalid",
            "local_altitude_invalid",
            "local_position_invalid",
            "local_velocity_invalid",
            "global_position_invalid",
            "home_position_invalid",
            "battery_unhealthy",
            "manual_control_signal_lost",
        )
        return [name for name in interesting if bool(getattr(msg, name, False))]

    # ------------------------------------------------------------------ #
    # command helpers                                                     #
    # ------------------------------------------------------------------ #

    def _send_command(self, command: int, **params) -> None:
        msg = VehicleCommand()
        msg.timestamp = self.now_us()
        msg.command = int(command)
        for i in range(1, 8):
            setattr(msg, f"param{i}", float(params.get(f"param{i}", 0.0)))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.pub_cmd.publish(msg)

    def arm(self) -> None:
        self._send_command(VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def disarm(self) -> None:
        self._send_command(VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)

    def request_offboard(self) -> None:
        self._send_command(VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def land(self) -> None:
        self._send_command(VEHICLE_CMD_NAV_LAND)

    # ------------------------------------------------------------------ #
    # setpoint construction                                               #
    # ------------------------------------------------------------------ #

    def _make_axis_setpoint(
        self,
        pos_enu: np.ndarray,
        vel_enu: np.ndarray,
        yaw_enu: float = 0.0,
    ) -> TrajectorySetpoint:
        """Per-axis setpoint: NaN leaves that ENU axis to the other channel.

        The dash used to command velocity on both horizontal axes, which holds
        the cross-track velocity at zero but never corrects cross-track
        *position* — there is no reference to return to, so any drift picked up
        during the run-in is frozen in. It showed up as 0.119 m of y error at
        handover, most of the total, against a training distribution of
        +-0.1 m. Only the travel axis needs velocity control; the other two
        should be held by position.

        The ENU->NED map is a permutation (x<->y, z negated), so NaNs land on
        the right components automatically.
        """
        msg = TrajectorySetpoint()
        msg.timestamp = self.now_us()

        p = np.asarray(pos_enu, dtype=np.float64) + self.origin_offset
        p_ned = ft.enu_to_ned(p)
        v_ned = ft.enu_to_ned(np.asarray(vel_enu, dtype=np.float64))

        msg.position = [float(p_ned[0]), float(p_ned[1]), float(p_ned[2])]
        msg.velocity = [float(v_ned[0]), float(v_ned[1]), float(v_ned[2])]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = ft.yaw_enu_to_px4(yaw_enu)
        msg.yawspeed = NAN
        return msg

    def _make_traj_setpoint(
        self,
        position_enu: np.ndarray | None = None,
        velocity_enu: np.ndarray | None = None,
        altitude_enu: float | None = None,
        yaw_enu: float = 0.0,
    ) -> TrajectorySetpoint:
        msg = TrajectorySetpoint()
        msg.position = [NAN, NAN, NAN]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]

        if position_enu is not None:
            ned = self._policy_enu_to_ned(position_enu)
            msg.position = [float(ned[0]), float(ned[1]), float(ned[2])]

        if velocity_enu is not None:
            v_ned = ft.enu_to_ned(np.asarray(velocity_enu, dtype=np.float64))
            msg.velocity = [float(v_ned[0]), float(v_ned[1]), NAN if altitude_enu is not None else float(v_ned[2])]

        if altitude_enu is not None:
            z_ned = self._policy_enu_to_ned(np.array([0.0, 0.0, altitude_enu]))[2]
            msg.position = [msg.position[0], msg.position[1], float(z_ned)]

        msg.yaw = ft.yaw_enu_to_px4(yaw_enu)
        msg.yawspeed = NAN
        return msg

    # ------------------------------------------------------------------ #
    # 100 Hz stream                                                       #
    # ------------------------------------------------------------------ #

    def _stream_tick(self) -> None:
        now = self.now_s()
        mode = OffboardControlMode()
        mode.timestamp = self.now_us()
        in_policy = self.phase is Phase.POLICY
        mode.position = not in_policy
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = in_policy
        self.pub_offboard.publish(mode)

        if in_policy:
            with self._lock:
                rates = self.rate_sp.copy()
                thrust = float(self.thrust_sp)
            msg = VehicleRatesSetpoint()
            msg.timestamp = self.now_us()
            msg.roll = float(rates[0])
            msg.pitch = float(rates[1])
            msg.yaw = float(rates[2])
            msg.thrust_body = [0.0, 0.0, float(-thrust)]  # FRD: up is negative
            self.pub_rates.publish(msg)
        else:
            with self._lock:
                sp = self.traj_sp
            sp.timestamp = self.now_us()
            self.pub_traj.publish(sp)

        self.stream_ticks += 1

        # PX4 wants a setpoint stream running *before* the mode switch.
        if self.phase is Phase.BOOT and self.auto_start:
            if self.stream_ticks > int(self.setpoint_rate * 1.5) and self.state_x is not None:
                self._set_phase(Phase.ARM)
        del now

    # ------------------------------------------------------------------ #
    # 50 Hz policy / state machine                                        #
    # ------------------------------------------------------------------ #

    def _policy_tick(self) -> None:
        """ROS timer: the cheap phases only.

        The POLICY phase is deliberately absent. A single policy evaluation
        blocks for ~100 ms on this machine, and running that inside an executor
        callback starves the timer that is supposed to schedule the next one:
        measured 36 steps in 13.3 s of wall time where 105 were due, with only
        30% of that wall time actually spent computing. The rest was the
        executor failing to re-fire. The work now lives in its own thread with
        a tight loop, which also lets rclpy keep servicing subscriptions while
        XLA runs with the GIL released.
        """
        if self.phase is Phase.POLICY:
            return
        if not self._policy_busy.acquire(blocking=False):
            self.overruns += 1
            return
        try:
            with self._lock:
                x = None if self.state_x is None else self.state_x.copy()
            if x is None:
                return
            handler = {
                Phase.ARM: self._do_arm,
                Phase.TAKEOFF: self._do_takeoff,
                Phase.LINEUP: self._do_lineup,
                Phase.DASH: self._do_dash,
                Phase.RECOVER: self._do_recover,
                Phase.LAND: self._do_land,
                Phase.ABORT: self._do_recover,
            }.get(self.phase)
            if handler is not None:
                handler(x)
        finally:
            self._policy_busy.release()

    def _policy_loop(self) -> None:
        """Dedicated thread: one policy evaluation per dt of simulation time."""
        while not self._stop_policy.is_set():
            if self.phase is not Phase.POLICY:
                self._stop_policy.wait(0.02)
                continue

            with self._lock:
                x = None if self.state_x is None else self.state_x.copy()
                odom_age = self.wall_s() - self.odom_stamp
            if x is None:
                self._stop_policy.wait(0.002)
                continue

            # odom_timeout is in simulation time; convert to the wall clock the
            # measurement was actually stamped against.
            deadline = self.odom_timeout / max(self.rtf, 0.02)
            if odom_age > deadline:
                self.get_logger().error(
                    f"Odometry stale by {odom_age * 1e3:.0f} ms wall "
                    f"(deadline {deadline * 1e3:.0f} ms at RTF {self.rtf:.2f}) — aborting."
                )
                self._abort()
                continue

            sim_now = self.now_s()
            next_t = self._last_policy_px4_t + self.policy.dt
            if sim_now < next_t:
                # Predicate-based wait. A bare wait() loses any notification
                # that arrives between checking the gate and re-entering the
                # wait, and at 125 Hz odometry that window swallowed roughly
                # half the wake-ups. wait_for re-tests the condition while
                # holding the lock, so a deadline that has already passed
                # returns immediately and a notification cannot slip through.
                with self._odom_cv:
                    self._odom_cv.wait_for(
                        lambda: self.now_s() >= next_t or self._stop_policy.is_set(),
                        timeout=0.05,
                    )
                continue

            # Advance by exactly dt rather than to the current time. Wake-ups
            # land on odometry arrivals (8 ms apart), so snapping the schedule
            # to them pushed each deadline a few milliseconds late and the
            # error compounded: a 20 ms period drifted out to 31 ms.
            # Accumulating exact dt keeps the period locked; the wake-up
            # granularity becomes jitter, not drift.
            if sim_now - next_t > 3.0 * self.policy.dt:
                # Fell far behind (a stall, not jitter) — resynchronise instead
                # of firing a burst of catch-up steps at stale references.
                self._last_policy_px4_t = sim_now
            else:
                self._last_policy_px4_t = next_t

            try:
                self._do_policy(x)
            except Exception as exc:  # never let the thread die silently
                self.get_logger().error(f"Policy loop error: {exc!r} — aborting.")
                self._abort()

    # -------------------------- phase handlers ------------------------- #

    @staticmethod
    def _tilt_deg(x: np.ndarray) -> float:
        """Angle between the body thrust axis and world up, in degrees."""
        up = ft.rotation_matrix(x[6:10]) @ np.array([0.0, 0.0, 1.0])
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def _do_arm(self, x: np.ndarray) -> None:
        # Keep a position setpoint streaming the whole time; PX4 refuses the
        # offboard mode switch without one.
        with self._lock:
            self.traj_sp = self._make_traj_setpoint(
                position_enu=np.array([x[0], x[1], self.takeoff_altitude])
            )

        now = self.now_s()

        # Gazebo does not reset when the ROS node restarts. After a crash the
        # vehicle is still lying wherever it ended up, and arming an inverted
        # airframe just spins the props into the ground while the altitude
        # readout sits at a constant value that looks like a stalled climb.
        tilt = self._tilt_deg(x)
        if tilt > self.max_prearm_tilt_deg:
            if now - self._arm_wait_logged > 5.0:
                self._arm_wait_logged = now
                self.get_logger().error(
                    f"Refusing to arm: vehicle is tilted {tilt:.0f}deg from upright "
                    f"(limit {self.max_prearm_tilt_deg:.0f}deg). It is probably still "
                    f"crashed from a previous run — restart PX4 SITL to respawn it."
                )
            return

        # One command per second. PX4 acts on the first one it likes, and
        # retrying at the 50 Hz loop rate just floods the console.
        if now - self._last_cmd_t >= 1.0:
            self._last_cmd_t = now

            if not self.offboard:
                # The mode switch must happen BEFORE the preflight gate below.
                # pre_flight_checks_pass is evaluated for the *current* mode
                # (Commander.cpp: canArm(_vehicle_status.nav_state)), and PX4
                # boots into Position, which requires manual control. Without an
                # RC link that check can never pass, so waiting for it before
                # switching to Offboard deadlocks: the mode that would clear the
                # check is the mode we refuse to request. Offboard carries no
                # manual-control requirement.
                self.request_offboard()
            elif not self.armed and self.preflight_ok:
                self.arm()

        if self.armed and self.offboard:
            self._set_phase(Phase.TAKEOFF)
            return

        # Periodic diagnostics, throttled so the console stays readable.
        if now - self._arm_wait_logged > 5.0:
            self._arm_wait_logged = now
            waited = now - self.phase_started
            if not self.offboard:
                self.get_logger().warn(
                    f"Still not in offboard mode after {waited:.0f}s. PX4 accepts the "
                    f"switch only while offboard setpoints are streaming — check "
                    f"'ros2 topic hz /fmu/in/offboard_control_mode --qos-reliability "
                    f"best_effort --qos-durability transient_local'."
                )
            elif not self.preflight_ok:
                blockers = self._blocking_flags()
                detail = ", ".join(blockers) if blockers else "no failsafe flag set"
                self.get_logger().warn(
                    f"In offboard mode but preflight still failing after {waited:.0f}s "
                    f"({detail}). Most arming-check failures are reported only over the "
                    f"MAVLink event stream, so connect QGroundControl to see the reason."
                )

    def _do_takeoff(self, x: np.ndarray) -> None:
        target = np.array([self.approach_pos[0], self.approach_pos[1], self.takeoff_altitude])
        with self._lock:
            self.traj_sp = self._make_traj_setpoint(position_enu=target)
        if float(np.linalg.norm(x[:3] - target)) < 1.0:
            self._set_phase(Phase.LINEUP)

    def _do_lineup(self, x: np.ndarray) -> None:
        with self._lock:
            self.traj_sp = self._make_traj_setpoint(position_enu=self.approach_pos)

        err = float(np.linalg.norm(x[:3] - self.approach_pos))
        speed = float(np.linalg.norm(x[3:6]))
        now = self.now_s()
        if err < self.lineup_position_tol and speed < 0.4:
            if self.lineup_in_tol_since is None:
                self.lineup_in_tol_since = now
            elif now - self.lineup_in_tol_since > self.lineup_settle_time:
                self.lineup_in_tol_since = None
                self._set_phase(Phase.DASH)
        else:
            self.lineup_in_tol_since = None

    def _dash_plan(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Position and velocity of the run-in reference at time t.

        Commanding a bare velocity setpoint leaves PX4's velocity loop to
        converge on its own, and its steady-state error showed up directly as
        0.2-0.3 m/s of entry error against a policy trained with no velocity
        perturbation at all. Handing PX4 a *moving position* target instead,
        with the velocity as feedforward, closes the loop on position: the
        integral term drives the error to zero and the vehicle inherits the
        reference's velocity exactly.

        The profile is the simple one that arrives correctly: constant
        acceleration to entry speed, then cruise, timed so the reference
        reaches the entry point at the moment it is travelling at entry speed.
        Returns (position, velocity, total duration).
        """
        v = max(self.entry_speed, 1e-6)
        a = max(self.dash_accel, 0.1)
        t_accel = v / a
        d_accel = 0.5 * a * t_accel * t_accel
        d_cruise = max(self.lineup_distance - d_accel, 0.0)
        t_cruise = d_cruise / v
        total = t_accel + t_cruise

        # Deliberately NOT clamped at the entry point. A vehicle chasing a
        # position target moving at 4.5 m/s settles into a steady following
        # distance behind it — measured at ~1.5 m. Freezing the target on
        # arrival turns that lag into a braking manoeuvre: the vehicle closes
        # the gap by decelerating, and reaches the entry point slow, while the
        # velocity feedforward still asks for 4.5 m/s and the controller fights
        # itself. Letting the reference cruise straight through keeps the lag
        # constant, so the vehicle crosses the entry point at full speed and
        # the handover triggers on the vehicle's own state, not the plan's
        # clock.
        tc = max(t, 0.0)
        if tc < t_accel:
            travelled = 0.5 * a * tc * tc
            speed = a * tc
        else:
            travelled = d_accel + v * (tc - t_accel)
            speed = v

        pos = self.approach_pos + self.entry_dir * travelled
        vel = self.entry_dir * speed
        return pos, vel, total

    def _do_dash(self, x: np.ndarray) -> None:
        t_dash = self.now_s() - self.phase_started
        pos_sp, vel_sp, plan_total = self._dash_plan(t_dash)
        # Require the run-in to have reached cruise and held it briefly, so the
        # position loop has settled into a steady lag before handover.
        t_settle = min(self.entry_speed / max(self.dash_accel, 0.1) + 1.0, plan_total)
        lag = float(np.dot(pos_sp - x[:3], self.entry_dir))
        with self._lock:
            self.traj_sp = self._make_axis_setpoint(pos_sp, vel_sp)

        along = float(np.dot(x[:3] - self.entry_pos, self.entry_dir))
        speed_err = float(np.linalg.norm(x[3:6] - self.entry_vel))
        alt_err = abs(float(x[2] - self.entry_pos[2]))
        tilt = self._tilt_deg(x)

        # Distance from the reference's own starting state, in full 3D.
        # `along` alone only says the vehicle has crossed the entry plane, not
        # how far past it: at 4.5 m/s, waiting an extra 0.2 s for the speed
        # gate puts the handover 0.9 m downrange, and the cross-track axis was
        # not bounded at all. The policy's training distribution perturbs
        # position by only +-0.1 m, and perturbs velocity and attitude not at
        # all (init_v_range and init_tilt_deg_range are both 0), so every gate
        # below is tied to that distribution rather than to a guess.
        pos_err = float(np.linalg.norm(x[:3] - self.entry_pos))
        horiz_gate = self.dash_pos_tol if not self.align_origin else float("inf")
        if (
            t_dash >= t_settle
            and along >= 0.0
            and float(np.linalg.norm((x[:2] - self.entry_pos[:2]))) < horiz_gate
            and speed_err < self.dash_speed_tol
            and alt_err < 0.1
            and tilt < self.dash_tilt_tol_deg
        ):
            if self.align_origin:
                # Translate the policy frame so the vehicle *is* at the
                # reference's starting point, instead of flying it to a fixed
                # spot in the world. The powerloop is invariant to horizontal
                # translation, so where it happens carries no physical meaning
                # and chasing an exact x,y was manufacturing an initial-state
                # error out of nothing.
                #
                # Altitude is deliberately excluded. The safety constraint
                # z <= z_max is expressed in this frame, so shifting z would
                # move the ceiling relative to the ground and change what the
                # experiment measures. The measured altitude error is ~1 mm
                # anyway, so there is nothing to gain and a guarantee to lose.
                with self._lock:
                    px4_enu = x[:3] + self.origin_offset
                    new_offset = self.origin_offset.copy()
                    new_offset[:2] = px4_enu[:2] - self.entry_pos[:2]
                    shift = new_offset[:2] - self.origin_offset[:2]
                    self.origin_offset = new_offset
                    x = x.copy()
                    x[:2] -= shift
                self.get_logger().info(
                    f"Aligned policy frame horizontally by "
                    f"({shift[0]:+.3f}, {shift[1]:+.3f}) m; altitude left physical."
                )
                pos_err = float(np.linalg.norm(x[:3] - self.entry_pos))

            with self._lock:
                self.policy_t0 = self.now_s()
                # Schedule the first step at t = 0, then every dt from there.
                self._last_policy_px4_t = self.policy_t0 - self.policy.dt
                self._policy_wall_t0 = self.wall_s()
            self.entry_dev_pos, self.entry_dev_speed, self.entry_dev_tilt = (
                pos_err, speed_err, tilt
            )
            self.get_logger().info(
                f"Entry captured: pos_err={pos_err:.3f} m (trained +-0.100), "
                f"speed_err={speed_err:.3f} m/s (trained 0.000), "
                f"tilt={tilt:.1f}deg (trained 0.0). "
                f"Handing over to PS2-RL for {self.policy.reference_duration:.2f} s."
            )
            if pos_err > 0.1 or speed_err > 0.05 or tilt > 1.0:
                self.get_logger().warn(
                    "Handover state is outside the policy's training distribution "
                    "(position was perturbed by +-0.1 m during training; velocity and "
                    "attitude were not perturbed at all). Some of the tracking error "
                    "that follows is an initial-condition mismatch, not plant dynamics."
                )
            self._set_phase(Phase.POLICY)
            return

        elapsed = self.now_s() - self.phase_started
        if elapsed < 0.0 or elapsed > 10.0 * self.dash_timeout:
            # Clock jumped (odometry reset, PX4 restart). Restart the window
            # rather than timing out instantly on a bogus elapsed time.
            self.phase_started = self.now_s()
            return
        if elapsed > self.dash_timeout:
            self.get_logger().warn(
                f"Dash timed out without a clean entry — returning to lineup. "
                f"State at timeout: lag behind reference {lag:.2f} m, "
                f"pos {pos_err:.3f}/{self.dash_pos_tol:.2f} m, "
                f"speed {speed_err:.3f}/{self.dash_speed_tol:.2f} m/s, "
                f"tilt {tilt:.1f}/{self.dash_tilt_tol_deg:.1f} deg"
            )
            self._set_phase(Phase.LINEUP)

    def _do_policy(self, x: np.ndarray) -> None:
        if not (self.armed and self.offboard):
            self.get_logger().error("Lost arming or offboard mode during policy phase.")
            self._abort()
            return

        t = self.now_s() - self.policy_t0

        if x[2] > self.policy.z_max + self.z_abort_margin:
            self.get_logger().error(f"Ceiling breach: z={x[2]:.2f} > {self.policy.z_max:.2f}+margin")
            self._abort()
            return
        if float(np.linalg.norm(x[:2])) > self.arena_radius:
            self.get_logger().error("Left the arena — aborting.")
            self._abort()
            return

        if x[2] > self.policy.z_max:
            self.violations += 1

        if self.dry_run_policy:
            # Control experiment: same loop, same publishing, no inference.
            # If this reaches 105/105 and the real policy does not, the cost is
            # the policy. If neither does, the cost is the surrounding
            # machinery and the algorithm is not the thing to optimise.
            ref, _, _, _ = self.policy.reference_at(t)
            hover = np.array([9.81, 0.0, 0.0, 0.0])
            out = PolicyOutput(
                u_safe=hover, u_raw=hover, slack=0.0, projected_norm=0.0, latency_s=0.0
            )
        else:
            out = self.policy.act(x, t)
        if not np.all(np.isfinite(out.u_safe)):
            self.get_logger().error("Policy produced a non-finite action — aborting.")
            self._abort()
            return

        a_cmd = float(out.u_safe[0])
        omega_flu = out.u_safe[1:4]
        thrust_norm, saturated = self.thrust.thrust_from_accel(a_cmd)
        omega_frd = ft.flu_to_frd(omega_flu)

        with self._lock:
            self.rate_sp = omega_frd
            self.thrust_sp = thrust_norm

        if saturated:
            self.saturations += 1
        self.max_latency = max(self.max_latency, out.latency_s)
        self.slack_max = max(self.slack_max, out.slack)
        self.latencies.append(out.latency_s)

        omega_meas = self.omega_meas_flu.copy()
        a_meas = self._measured_a_cmd(x)
        if self.omega_meas_valid:
            self.rate_err.append(omega_flu - omega_meas)
        if np.isfinite(a_meas):
            self.accel_err.append(a_cmd - a_meas)

        if self._csv is not None:
            ref, _, _, _ = self.policy.reference_at(t)
            self._csv.writerow(
                [f"{t:.4f}"]
                + [f"{v:.6f}" for v in x]
                + [f"{v:.6f}" for v in ref[:3]]
                + [f"{v:.6f}" for v in out.u_safe]
                + [f"{v:.6f}" for v in out.u_raw]
                + [f"{v:.6f}" for v in omega_meas]
                + [f"{a_meas:.6f}"]
                + [
                    f"{thrust_norm:.6f}",
                    int(saturated),
                    f"{out.slack:.6e}",
                    f"{out.projected_norm:.6f}",
                    f"{out.latency_s * 1e3:.3f}",
                ]
            )

        if t >= self.policy.reference_duration:
            lat = np.asarray(self.latencies, dtype=np.float64) * 1e3
            steps = int(lat.size)
            expected = int(round(self.policy.reference_duration * self.policy_rate))
            self.get_logger().info(
                f"Reference complete. max_z={self.max_z_seen:.3f} (limit {self.policy.z_max:.2f}), "
                f"violations={self.violations}, thrust_saturations={self.saturations}/{steps}, "
                f"max_slack={self.slack_max:.3e}"
            )
            if steps:
                wall = max(self.wall_s() - getattr(self, "_policy_wall_t0", self.wall_s()), 1e-6)
                rtf = self.policy.reference_duration / wall
                iv = np.asarray(self._odom_intervals[-400:], dtype=np.float64)
                odom_hz = 1.0 / float(np.median(iv)) if iv.size else float("nan")
                sim_elapsed = self.now_s() - self.policy_t0
                self.get_logger().info(
                    f"Clock: {sim_elapsed:.3f}s sim elapsed over {wall:.2f}s wall "
                    f"(ratio {sim_elapsed / max(wall, 1e-6):.2f}, rtf estimate {self.rtf:.2f})"
                )
                self.get_logger().info(
                    f"State feedback: odometry at {odom_hz:.0f} Hz in sim time "
                    f"({steps} control steps over {self.policy.reference_duration:.2f}s)"
                )
                self.get_logger().info(
                    f"Timing: {steps}/{expected} steps ran, {self.overruns} dropped | "
                    f"latency mean={lat.mean():.1f} p95={np.percentile(lat, 95):.1f} "
                    f"max={lat.max():.1f} ms (sim budget {1e3 / self.policy_rate:.0f} ms) | "
                    f"sim ran at {rtf:.2f}x real time over {wall:.2f}s wall"
                )
            # Explicit pass/fail on the measurement conditions. Run-to-run
            # variation here is large — the same build has produced 106/105
            # steps at 125 Hz odometry and 89/105 at 62 Hz within minutes of
            # each other, purely from thermal state and background load — and
            # the tracking numbers move with it. A trial that did not hold the
            # control rate is not a measurement of the plant.
            step_ratio = steps / max(expected, 1)
            reasons = []
            if step_ratio < 0.97:
                reasons.append(f"only {steps}/{expected} control steps")
            if odom_hz == odom_hz and odom_hz < 90.0:
                reasons.append(f"odometry degraded to {odom_hz:.0f} Hz")
            if lat.size and np.percentile(lat, 95) > 0.5 * (1e3 / self.policy_rate):
                reasons.append(f"p95 latency {np.percentile(lat, 95):.0f} ms")
            if reasons:
                self.get_logger().warn(
                    "Run quality: DEGRADED — consider discarding this trial ("
                    + "; ".join(reasons) + ")"
                )
            else:
                self.get_logger().info(
                    f"Run quality: GOOD — {steps}/{expected} steps, "
                    f"odometry {odom_hz:.0f} Hz, p95 latency "
                    f"{np.percentile(lat, 95):.1f} ms"
                )

            self.get_logger().info(
                f"Handover deviation: pos {self.entry_dev_pos:.3f} m, "
                f"speed {self.entry_dev_speed:.3f} m/s, tilt {self.entry_dev_tilt:.1f} deg"
            )

            if self.rate_err:
                re = np.asarray(self.rate_err)
                rms = np.sqrt(np.mean(np.square(re), axis=0))
                self.get_logger().info(
                    f"Rate tracking error (commanded - achieved, RMS): "
                    f"x={rms[0]:.2f} y={rms[1]:.2f} z={rms[2]:.2f} rad/s "
                    f"| peak |err|={np.abs(re).max():.2f} rad/s"
                )
            if self.accel_err:
                ae = np.asarray(self.accel_err)
                self.get_logger().info(
                    f"Thrust tracking error: RMS={np.sqrt(np.mean(ae ** 2)):.2f} "
                    f"peak={np.abs(ae).max():.2f} m/s^2"
                )
            self._set_phase(Phase.RECOVER)

    def _do_recover(self, x: np.ndarray) -> None:
        with self._lock:
            self.traj_sp = self._make_traj_setpoint(
                velocity_enu=np.array([0.0, 0.0, 0.0]), altitude_enu=self.recover_altitude
            )
        if self.now_s() - self.phase_started > self.recover_time:
            hold = np.array([x[0], x[1], self.recover_altitude])
            with self._lock:
                self.traj_sp = self._make_traj_setpoint(position_enu=hold)
            self._set_phase(Phase.LAND)

    def _do_land(self, x: np.ndarray) -> None:
        if self.now_s() - self.phase_started < 0.5:
            self.land()

    def _abort(self) -> None:
        self._set_phase(Phase.ABORT)

    # ------------------------------------------------------------------ #

    def _supervise(self) -> None:
        if self.phase is Phase.POLICY:
            return
        self.get_logger().debug(
            f"{self.phase.name} armed={self.armed} offboard={self.offboard} "
            f"rtf={self.rtf:.2f} z_max_seen={self.max_z_seen:.2f}"
        )

    def destroy_node(self) -> bool:
        stop = getattr(self, "_stop_policy", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_policy_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if self._csv_file is not None:
            self._csv_file.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PS2RLBridge()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # launch delivers SIGINT and rclpy's default handler may already have
        # torn the context down; calling shutdown() again raises RCLError and
        # makes a clean Ctrl-C look like a crash (exit code 1).
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

#!/usr/bin/env python3
"""Filtered landing <-> PX4 offboard bridge for Gazebo SITL.

Flight sequence (decided by ``ps2rl.deploy.landing_flight``, not here):

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> LAND -> DONE
                                              \\-> RECOVER -> LAND

Up to GOTO_START PX4's position controller flies the vehicle (``TrajectorySetpoint``). In POLICY
a nominal controller passed through the landing safety filter commands body rates and collective
thrust (``VehicleRatesSetpoint``) every control step. On touchdown PX4's land mode takes over.

This node is a thin shell: it converts PX4's NED/FRD messages to the z-up frames the flight logic
uses, calls it once per control step, and publishes what it returns. The clock, quality-of-service
and arming conventions are the ones of ``bridge_node.py`` (the powerloop bridge), which is left
untouched.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import sys
import threading

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor

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
    from px4_msgs.msg import VehicleLandDetected

    _HAS_LAND_DETECTED = True
except ImportError:  # older px4_msgs
    VehicleLandDetected = None
    _HAS_LAND_DETECTED = False

from ps2rl_px4_bridge import frame_transforms as ft
from ps2rl_px4_bridge.thrust_model import ThrustModel

NAN = float("nan")

VEHICLE_CMD_NAV_LAND = 21
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400

# parameters of the landing runtime: name -> default (see ps2rl/deploy/landing_runtime.py)
RUNTIME_PARAMS = {
    "backup": "",
    "filter": "alpha=10,alpha_floor=20",
    "use_filter": True,
    "qp_solve_dtype": "float64",
    "nominal": "lqr_target",
    "target": "pad_centre",
    "max_position_error": 0.0,
    "reference_path": "",
    "jax_platform": "cpu",
}
# PX4 topics: parameter "topic_<key>" -> default name (PX4 v1.16 appends a version suffix to some)
TOPICS = {
    "odometry": "/fmu/out/vehicle_odometry",
    "local_position": "/fmu/out/vehicle_local_position",
    "status": "/fmu/out/vehicle_status_v1",
    "land_detected": "/fmu/out/vehicle_land_detected",   # optional: empty to leave it out
    "offboard_control_mode": "/fmu/in/offboard_control_mode",
    "trajectory_setpoint": "/fmu/in/trajectory_setpoint",
    "rates_setpoint": "/fmu/in/vehicle_rates_setpoint",
    "vehicle_command": "/fmu/in/vehicle_command",
}
# FlightConfig fields given as "x,y,z" strings (ROS parameters cannot be empty arrays)
VECTOR_PARAMS = ("start_position", "pad_offset")


def _coerce(value, default, name: str):
    """Bring a parameter to the type of its default.

    Parameters are declared with dynamic typing because a value typed on the command line
    (``max_position_error:=1``, ``use_filter:=false``) reaches the node with whatever type YAML
    infers for the text, and a strictly typed parameter would refuse an integer for a float.
    """
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("true", "1", "yes", "on"):
                return True
            if text in ("false", "0", "no", "off"):
                return False
            raise ValueError("not a boolean")
        if isinstance(default, int):
            number = float(value)
            if not number.is_integer():
                raise ValueError("not an integer")
            return int(number)
        if isinstance(default, float):
            if isinstance(value, bool):
                raise ValueError("not a number")
            return float(value)
        return str(value).strip()
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"parameter '{name}': cannot read {value!r} as {type(default).__name__} ({e})") from e


def _vector(text: str, name: str) -> tuple:
    """'x,y,z' -> tuple; '' or 'auto' -> () (the flight logic's own default)."""
    text = str(text).strip()
    if not text or text.lower() == "auto":
        return ()
    try:
        return tuple(float(t) for t in text.split(","))
    except ValueError as e:
        raise RuntimeError(f"parameter '{name}' must be 'x,y,z', got '{text}'") from e


class LandingBridge(Node):
    # Task-specific pieces, overridden by other tasks' nodes (cone_obstacle_node.py): node name, runtime
    # parameters, "x,y,z" flight parameters, record label, and the _task_imports / _make_runtime hooks.
    NODE_NAME = "ps2rl_landing"
    RUNTIME_PARAMS = RUNTIME_PARAMS
    VECTOR_PARAMS = VECTOR_PARAMS
    DEFAULT_LABEL = "landing"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # ----------------------------- parameters ----------------------- #
        self._defaults: dict = {}
        self._values: dict = {}

        def declare(name: str, default) -> None:
            self._defaults[name] = default
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))

        declare("ps2rl_path", "")
        for name, default in self.RUNTIME_PARAMS.items():
            declare(name, default)
        declare("jax_single_thread", True)

        declare("policy_rate", 50.0)
        declare("setpoint_rate", 50.0)

        declare("thrust_model", "linear")
        declare("hover_thrust", 0.60)
        declare("thrust_k0", 0.0)
        declare("thrust_k1", 0.0)
        declare("thrust_k2", 0.0)
        declare("thrust_min", 0.02)
        declare("thrust_max", 1.0)

        declare("odom_timeout", 0.30)
        for key, topic in TOPICS.items():
            declare(f"topic_{key}", topic)
        declare("log_root", "")
        declare("label", self.DEFAULT_LABEL)
        declare("exit_when_done", True)

        ps2rl_path = self.param("ps2rl_path")
        self._ps2rl_root = Path(ps2rl_path).expanduser().resolve() if ps2rl_path else None
        if self._ps2rl_root is not None and str(self._ps2rl_root) not in sys.path:
            sys.path.insert(0, str(self._ps2rl_root))
        self._check_task_params()

        # Imported here: ps2rl is found through ps2rl_path, and the runtime sets JAX's
        # environment before JAX is first imported.
        FlightConfig, FlightClass, Phase = self._task_imports()

        self._Phase = Phase
        flight_defaults = FlightConfig()
        for f in fields(FlightConfig):
            if f.name in self.VECTOR_PARAMS:
                declare(f.name, ",".join(f"{v:g}" for v in getattr(flight_defaults, f.name)))
            elif f.name != "log_dir":
                declare(f.name, getattr(flight_defaults, f.name))

        self.policy_rate = self.param("policy_rate")
        self.setpoint_rate = self.param("setpoint_rate")
        self.odom_timeout = self.param("odom_timeout")
        self.exit_when_done = self.param("exit_when_done")

        ft.test_frame_transforms()
        self.get_logger().info("Frame transforms self-check passed.")

        self.thrust = ThrustModel(
            model=self.param("thrust_model"),
            hover_thrust=self.param("hover_thrust"),
            k0=self.param("thrust_k0"),
            k1=self.param("thrust_k1"),
            k2=self.param("thrust_k2"),
            thrust_min=self.param("thrust_min"),
            thrust_max=self.param("thrust_max"),
        )
        self.get_logger().info(f"Thrust model: {self.thrust.describe()}")

        self.runtime = self._make_runtime()
        if abs(self.policy_rate * self.runtime.dt - 1.0) > 1e-6:
            raise RuntimeError(
                f"policy_rate {self.policy_rate:g} Hz does not match the backup's control step "
                f"{self.runtime.dt:g} s ({1.0 / self.runtime.dt:g} Hz)."
            )
        self._check_thrust_range()

        log_root = self.param("log_root")
        self.log_dir = ""
        if log_root:
            stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            self.log_dir = str(Path(log_root).expanduser() / f"{stamp}-{self.param('label') or self.DEFAULT_LABEL}")
        flight_kwargs = {}
        for f in fields(FlightConfig):
            if f.name == "log_dir":
                flight_kwargs[f.name] = self.log_dir
            elif f.name in self.VECTOR_PARAMS:
                flight_kwargs[f.name] = _vector(self.param(f.name), f.name)
            else:
                flight_kwargs[f.name] = self.param(f.name)
        self.flight = FlightClass(
            self.runtime, FlightConfig(**flight_kwargs), logger=self.get_logger(),
            thrust_map=self.thrust.thrust_from_accel,
        )

        # ----------------------------- state ---------------------------- #
        self.px4_time_s = 0.0            # PX4 / simulation clock, drives control
        self.px4_time_valid = False
        self.rtf = 1.0                   # simulation time / wall time, for wall-clock deadlines only
        self._rtf_px4_ref = 0.0
        self._rtf_wall_ref = 0.0
        self._clock_offset = 0.0         # keeps now_s() from stepping back after a gap in odometry
        self._px4_restarted = False

        self._lock = threading.Lock()          # odometry and the latest command
        self._flight_lock = threading.Lock()   # one flight-logic step at a time
        self._odom_cv = threading.Condition()

        self.state_x: np.ndarray | None = None   # [p, v, q], east-north-up, body z up
        self.odom_stamp = 0.0
        self.omega_meas_flu = np.full(3, NAN)
        self.accel_ned = np.zeros(3)
        self.accel_valid = False
        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self.landed: bool | None = None
        self.landing = False             # PX4 is in its land mode
        self._status_seen = False
        self._status_stamp = 0
        self._reset_counter: int | None = None
        self._local_pos_note = False
        self._logic_failed = False
        self._failsafe_land_wall = -1e9

        self._cmd_mode = "position"
        self._traj_sp = self._traj_msg(np.zeros(3), 0.0)
        self._have_setpoint = False
        self._rate_sp = np.zeros(3)      # FRD rad/s
        self._thrust_sp = 0.0
        self._last_policy_px4_t = -1e9
        self._policy_scheduled = False
        self.done_event = threading.Event()
        self._done_logged = False

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
        cb_status = MutuallyExclusiveCallbackGroup()   # armed / mode / landed must be applied in the order they arrive

        topic = {key: self.param(f"topic_{key}") for key in TOPICS}
        self._topic = topic
        self.pub_offboard = self.create_publisher(OffboardControlMode, topic["offboard_control_mode"], qos_pub)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, topic["trajectory_setpoint"], qos_pub)
        self.pub_rates = self.create_publisher(VehicleRatesSetpoint, topic["rates_setpoint"], qos_pub)
        self.pub_cmd = self.create_publisher(VehicleCommand, topic["vehicle_command"], qos_pub)

        self.create_subscription(VehicleOdometry, topic["odometry"], self._on_odom, qos_sub, callback_group=cb_sub)
        self.create_subscription(VehicleLocalPosition, topic["local_position"], self._on_local_pos, qos_sub, callback_group=cb_sub)
        self.create_subscription(VehicleStatus, topic["status"], self._on_status, qos_sub, callback_group=cb_status)
        if _HAS_LAND_DETECTED and topic["land_detected"]:
            self.create_subscription(VehicleLandDetected, topic["land_detected"], self._on_land_detected, qos_sub, callback_group=cb_status)

        cb_stream = ReentrantCallbackGroup()
        cb_logic = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / self.setpoint_rate, self._stream_tick, callback_group=cb_stream)
        self.create_timer(1.0 / self.policy_rate, self._logic_tick, callback_group=cb_logic)
        self.create_timer(5.0, self._health_tick, callback_group=cb_stream)

        if self.log_dir:   # the flight logic has created the folder; keep what this node was run with next to its records
            record = {"parameters": {name: self.param(name) for name in self._defaults}, "thrust_model": self.thrust.describe(),
                      "thrust_at_action_limits": [self.thrust.thrust_from_accel(float(a))[0]
                                                  for a in (self.runtime.action_low[0], self.runtime.gravity, self.runtime.action_high[0])]}
            (Path(self.log_dir) / "bridge_params.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        self._stop_policy = threading.Event()
        self._policy_thread = threading.Thread(target=self._policy_loop, name="ps2rl_landing_policy", daemon=True)
        self._policy_thread.start()
        self.get_logger().info("Landing bridge up. Streaming setpoints before requesting offboard.")

    # ------------------------------------------------------------------ #
    # utilities                                                           #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ task hooks (landing)
    def _check_task_params(self) -> None:
        backup = self._path(self.param("backup"))
        if not backup:
            raise RuntimeError("Parameter 'backup' is required (a landing Phase-I run directory).")
        if not Path(backup).exists():
            raise RuntimeError(f"Parameter 'backup': '{backup}' does not exist.")

    def _task_imports(self):
        """(flight config class, flight logic class, phase enum) of this task."""
        from ps2rl.deploy.landing_flight import FlightConfig, LandingFlight, Phase
        return FlightConfig, LandingFlight, Phase

    def _make_runtime(self):
        from ps2rl.deploy.landing_runtime import LandingRuntime, LandingRuntimeConfig
        runtime_values = {k: self.param(k) for k in self.RUNTIME_PARAMS if k != "backup"}
        runtime_values["reference_path"] = self._path(runtime_values["reference_path"])
        runtime_cfg = LandingRuntimeConfig(backup=self._path(self.param("backup")), single_thread=self.param("jax_single_thread"),
                                           **runtime_values)
        return LandingRuntime(runtime_cfg, logger=self.get_logger())

    def param(self, name: str):
        """Value of a declared parameter, in the type of its default."""
        if name not in self._values:
            self._values[name] = _coerce(self.get_parameter(name).value, self._defaults[name], name)
        return self._values[name]

    def _path(self, text: str) -> str:
        """A path parameter; a relative one is taken from ``ps2rl_path``."""
        if not text:
            return ""
        path = Path(text).expanduser()
        if not path.is_absolute() and self._ps2rl_root is not None:
            path = self._ps2rl_root / path
        return str(path)

    def _check_thrust_range(self) -> None:
        """The filter assumes its thrust commands are delivered; warn if the map clips them."""
        lo, hi = float(self.runtime.action_low[0]), float(self.runtime.action_high[0])
        t_lo, sat_lo = self.thrust.thrust_from_accel(lo)
        t_hi, sat_hi = self.thrust.thrust_from_accel(hi)
        self.get_logger().info(
            f"Thrust commands {lo:.2f}..{hi:.2f} m/s^2 map to normalized thrust {t_lo:.3f}..{t_hi:.3f}; "
            f"hover {self.runtime.gravity:.2f} m/s^2 -> {self.thrust.thrust_from_accel(self.runtime.gravity)[0]:.3f}"
        )
        if sat_lo or sat_hi:
            self.get_logger().warn(
                "The thrust map saturates inside the controller's thrust range: the vehicle cannot deliver "
                "what the filter assumes. Check the thrust coefficients for this airframe."
            )

    def wall_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def now_s(self) -> float:
        """The vehicle clock: the stamp of the latest odometry message, not extrapolated, so control
        steps follow simulation time exactly. If odometry stops, the clock keeps moving from wall
        time (scaled by the real-time factor) so that time-outs after an abort still run."""
        if self.px4_time_valid:
            stale = self.wall_s() - self.odom_stamp - self.odom_timeout / max(self.rtf, 0.02)
            return self.px4_time_s + self._clock_offset + max(0.0, stale) * self.rtf
        return self.wall_s()

    def now_us(self) -> int:
        if self.px4_time_valid:
            return int(self.px4_time_s * 1e6)
        return int(self.get_clock().now().nanoseconds / 1000)

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
        x = ft.odometry_to_ps2rl_state(pos_ned, vel, q)
        reset_counter = getattr(msg, "reset_counter", None)
        with self._lock:
            if stamp_us > 0:
                before = self.now_s() if self.px4_time_valid else None
                if before is not None and stamp_us * 1e-6 < self.px4_time_s - 1.0:
                    self._px4_restarted = True       # PX4's clock went back: the autopilot was restarted
                self.px4_time_s = stamp_us * 1e-6
                if before is not None and self.px4_time_s + self._clock_offset < before:
                    # the clock ran on wall time while odometry was missing (a paused simulation): do not step back
                    self._clock_offset = before - self.px4_time_s
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
            self.state_x = x
            self.odom_stamp = self.wall_s()
            reset_changed = reset_counter is not None and self._reset_counter is not None and int(reset_counter) != self._reset_counter
            if reset_counter is not None:
                self._reset_counter = int(reset_counter)
        if reset_changed and self.flight.shift is not None:
            # The pad was placed in the local frame before arming; a reset of the estimate moves that frame.
            with self._flight_lock:
                self.flight.add_event(self.now_s(), "the autopilot reset part of its state estimate (odometry reset_counter changed) "
                                                    "after the ground point was taken")
        with self._odom_cv:
            self._odom_cv.notify_all()

    def _on_local_pos(self, msg: VehicleLocalPosition) -> None:
        a = np.array([msg.ax, msg.ay, msg.az], dtype=np.float64)
        if np.all(np.isfinite(a)):
            self.accel_ned = a
            self.accel_valid = True

    def _measured_a_cmd(self, x: np.ndarray) -> float | None:
        """Thrust acceleration achieved along the body up axis (specific force from the EKF)."""
        if not self.accel_valid:
            return None
        a_enu = ft.ned_to_enu(self.accel_ned) + np.array([0.0, 0.0, self.runtime.gravity])
        up = ft.rotation_matrix(x[6:10]) @ np.array([0.0, 0.0, 1.0])
        return float(np.dot(a_enu, up))

    def _on_status(self, msg: VehicleStatus) -> None:
        armed_value = int(getattr(VehicleStatus, "ARMING_STATE_ARMED", 2))
        offboard_value = int(getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14))
        land_value = int(getattr(VehicleStatus, "NAVIGATION_STATE_AUTO_LAND", 18))
        stamp = int(getattr(msg, "timestamp", 0))
        if 0 < stamp < self._status_stamp and not self._px4_restarted:
            return                                   # older than the one already applied
        self._status_stamp = stamp
        self.armed = int(msg.arming_state) == armed_value
        self.offboard = int(msg.nav_state) == offboard_value
        self.landing = int(msg.nav_state) == land_value
        self.preflight_ok = bool(getattr(msg, "pre_flight_checks_pass", True))
        self._status_seen = True

    def _on_land_detected(self, msg) -> None:
        self.landed = bool(getattr(msg, "landed", False))

    # ------------------------------------------------------------------ #
    # commands to PX4                                                     #
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

    def _execute_requests(self, requests: list) -> None:
        for name in requests:
            if name == "offboard":
                self._send_command(VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            elif name == "arm":
                self._send_command(VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            elif name == "land":
                self._send_command(VEHICLE_CMD_NAV_LAND)
            else:
                self.get_logger().error(f"unknown request '{name}' from the flight logic")

    def _traj_msg(self, position_enu: np.ndarray, yaw_enu: float) -> TrajectorySetpoint:
        msg = TrajectorySetpoint()
        ned = ft.enu_to_ned(np.asarray(position_enu, dtype=np.float64))
        msg.position = [float(ned[0]), float(ned[1]), float(ned[2])]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = ft.yaw_enu_to_px4(float(yaw_enu))
        msg.yawspeed = NAN
        return msg

    def _apply(self, cmd) -> None:
        """Store the flight logic's command for the setpoint stream and send its mode requests."""
        if cmd.mode == "rates":
            thrust_norm, _ = self.thrust.thrust_from_accel(float(cmd.a_cmd))
            omega_frd = ft.flu_to_frd(np.asarray(cmd.rates, dtype=np.float64))
            with self._lock:
                self._rate_sp, self._thrust_sp, self._cmd_mode = omega_frd, float(thrust_norm), "rates"
            self._publish_setpoint()   # at once: waiting for the next stream tick would add up to one period of delay
        else:
            sp = self._traj_msg(cmd.position, cmd.yaw)
            with self._lock:
                self._traj_sp, self._cmd_mode, self._have_setpoint = sp, "position", True
        self._execute_requests(cmd.requests)

    # ------------------------------------------------------------------ #
    # setpoint stream                                                     #
    # ------------------------------------------------------------------ #

    def _stream_tick(self) -> None:
        self._publish_setpoint()

    def _publish_setpoint(self) -> None:
        """Publish the offboard mode and the latest setpoint (stream timer, and each new rate command)."""
        handed_back = self._logic_failed or self.flight.phase in (self._Phase.LAND, self._Phase.DONE)
        if handed_back and self._status_seen and not self.offboard:
            return   # PX4 has left offboard (its land mode flies now): do not write setpoints under it
        with self._lock:
            mode_rates = self._cmd_mode == "rates"
            rates, thrust, sp, have = self._rate_sp.copy(), float(self._thrust_sp), self._traj_sp, self._have_setpoint
        mode = OffboardControlMode()
        mode.timestamp = self.now_us()
        mode.position = not mode_rates
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = mode_rates
        self.pub_offboard.publish(mode)

        if mode_rates:
            msg = VehicleRatesSetpoint()
            msg.timestamp = self.now_us()
            msg.roll = float(rates[0])
            msg.pitch = float(rates[1])
            msg.yaw = float(rates[2])
            msg.thrust_body = [0.0, 0.0, float(-thrust)]  # FRD: up is negative
            self.pub_rates.publish(msg)
        elif have:
            sp.timestamp = self.now_us()
            self.pub_traj.publish(sp)

    # ------------------------------------------------------------------ #
    # flight logic                                                        #
    # ------------------------------------------------------------------ #

    def _step(self) -> None:
        """One step of the flight logic with the latest state."""
        with self._lock:
            x = None if self.state_x is None else self.state_x.copy()
            omega = self.omega_meas_flu.copy()
        if x is None:
            return
        # The command is stored under the same lock as the step that produced it: at the hand-over
        # two threads step in turn, and the stream must carry their commands in that order.
        with self._flight_lock:
            self.flight.update_status(armed=self.armed, offboard=self.offboard, preflight_ok=self.preflight_ok, landed=self.landed,
                                      landing=self.landing if self._status_seen else None)
            cmd = self.flight.step(self.now_s(), x, omega_meas=omega if np.all(np.isfinite(omega)) else None,
                                   a_meas=self._measured_a_cmd(x))
            self._apply(cmd)
        if self.flight.phase is self._Phase.DONE and not self._done_logged:
            self._done_logged = True
            self.done_event.set()

    def _health_tick(self) -> None:
        """Nothing can start without odometry and the vehicle status: say which one is missing."""
        if self.state_x is None:
            self.get_logger().warn(f"no odometry on {self._topic['odometry']} yet: is PX4 running and the DDS agent connected "
                                   "('ros2 topic list' should show /fmu/out/...)?")
        elif not self._status_seen:
            self.get_logger().warn(f"no vehicle status on {self._topic['status']} yet: arming and the offboard switch cannot be "
                                   "confirmed. Check the topic name with 'ros2 topic list' (parameter topic_status).")
        elif not self.accel_valid and not self._local_pos_note:
            self._local_pos_note = True
            self.get_logger().warn(f"no acceleration on {self._topic['local_position']}: the measured thrust will be missing from "
                                   "the records (the flight does not need it).")

    def _logic_tick(self) -> None:
        """ROS timer: every phase except POLICY, which runs in its own thread on the vehicle clock."""
        if self._logic_failed:
            self._failsafe_tick()
            return
        if self._px4_restarted:
            self.get_logger().error("PX4's clock went back: the autopilot was restarted. The ground point and the pad are no "
                                    "longer valid - stopping. Start this node again.")
            self._fail("aborted: the autopilot was restarted")
            return
        if self.flight.phase is self._Phase.POLICY:
            return
        try:
            self._step()
        except Exception as exc:
            # Leaving here would end the setpoint stream with the vehicle in the air. Keep the last
            # position setpoint streaming and hand the vehicle to PX4's land mode.
            self.get_logger().error(f"flight logic error: {exc!r} - asking PX4 to land.")
            self._fail(f"aborted: flight logic error {exc!r}")
            return
        if self.flight.phase is self._Phase.POLICY:   # hand-over: wake the control thread now
            with self._odom_cv:
                self._odom_cv.notify_all()

    def _fail(self, outcome: str) -> None:
        """Stop stepping the flight logic, record why, and hand the vehicle to PX4."""
        self._logic_failed = True
        with self._lock:
            x = None if self.state_x is None else self.state_x.copy()
        try:
            with self._flight_lock:
                self.flight.end(self.now_s(), x, outcome)
        except Exception as exc:   # the records are lost, the landing below is what matters
            self.get_logger().error(f"could not write the flight summary: {exc!r}")
        self._failsafe_tick()

    def _failsafe_tick(self) -> None:
        """The flight logic has stopped: ask PX4 to land until it has, then end."""
        if self._status_seen and self.armed:
            wall = self.wall_s()
            if not self.landing and wall - self._failsafe_land_wall > 1.0:
                self._failsafe_land_wall = wall
                with self._lock:
                    if self._cmd_mode == "rates" and self.state_x is not None:   # never leave a rate command standing
                        self._traj_sp, self._cmd_mode, self._have_setpoint = self._traj_msg(self.state_x[0:3], 0.0), "position", True
                self._execute_requests(["land"])
            return
        self.done_event.set()

    def _policy_loop(self) -> None:
        """Dedicated thread: one filtered control step per dt of simulation time.

        Kept out of the executor because a control step blocks for several milliseconds; inside a
        timer callback that starves the timer that should schedule the next step.
        """
        dt = self.runtime.dt
        while not self._stop_policy.is_set():
            if self.flight.phase is not self._Phase.POLICY or self._logic_failed:
                self._policy_scheduled = False
                with self._odom_cv:   # woken by the hand-over (and by every odometry message)
                    self._odom_cv.wait_for(lambda: (self.flight.phase is self._Phase.POLICY and not self._logic_failed)
                                           or self._stop_policy.is_set(), timeout=0.05)
                continue
            if not self._policy_scheduled:   # first step right after the hand-over
                self._policy_scheduled = True
                self._last_policy_px4_t = self.now_s() - dt

            with self._lock:
                x = None if self.state_x is None else self.state_x.copy()
                odom_age = self.wall_s() - self.odom_stamp
            deadline = self.odom_timeout / max(self.rtf, 0.02)   # odom_timeout is in simulation time
            if x is not None and odom_age > deadline:
                self.get_logger().error(f"Odometry stale by {odom_age * 1e3:.0f} ms wall (deadline {deadline * 1e3:.0f} ms) - aborting.")
                with self._flight_lock:
                    self._apply(self.flight.abort(self.now_s(), x, "stale odometry"))
                continue

            sim_now = self.now_s()
            next_t = self._last_policy_px4_t + dt
            if sim_now < next_t:
                with self._odom_cv:
                    self._odom_cv.wait_for(lambda: self.now_s() >= next_t or self._stop_policy.is_set(), timeout=0.05)
                continue
            # Advance by exactly dt, so that the arrival time of odometry gives jitter and not drift.
            # After a stall start again from now: the controller takes the time explicitly, and
            # steps squeezed together to catch up would not be the control period the filter assumes.
            self._last_policy_px4_t = sim_now if sim_now - next_t > dt else next_t

            try:
                self._step()
            except Exception as exc:  # never let the thread die silently
                self.get_logger().error(f"Control step error: {exc!r} - aborting.")
                with self._lock:
                    x = None if self.state_x is None else self.state_x.copy()
                if x is not None:
                    with self._flight_lock:
                        self._apply(self.flight.abort(self.now_s(), x, f"exception {exc!r}"))

    # ------------------------------------------------------------------ #

    def destroy_node(self) -> bool:
        stop = getattr(self, "_stop_policy", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_policy_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        flight = getattr(self, "flight", None)
        if flight is not None:
            flight.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LandingBridge()
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

#!/usr/bin/env python3
"""Measure the airframe's thrust -> acceleration curve, then fit the model.

Why this exists: PS2-RL commands ``a_cmd`` in m/s^2 but PX4 wants a unitless
[0, 1] collective thrust.  Assuming the two are related linearly through hover
thrust is only correct *at* hover; the powerloop spends most of its time well
above that, and the error there directly becomes tracking error the policy was
never trained to see.

Method: climb to a safe altitude, then apply a sequence of short open-loop
thrust pulses at level attitude (zero body rates).  During each pulse the body
z specific force read by the IMU *is* the achieved ``a_cmd``.  Between pulses
the vehicle returns to position hold so it never drifts far.

    ros2 run ps2rl_px4_bridge thrust_calib --ros-args \
        -p calib_altitude:=15.0 -p output_yaml:=/tmp/thrust_fit.yaml

Run it in an empty world with plenty of vertical clearance.
"""

from __future__ import annotations

from enum import Enum, auto
import math

import numpy as np
import rclpy
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

# /fmu/out/vehicle_acceleration is NOT in PX4's default dds_topics.yaml, so we
# cannot rely on it. VehicleLocalPosition.az (the EKF's NED down-acceleration)
# is always available and, at level attitude, gives the same quantity:
#     a_cmd = -az_ned + g
# If you enable vehicle_acceleration in dds_topics.yaml and rebuild PX4, this
# node prefers it automatically — raw IMU specific force is less filtered and
# less lagged than the EKF derivative.
try:
    from px4_msgs.msg import VehicleAcceleration

    _HAS_VEHICLE_ACCEL = True
except ImportError:
    VehicleAcceleration = None
    _HAS_VEHICLE_ACCEL = False

from ps2rl_px4_bridge import frame_transforms as ft
from ps2rl_px4_bridge.thrust_model import fit_quadratic

NAN = float("nan")
VEHICLE_CMD_NAV_LAND = 21
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400


class Stage(Enum):
    BOOT = auto()
    ARM = auto()
    CLIMB = auto()
    SETTLE = auto()
    PULSE = auto()
    DONE = auto()


class ThrustCalib(Node):
    def __init__(self) -> None:
        super().__init__("ps2rl_thrust_calib")

        self.declare_parameter("thrust_levels", [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        self.declare_parameter("calib_altitude", 15.0)
        self.declare_parameter("pulse_duration", 0.40)
        self.declare_parameter("pulse_settle_skip", 0.12)
        self.declare_parameter("settle_duration", 6.0)
        self.declare_parameter("climb_timeout", 45.0)
        self.declare_parameter("min_calib_altitude", 8.0)
        self.declare_parameter("climb_tolerance", 1.0)
        self.declare_parameter("output_yaml", "")

        self.levels = [float(v) for v in self.get_parameter("thrust_levels").value]
        self.calib_altitude = float(self.get_parameter("calib_altitude").value)
        self.pulse_duration = float(self.get_parameter("pulse_duration").value)
        self.pulse_skip = float(self.get_parameter("pulse_settle_skip").value)
        self.settle_duration = float(self.get_parameter("settle_duration").value)
        self.climb_timeout = float(self.get_parameter("climb_timeout").value)
        self.min_calib_altitude = float(self.get_parameter("min_calib_altitude").value)
        self.climb_tolerance = float(self.get_parameter("climb_tolerance").value)
        self.output_yaml = str(self.get_parameter("output_yaml").value).strip()

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

        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odom, qos_sub)
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_local_pos, qos_sub
        )
        if _HAS_VEHICLE_ACCEL:
            self.create_subscription(
                VehicleAcceleration, "/fmu/out/vehicle_acceleration", self._on_accel, qos_sub
            )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, qos_sub
        )
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self._on_status, qos_sub)

        self.stage = Stage.BOOT
        self.stage_t0 = self.now_s()
        self.ticks = 0
        self.pos_enu = None
        self.hold_xy = np.zeros(2)
        self.gravity = 9.81
        self.accel_cmd = 0.0            # measured thrust acceleration [m/s^2]
        self.accel_src = "none"
        self.accel_stamp = 0.0
        self.imu_accel_stamp = 0.0
        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self._last_cmd_t = 0.0
        self._arm_wait_logged = 0.0

        self.vel_enu = np.zeros(3)
        self.tilt_deg = 0.0
        self.best_altitude = -1e9
        self._climb_logged = 0.0
        self.level_idx = 0
        self.samples: list[float] = []
        self.results: list[tuple[float, float]] = []

        self.create_timer(0.01, self._tick)
        self.get_logger().info(f"Calibrating thrust levels: {self.levels}")

    # ------------------------------------------------------------------ #

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _stage(self, stage: Stage) -> None:
        if stage is not self.stage:
            self.get_logger().info(f"{self.stage.name} -> {stage.name}")
            self.stage = stage
            self.stage_t0 = self.now_s()

    def _on_odom(self, msg: VehicleOdometry) -> None:
        p = np.asarray(msg.position, dtype=np.float64)
        if np.all(np.isfinite(p)):
            self.pos_enu = ft.ned_to_enu(p)
            self.best_altitude = max(self.best_altitude, float(self.pos_enu[2]))
        v = np.asarray(msg.velocity, dtype=np.float64)
        if np.all(np.isfinite(v)):
            self.vel_enu = ft.ned_to_enu(v)
        q = np.asarray(msg.q, dtype=np.float64)
        if np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-6:
            up = ft.rotation_matrix(ft.px4_quat_to_enu_flu(q)) @ np.array([0.0, 0.0, 1.0])
            self.tilt_deg = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def _on_accel(self, msg) -> None:
        # Body FRD specific force; hovering level reads (0, 0, -g), so the
        # thrust acceleration is simply the negated z component.
        self.accel_cmd = -float(msg.xyz[2])
        self.accel_src = "imu"
        self.accel_stamp = self.now_s()
        self.imu_accel_stamp = self.accel_stamp

    def _on_local_pos(self, msg: VehicleLocalPosition) -> None:
        # Prefer the raw IMU source when it is actually arriving.
        if self.now_s() - self.imu_accel_stamp < 0.5:
            return
        az = float(msg.az)
        if not math.isfinite(az):
            return
        # az is the NED *down* acceleration. At level attitude the thrust axis
        # is world-up, so a_cmd = (upward acceleration) + g = -az + g.
        self.accel_cmd = -az + self.gravity
        self.accel_src = "ekf"
        self.accel_stamp = self.now_s()

    def _on_status(self, msg: VehicleStatus) -> None:
        self.armed = int(msg.arming_state) == int(getattr(VehicleStatus, "ARMING_STATE_ARMED", 2))
        self.offboard = int(msg.nav_state) == int(
            getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14)
        )
        self.preflight_ok = bool(getattr(msg, "pre_flight_checks_pass", True))

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

    def _publish_mode(self, body_rate: bool) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self.now_us()
        msg.position = not body_rate
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = body_rate
        self.pub_offboard.publish(msg)

    def _publish_hold(self) -> None:
        msg = TrajectorySetpoint()
        msg.timestamp = self.now_us()
        ned = ft.enu_to_ned(np.array([self.hold_xy[0], self.hold_xy[1], self.calib_altitude]))
        msg.position = [float(ned[0]), float(ned[1]), float(ned[2])]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = 0.0
        msg.yawspeed = NAN
        self.pub_traj.publish(msg)

    def _publish_thrust(self, thrust: float) -> None:
        msg = VehicleRatesSetpoint()
        msg.timestamp = self.now_us()
        msg.roll = 0.0
        msg.pitch = 0.0
        msg.yaw = 0.0
        msg.thrust_body = [0.0, 0.0, float(-thrust)]
        self.pub_rates.publish(msg)

    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        self.ticks += 1
        elapsed = self.now_s() - self.stage_t0

        if self.stage is Stage.BOOT:
            self._publish_mode(False)
            self._publish_hold()
            if self.ticks > 150 and self.pos_enu is not None:
                self.hold_xy = self.pos_enu[:2].copy()
                self._stage(Stage.ARM)
            return

        if self.stage is Stage.ARM:
            self._publish_mode(False)
            self._publish_hold()
            now = self.now_s()
            # Gazebo keeps the crashed pose across node restarts; arming an
            # upside-down airframe looks exactly like a climb that never starts.
            if self.tilt_deg > 30.0:
                if now - self._arm_wait_logged > 5.0:
                    self._arm_wait_logged = now
                    self.get_logger().error(
                        f"Refusing to arm: vehicle is tilted {self.tilt_deg:.0f}deg from "
                        f"upright. Restart PX4 SITL to respawn it."
                    )
                return
            # Switch mode first: pre_flight_checks_pass is evaluated for the
            # current mode, and PX4 boots into Position, which requires an RC
            # link. Gating the mode switch on it would deadlock in SITL.
            if now - self._last_cmd_t >= 1.0:
                self._last_cmd_t = now
                if not self.offboard:
                    self._send_command(VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                elif not self.armed and self.preflight_ok:
                    self._send_command(VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            if not (self.armed and self.offboard) and now - self._arm_wait_logged > 5.0:
                self._arm_wait_logged = now
                state = "not in offboard" if not self.offboard else "preflight failing"
                self.get_logger().warn(
                    f"Waiting to arm after {now - self.stage_t0:.0f}s ({state}). "
                    f"Connect QGroundControl to see arming-check details."
                )
            if self.armed and self.offboard:
                self._stage(Stage.CLIMB)
            return

        if self.stage is Stage.CLIMB:
            self._publish_mode(False)
            self._publish_hold()
            if self.pos_enu is None:
                return

            alt = float(self.pos_enu[2])
            vz = float(self.vel_enu[2])

            # Climbing used to be a silent black box: if the vehicle stalled
            # short of the target there was no altitude, no rate, no timeout,
            # just a stage that never advanced.
            now = self.now_s()
            if now - self._climb_logged > 2.0:
                self._climb_logged = now
                self.get_logger().info(
                    f"  climbing: {alt:5.2f} m / {self.calib_altitude:.1f} m "
                    f"(vz={vz:+.2f} m/s, {elapsed:.0f}s elapsed)"
                )

            if abs(alt - self.calib_altitude) < self.climb_tolerance and abs(vz) < 0.8:
                self._stage(Stage.SETTLE)
                return

            if elapsed > self.climb_timeout:
                if alt >= self.min_calib_altitude:
                    self.get_logger().warn(
                        f"Could not reach {self.calib_altitude:.1f} m in {elapsed:.0f}s "
                        f"(stalled at {alt:.2f} m, peak {self.best_altitude:.2f} m). "
                        f"Calibrating at the current altitude instead."
                    )
                    self.calib_altitude = alt
                    self._stage(Stage.SETTLE)
                else:
                    self.get_logger().error(
                        f"Climb failed: only {alt:.2f} m after {elapsed:.0f}s "
                        f"(peak {self.best_altitude:.2f} m, need >= "
                        f"{self.min_calib_altitude:.1f} m). Aborting and landing. "
                        f"Check MPC_THR_HOVER against the airframe's actual hover thrust."
                    )
                    self._send_command(VEHICLE_CMD_NAV_LAND)
                    self._stage(Stage.DONE)
            return

        if self.stage is Stage.SETTLE:
            self._publish_mode(False)
            self._publish_hold()
            if elapsed > self.settle_duration:
                if self.level_idx >= len(self.levels):
                    self._finish()
                    return
                self.samples = []
                self._stage(Stage.PULSE)
            return

        if self.stage is Stage.PULSE:
            thrust = self.levels[self.level_idx]
            self._publish_mode(True)
            self._publish_thrust(thrust)
            if elapsed > self.pulse_skip and self.now_s() - self.accel_stamp < 0.3:
                self.samples.append(self.accel_cmd)
            if elapsed > self.pulse_duration:
                accel = float(np.median(self.samples)) if self.samples else float("nan")
                self.results.append((thrust, accel))
                self.get_logger().info(
                    f"  T={thrust:.2f} -> a={accel:.2f} m/s^2 ({accel / 9.81:.2f} g), "
                    f"n={len(self.samples)} src={self.accel_src}"
                )
                if not self.samples:
                    self.get_logger().error(
                        "No acceleration samples for this pulse. Check that "
                        "/fmu/out/vehicle_local_position is publishing."
                    )
                self.level_idx += 1
                self._stage(Stage.SETTLE)
            return

        if self.stage is Stage.DONE:
            self._publish_mode(False)
            self._publish_hold()

    def _finish(self) -> None:
        self._stage(Stage.DONE)
        valid = [(t, a) for t, a in self.results if math.isfinite(a)]
        if len(valid) < 3:
            self.get_logger().error("Not enough valid samples to fit.")
            return

        thrusts = [t for t, _ in valid]
        accels = [a for _, a in valid]
        model = fit_quadratic(thrusts, accels)

        residuals = [model.accel_from_thrust(t) - a for t, a in valid]
        rms = float(np.sqrt(np.mean(np.square(residuals))))

        self.get_logger().info("=" * 62)
        self.get_logger().info(f"  thrust_model: quadratic")
        self.get_logger().info(f"  thrust_k2: {model.k2:.6f}")
        self.get_logger().info(f"  thrust_k1: {model.k1:.6f}")
        self.get_logger().info(f"  thrust_k0: {model.k0:.6f}")
        self.get_logger().info(f"  fit RMS residual: {rms:.3f} m/s^2")
        self.get_logger().info(f"  {model.describe()}")
        if model.max_accel < 23.4:
            self.get_logger().warn(
                "  This airframe cannot reach the 2.4 g the powerloop reference peaks at."
            )
        self.get_logger().info("=" * 62)

        if self.output_yaml:
            with open(self.output_yaml, "w", encoding="utf-8") as f:
                f.write("/ps2rl_px4_bridge:\n  ros__parameters:\n")
                f.write('    thrust_model: "quadratic"\n')
                f.write(f"    thrust_k2: {model.k2:.6f}\n")
                f.write(f"    thrust_k1: {model.k1:.6f}\n")
                f.write(f"    thrust_k0: {model.k0:.6f}\n")
            self.get_logger().info(f"Wrote {self.output_yaml}")

        self._send_command(VEHICLE_CMD_NAV_LAND)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ThrustCalib()
    try:
        rclpy.spin(node)
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

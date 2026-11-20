#!/usr/bin/env python3
"""A stand-in PX4 on real ROS 2 topics, for measuring a bridge node without PX4, Gazebo or a DDS agent.

It runs PS2-RL's stand-in vehicle (``ps2rl/deploy/landing_plant_sim.py``: rate / thrust lag, delay, a rough
position controller, modes) in real time and talks to the bridge the way PX4's uXRCE-DDS client does:

  out  /fmu/out/vehicle_odometry (100 Hz), vehicle_local_position[_vN] (50 Hz), vehicle_status[_vN] (5 Hz),
       vehicle_land_detected (2 Hz)
  in   /fmu/in/offboard_control_mode, vehicle_rates_setpoint, trajectory_setpoint, vehicle_command

Topic names follow the MESSAGE_VERSION of the installed px4_msgs, as PX4 names them.

Time stamps as PX4 v1.16 / v1.17 handle them (Tools/msg/templates/ucdr/msg.h.em): an outgoing stamp is PX4's time plus
the time-sync offset; an incoming stamp of 0 becomes PX4's time of receipt, any other one min(stamp - offset, now). The
offboard signal counts as lost when now >= offboard_control_mode.timestamp + COM_OF_LOSS_T (commander's offboardCheck),
and PX4 then lands. ``--offset_jumps 15:+6,22:+3`` shifts the time-sync offset by +6 s at 15 s and +3 s at 22 s after
the start, as the Pi's loaded time sync did (2026-11-20).

The thrust command is turned back into an acceleration with a linear map, so run the bridge with
``thrust_model:=linear hover_thrust:=<the same value>``.

    python3 fake_px4.py --ps2rl_path ~/ws_shared/PS2-RL --hover_thrust 0.5 --duration 90
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLandDetected,
                          VehicleLocalPosition, VehicleOdometry, VehicleRatesSetpoint, VehicleStatus)


def topic(base: str, cls) -> str:
    v = int(getattr(cls, "MESSAGE_VERSION", 0) or 0)
    return f"{base}_v{v}" if v > 0 else base


class FakePX4(Node):
    def __init__(self, args) -> None:
        super().__init__("fake_px4")
        sys.path.insert(0, args.ps2rl_path)
        from ps2rl.deploy.landing_plant_sim import FakeVehicle, PlantConfig
        from ps2rl_px4_bridge import frame_transforms as ft
        self.ft = ft
        self.args = args
        self.g = 9.81
        self.veh = FakeVehicle(PlantConfig(gravity=self.g, thrust_accel_min=0.0, thrust_accel_max=16.6, seed=args.seed))
        self.lock = threading.Lock()
        self.boot_wall = time.monotonic()
        self.px4_t0_us = 5_000_000                      # PX4 has been up 5 s
        self.offset_us = int(time.time() * 1e6) - self.px4_t0_us if args.synced else 0
        self.jumps = sorted((float(a), float(b)) for a, b in (j.split(":") for j in args.offset_jumps.split(",") if j))
        self.ocm_stamp_us = None
        self.ocm_rates = False
        self.events: list[str] = []
        self.n_rates = 0
        self.rate_arrivals: list[float] = []
        self.failsafe = False

        qos_pub = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_sub = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub_odom = self.create_publisher(VehicleOdometry, topic("/fmu/out/vehicle_odometry", VehicleOdometry), qos_pub)
        self.pub_lpos = self.create_publisher(VehicleLocalPosition, topic("/fmu/out/vehicle_local_position", VehicleLocalPosition), qos_pub)
        self.pub_status = self.create_publisher(VehicleStatus, topic("/fmu/out/vehicle_status", VehicleStatus), qos_pub)
        self.pub_land = self.create_publisher(VehicleLandDetected, topic("/fmu/out/vehicle_land_detected", VehicleLandDetected), qos_pub)
        self.create_subscription(OffboardControlMode, topic("/fmu/in/offboard_control_mode", OffboardControlMode), self.on_ocm, qos_sub)
        self.create_subscription(VehicleRatesSetpoint, topic("/fmu/in/vehicle_rates_setpoint", VehicleRatesSetpoint), self.on_rates, qos_sub)
        self.create_subscription(TrajectorySetpoint, topic("/fmu/in/trajectory_setpoint", TrajectorySetpoint), self.on_traj, qos_sub)
        self.create_subscription(VehicleCommand, topic("/fmu/in/vehicle_command", VehicleCommand), self.on_cmd, qos_sub)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ PX4's clocks
    def px4_us(self) -> int:
        return self.px4_t0_us + int((time.monotonic() - self.boot_wall) * 1e6)

    def stamp_in(self, stamp: int) -> int:
        now = self.px4_us()
        return now if stamp == 0 else min(int(stamp) - self.offset_us, now)

    def note(self, text: str) -> None:
        self.events.append(f"{time.monotonic() - self.boot_wall:7.2f} s  {text}")
        self.get_logger().info(text)

    # ------------------------------------------------------------------ incoming
    def on_ocm(self, msg) -> None:
        with self.lock:
            self.ocm_stamp_us = self.stamp_in(msg.timestamp)
            self.ocm_rates = bool(msg.body_rate)

    def on_rates(self, msg) -> None:
        a = self.g * float(-msg.thrust_body[2]) / self.args.hover_thrust
        rates = self.ft.frd_to_flu(np.array([msg.roll, msg.pitch, msg.yaw], dtype=np.float64))
        with self.lock:
            self.veh.apply(SimpleNamespace(requests=[], mode="rates", a_cmd=a, rates=rates))
            self.n_rates += 1
            self.rate_arrivals.append(time.monotonic())

    def on_traj(self, msg) -> None:
        p = self.ft.ned_to_enu(np.asarray(msg.position, dtype=np.float64))
        yaw = self.ft.yaw_enu_to_px4(float(msg.yaw))   # pi/2 - yaw, its own inverse
        with self.lock:
            if np.all(np.isfinite(p)):
                self.veh.apply(SimpleNamespace(requests=[], mode="position", position=p, yaw=yaw))

    def on_cmd(self, msg) -> None:
        name = None
        if msg.command == 176 and abs(msg.param2 - 6.0) < 0.5:
            name = "offboard"
        elif msg.command == 400 and msg.param1 > 0.5:
            name = "arm"
        elif msg.command == 21:
            name = "land"
        if name:
            with self.lock:
                self.veh._pending.append((self.veh.t + self.veh.cfg.mode_delay, name))

    # ------------------------------------------------------------------ the vehicle, in real time
    def loop(self) -> None:
        ft, veh = self.ft, self.veh
        dt = veh.cfg.dt_sim
        t_odom = t_lpos = t_status = t_land = 0.0
        jumps = list(self.jumps)
        while not self.stop.is_set():
            wall = time.monotonic() - self.boot_wall
            with self.lock:
                while veh.t + dt <= wall:
                    veh.advance(dt)
                # PX4's offboard check on the message stamp
                if veh.offboard and self.ocm_stamp_us is not None and not self.failsafe:
                    if self.px4_us() >= self.ocm_stamp_us + int(self.args.of_loss_t * 1e6):
                        self.failsafe = True
                        veh.offboard, veh.nav, veh._land_z = False, "land", float(veh.p[2])
                        self.note(f"offboard signal lost (stamp {(self.px4_us() - self.ocm_stamp_us) * 1e-6:.2f} s old) -> land")
                while jumps and wall >= jumps[0][0]:
                    t_j, d_j = jumps.pop(0)
                    self.offset_us += int(d_j * 1e6)
                    self.note(f"time-sync offset {d_j:+.2f} s")
                px4 = self.px4_us()
                out = px4 + self.offset_us
                x, omega, a = veh.estimate()
                armed, nav, landed = veh.armed, veh.nav, veh.landed
                acc_enu = (a * _body_z(x[6:10]) - np.array([0.0, 0.0, self.g])) if not veh.on_ground else np.zeros(3)
            if wall >= t_odom:
                t_odom += 1.0 / self.args.odom_rate
                m = VehicleOdometry()
                m.timestamp = out
                m.timestamp_sample = out
                m.position = ft.enu_to_ned(x[0:3]).astype(np.float32)
                m.velocity = ft.enu_to_ned(x[3:6]).astype(np.float32)
                m.q = ft.enu_flu_to_px4_quat(x[6:10]).astype(np.float32)
                m.velocity_frame = 1
                m.angular_velocity = ft.flu_to_frd(omega).astype(np.float32)
                m.reset_counter = 0
                self.pub_odom.publish(m)
            if wall >= t_lpos:
                t_lpos += 1.0 / self.args.lpos_rate
                lp = VehicleLocalPosition()
                lp.timestamp = out
                ned = ft.enu_to_ned(acc_enu)
                lp.ax, lp.ay, lp.az = float(ned[0]), float(ned[1]), float(ned[2])
                self.pub_lpos.publish(lp)
            if wall >= t_status:
                t_status += 1.0 / self.args.status_rate
                s = VehicleStatus()
                s.timestamp = out
                s.arming_state = 2 if armed else 1
                s.nav_state = 14 if nav == "offboard" else (18 if nav == "land" else 4)
                s.pre_flight_checks_pass = True
                self.pub_status.publish(s)
            if wall >= t_land:
                t_land += 1.0 / self.args.land_rate
                ld = VehicleLandDetected()
                ld.timestamp = out
                ld.landed = bool(landed)
                self.pub_land.publish(ld)
            nxt = min(t_odom, t_lpos, t_status, t_land, veh.t + dt)
            time.sleep(max(0.0, nxt - (time.monotonic() - self.boot_wall)))

    def report(self) -> dict:
        arr = np.diff(np.asarray(self.rate_arrivals)) * 1e3 if len(self.rate_arrivals) > 2 else np.zeros(1)
        return {"rate_setpoints": self.n_rates, "rate_gap_ms_median": float(np.median(arr)),
                "rate_gap_ms_p95": float(np.percentile(arr, 95)), "rate_gap_ms_max": float(arr.max()),
                "failsafe": self.failsafe, "events": self.events,
                "min_height": float(self.veh.min_height_airborne) if math.isfinite(self.veh.min_height_airborne) else None}


def _body_z(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ps2rl_path", required=True)
    ap.add_argument("--hover_thrust", type=float, default=0.5)
    ap.add_argument("--odom_rate", type=float, default=100.0)
    ap.add_argument("--lpos_rate", type=float, default=50.0)
    ap.add_argument("--status_rate", type=float, default=5.0)
    ap.add_argument("--land_rate", type=float, default=2.0)
    ap.add_argument("--of_loss_t", type=float, default=1.0, help="COM_OF_LOSS_T [s]")
    ap.add_argument("--synced", type=int, default=1, help="1: stamps in the agent's clock, as after time sync")
    ap.add_argument("--offset_jumps", default="", help="t:delta,... [s]")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    rclpy.init()
    node = FakePX4(args)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    t_end = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < t_end:
            ex.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.stop.set()
    rep = node.report()
    if args.report:
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "events"}))
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()

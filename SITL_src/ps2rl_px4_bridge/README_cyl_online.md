# Obstacle-only demo in Gazebo (stock x500): online backup filter, virtual cylinders

`ps2rl_cyl_online` flies east from the take-off point at 1.5 m height, past virtual cylinders, to a goal. It sends
body-rate and collective-thrust commands at 25 Hz (`control_period` 0.04 s; 50 Hz at 0.02). The commands come from a go-to-goal nominal passed through the
online backup filter (PS2-RL `docker_batch/README_cyl_online.md`, the learned 128-wide backup, blends 1 / 0, horizon
1 s). The cylinders exist only for the filter and the contact check; nothing is placed in Gazebo.

```
BOOT -> ARM -> TAKEOFF -> GOTO_START (hold 8 s) -> POLICY -> RECOVER (hover 3 s) -> LAND -> DONE
                                                    \-> RECOVER (hover where it is, 3 s) -> LAND     (abort)
```

| course | cylinders (x east, y north, r) [m] | goal [m] | simulated (planner) |
|---|---|---|---|
| O3 | (1.3, -0.1, 0.1) | (3.3, 0) | 4.7 s, the filter never steps in |
| S1 | (1.1, -0.1, 0.1), (3.1, 1.6, 0.1) | (4.3, 0) | 6.3 s weave, the filter steps in right after the start |

- **nominal.** `planner` keeps 1 m from the surfaces by itself. `straight` aims at the goal; the filter alone takes
  it round the cylinder (O3: about 10 s, lam switching between 1 and 0).
- **End.** Inside the 0.5 m goal circle and below 0.25 m/s, or after 30 s. Then PX4 position hold where the vehicle is
  for 3 s, then PX4's land mode lands it.
- **Aborts.** Any of these sends the vehicle back to PX4 hold where it is, then it lands:
  - contact with a virtual cylinder;
  - tilt above 35°, or speed above 2 m/s;
  - lower than 0.3 m above the take-off point;
  - a non-finite state or command;
  - stale odometry (0.3 s), or an exception;
  - lost arming or offboard mode.
- **Footprint.** x 0–4.5 m east, y -0.2–1.7 m north of the take-off point.

The frames, clock, arming, setpoint stream, fail-safes, PX4 clock-jump guard and CPU record are the slow-track
bridge's: `CylOnlineBridge` subclasses `SlowTrackBridge`.

**Compute (2026-11-20, for the Raspberry Pi).** The YAML defaults are now:
- `control_period: 0.04`: one command every 40 ms (25 Hz), held by PX4 in between. The filter's rollouts hold each
  command just as long. The node sets `policy_rate` from it.
- `process: true`: the controller runs in its own process. `worker_cpu` pins it to a core.
- `io: poll` (second round, below): no rclpy executor. `io: executor` keeps it, with `executor: single` / `multi`
  (the environment variable `PS2RL_EXECUTOR` overrides that if set).
- `setpoint_rate: 25`, and the stream stays quiet while the control steps publish: PX4 gets each command once.
- `outgoing_stamp: receipt` (below).
- `bridge_cpu` pins the bridge process to a core (-1: no).

**Second round (2026-11-20, after the Pi measurements).** With process + single executor + 40 ms the Pi's bridge
process alone still filled its core (92 %), about three quarters of it inside rclpy's executor. The control steps fell
behind (72 % done, step p95 133 ms). Two fixes:

1. **`io: poll`** (`ps2rl_px4_bridge/polled_loop.py`). One loop does the node's work without an executor.
   - In the controller phase it sleeps until shortly before the odometry message that makes the next step due. It
     predicts that message from the odometry stamps, their spacing and the middleware's receive time. It then waits
     for the message and steps at once, in the same thread (about 50 wake-ups/s instead of about 225 executor passes).
   - In the other phases it wakes only at the timer deadlines.
   - The control-step rule is `LandingBridge._policy_loop`'s: a step when the vehicle clock reaches it.
2. **Outgoing stamps.** PX4 v1.16 / v1.17 turn an incoming stamp of 0 into their own time of receipt; any other stamp
   becomes `min(stamp - time-sync offset, now)`. The commander counts the offboard signal as lost once
   `now >= offboard_control_mode.timestamp + COM_OF_LOSS_T`. The old stamp (the guarded odometry clock) was 6 s old after
   a +6 s time-sync jump, so all five Pi flights lost offboard. `outgoing_stamp`:
   - `receipt` (default): stamp 0.
   - `node`: the computer's clock (px4_ros_com style). Wrong while the time sync re-converges.
   - `px4`: the newest raw odometry stamp (the Pi-side fix).

   This fix sits in `slow_track_node.py`, so the slow-track node has it too.
3. **Topic names** follow the installed px4_msgs (`_v<N>` for MESSAGE_VERSION N > 0). On v1.17 that gives
   `/fmu/out/vehicle_local_position_v1`, so no `sed` is needed. A `topic_*` parameter set explicitly is used as given.

**Measured** with `tools/bench_fake.py`: real ROS 2 Jazzy + px4_msgs 1.17, against `tools/fake_px4.py` (a stand-in PX4
on the real topics). Course: one cylinder r 0.27 at (1.3, -0.1), straight nominal, 40 ms.
- "Pi core": the bridge and the controller process each limited to 15 % of an x86 core. With that limit, the
  pushed code reproduced the Pi's failure.
- CPU is given as % of that core (≈ measured % / 0.15; / 0.10 for the slower one).

| | bridge CPU (% of its core) | steps done | whole step median / p95 |
|---|---|---|---|
| pushed (executor single, 50 Hz stream) | ≈ 73 % (at its limit) | 256 / 312 | 40 / 70 ms |
| `io: poll` (new defaults) | ≈ 40 % | 334 / 334 | 4.9 / 14.8 ms |
| `io: poll`, a 1.5× slower core | ≈ 60 % | 337 / 338 | 11.8 / 22.9 ms |
| `io: poll`, `control_period: 0.02` | ≈ 53 % | 659 / 659 | 2.5 / 13.2 ms |
| `io: poll`, `process: false` | ≈ 60 % (bridge + controller) | 338 / 338 | 1.5 / 6.9 ms |

- Time-sync offset jumps (+6, +3, +2.5 s) during GOTO_START:
  - the pushed code lost offboard at the first jump, as on the Pi;
  - `receipt` and `px4` flew to the goal;
  - `node` lost offboard.
- The slow-track node flew its reference through a +6 s jump.

To fly as at 13:46 on 2026-11-20 (the first SITL flights):

`control_period:=0.02 process:=false io:=executor executor:=multi setpoint_rate:=50 outgoing_stamp:=px4`

### Measuring the bridge without PX4

`tools/fake_px4.py` (not installed; run from the source tree) is a stand-in PX4 on the real ROS topics. It runs PS2-RL's
stand-in vehicle in real time and handles stamps, the offboard check and the modes as PX4 v1.17 does.
`tools/bench_fake.py` starts it with the bridge and prints the outcome, the steps, the command gaps PX4 saw, the step
times and the CPU of both processes. It needs a sourced ROS 2 with px4_msgs:

```bash
cd ~/ws_shared/dual_stage_rl/SITL_src/ps2rl_px4_bridge
python3 tools/bench_fake.py --ps2rl_path ~/ws_shared/PS2-RL --label poll
python3 tools/bench_fake.py --ps2rl_path ~/ws_shared/PS2-RL --label exec -- io:=executor
python3 tools/bench_fake.py --ps2rl_path ~/ws_shared/PS2-RL --label jumps --offset_jumps 14:+6,17:+3 -- outgoing_stamp:=receipt
```

On the Pi this measures the bridge with nothing else loaded (PX4 runs on the Pixhawk in flight). For a core layout add
`--bridge_cpu 2 --fake_cpu 0` and `worker_cpu:=3`.

## Run

Every terminal is `docker exec -it px4sitl bash` from the host. The bridge runs with the container's system `python3`,
**not** the training venv.

**Once.** numba is optional but recommended: about 0.4 ms per decision against 4–30 ms in numpy. The straight-line
runs need it. Check, then install if it is missing:

```bash
python3 -c "import numba, numpy; print(numba.__version__, numpy.__version__)"
python3 -m pip install --user --break-system-packages numba==0.60.0
```

numba 0.60 keeps the system numpy 1.26.4. Then build the bridge, and open a new shell afterwards:

```bash
cd ~/ws_shared
colcon build --packages-select ps2rl_px4_bridge --base-paths src
```

**Offline check.** No ROS, PX4 or Gazebo; about 30 s:

```bash
cd ~/ws_shared/PS2-RL
python3 scripts/rehearse_cyl_online_flight.py --all
```

**Terminal 1**, DDS:

```bash
pkill -f MicroXRCEAgent; sleep 1
MicroXRCEAgent udp4 -p 8888
```

**Terminal 2**, PX4 SITL. Restart it before every flight:

```bash
pkill -f 'px4|gz sim|ruby'; sleep 2
cd ~/PX4-Autopilot && make px4_sitl gz_x500
```

**Terminal 3**, bridge, in a fresh shell. Run one flight per PX4 start; the node exits after landing:

```bash
deactivate 2>/dev/null || true
ros2 launch ps2rl_px4_bridge ps2rl_cyl_online.launch.py course:=O3 label:=O3_planner
ros2 launch ps2rl_px4_bridge ps2rl_cyl_online.launch.py course:=S1 label:=S1_planner
ros2 launch ps2rl_px4_bridge ps2rl_cyl_online.launch.py course:=O3 nominal:=straight label:=O3_straight
```

**After a flight**, draw it. This writes `flight.png` into the flight folder:

```bash
cd ~/ws_shared/experiments/cyl_online
python3 ~/ws_shared/PS2-RL/scripts/plot_cyl_online_flight.py $(ls -td 2026*/ | head -1)
```

The startup log should show:
- `controller process: pid ...` and `policy_rate 50 Hz -> 25 Hz (control_period 0.04 s)`;
- `PX4 topics: ... local_position /fmu/out/vehicle_local_position[_v1] ...; outgoing stamps: receipt`;
- `io: poll; setpoint stream 25 Hz; control 25 Hz`;
- `cyl-online runtime: ... control period 40 ms, numba`;
- the cylinders and the goal;
- `start: 1.00 m from the nearest surface` for S1, or 1.20 m for O3.

During GOTO_START, `at the start ...; holding for 8 s`. Then `hand-over to the filtered controller`. At the end:
- `filtered descent ended: goal reached ...`;
- the filter line: lam, modes, rollout steps;
- the compute line: decision and step times against the control period;
- after landing: the CPU use of the bridge process and of the controller process, and the loop line
  (`bridge loop (io poll) ...: N wake-ups/s; a pass takes ...`).

The wording "descent", "pad" and "cone margin" comes from the landing logic this node shares; "cone margin" is the
clearance from the cylinders.

Records: `~/ws_shared/experiments/cyl_online/<time>-<label>/{flight.csv, summary.json, bridge_params.json}`.

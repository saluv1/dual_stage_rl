# Obstacle-only demo in Gazebo (stock x500): online backup filter, virtual cylinders

`ps2rl_cyl_online` flies east from the take-off point at 1.5 m height, past virtual cylinders, to a goal. It sends
body-rate and collective-thrust commands at 50 Hz. The commands come from a go-to-goal nominal passed through the
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
- `cyl-online runtime: ... numba`;
- the cylinders and the goal;
- `start: 1.00 m from the nearest surface` for S1, or 1.20 m for O3.

During GOTO_START, `at the start ...; holding for 8 s`. Then `hand-over to the filtered controller`. At the end:
- `filtered descent ended: goal reached ...`;
- the filter line: lam, modes, rollout steps;
- the compute line: decision and step times.

The wording "descent", "pad" and "cone margin" comes from the landing logic this node shares; "cone margin" is the
clearance from the cylinders.

Records: `~/ws_shared/experiments/cyl_online/<time>-<label>/{flight.csv, summary.json, bridge_params.json}`.

# Slow forward + loop tracking in Gazebo (stock x500)

`ps2rl_slow_track` flies a slow reference with body-rate and collective-thrust commands at 50 Hz. The reference is a hover, 2 m straight east at 0.4 m/s, one 1.5 m-radius loop to the left, 2 m straight, then a hover; 46 s in all. It flies it with either:
- the PD controller (`controller:=pd`), the first check of the whole chain; or
- a trained RL policy (`controller:=policy run:=...`).

There are no obstacles.

```
BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY (reference + 2 s) -> RECOVER (hover 3 s) -> LAND -> DONE
                                          \-> RECOVER (hover where it is, 3 s) -> LAND     (abort)
```

| phase | who flies | what happens |
|---|---|---|
| BOOT | – | the vehicle stands disarmed; its position is the take-off point |
| ARM, TAKEOFF, GOTO_START | PX4 position control | climb to 2 m above the take-off point at yaw 0 (east), and hold until position, speed and tilt are inside the gates for 2 s; the disturbance observer already runs on the measured thrust |
| POLICY | PD or policy, 50 Hz | the reference, starting from the hand-over point (task-frame origin, x east) |
| RECOVER | PX4 position control | hover where the vehicle is, then PX4's land mode lands it |

**Aborts.** Any of these sends the vehicle straight back to PX4 position hold where it is, then it lands:
- tracking error above 0.8 m;
- tilt above 35°;
- speed above 1.5 m/s;
- lower than 0.7 m above the take-off point;
- a non-finite state or command;
- stale odometry (0.3 s);
- an exception in the controller;
- loss of arming or offboard mode.

**Footprint.** The vehicle flies x 0–5 m (east) and y 0–3 m (north) from the take-off point, at 2 m height. The straight line needs x 0–2 m only.

## Run

This follows `ws_shared/PS2RL_SITL_final.md`. Every terminal is `docker exec -it px4sitl bash` from the host. The bridge runs with the container's system `python3`, **not** the training venv. It needs only numpy; the policy's actor is evaluated in numpy.

**Once**, after unzipping the bundle on the host (`~/px4-ros-docker`), in the container:

```bash
cd ~/ws_shared
colcon build --packages-select ps2rl_px4_bridge --base-paths src
```

Open a new shell afterwards so it sources `ws_shared/install`. Rebuild after every change to the node, the YAML or the launch file; changes under `PS2-RL/` need no rebuild. After a `docker compose down` / `up`, run `bash ~/ws_shared/bootstrap.sh` again: it re-creates the `src` symlink and builds the bridge. The stock `gz_x500` used here needs none of the powerloop modifications.

**Terminal 1**, DDS:

```bash
pkill -f MicroXRCEAgent; sleep 1
MicroXRCEAgent udp4 -p 8888
```

**Terminal 2**, PX4 SITL with the stock x500. Restart it cleanly before every flight:

```bash
pkill -f 'px4|gz sim|ruby'; sleep 2
cd ~/PX4-Autopilot && make px4_sitl gz_x500
```

**Terminal 3**, bridge:

```bash
deactivate 2>/dev/null || true
# PD first:
ros2 launch ps2rl_px4_bridge ps2rl_slow_track.launch.py
# then the trained policy (the loop policy that flew: v4_floor_s0; published copy: checkpoints/slow_track/v4_floor_s0):
ros2 launch ps2rl_px4_bridge ps2rl_slow_track.launch.py controller:=policy run:=outputs/slow_track/v4_floor_s0 label:=policy_v4_floor
```

**Straight-line test** (2 m along x at up to 0.3 m/s, 16.7 s; the first test on the vehicle). The PD takes the reference from `reference_path`; a policy takes it from its own `configs.json`:

```bash
ros2 launch ps2rl_px4_bridge ps2rl_slow_track.launch.py reference_path:=ps2rl/envs/assets/slow_line/quadrotor_slow_line_reference.npz label:=pd_line
ros2 launch ps2rl_px4_bridge ps2rl_slow_track.launch.py controller:=policy run:=outputs/slow_track/line_v4_s0 label:=policy_line
```

`PS2-RL/SLOW_TRACK_ONBOARD.md` says which policy to use on the vehicle and what to change there.

The node exits by itself once PX4 has landed and disarmed.

**Figure and report** of the last flight, or several overlaid (system `python3`):

```bash
python3 ~/ws_shared/PS2-RL/scripts/plot_slow_track_flight.py --latest ~/ws_shared/experiments/slow_track
python3 ~/ws_shared/PS2-RL/scripts/plot_slow_track_flight.py <pd flight dir> <policy flight dir>
```

**Records.** Each flight writes to `~/ws_shared/experiments/slow_track/<time>-<label>/` (git-ignored by `ws_shared/.gitignore`):
- `flight.csv`: every step, with state, reference, error, commands, PD command, measured rates and thrust, normalised thrust and `d_hat`;
- `summary.json`: outcome, error statistics, command ranges, latency and the fitted rate response;
- `bridge_params.json`.

At the end of the flight the node prints the measured roll/pitch rate response (delay + lag). The policy was trained for 0–40 ms of command delay plus rate lags of 20–80 ms. If SITL shows more than that, tell me before flying the policy.

**Thrust map.** It is the stock-x500 fit used for the landing flights (`thrust_k*` in `config/slow_track_x500.yaml`). The command box of 0.5–1.5 g is inside the x500's 1.69 g.

## If the height sags in SITL

The first five policy flights (2026-11-18) sagged 7–27 cm while the PD flight held ±2 cm. Cause: numpy's OpenBLAS (older builds, e.g. numpy 1.26) split the policy's 256 x 256 products over every core and kept those threads spinning between the 50 Hz steps. The bridge took most of the computer, Gazebo delivered IMU samples late, PX4's gz bridge stamps them on arrival, and EKF2 caps the integration time of a long sample at 20 ms for gravity but not for the measured velocity change. Each late sample therefore kicks PX4's vertical velocity up by a few cm/s, and any controller that damps vertical velocity sinks. The same starvation caused the uXRCE-DDS time-sync resets.

`ps2rl_px4_bridge/__init__.py` now sets `OPENBLAS_NUM_THREADS=1` (and the OpenMP, MKL and BLIS equivalents) for every node, and the slow-track node records its own CPU use during the controller phase as a flight event (expect a few % of one core). Checks:

```bash
python3 ~/ws_shared/PS2-RL/scripts/check_blas_threads.py            # same python3 as the bridge
python3 ~/ws_shared/PS2-RL/scripts/check_sim_timing.py <flight folders> --plot timing_check.png
```

## Files

| | |
|---|---|
| `ps2rl_px4_bridge/__init__.py` | single-threaded BLAS for every node |
| `ps2rl_px4_bridge/slow_track_node.py` | ROS 2 node (subclass of the landing node: frames, clock, arming, setpoints and fail-safes are shared) |
| `config/slow_track_x500.yaml` | parameters |
| `launch/ps2rl_slow_track.launch.py` | launch file with command-line overrides |
| `setup.py` | one new entry point, `ps2rl_slow_track` |
| `PS2-RL/ps2rl/deploy/slow_track_runtime.py`, `slow_track_flight.py` | controller and flight logic (plain Python, rehearsed offline with `scripts/rehearse_slow_track_flight.py`) |

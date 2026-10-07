# Filtered landing in Gazebo (stock x500), without a Phase-II policy

`ps2rl_landing` flies a landing with the **landing safety filter in the loop** on the stock
`gz_x500`. The nominal controller is the hover LQR of the Phase-I backup checkpoint, aimed at the
pad; every command passes through the BCBF filter (cone, floor, backup policy) before it goes to
PX4 as body rates and thrust. No Phase-II policy is needed. The powerloop bridge
(`ps2rl_bridge`) is unchanged.

```
BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> LAND -> DONE
                                          \-> RECOVER -> LAND      (abort)
```

| phase | who flies | what happens |
|---|---|---|
| BOOT | – | vehicle stands disarmed; its position is taken as the ground point |
| ARM, TAKEOFF, GOTO_START | PX4 position control | climb, fly to the start point, hold until position, speed and tilt are inside the gates and the state is recoverable by the backup |
| POLICY | LQR + filter, 50 Hz body rates and thrust | descent to the pad plane |
| LAND | PX4 land mode | the last centimetres, disarm |
| RECOVER | PX4 position control | after an abort: hold 1.5 m above the pad, then land |

## Where things are

| | |
|---|---|
| `ps2rl_px4_bridge/landing_node.py` | ROS 2 node: frame conversion (NED/FRD to z-up), clock, publishing. No flight decisions. |
| `config/landing_x500.yaml` | parameters for the stock x500 |
| `launch/ps2rl_landing.launch.py` | launch file with command-line overrides |
| `PS2-RL/ps2rl/deploy/landing_runtime.py` | nominal controller + filter for one state |
| `PS2-RL/ps2rl/deploy/landing_flight.py` | flight logic (phases, gates, touchdown test, aborts, records). Plain Python. |
| `PS2-RL/ps2rl/deploy/landing_plant_sim.py`, `PS2-RL/scripts/rehearse_landing_flight.py` | the same flight logic against a stand-in vehicle, without PX4, Gazebo or ROS |
| `PS2-RL/scripts/plot_landing_flight.py` | figure and report from a recorded flight |
| `PS2-RL/checkpoints/landing_phase1/x500_rec10_td3_seed1/` | the backup checkpoint from the x500 GPU batch (seed 1) |

## Geometry

- **Pad centre** = the point where the vehicle stands before arming, plus `pad_offset`.
  The default `pad_offset: "0,0,0.05"` puts the pad plane, which is the filter's floor, 5 cm above
  the ground. That is a margin against height-estimate and thrust errors; PX4's land mode flies
  the last 5 cm.
- **Start** of the filtered descent: `start_position`, relative to the pad centre
  (default 1.65 m east, 2 m up).
- **Yaw** is held at east (body x along local x). The cone is symmetric, so this only fixes the
  direction of a reference trajectory.
- So the vehicle takes off, flies 1.65 m east, and lands back where it stood.

## Set up (once, after copying the files)

In the container:

```
cd ~/ws_shared
colcon build --packages-select ps2rl_px4_bridge --base-paths src
source install/setup.bash
```

Rebuild after every change to the node, the launch file or the YAML. Changes under `PS2-RL/`
need no rebuild.

The node runs in the same Python as the powerloop bridge (system Python 3.12 with `rclpy`, CPU
`jax`, `qpax`, `numpy`, `scipy`); it needs nothing else. It talks to PX4 through the same offboard
interface as the thrust calibration that was run on the stock x500, so PX4 should need no new
parameters.

### Optional: rehearse without Gazebo

```
cd ~/ws_shared/PS2-RL
python3 scripts/rehearse_landing_flight.py --backup checkpoints/landing_phase1/x500_rec10_td3_seed1 \
    --start 1.65,0,2 --target 0,0,-0.1 --max_position_error 1 --pad_offset 0,0,0.05 --sweep
```

This flies 14 variations of a stand-in vehicle (rate and thrust lags, delay, thrust-map error,
height drift, noise) and prints one line per flight.

## Fly

Terminal 1, DDS agent:

```
pkill -f MicroXRCEAgent; sleep 1
MicroXRCEAgent udp4 -p 8888
```

Terminal 2, PX4 with the **stock** x500:

```
pkill -f 'px4|gz sim|ruby'; sleep 2
cd ~/PX4-Autopilot && make px4_sitl gz_x500
```

Terminal 3, the landing bridge:

```
ros2 launch ps2rl_px4_bridge ps2rl_landing.launch.py label:=A_default
```

The node exits when the vehicle has disarmed. A flight takes about 25 s. To fly again, launch
again; if PX4 refuses to arm, restart terminal 2.

A healthy run prints (times and numbers will differ):

```
landing runtime ready | backup x500_rec10_td3_seed1 | nominal lqr_target -> [0.0, 0.0, -0.1] (position error capped at 1 m) | filter on ... | warm latency 8.3 ms
Thrust commands 1.99..14.95 m/s^2 map to normalized thrust 0.160..0.934; hover 9.81 m/s^2 -> 0.728
ground point [...] (local frame, 101 samples); pad centre at [...]
phase: BOOT -> ARM
phase: ARM -> TAKEOFF
phase: TAKEOFF -> GOTO_START
hand-over to the filtered controller: position 0.03/0.15 m, speed 0.04/0.15 m/s, tilt 0.1/5.0 deg; cone margin 0.66 m, height 1.98 m
phase: GOTO_START -> POLICY
filtered descent ended: touchdown
  161 control steps in 3.21 s (expected 161); cone margin min +0.247 m, height min +0.006 m, ...
  filter: active on 2 % of steps, largest slack 7.6e-05, QP fallbacks 0, safeguard blends 1, thrust saturated 0 steps; latency median 10.5 / p95 14.8 / max 22.8 ms
phase: POLICY -> LAND
phase: LAND -> DONE
flight complete (touchdown); records in /home/dev/ws_shared/experiments/landing/20261007_151712-A_default
```

## Test sequence

Fly A first. C and D are the pair that shows what the filter does at the cone wall; B shows it at the floor.

| | command (after `ros2 launch ps2rl_px4_bridge ps2rl_landing.launch.py`) | what it tests |
|---|---|---|
| A | `label:=A_default` | the ROS side, not the filter: the capped LQR flies a straight line well inside the cone, so the filter only acts in the last steps at the pad plane. Frames, clock, thrust map, hand-over, touchdown. |
| B | `target:=apex max_position_error:=0 label:=B_apex` | the floor: the plain re-centred LQR aimed at the cone's apex arrives fast and would go through the pad plane; the filter brakes it there |
| C | `nominal:=track_reference reference_path:=ps2rl/envs/assets/quadrotor_landing_cornercut_reference.npz start_position:=auto label:=C_track` | the cone wall: tracking a reference that is outside the cone for 0.76 s of its 2 s (by up to 15 cm), with the filter |
| D | the same as C plus `use_filter:=false label:=D_nofilter` | the same reference without the filter (simulation only) |

What the stand-in vehicle gave here (not Gazebo; treat them as the order of magnitude to expect):

| | length of the descent | lowest cone margin | filter changed the command | speed 2 cm above the pad plane |
|---|---|---|---|---|
| A | 3.2 s | +0.25 m | 2 % of steps (at the floor) | 0.25 m/s |
| B | 2.2 s | +0.06 m | 25 % | 0.6–0.7 m/s, dips 1–2 cm below the pad plane |
| C | 1.75 s | 0 (rides the wall; within 1 cm of it) | 15 % | 0.26 m/s |
| D | 1.7 s | −0.05 to −0.07 m (outside the cone) | – | 0.7–0.9 m/s, dips 4–5 cm below the pad plane |

## Records and analysis

Each flight writes `<log_root>/<time>-<label>/`:

| file | content |
|---|---|
| `flight.csv` | one row per step: phase, state (pad frame), target, commanded / nominal / measured thrust and rates, slack, QP status, safeguard, cone margin, height, computation time |
| `summary.json` | outcome, statistics of the filtered descent, geometry, every setting |
| `bridge_params.json` | the node's parameters and thrust map |

Figure and report (system Python, needs matplotlib):

```
python3 ~/ws_shared/PS2-RL/scripts/plot_landing_flight.py --latest ~/ws_shared/experiments/landing
python3 ~/ws_shared/PS2-RL/scripts/plot_landing_flight.py RUN_C RUN_D --compare ~/ws_shared/experiments/landing/C_vs_D.png
```

It adds `flight.png` and `analysis.txt` to the flight's folder.

**To send back:** the folder `~/ws_shared/experiments/landing/` (zipped), and the bridge
terminal output of any flight that did not end in `touchdown`.

## What to look at in the first flights

These are the things the stand-in vehicle could only assume.

1. **Thrust map.** `thrust   commanded - measured: mean ...` in the report. The filter has no
   integral action, so a steady offset shifts where the vehicle settles: about 2.8 cm per
   0.1 m/s². The coefficients are those of `ws_shared/thrust_fit_x500.yaml`; at low thrust they
   disagree with the rotor model, so a descent with low thrust (B) is the informative one.
2. **Rate tracking.** `rates   commanded - measured, RMS`. The stand-in assumed a 0.05 s lag in
   roll and pitch and gave 0.2–0.8 rad/s RMS.
3. **Timing.** `steps ... interval mean / min / max`. Control steps are scheduled on PX4's
   clock, which the node reads from odometry stamps. With odometry every 8 ms the 20 ms period
   comes out as alternating 16 and 24 ms; the mean must be 20 ms. `latency` must stay well under
   20 ms.
4. **Height.** `floor   lowest height above the pad plane`, and whether the vehicle touched the
   ground before PX4's land mode took over (it should not with the 5 cm offset).
5. **Outcome `at rest ... (no touchdown)`** means the vehicle settled above the pad plane, which
   points to a thrust surplus. A target further below the pad (`target:="0,0,-0.2"`) pushes harder.

## Parameters

Launch arguments (each overrides the YAML only when given):
`ps2rl_path`, `backup`, `filter`, `use_filter`, `qp_solve_dtype`, `nominal`, `target`,
`max_position_error`, `reference_path`, `start_position`, `pad_offset`, `policy_timeout`,
`log_root`, `label`, `exit_when_done`, and `config` for another YAML.
`ros2 launch ps2rl_px4_bridge ps2rl_landing.launch.py --show-args` lists them with a description.
`start_position:=auto` asks for the default start (the reference's first state, or the base-set
hover point); `reference_path:=none` clears a path.

Other parameters are set in the YAML:

- every field of `FlightConfig` in `PS2-RL/ps2rl/deploy/landing_flight.py` (gates, touchdown
  test, abort limits, recovery), which also documents them;
- thrust map (`thrust_model`, `thrust_k2`, `thrust_k1`, `thrust_k0`, `thrust_min`, `thrust_max`);
- PX4 topic names (`topic_status`, `topic_odometry`, ...), in case the PX4 version names them
  differently;
- `filter`: any field of the landing BCBF config, for example
  `filter:="alpha=10,alpha_floor=20,backup_policy_mode=base_controller"` uses the hover LQR as
  the backup policy (about 3 ms per step instead of 9).

Relative paths (`backup`, `reference_path`) are taken from `ps2rl_path`.

## If something goes wrong

| message | meaning |
|---|---|
| `no odometry on ... yet` | PX4 or the DDS agent is not running, or the topic name differs |
| `no vehicle status on ... yet` | status topic name differs (`ros2 topic list`); set `topic_status` |
| `still waiting after N s: offboard False ...` | PX4 accepts offboard only while setpoints arrive; check the agent |
| `... pre-flight checks FAIL` | PX4 refuses to arm; connect QGroundControl to see why |
| `start ... is outside the safe set` / `hovering at the start ... is not recoverable` (at start-up) | choose a start further inside the cone |
| `the state at the start is not recoverable by the backup; waiting` | the vehicle has not settled; it hands over when the state becomes recoverable, or lands after `start_timeout` |
| `filter latency ... is close to the control period` | the CPU is too slow or busy; try `filter:="...,backup_policy_mode=base_controller"` |
| `The thrust map saturates inside the controller's thrust range` | thrust coefficients do not fit this airframe |
| `aborted: ...` | the descent was stopped (cone left by more than 0.3 m, 0.3 m below the pad plane, tilt above 75°, stale odometry, an exception); PX4 holds 1.5 m above the pad and lands |
| `time-out after 15.0 s` | the vehicle neither touched down nor came to rest within `policy_timeout`; PX4 lands it from where it is |
| `flight logic error ... asking PX4 to land` | a bug outside the filtered descent; the node keeps the last position setpoint, sends PX4 to its land mode and exits after the disarm. Send the terminal output. |
| `PX4's clock went back: the autopilot was restarted` | PX4 was restarted while the node ran; the ground point is no longer valid. Launch the node again. |
| `the autopilot reset part of its state estimate ...` | PX4's estimator reset something after the ground point was taken (also listed under `events` in `summary.json`). The flight continues; if the pad looks shifted in the figure, this is why. |

## Limits of this test

- The action limits are those the Phase-I backup was trained with for the stock x500
  (thrust 1.99–14.95 m/s², roll/pitch 3.84 rad/s, yaw 0.203 rad/s). They are placeholders: the
  hardware will be a Tarot 450, and its limits, thrust map and backup will replace them.
- The nominal controller is an LQR, not a learned policy. A Phase-II policy will plug into
  `landing_runtime.py` as a third nominal controller once it exists.
- The node was tested without ROS against a fake PX4 that speaks the same messages (all phases,
  slow simulation, stale odometry, exceptions in the control step and in the flight logic, a
  restarted autopilot, missing land detector), and its ROS, launch and PX4 calls were checked
  against the Jazzy and PX4 v1.16 sources. It has not yet run against the real PX4.

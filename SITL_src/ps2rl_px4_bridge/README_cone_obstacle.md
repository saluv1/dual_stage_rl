# Outside-cone task in Gazebo (light x500_powerloop airframe)

`ps2rl_cone_obstacle` flies the trained outside-cone policies in PX4 SITL. A Phase-II policy
(or the stage-1 tracker) commands body rates and collective thrust at 50 Hz for one training
episode (reference L3-1-f, 5.6 s + 2 s), through the cone-obstacle safety filter (UE-bCBF by
default) with its disturbance observer. The funnel and the target cylinder are **virtual**:
nothing stands in the Gazebo world; the filter keeps the vehicle out of the funnel it believes is there,
and the records say what the margins were.

```
BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY (one episode) -> RECOVER (hold 3 s) -> LAND -> DONE
                                          \-> RECOVER -> LAND      (abort)
```

| phase | who flies | what happens |
|---|---|---|
| BOOT | – | vehicle stands disarmed; its position is the ground point = the task anchor |
| ARM, TAKEOFF, GOTO_START | PX4 position control | climb to the reference start (2.2 m), hold until position, speed and tilt are inside the gates and the state is recoverable by the backup; the disturbance observer already runs on the measured thrust |
| POLICY | policy + filter, 50 Hz | one training episode; touching the target is recorded, not an end condition |
| RECOVER | PX4 position control | hold where the episode ended, then land |

**Airframe.** The policies were trained with the 4 g box (thrust 0–39.24 m/s², body rates
±18 rad/s on all axes). The light airframe (1 kg, rotor speed up to 1500 rad/s, rate limits
1100 °/s; `setup_px4_powerloop.sh`) can deliver it: 39.24 m/s² is a normalised thrust of about
0.43 with its thrust map. The stock x500 (1.7 g at full throttle) cannot.

**Geometry.** Take-off point = the ground point below the reference start. The funnel axis is
3 m along local x (east) from it; its tip is 1.2 m above the ground, the floor 0.5 m. The vehicle
flies east, slides under the funnel's wall and touches the tip from outside; it ends about 3 m
east of where it took off, about 0.9 m up.

## Where things are

| | |
|---|---|
| `ps2rl_px4_bridge/cone_obstacle_node.py` | ROS 2 node (subclass of the landing node: frames, clock, arming, setpoints and fail-safes are shared) |
| `config/cone_obstacle_x500_powerloop.yaml` | parameters for the light airframe |
| `launch/ps2rl_cone_obstacle.launch.py` | launch file with command-line overrides |
| `PS2-RL/ps2rl/deploy/cone_obstacle_runtime.py` | policy + filter + observer (+ delay compensation) for one state |
| `PS2-RL/ps2rl/deploy/cone_obstacle_flight.py` | flight logic, aborts, records. Plain Python. |
| `PS2-RL/scripts/rehearse_cone_obstacle_flight.py` | the same flight logic against a stand-in vehicle, without PX4, Gazebo or ROS |

## Set up (once, after copying the files)

The policy runs live under `PS2-RL/outputs/cone_obstacle_phase2_ue/` (the GPU/CPU batch writes
them there) and the Phase-I choice under `PS2-RL/outputs/cone_obstacle_phase1/compare/chosen_backup.json`.

```
cd ~/ws_shared
colcon build --packages-select ps2rl_px4_bridge --base-paths src
source install/setup.bash
```

Rebuild after every change to the node, the launch file or the YAML (changes under `PS2-RL/` need no rebuild).
The node runs in the system Python with `rclpy` (CPU `jax`), not in the training venv.

### Optional: rehearse without Gazebo

```
cd ~/ws_shared/PS2-RL
python3 scripts/rehearse_cone_obstacle_flight.py --run outputs/cone_obstacle_phase2_ue/cone_ue_ps2_s0 \
    --backup_from outputs/cone_obstacle_phase1/compare/chosen_backup.json --predict_ahead 0.04 --sweep
```

## Fly

Terminal 1: `pkill -f MicroXRCEAgent; sleep 1; MicroXRCEAgent udp4 -p 8888`

Terminal 2: `pkill -f 'px4|gz sim|ruby'; sleep 2; cd ~/PX4-Autopilot && make px4_sitl gz_x500_powerloop`

Terminal 3:

```
ros2 launch ps2rl_px4_bridge ps2rl_cone_obstacle.launch.py label:=A_ps2_ue
```

A flight takes about 25 s; the node exits when the vehicle has disarmed. Records:
`~/ws_shared/experiments/cone_obstacle/<time>-<label>/` (`flight.csv`, `summary.json`, `bridge_params.json`).

A healthy run ends like this (numbers from the stand-in vehicle):

```
touched the target cylinder at t = 4.35 s
filtered descent ended: episode complete after 7.59 s (touched the target at 4.35 s)
  380 control steps in 7.59 s (expected 381); funnel margin min +0.033 m, floor margin min +0.359 m; touched the target at 4.35 s
  filter: active on 42 % of steps, mean correction 5.07, largest slack 2.8e-04, QP fallbacks 0, safeguard blends 0, ...
  roll/pitch rate response: delay 20 ms + first-order lag 38 ms = effective delay 58 ms (delay compensation in this flight: 40 ms; ...)
```

## Test sequence

| | command (after `ros2 launch ps2rl_px4_bridge ps2rl_cone_obstacle.launch.py`) | what it shows |
|---|---|---|
| A | `label:=A_ps2_ue` | Phase-II policy through the UE-bCBF (the result) |
| B | `run:=outputs/cone_obstacle_phase2_ue/cone_ue_vanilla_s0_rs0p01 label:=B_tracker_ue` | stage-1 tracker through the UE-bCBF (Phase II's starting point) |
| C | `run:=outputs/cone_obstacle_phase2_ue/cone_ue_vanilla_s0_rs0p01 filter:=none label:=C_tracker_none` | tracker alone: flies through the (virtual) funnel, about 1 m deep |
| D | `filter:=nominal label:=D_ps2_nominal` | Phase-II policy through the plain bCBF |

## Delay compensation (`predict_ahead`)

The policies were trained on a model where a commanded body rate acts at once. PX4 adds a delay
(computation, DDS, the rate loop's cycle) and the rate loop adds a lag. In the stand-in vehicle
this decides the outcome:

| rate lag \ command delay | 0 | 10 ms | 20 ms | 30 ms | 40 ms |
|---|---|---|---|---|---|
| 20 ms, no compensation | ok | ok | ok | tilt abort | – |
| 30 ms, no compensation | ok | ok | ok, but 23 cm under the floor | tilt abort | – |
| 50 ms, no compensation | ok | ok | under the floor, abort | tilt abort | – |
| 20–50 ms, compensation = delay | | | ok | ok | ok |

With compensation the controller acts on the state predicted `predict_ahead` seconds ahead (model,
disturbance estimate and the commands still on their way; the first segment uses the measured
rates and thrust). Over-compensating is mild (an ideal vehicle with 80 ms still flies and touches;
a 10 ms delay + 30 ms lag vehicle with 40 ms flies like the ideal one), under-compensating is not. The
config therefore sets **40 ms**. Every flight prints the fitted roll/pitch rate response
(delay + first-order lag = "effective delay"); if a flight oscillates or aborts on tilt, fly it
again with `predict_ahead:=` about that effective delay (e.g. `predict_ahead:=0.08`).

Stand-in sweep with 20 ms delay + 50 ms rate lag as the base (`--sweep`): without compensation all
11 variations except the ideal vehicle abort; with 20 ms compensation 9 of 11 finish and touch the
target (the two 60 ms-delay variations finish with `predict_ahead` 0.08 / 0.16). The one that
touched the funnel is the noisy estimate (1.5 cm / 0.05 m/s white noise): 4 mm inside at worst.

## Other things to know

- **Observer noise.** The disturbance observer's gain (λ = 20) multiplies velocity-estimate noise
  by 20: 0.02 m/s of white noise gives |d̂| ≈ 0.7 m/s² with no disturbance at all. The flights
  still finish with margin in the stand-in, but `dhat_*` in `flight.csv` is worth a look; the
  design bound is 0.5 m/s². `observer:=commanded` (the training form) is worse with actuator lag
  (it sees the lag as a disturbance); `observer:=off` sets d̂ = 0.
- **Timing.** One control step (policy + UE-bCBF, float64 QP) takes 7–11 ms median on a 2-core
  CPU; the budget is 20 ms. The summary prints the latency.
- **Comparison flights without the filter** (C) are expected to cross the virtual funnel; the
  funnel abort is only active in filtered flights. Floor (0.3 m under the 0.5 m floor) and tilt
  (110°) aborts are always active.

# ps2rl_px4_bridge

PS2-RL의 학습된 Phase-2 정책을 **PX4 + Gazebo SITL**에서 돌리기 위한 ROS 2 (Jazzy) 브리지.

정책의 액션 `u = [a_cmd, ωx, ωy, ωz]`을 PX4 offboard **body-rate + collective thrust**
setpoint로 변환해서 내보냅니다. CIL(BCBF-QP) 투영은 **런타임에 매 스텝 실행**됩니다 —
이걸 빼면 방법론의 안전 보증이 통째로 사라지기 때문에 끄는 옵션은 없습니다.

## 검증된 인터페이스

`checkpoints/deployed_ps2/quadrotor_ps2_learned`로 실제 로딩 테스트를 통과한 값들:

| 항목 | 값 |
|---|---|
| 관측 차원 | 26 (`x(10) + ref_state(10) + ref_omega(3) + [t, sin φ, cos φ]`) |
| 제어 주기 | `env_cfg.dt = 0.02 s` → 50 Hz |
| 상태 규약 | ENU (z-up), 쿼터니언 `[w,x,y,z]`, FLU 바디 |
| 액션 | `a_cmd ∈ [0, 4g] m/s²`, `ω ∈ ±18 rad/s` (FLU 바디) |
| 안전 제약 | `z ≤ z_max = 3.0 m` |
| 레퍼런스 | 파워루프, 2.10 s, 진입 `p=(0,0,0.5)`, `v=(-4.5,0,0)` |
| CPU 추론 지연 | 약 5 ms (20 ms 예산 대비 여유) |

---

## 1. 설치

### 1-1. PX4 + Gazebo

```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot
cd ~/PX4-Autopilot && bash ./Tools/setup/ubuntu.sh
make px4_sitl gz_x500          # Gazebo(gz sim) SITL 빌드 및 실행
```

### 1-2. uXRCE-DDS agent

```bash
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent && mkdir build && cd build
cmake .. && make && sudo make install && sudo ldconfig /usr/local/lib/
```

### 1-3. ROS 2 워크스페이스

**중요:** `px4_msgs`는 ROS 배포판이 아니라 **PX4 릴리스 라인**에 맞춰 브랜치를 골라야 합니다.
메시지 정의가 어긋나면 조용히 잘못된 필드를 읽습니다.

```bash
mkdir -p ~/ps2rl_ws/src && cd ~/ps2rl_ws/src
git clone https://github.com/PX4/px4_msgs.git      # 쓰는 PX4 버전 브랜치로 체크아웃
cp -r /path/to/ps2rl_px4_bridge .

cd ~/ps2rl_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select px4_msgs ps2rl_px4_bridge
source install/setup.bash
```

### 1-4. PS2-RL 의존성

ROS 2 Jazzy의 Python 3.12 환경에 설치하세요 (별도 venv를 쓰면 rclpy가 안 보입니다):

```bash
pip install --user "jax==0.6.2" "jaxlib==0.6.2" "qpax==0.0.9" numpy scipy
```

`ps2rl` 패키지 자체는 `ps2rl_path` 파라미터로 `sys.path`에 주입되므로 별도 설치가 필요 없습니다.

---

## 1-5. PX4 파라미터 (SITL 전용)

```
pxh> param set COM_RCL_EXCEPT 4   # RC 없이 offboard 허용
pxh> param set NAV_RCL_ACT 0      # RC 링크 로스 failsafe 끄기
pxh> param set NAV_DLL_ACT 0      # 데이터링크 로스 failsafe 끄기
pxh> param set FD_FAIL_R 0        # roll 실패 검출 끄기 (곡예비행 필수)
pxh> param set FD_FAIL_P 0        # pitch 실패 검출 끄기 (곡예비행 필수)
pxh> param save
```

`FD_FAIL_R` / `FD_FAIL_P`는 기본 60도입니다. 파워루프는 피치가 180도까지 가므로
FailureDetector가 "기체 전복"으로 판단해 flight termination을 발동합니다
(`Preflight Fail: Attitude failure (roll)` → `Landing at current position`).
0으로 두면 검사가 비활성화됩니다.

`NAV_DLL_ACT`를 빼먹으면 `Preflight Fail: No connection to the GCS`로 arming이
막힙니다. PX4의 `rcAndDataLinkCheck.cpp`는 `NAV_DLL_ACT > 0`일 때 GCS 연결을
모든 모드의 arming 필수 조건으로 취급합니다. QGroundControl을 띄워도 해결됩니다.

`param save`를 빠뜨리면 SITL 재시작 시 전부 날아갑니다.

**확인:** `commander check` → `Preflight check: OK`

## 2. 실행

터미널 3개:

```bash
# 1) SITL
cd ~/PX4-Autopilot && make px4_sitl gz_x500

# 2) DDS agent
MicroXRCEAgent udp4 -p 8888

# 3) 브리지
source ~/ps2rl_ws/install/setup.bash
ros2 launch ps2rl_px4_bridge ps2rl_sitl.launch.py \
  ps2rl_path:=$HOME/PS2-RL \
  run_dir:=$HOME/PS2-RL/checkpoints/deployed_ps2/quadrotor_ps2_learned
```

브리지는 자동으로 `BOOT → ARM → TAKEOFF → LINEUP → DASH → POLICY → RECOVER → LAND`를 진행합니다.

### 비행 시퀀스가 이렇게 생긴 이유

파워루프 레퍼런스는 **정지 상태에서 시작하지 않습니다.** `t=0`에 이미 고도 0.5 m에서
4.5 m/s로 날고 있어야 합니다. 그래서 `LINEUP`이 진입점 상류 10 m에 자리를 잡고,
`DASH`가 PX4 위치제어로 진입 속도까지 가속한 뒤, 진입 평면을 통과하는 순간
정책 시계를 `t=0`으로 열고 body-rate 제어권을 넘깁니다.

`DASH`가 반복적으로 타임아웃되면 `lineup_distance`를 늘리거나 `dash_speed_tol`을 완화하세요.

---

## 3. 추력 캘리브레이션 (권장 필수)

`thrust_model: "linear"`는 호버에서만 정확합니다. 파워루프는 최대 **2.4 g**를 요구하는데
이 영역에서 선형 근사는 크게 어긋나고, 그 오차가 정책이 학습 중 본 적 없는 추적 오차로
그대로 들어갑니다.

```bash
ros2 launch ps2rl_px4_bridge thrust_calib.launch.py \
  calib_altitude:=15.0 output_yaml:=/tmp/thrust_fit.yaml
```

안전 고도로 상승 → 짧은 개루프 추력 펄스를 계단식으로 인가 → 각 펄스 중 IMU가 읽는 바디 z
비추력을 기록 → 2차 피팅. 나온 `thrust_k0/k1/k2`를 `config/bridge.yaml`에 넣고
`thrust_model: "quadratic"`으로 바꾸세요.

캘리브레이션이 **"full thrust에서 2.4 g에 못 미친다"**고 경고하면, x500 모델 SDF의
`motorConstant`를 올리거나 추중비가 더 높은 기체를 쓰셔야 합니다. 이건 튜닝으로
해결되는 문제가 아닙니다.

---

## 4. 시뮬레이션과 학습 모델의 갭

PS2-RL의 모델은 **각속도 명령이 즉시 반영된다**고 가정합니다 (`ẋ = f(x) + g(x)u`,
`u`에 ω가 직접 들어감). PX4의 내부 rate PID는 그렇지 않고, 이 지연은 CBF 보증이
기대는 전제를 깹니다. 항력과 모터 다이내믹스도 모델에 없습니다.

최소한 아래 두 가지는 하고 시작하세요:

1. **Rate 루프를 빠르게** — `MC_ROLLRATE_P`, `MC_PITCHRATE_P`, `MC_ROLLRATE_D` 등을
   올려서 명령 추종 지연을 줄입니다. 620°/s 기동이라 기본 게인으론 부족합니다.
2. **안전집합에 마진** — 첫 비행은 학습 때의 `z_max=3.0`보다 낮춰서 검증하세요.
   `run_dir`의 `configs.json`에서 `cbf.z_max`를 2.6~2.7로 복사본을 만들어 쓰면 됩니다.

여유가 되면 rate 루프의 1차 지연을 상태에 포함한 모델로 Phase-2를 재학습하는 게 정석입니다.

### 로그 분석

`log_csv`에 매 스텝 기록됩니다: 상태, 레퍼런스 위치, `u_raw` vs `u_safe`, QP slack,
추력 포화 여부, 추론 지연.

- `slack > 0`이 지속 → QP가 안전 행을 완화하고 있음. 모델 갭이 크다는 신호
- `saturated = 1`이 자주 → 추력 부족. 투영이 가정한 액션이 실제로 인가되지 않음
- `proj_norm`이 크다 → CIL이 정책을 강하게 잘라내는 중 (천장 근처에선 정상)

---

## 5. 파일 구성

```
ps2rl_px4_bridge/
├── frame_transforms.py   NED/FRD ↔ ENU/FLU. 행렬식과 대조하는 self-test 포함
├── thrust_model.py       a_cmd [m/s²] ↔ 정규화 추력. linear / quadratic
├── policy_runner.py      configs.json + weights.pkl 로딩, 관측 조립, CIL 실행
├── bridge_node.py        상태기계 + 50 Hz 정책 / 100 Hz setpoint 스트림
└── thrust_calib_node.py  추력 스윕 및 2차 피팅
```

`frame_transforms.py`는 노드 기동 시 self-test를 실행합니다. 좌표계는 이런 통합에서
가장 흔한 실패 지점이고, 조용히 틀리면 원인 모를 추락으로만 나타나기 때문입니다.

## 안전 워치독

`POLICY` 구간에서 아래 중 하나라도 걸리면 즉시 위치 홀드로 중단합니다:

- 오도메트리 100 ms 이상 정체
- `z > z_max + z_abort_margin`
- `arena_radius` 이탈
- 정책 출력에 NaN/Inf
- arming 또는 offboard 모드 상실

## 주의

이 패키지는 **SITL 검증용**입니다. 실기체에 올리기 전에 추력 캘리브레이션,
rate 루프 튜닝, 안전집합 마진, 그리고 충분한 시뮬레이션 로그 분석을 먼저 끝내세요.

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.dynamics import Dynamics


class Box:

    def __init__(self, low, high):

        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.shape = self.low.shape

    def sample(self):

        return np.random.uniform(self.low, self.high).astype(np.float32)


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def normalize_quat(q):

    q = np.array(q, dtype=float)
    q_norm = np.linalg.norm(q)

    if q_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / q_norm


def canonicalize_quat(q):
    """
    Canonicalize the quaternion sign so that q_w >= 0 (FIX #2).

    q and -q are the same rotation, but appear as opposite input vectors. The
    powerloop performs a full 360-degree flip, so q_w crosses zero and ~half
    the trajectory has q_w < 0. Feeding the raw sign makes the network waste
    capacity learning that antipodal inputs mean the same attitude, and creates
    a discontinuity in the observation right where the flip happens. Negating
    all four components when q_w < 0 removes it.
    """

    q = normalize_quat(q)

    if q[0] < 0.0:
        q = -q

    return q


def quat_conjugate(q):

    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mult(q1, q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def quat_from_axis_angle(axis, angle):

    axis = np.array(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)

    half = 0.5 * angle

    return normalize_quat(np.array([
        np.cos(half),
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half)
    ]))


def attitude_error(q, q_ref):
    """
    e_att = sgn(qe_w) qe_xyz, qe = q_ref ⊗ q^{-1}.
    """

    q = normalize_quat(q)
    q_ref = normalize_quat(q_ref)

    q_err = quat_mult(q_ref, quat_conjugate(q))
    q_err = normalize_quat(q_err)

    sign = 1.0
    if q_err[0] < 0.0:
        sign = -1.0

    return sign * q_err[1:4]


class PowerLoopEnv:
    """
    Vanilla SAC powerloop tracking environment without CIL.

    State:
        x = [p, v, q] in R^10

    Action:
        u = [a_cmd, wx, wy, wz]
        a_cmd in [0, 4g]
        body rates in [-18, 18] rad/s

    Episode horizon:
        106 steps at dt = 0.02 sec
    """

    def __init__(
            self,
            seed=0,
            horizon=106,
            term_pos_err=1.0,
            term_att_deg=90.0,
            reward_overrides=None,
            ref_mode="flat",
            survival_bonus=0.0
    ):
        # ref_mode selects the reference trajectory:
        #   "circular": horizontal differentially-flat circular trajectory with
        #          fixed yaw. This is the easiest feasible reference. It uses a
        #          constant-altitude circle, yaw psi=0, thrust direction from
        #          a_ref + g e3, and body-rate reference from finite-difference
        #          attitude kinematics. The default radius/speed below are well
        #          inside thrust and body-rate limits.
        #   "flat" (Version A): vertical differential-flatness loop. Attitude is
        #          derived from the required thrust direction. This is feasible
        #          but much harder because the vehicle can approach inversion.
        #   "flip" (Version B): geometric vertical circle + independent 360 deg
        #          flip. This reference is intentionally not dynamically feasible.
        self.ref_mode = ref_mode

        # Survival bonus (added to every step's reward). The tracking reward is
        # always negative, so with early termination (which zeros the future),
        # a state's value satisfies V(diverge)=0 > V(keep tracking)<0: the agent
        # is incentivized to crash on purpose to escape the stream of negative
        # rewards. Adding a positive constant per step makes a well-tracked step
        # net-positive, so surviving accumulates value and terminating (value 0)
        # becomes worse than living. This removes the "suicidal agent" motive.
        # It does not change the tracking objective (only the reward zero-point),
        # and it is NOT subtracted from the cost_* fields in last_info, so the
        # logged tracking errors are unaffected.
        self.survival_bonus = survival_bonus

        self.rng = np.random.default_rng(seed)

        self.dyn = Dynamics()
        self.dt = self.dyn.del_t
        self.g = self.dyn.g
        self.horizon = horizon
        self.step_count = 0

        # Early-termination thresholds (fix C). The vanilla tracker has no
        # safety layer, so an uncontrolled policy lets the state diverge and the
        # quadratic tracking cost explodes (per-step cost from ~20 near the
        # reference to ~90,000 when |v| reaches ~150 m/s). Those far-field states
        # dominate the critic's regression targets and destabilize SAC (alpha
        # ran away to 500+ in the un-terminated run). Ending the episode as a
        # REAL terminal the moment the state diverges keeps those states out of
        # the buffer entirely, which fixes the instability at its source rather
        # than by rescaling the reward.
        #
        # OR semantics: terminate if EITHER position OR attitude error exceeds
        # its bound, since divergence in either one produces the blow-up.
        #
        # term_att_deg is stored as the equivalent |e_att| = 2 sin(theta/2)
        # value so the check is a direct comparison against attitude_error(...).
        self.term_pos_err = term_pos_err
        self.term_att_deg = term_att_deg
        self.term_att_err = 2.0 * np.sin(np.deg2rad(term_att_deg) / 2.0)

        self.radius = 1.5
        self.center = np.array([0.0, 0.0, 2.0])
        self.v_tangent = 4.5
        self.omega_loop = self.v_tangent / self.radius

        # Easier fixed-yaw horizontal circle. These values are deliberately
        # conservative:
        #   centripetal acceleration = v^2 / r = 2.67 m/s^2
        #   thrust = sqrt(g^2 + a_c^2) = 10.17 m/s^2 < 4g
        #   tilt = atan(a_c / g) = 15.2 deg
        self.circular_radius = 1.5
        self.circular_speed = 2.0
        self.circular_omega = self.circular_speed / self.circular_radius
        self.circular_center = np.array([0.0, 0.0, 2.0])
        self.circular_yaw = 0.0

        # For the circular task, use one full lap by default. Keep the old
        # 106-step horizon for the vertical loop modes.
        if self.ref_mode == "circular" and horizon == 106:
            self.horizon = int(np.ceil((2.0 * np.pi / self.circular_omega) / self.dt)) + 1

        self.action_low = np.array([0.0, -18.0, -18.0, -18.0], dtype=np.float32)
        self.action_high = np.array([4.0 * self.g, 18.0, 18.0, 18.0], dtype=np.float32)

        # Observation is the 10-D raw state augmented with the 9-D tracking
        # error block (e_p, e_v, e_att); see get_obs (FIX #1).
        self.obs_dim = 19

        self.action_space = Box(self.action_low, self.action_high)
        self.observation_space = Box(
            -np.inf * np.ones(self.obs_dim),
            np.inf * np.ones(self.obs_dim)
        )

        self.wp_xy = 2.5
        self.wp_z = 2.0
        self.wv = 4.0
        self.watt = 16.0
        self.Womega = np.diag([0.10, 0.20, 0.05])
        self.wa = 0.0 # 0.01
        self.wOmega = 0.0 # 0.01

        # Diagnostic override: allow any reward weight to be replaced without
        # editing this file. reward_overrides is a dict like
        # {"watt": 100.0, "Womega": np.zeros((3,3))}.
        if reward_overrides:
            for key, val in reward_overrides.items():
                setattr(self, key, val)

        self.last_info = {}

    def reset_dynamics_state(self, state):

        self.dyn.state = state.copy()

        if hasattr(self.dyn, "curr_step"):
            self.dyn.curr_step = 0

        if hasattr(self.dyn, "xlist"):
            self.dyn.xlist = []

        if hasattr(self.dyn, "vlist"):
            self.dyn.vlist = []

        if hasattr(self.dyn, "qlist"):
            self.dyn.qlist = []

    def _rotmat_to_quat(self, R):
        """Convert a rotation matrix (columns = body axes in world) to [w,x,y,z]."""
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
            if i == 0:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / S; x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S; z = (R[0, 2] + R[2, 0]) / S
            elif i == 1:
                S = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / S; x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S; z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
                w = (R[1, 0] - R[0, 1]) / S; x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S; z = 0.25 * S
        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)

    def _flat_output_attitude(self, a_ref, yaw):
        """Return attitude from flat outputs p_ddot and fixed yaw.

        Dynamics convention in this repo:
            v_dot = -g e3 + R(q) [0, 0, a_cmd]^T

        Therefore the desired body z-axis must align with
            f_des = a_ref + g e3.

        For fixed yaw, use the standard flatness construction with
        b2_des = [-sin(yaw), cos(yaw), 0].
        """
        f_des = np.asarray(a_ref, dtype=float) + np.array([0.0, 0.0, self.g])
        thrust = np.linalg.norm(f_des)
        if thrust < 1e-9:
            thrust = 1e-9
        zb = f_des / thrust

        b2_des = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        xb = np.cross(b2_des, zb)
        nxb = np.linalg.norm(xb)
        if nxb < 1e-9:
            xb = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        else:
            xb = xb / nxb
        yb = np.cross(zb, xb)
        yb = yb / (np.linalg.norm(yb) + 1e-9)

        R = np.column_stack([xb, yb, zb])
        q = self._rotmat_to_quat(R)
        return R, q, thrust

    def _vee(self, M):
        return np.array([
            M[2, 1],
            M[0, 2],
            M[1, 0]
        ])

    def _circular_kinematics(self, t):
        r = self.circular_radius
        w = self.circular_omega
        c, s = np.cos(w * t), np.sin(w * t)

        p_ref = self.circular_center + np.array([r * c, r * s, 0.0])
        v_ref = np.array([-r * w * s, r * w * c, 0.0])
        a_ref = np.array([-r * w**2 * c, -r * w**2 * s, 0.0])
        return p_ref, v_ref, a_ref

    def _reference_circular(self, k):
        """Fixed-yaw, differentially-flat, feasible horizontal circle.

        Flat outputs:
            p(t) = [r cos(wt), r sin(wt), z0]
            yaw(t) = 0

        With r=1.5 m and v=2.0 m/s, the commanded thrust and body
        rates are well within the input limits [0, 4g] and [-18, 18] rad/s.
        """
        t = k * self.dt
        p_ref, v_ref, a_ref = self._circular_kinematics(t)
        R, q_ref, thrust = self._flat_output_attitude(a_ref, self.circular_yaw)

        # Body-rate reference from R_dot = R * omega_hat. Use central finite
        # difference. This avoids deriving snap/jerk formulas and stays accurate
        # at dt=0.02 for this smooth circle.
        t_minus = max(0.0, t - self.dt)
        t_plus = t + self.dt
        _, _, a_minus = self._circular_kinematics(t_minus)
        _, _, a_plus = self._circular_kinematics(t_plus)
        R_minus, _, _ = self._flat_output_attitude(a_minus, self.circular_yaw)
        R_plus, _, _ = self._flat_output_attitude(a_plus, self.circular_yaw)

        if t <= 0.0:
            R_dot = (R_plus - R) / self.dt
        else:
            R_dot = (R_plus - R_minus) / (2.0 * self.dt)

        omega_hat = R.T @ R_dot
        omega_ref = self._vee(omega_hat)

        return p_ref, v_ref, q_ref, omega_ref

    def _reference_flat(self, k):
        """Version A: differential-flatness circle (feasible)."""
        t = k * self.dt
        w = self.omega_loop
        th = -0.5 * np.pi + w * t
        c, s = np.cos(th), np.sin(th)
        r = self.radius

        p_ref = self.center + r * np.array([c, 0.0, s])
        v_ref = r * w * np.array([-s, 0.0, c])
        a_ref = r * w**2 * np.array([-c, 0.0, -s])          # acceleration
        j_ref = r * w**3 * np.array([s, 0.0, -c])           # jerk

        # Thrust vector must produce a_ref against gravity; body-z aligns to it.
        f = a_ref + np.array([0.0, 0.0, self.g])
        thrust = np.linalg.norm(f)
        zb = f / thrust

        # Planar loop: keep body-y along world-y, resolve x from z.
        xb = np.cross(np.array([0.0, 1.0, 0.0]), zb)
        nxb = np.linalg.norm(xb)
        xb = np.array([1.0, 0.0, 0.0]) if nxb < 1e-6 else xb / nxb
        yb = np.cross(zb, xb)
        R = np.column_stack([xb, yb, zb])
        q_ref = self._rotmat_to_quat(R)

        # Body rates from jerk (yaw-flat = 0).
        h = (j_ref - np.dot(zb, j_ref) * zb) / thrust
        omega_ref = np.array([-np.dot(h, yb), np.dot(h, xb), 0.0])

        return p_ref, v_ref, q_ref, omega_ref

    def _reference_flip(self, k):
        """Version B: geometric circle + independent 360 deg flip (INFEASIBLE
        target, pursued as closely as possible)."""
        t = k * self.dt
        w = self.omega_loop
        theta = -0.5 * np.pi + w * t

        p_ref = np.array([
            self.center[0] + self.radius * np.cos(theta),
            self.center[1],
            self.center[2] + self.radius * np.sin(theta)
        ])
        v_ref = np.array([
            -self.radius * w * np.sin(theta),
            0.0,
            self.radius * w * np.cos(theta)
        ])
        flip_angle = theta + 0.5 * np.pi
        q_ref = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), flip_angle)
        omega_ref = np.array([0.0, w, 0.0])
        return p_ref, v_ref, q_ref, omega_ref

    def reference(self, k):
        """
        Returns (p_ref, v_ref, q_ref, omega_ref) at index k.

        ref_mode = "circular" -> fixed-yaw feasible horizontal circle
        ref_mode = "flat"     -> feasible vertical flatness loop
        ref_mode = "flip"     -> infeasible circle+360-flip target
        """
        if self.ref_mode == "circular":
            return self._reference_circular(k)
        if self.ref_mode == "flip":
            return self._reference_flip(k)
        return self._reference_flat(k)

    def get_obs(self):
        """
        Observation (19-D):

            [ p, v, q_canon,          # raw state (10)
              e_p (=p - p_ref),       # position error (3)
              e_v (=v - v_ref),       # velocity error (3)
              e_att ]                 # attitude error (3)

        FIX #1 (Markov observation): the reward at step k scores the state
        against reference(k), but the raw state [p, v, q] contains neither k
        nor reference(k). Two timesteps with the same physical state but
        different loop phase then look identical to the policy while being
        scored against different targets -- the optimal tracker is literally
        unrepresentable. Appending the reference-relative errors fixes this.

        The reference itself does not need to be added separately: the raw
        state is still present, so the policy can recover p_ref = p - e_p,
        v_ref = v - e_v, etc. The errors are exactly the hidden information.

        The observation uses the reference at the CURRENT step_count, i.e. the
        reference the NEXT action will be scored against, so the policy sees
        the target it is about to be judged on. See the note in step().

        FIX #2: the quaternion in the raw-state block is sign-canonicalized.
        """

        state = np.array(self.dyn.state, dtype=float)
        state[6:10] = canonicalize_quat(state[6:10])

        p = state[0:3]
        v = state[3:6]
        q = state[6:10]

        ref_index = min(self.step_count, self.horizon - 1)
        p_ref, v_ref, q_ref, omega_ref = self.reference(ref_index)

        e_p = p - p_ref
        e_v = v - v_ref
        e_att = attitude_error(q, q_ref)

        obs = np.concatenate([
            state,     # 10
            e_p,       # 3
            e_v,       # 3
            e_att      # 3
        ]).astype(np.float32)

        return obs

    def get_raw_state(self):
        """
        The raw 10-D physical state [p, v, q_canon], WITHOUT the tracking-error
        augmentation that get_obs adds.

        This is what PS2-RL Phase I expects when it builds its design region
        Omega from vanilla-SAC rollouts (App. F.3.3): it needs actual visited
        states, not policy observations. Use this, not get_obs(), when saving
        traces for Phase I.
        """

        state = np.array(self.dyn.state, dtype=float)
        state[6:10] = canonicalize_quat(state[6:10])
        return state

    def reset(self):

        self.step_count = 0

        p_ref, v_ref, q_ref, omega_ref = self.reference(0)

        p0 = p_ref.copy()
        p0 += self.rng.uniform(-0.1, 0.1, size=3)

        state = np.array([
            p0[0], p0[1], p0[2],
            v_ref[0], v_ref[1], v_ref[2],
            q_ref[0], q_ref[1], q_ref[2], q_ref[3]
        ])

        self.reset_dynamics_state(state)
        return self.get_obs()

    def reward(self, state, action, ref_index):

        p = state[0:3]
        v = state[3:6]
        q = state[6:10]

        a_cmd = action[0]
        omega_cmd = action[1:4]

        p_ref, v_ref, q_ref, omega_ref = self.reference(ref_index)

        e_p_xy = p[0:2] - p_ref[0:2]
        e_p_z = p[2] - p_ref[2]
        e_v = v - v_ref
        e_att = attitude_error(q, q_ref)
        e_omega = omega_cmd - omega_ref

        cost_p_xy = self.wp_xy * np.dot(e_p_xy, e_p_xy)
        cost_p_z = self.wp_z * e_p_z**2
        cost_v = self.wv * np.dot(e_v, e_v)
        cost_att = self.watt * np.dot(e_att, e_att)
        cost_omega_ref = e_omega.T @ self.Womega @ e_omega
        cost_a = self.wa * a_cmd**2
        cost_omega = self.wOmega * np.dot(omega_cmd, omega_cmd)

        cost = (
            cost_p_xy
            + cost_p_z
            + cost_v
            + cost_att
            + cost_omega_ref
            + cost_a
            + cost_omega
        )

        self.last_info = {
            "cost": cost,
            "cost_p_xy": cost_p_xy,
            "cost_p_z": cost_p_z,
            "cost_v": cost_v,
            "cost_att": cost_att,
            "cost_omega_ref": cost_omega_ref,
            "cost_a": cost_a,
            "cost_omega": cost_omega,
            "p_ref": p_ref,
            "v_ref": v_ref,
            "q_ref": q_ref,
            "omega_ref": omega_ref,
            "tracking_error_pos": np.linalg.norm(p - p_ref),
            "tracking_error_v": np.linalg.norm(e_v),
            "tracking_error_att": np.linalg.norm(e_att),
            "unsafe": float(p[2] > 3.0),
        }

        # Survival bonus is added to the RETURNED reward only (not to the
        # cost_* fields above), so logged tracking errors stay unaffected.
        return -float(cost) + self.survival_bonus

    def step(self, action):

        action = np.array(action, dtype=float)
        action = np.clip(action, self.action_low, self.action_high)

        next_state = self.dyn.step(action).copy()
        next_state[6:10] = normalize_quat(next_state[6:10])

        self.step_count += 1

        ref_index = min(self.step_count, self.horizon - 1)
        reward = self.reward(next_state, action, ref_index)

        # Divergence check (fix C). last_info was just populated by reward().
        pos_err = self.last_info["tracking_error_pos"]
        att_err = self.last_info["tracking_error_att"]

        diverged = (
            pos_err > self.term_pos_err
            or att_err > self.term_att_err
        )

        reached_horizon = self.step_count >= self.horizon

        # A divergence is a REAL terminal: the rollout genuinely failed, so its
        # value should NOT bootstrap (mask = 0 in the trainer). The horizon is a
        # time limit -> truncation, which keeps bootstrapping (mask = 1). These
        # are mutually exclusive; divergence takes precedence if both trip on the
        # same step.
        terminated = bool(diverged)
        truncated = bool(reached_horizon and not diverged)

        done = terminated or truncated

        obs = self.get_obs()
        info = self.last_info.copy()
        info["step"] = self.step_count
        info["truncated"] = truncated
        info["terminated"] = terminated
        info["diverged"] = bool(diverged)

        return obs, reward, done, info
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

    def __init__(self, seed=0, horizon=106):

        self.rng = np.random.default_rng(seed)

        self.dyn = Dynamics()
        self.dt = self.dyn.del_t
        self.g = self.dyn.g
        self.horizon = horizon
        self.step_count = 0

        self.radius = 1.5
        self.center = np.array([0.0, 0.0, 2.0])
        self.v_tangent = 4.5
        self.omega_loop = self.v_tangent / self.radius

        self.action_low = np.array([0.0, -18.0, -18.0, -18.0], dtype=np.float32)
        self.action_high = np.array([4.0 * self.g, 18.0, 18.0, 18.0], dtype=np.float32)

        self.action_space = Box(self.action_low, self.action_high)
        self.observation_space = Box(-np.inf * np.ones(10), np.inf * np.ones(10))

        self.wp_xy = 2.5
        self.wp_z = 2.0
        self.wv = 4.0
        self.watt = 16.0
        self.Womega = np.diag([0.10, 0.20, 0.05])
        self.wa = 0.01
        self.wOmega = 0.01

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

    def reference(self, k):
        """
        Returns reference at index k.

        The loop is in the x-z plane and starts at the bottom:
            p_ref(0) = [0, 0, 0.5]
            v_ref(0) = [4.5, 0, 0]
        """

        t = k * self.dt
        theta = -0.5 * np.pi + self.omega_loop * t

        p_ref = np.array([
            self.center[0] + self.radius * np.cos(theta),
            self.center[1],
            self.center[2] + self.radius * np.sin(theta)
        ])

        v_ref = np.array([
            -self.radius * self.omega_loop * np.sin(theta),
            0.0,
            self.radius * self.omega_loop * np.cos(theta)
        ])

        flip_angle = theta + 0.5 * np.pi
        q_ref = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), flip_angle)

        omega_ref = np.array([0.0, self.omega_loop, 0.0])

        return p_ref, v_ref, q_ref, omega_ref

    def get_obs(self):

        state = np.array(self.dyn.state, dtype=np.float32)
        state[6:10] = normalize_quat(state[6:10]).astype(np.float32)
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

        return -float(cost)

    def step(self, action):

        action = np.array(action, dtype=float)
        action = np.clip(action, self.action_low, self.action_high)

        next_state = self.dyn.step(action).copy()
        next_state[6:10] = normalize_quat(next_state[6:10])

        self.step_count += 1

        ref_index = min(self.step_count, self.horizon - 1)
        reward = self.reward(next_state, action, ref_index)

        done = self.step_count >= self.horizon

        obs = self.get_obs()
        info = self.last_info.copy()
        info["step"] = self.step_count

        return obs, reward, done, info

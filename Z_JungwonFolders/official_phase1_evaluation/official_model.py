"""NumPy port of the official PS2-RL quadrotor Phase-I model and DLQR.

The implementation follows the released PS2-RL state/action conventions:
state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
action = [a_cmd, omega_x, omega_y, omega_z]

The paper evaluation uses a fixed forward-Euler step. The only dependency on
this repository is the learned PyTorch actor; official dynamics and base-set
logic are kept in this folder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import solve_discrete_are


@dataclass(frozen=True)
class OfficialQuadrotorConfig:
    dt: float = 0.02
    gravity: float = 9.81
    z_des: float = 2.0
    z_max: float = 3.0
    a_cmd_min: float = 0.0
    a_cmd_max: float = 39.24
    omega_max: float = 18.0
    base_set_c: float = 8.0
    base_set_smooth_gain: float = 10.0
    lqr_q_z: float = 1.0
    lqr_q_vx: float = 0.16
    lqr_q_vy: float = 0.16
    lqr_q_vz: float = 0.4
    lqr_q_thetax: float = 0.8
    lqr_q_thetay: float = 0.8
    lqr_q_thetaz: float = 0.16
    lqr_r_a_cmd: float = 0.02
    lqr_r_omega_x: float = 0.012
    lqr_r_omega_y: float = 0.012
    lqr_r_omega_z: float = 0.004

    @classmethod
    def from_reset_payload(cls, payload: dict | None) -> "OfficialQuadrotorConfig":
        raw = payload or {}

        def value(key: str, default: float) -> float:
            try:
                return float(raw.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        gravity = value("gravity", cls.gravity)
        return cls(
            dt=value("dt", cls.dt),
            gravity=gravity,
            z_des=value("z_des", cls.z_des),
            z_max=value("z_max", cls.z_max),
            a_cmd_min=value("a_cmd_min", cls.a_cmd_min),
            a_cmd_max=value("a_cmd_max", 4.0 * gravity),
            omega_max=value("omega_max", cls.omega_max),
            base_set_c=value("base_set_c", cls.base_set_c),
            base_set_smooth_gain=value(
                "base_set_smooth_gain", cls.base_set_smooth_gain
            ),
            lqr_q_z=value("lqr_q_z", cls.lqr_q_z),
            lqr_q_vx=value("lqr_q_vx", cls.lqr_q_vx),
            lqr_q_vy=value("lqr_q_vy", cls.lqr_q_vy),
            lqr_q_vz=value("lqr_q_vz", cls.lqr_q_vz),
            lqr_q_thetax=value("lqr_q_thetax", cls.lqr_q_thetax),
            lqr_q_thetay=value("lqr_q_thetay", cls.lqr_q_thetay),
            lqr_q_thetaz=value("lqr_q_thetaz", cls.lqr_q_thetaz),
            lqr_r_a_cmd=value("lqr_r_a_cmd", cls.lqr_r_a_cmd),
            lqr_r_omega_x=value("lqr_r_omega_x", cls.lqr_r_omega_x),
            lqr_r_omega_y=value("lqr_r_omega_y", cls.lqr_r_omega_y),
            lqr_r_omega_z=value("lqr_r_omega_z", cls.lqr_r_omega_z),
        )

    @property
    def action_low(self) -> np.ndarray:
        return np.array(
            [self.a_cmd_min, -self.omega_max, -self.omega_max, -self.omega_max],
            dtype=np.float64,
        )

    @property
    def action_high(self) -> np.ndarray:
        return np.array(
            [self.a_cmd_max, self.omega_max, self.omega_max, self.omega_max],
            dtype=np.float64,
        )

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if self.a_cmd_min >= self.a_cmd_max:
            raise ValueError("a_cmd_min must be smaller than a_cmd_max")
        if self.omega_max <= 0.0:
            raise ValueError("omega_max must be positive")
        if self.base_set_c <= 0.0:
            raise ValueError("base_set_c must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    bad = norm[:, 0] < 1e-12
    norm[bad] = 1.0
    out = q / norm
    out[bad] = np.array([1.0, 0.0, 0.0, 0.0])
    return out[0] if single else out


def clip_physical_action(
    action: np.ndarray, cfg: OfficialQuadrotorConfig
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    single = action.ndim == 1
    if single:
        action = action[None, :]
    if action.ndim != 2 or action.shape[1] != 4:
        raise ValueError(f"Expected action shape (4,) or (N, 4), got {action.shape}")
    action = np.nan_to_num(
        action,
        nan=0.0,
        posinf=np.broadcast_to(cfg.action_high, action.shape),
        neginf=np.broadcast_to(cfg.action_low, action.shape),
    )
    out = np.clip(action, cfg.action_low, cfg.action_high)
    return out[0] if single else out


def thrust_axis_world(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    qw, qx, qy, qz = q.T
    axis = np.stack(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        axis=1,
    )
    return axis[0] if single else axis


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    omega = np.asarray(omega, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]
        omega = omega[None, :]
    qw, qx, qy, qz = q.T
    wx, wy, wz = omega.T
    q_dot = 0.5 * np.stack(
        [
            -qx * wx - qy * wy - qz * wz,
            qw * wx - qz * wy + qy * wz,
            qz * wx + qw * wy - qx * wz,
            -qy * wx + qx * wy + qw * wz,
        ],
        axis=1,
    )
    return q_dot[0] if single else q_dot


def step_official_euler(
    state: np.ndarray,
    action: np.ndarray,
    cfg: OfficialQuadrotorConfig,
) -> np.ndarray:
    """Advance one official fixed forward-Euler step."""
    state = np.asarray(state, dtype=np.float64)
    single = state.ndim == 1
    if single:
        state = state[None, :]
    if state.ndim != 2 or state.shape[1] != 10:
        raise ValueError(f"Expected state shape (10,) or (N, 10), got {state.shape}")

    action = clip_physical_action(action, cfg)
    if action.ndim == 1:
        action = action[None, :]
    if action.shape[0] != state.shape[0]:
        raise ValueError("State and action batch sizes do not match")

    derivative = np.zeros_like(state)
    derivative[:, 0:3] = state[:, 3:6]
    derivative[:, 3:6] = (
        -cfg.gravity * np.array([0.0, 0.0, 1.0])[None, :]
        + action[:, 0:1] * thrust_axis_world(state[:, 6:10])
    )
    derivative[:, 6:10] = quaternion_derivative(
        state[:, 6:10], action[:, 1:4]
    )

    next_state = state + cfg.dt * derivative
    next_state[:, 6:10] = normalize_quaternion(next_state[:, 6:10])
    return next_state[0] if single else next_state


def safety_margin(state: np.ndarray, cfg: OfficialQuadrotorConfig) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    return cfg.z_max - state[..., 2]


def is_safe(state: np.ndarray, cfg: OfficialQuadrotorConfig) -> np.ndarray:
    return safety_margin(state, cfg) >= 0.0


class OfficialDLQR:
    """Official forward-Euler hover DLQR and ellipsoidal base set."""

    def __init__(self, cfg: OfficialQuadrotorConfig):
        cfg.validate()
        self.cfg = cfg
        dt = cfg.dt
        g = cfg.gravity

        # xe = [z-z_des, vx, vy, vz, theta_x, theta_y, theta_z]
        a_cont = np.zeros((7, 7), dtype=np.float64)
        a_cont[0, 3] = 1.0
        a_cont[1, 5] = -g
        a_cont[2, 4] = g

        b_cont = np.zeros((7, 4), dtype=np.float64)
        b_cont[3, 0] = 1.0
        b_cont[4, 1] = -1.0
        b_cont[5, 2] = -1.0
        b_cont[6, 3] = -1.0

        self.ad = np.eye(7) + dt * a_cont
        self.bd = dt * b_cont
        self.q_matrix = np.diag(
            [
                cfg.lqr_q_z,
                cfg.lqr_q_vx,
                cfg.lqr_q_vy,
                cfg.lqr_q_vz,
                cfg.lqr_q_thetax,
                cfg.lqr_q_thetay,
                cfg.lqr_q_thetaz,
            ]
        )
        self.r_matrix = np.diag(
            [
                cfg.lqr_r_a_cmd,
                cfg.lqr_r_omega_x,
                cfg.lqr_r_omega_y,
                cfg.lqr_r_omega_z,
            ]
        )
        p = solve_discrete_are(self.ad, self.bd, self.q_matrix, self.r_matrix)
        self.p_matrix = 0.5 * (np.real(p) + np.real(p).T)
        self.k_matrix = np.linalg.solve(
            self.r_matrix + self.bd.T @ self.p_matrix @ self.bd,
            self.bd.T @ self.p_matrix @ self.ad,
        )
        self.u_star = np.array([cfg.gravity, 0.0, 0.0, 0.0])

    def error_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        single = state.ndim == 1
        if single:
            state = state[None, :]
        q = normalize_quaternion(state[:, 6:10])
        q_err = q.copy()
        q_err[:, 1:4] *= -1.0
        sign = np.where(q_err[:, 0] >= 0.0, 1.0, -1.0)
        theta = 2.0 * sign[:, None] * q_err[:, 1:4]
        error = np.column_stack(
            [
                state[:, 2] - self.cfg.z_des,
                state[:, 3],
                state[:, 4],
                state[:, 5],
                theta[:, 0],
                theta[:, 1],
                theta[:, 2],
            ]
        )
        return error[0] if single else error

    def quadratic_value(self, state: np.ndarray) -> np.ndarray:
        error = self.error_state(state)
        single = error.ndim == 1
        if single:
            error = error[None, :]
        value = np.einsum("bi,ij,bj->b", error, self.p_matrix, error)
        return value[0] if single else value

    def base_margin(self, state: np.ndarray) -> np.ndarray:
        return self.cfg.base_set_c - self.quadratic_value(state)

    def contains(self, state: np.ndarray) -> np.ndarray:
        return self.base_margin(state) >= 0.0

    def action(self, state: np.ndarray) -> np.ndarray:
        error = self.error_state(state)
        single = error.ndim == 1
        if single:
            error = error[None, :]
        control = self.u_star[None, :] - error @ self.k_matrix.T
        control = clip_physical_action(control, self.cfg)
        return control[0] if single else control

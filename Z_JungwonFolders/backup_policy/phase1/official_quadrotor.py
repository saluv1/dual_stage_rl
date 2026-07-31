"""Numpy/SciPy reproduction of the official quadrotor Phase-I benchmark model.

This module contains only evaluation-side code. It does not modify the current
training dynamics, replay buffer, TD3 update, or saved checkpoints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are


def _cfg_value(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if value is None:
        return float(default)
    return float(value)


@dataclass(frozen=True)
class OfficialQuadrotorConfig:
    """Official Phase-I dynamics, set, action, and LQR parameters."""

    dt: float = 0.02
    gravity: float = 9.81
    z_des: float = 2.0
    z_max: float = 3.0
    a_cmd_min: float = 0.0
    a_cmd_max: float = 39.24
    omega_max: float = 18.0
    base_set_c: float = 8.0
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
    def from_reset_payload(
        cls, payload: dict[str, Any]
    ) -> "OfficialQuadrotorConfig":
        gravity = _cfg_value(payload, "gravity", 9.81)
        cfg = cls(
            dt=_cfg_value(payload, "dt", 0.02),
            gravity=gravity,
            z_des=_cfg_value(payload, "z_des", 2.0),
            z_max=_cfg_value(payload, "z_max", 3.0),
            a_cmd_min=_cfg_value(payload, "a_cmd_min", 0.0),
            a_cmd_max=_cfg_value(payload, "a_cmd_max", 4.0 * gravity),
            omega_max=_cfg_value(payload, "omega_max", 18.0),
            base_set_c=_cfg_value(payload, "base_set_c", 8.0),
            lqr_q_z=_cfg_value(payload, "lqr_q_z", 1.0),
            lqr_q_vx=_cfg_value(payload, "lqr_q_vx", 0.16),
            lqr_q_vy=_cfg_value(payload, "lqr_q_vy", 0.16),
            lqr_q_vz=_cfg_value(payload, "lqr_q_vz", 0.4),
            lqr_q_thetax=_cfg_value(payload, "lqr_q_thetax", 0.8),
            lqr_q_thetay=_cfg_value(payload, "lqr_q_thetay", 0.8),
            lqr_q_thetaz=_cfg_value(payload, "lqr_q_thetaz", 0.16),
            lqr_r_a_cmd=_cfg_value(payload, "lqr_r_a_cmd", 0.02),
            lqr_r_omega_x=_cfg_value(payload, "lqr_r_omega_x", 0.012),
            lqr_r_omega_y=_cfg_value(payload, "lqr_r_omega_y", 0.012),
            lqr_r_omega_z=_cfg_value(payload, "lqr_r_omega_z", 0.004),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.gravity <= 0.0:
            raise ValueError(f"gravity must be positive, got {self.gravity}")
        if not self.a_cmd_min < self.a_cmd_max:
            raise ValueError("a_cmd_min must be smaller than a_cmd_max")
        if self.omega_max <= 0.0:
            raise ValueError("omega_max must be positive")
        if self.base_set_c <= 0.0:
            raise ValueError("base_set_c must be positive")

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(
            [self.a_cmd_min, -self.omega_max, -self.omega_max, -self.omega_max],
            dtype=np.float64,
        )

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(
            [self.a_cmd_max, self.omega_max, self.omega_max, self.omega_max],
            dtype=np.float64,
        )


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q_arr))
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q_arr / norm


def quaternion_rate_matrix(q: np.ndarray) -> np.ndarray:
    """Return Xi(q) such that q_dot = 0.5 * Xi(q) * omega."""

    qw, qx, qy, qz = normalize_quaternion(q)
    return np.asarray(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ],
        dtype=np.float64,
    )


def thrust_axis_world(q: np.ndarray) -> np.ndarray:
    """World-frame direction of the body +z thrust axis."""

    qw, qx, qy, qz = normalize_quaternion(q)
    return np.asarray(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dtype=np.float64,
    )


def clip_physical_action(
    action: np.ndarray, cfg: OfficialQuadrotorConfig
) -> np.ndarray:
    u = np.nan_to_num(
        np.asarray(action, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if u.shape != (4,):
        raise ValueError(f"Expected physical action shape (4,), got {u.shape}")
    return np.clip(u, cfg.action_low, cfg.action_high)


def quadrotor_derivative(
    state: np.ndarray, action: np.ndarray, cfg: OfficialQuadrotorConfig
) -> np.ndarray:
    """Official 10D control-affine dynamics in numpy."""

    x = np.asarray(state, dtype=np.float64)
    if x.shape != (10,):
        raise ValueError(f"Expected state shape (10,), got {x.shape}")
    u = clip_physical_action(action, cfg)
    q = normalize_quaternion(x[6:10])

    derivative = np.zeros(10, dtype=np.float64)
    derivative[0:3] = x[3:6]
    derivative[3:6] = thrust_axis_world(q) * u[0]
    derivative[5] -= cfg.gravity
    derivative[6:10] = 0.5 * quaternion_rate_matrix(q) @ u[1:4]
    return derivative


def step_official_euler(
    state: np.ndarray, action: np.ndarray, cfg: OfficialQuadrotorConfig
) -> np.ndarray:
    """The fixed-step Euler transition used by the official Phase-I evaluator."""

    # The official JAX evaluator runs the held-out rollouts in float32.
    x = np.asarray(state, dtype=np.float32)
    u = clip_physical_action(action, cfg).astype(np.float32)
    derivative = quadrotor_derivative(x, u, cfg).astype(np.float32)
    x_next = x + np.float32(cfg.dt) * derivative
    x_next[6:10] = normalize_quaternion(x_next[6:10])
    return x_next


def step_rk45(
    state: np.ndarray, action: np.ndarray, cfg: OfficialQuadrotorConfig
) -> np.ndarray:
    """RK45 comparison using the same equations and action bounds."""

    x = np.asarray(state, dtype=np.float64)
    u = clip_physical_action(action, cfg)
    solution = solve_ivp(
        fun=lambda _t, y: quadrotor_derivative(y, u, cfg),
        t_span=(0.0, cfg.dt),
        y0=x,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"RK45 integration failed: {solution.message}")
    x_next = solution.y[:, -1]
    x_next[6:10] = normalize_quaternion(x_next[6:10])
    return x_next


def make_integrator(
    name: str, cfg: OfficialQuadrotorConfig
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if name == "official_euler":
        return lambda x, u: step_official_euler(x, u, cfg)
    if name == "rk45":
        return lambda x, u: step_rk45(x, u, cfg)
    raise ValueError(f"Unknown integrator '{name}'. Use official_euler or rk45.")


class OfficialDLQR:
    """Official forward-Euler hover DLQR and its ellipsoidal base set."""

    def __init__(self, cfg: OfficialQuadrotorConfig):
        self.cfg = cfg

        g = cfg.gravity
        a_cont = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -g, 0.0],
                [0.0, 0.0, 0.0, 0.0, g, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        b_cont = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

        self.a_d = np.eye(7, dtype=np.float64) + cfg.dt * a_cont
        self.b_d = cfg.dt * b_cont
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

        p = solve_discrete_are(self.a_d, self.b_d, self.q_matrix, self.r_matrix)
        p_symmetric = 0.5 * (np.real(p) + np.real(p).T)
        k = np.linalg.solve(
            self.r_matrix + self.b_d.T @ p_symmetric @ self.b_d,
            self.b_d.T @ p_symmetric @ self.a_d,
        )
        # The official DiscreteLQR stores the DARE certificate and gain at f32.
        self.p_matrix = np.asarray(p_symmetric, dtype=np.float32)
        self.k_matrix = np.asarray(k, dtype=np.float32)
        self.u_star = np.asarray([cfg.gravity, 0.0, 0.0, 0.0])

    def error_state(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32)
        q = normalize_quaternion(x[6:10]).astype(np.float32)
        # q_err = conjugate(q), then canonicalize q_err so its scalar part is
        # non-negative, exactly as the official implementation does.
        q_err = np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
        sign = 1.0 if q_err[0] >= 0.0 else -1.0
        theta_err = 2.0 * sign * q_err[1:4]
        return np.concatenate(
            [
                np.asarray([x[2] - self.cfg.z_des]),
                x[3:6],
                theta_err,
            ]
        )

    def quadratic_value(self, state: np.ndarray) -> float:
        error = self.error_state(state)
        return float(error.T @ self.p_matrix @ error)

    def base_margin(self, state: np.ndarray) -> float:
        return float(self.cfg.base_set_c - self.quadratic_value(state))

    def contains(self, state: np.ndarray, atol: float = 0.0) -> bool:
        return bool(self.base_margin(state) >= -float(atol))

    def action(self, state: np.ndarray) -> np.ndarray:
        control = self.u_star - self.k_matrix @ self.error_state(state)
        return clip_physical_action(control, self.cfg)


def safety_margin(state: np.ndarray, cfg: OfficialQuadrotorConfig) -> float:
    return float(cfg.z_max - np.asarray(state, dtype=np.float64)[2])


def is_safe(state: np.ndarray, cfg: OfficialQuadrotorConfig) -> bool:
    return bool(np.asarray(state, dtype=np.float64)[2] <= cfg.z_max)

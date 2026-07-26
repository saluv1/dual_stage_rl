"""Vertical power-loop reference for the reduced-order quadrotor model.

Geometry follows Deep Drone Acrobatics computeVerticalCircleTrajectory.
Quaternion convention is scalar-first: [qw, qx, qy, qz].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_EPS = 1e-8


@dataclass(frozen=True)
class PowerLoopReference:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    snap: np.ndarray
    quaternion: np.ndarray
    body_rates: np.ndarray
    cpp_body_rates: np.ndarray
    phi: np.ndarray
    path_omega: float

    def __len__(self) -> int:
        return int(self.time.shape[0])

    @property
    def state(self) -> np.ndarray:
        return np.concatenate(
            [self.position, self.velocity, self.quaternion], axis=-1
        ).astype(np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < _EPS:
        raise ValueError("Invalid quaternion.")
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 tensor-product q2."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Equivalent to Eigen::Quaterniond::FromTwoVectors(a, b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if dot < -1.0 + 1e-7:
        candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a, candidate))) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, candidate)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis], dtype=np.float64)

    q = np.concatenate(
        [np.array([1.0 + dot], dtype=np.float64), np.cross(a, b)]
    )
    return quat_normalize(q)


def xi_matrix(q: np.ndarray) -> np.ndarray:
    """Xi(q) for q_dot = 0.5 Xi(q) omega_body."""
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ],
        dtype=np.float64,
    )


def rotation_z(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def generate_powerloop_reference(
    center: Sequence[float] = (0.0, 0.0, 2.0),
    orientation: float = 0.0,
    radius: float = 1.5,
    speed: float = 4.5,
    phi_start: float = np.pi / 2.0,
    phi_end: float = -3.0 * np.pi / 2.0,
    sampling_frequency: float = 50.0,
    gravity: float = 9.81,
) -> PowerLoopReference:
    """Generate one 360-degree vertical loop.

    Defaults reproduce the task geometry:
      center [0, 0, 2], radius 1.5 m, speed 4.5 m/s,
      path angular velocity -3 rad/s, and 106 samples.
    """
    if radius <= 0.0 or speed <= 0.0 or sampling_frequency <= 0.0:
        raise ValueError("radius, speed, and sampling_frequency must be positive.")

    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,).")

    phi_total = float(phi_end - phi_start)
    if abs(phi_total) < _EPS:
        raise ValueError("phi_start and phi_end must differ.")

    direction = float(np.sign(phi_total))
    omega = direction * abs(speed / radius)
    angle_step = abs(omega / sampling_frequency)

    # Same sampling rule as the C++ source: strict loop, then exact endpoint.
    d_phi = np.arange(0.0, abs(phi_total), angle_step, dtype=np.float64)
    phi = phi_start + direction * d_phi
    phi = np.concatenate([phi, np.array([phi_end], dtype=np.float64)])
    time = np.abs((phi - phi_start) / omega)

    c = np.cos(phi)
    s = np.sin(phi)
    rz = rotation_z(orientation)

    p_local = radius * np.stack([c, np.zeros_like(c), -s], axis=-1)
    v_local = radius * omega * np.stack([-s, np.zeros_like(c), -c], axis=-1)
    a_local = radius * omega**2 * np.stack([-c, np.zeros_like(c), s], axis=-1)
    j_local = radius * omega**3 * np.stack([s, np.zeros_like(c), c], axis=-1)
    snap_local = radius * omega**4 * np.stack([c, np.zeros_like(c), -s], axis=-1)

    position = p_local @ rz.T + center
    velocity = v_local @ rz.T
    acceleration = a_local @ rz.T
    jerk = j_local @ rz.T
    snap = snap_local @ rz.T

    e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    q_yaw = np.array(
        [np.cos(0.5 * orientation), 0.0, 0.0, np.sin(0.5 * orientation)],
        dtype=np.float64,
    )

    q_list = []
    for a_ref in acceleration:
        q_align = quat_from_two_vectors(e3, a_ref + gravity * e3)
        q_ref = quat_normalize(quat_multiply(q_align, q_yaw))
        if q_list and np.dot(q_list[-1], q_ref) < 0.0:
            q_ref = -q_ref
        q_list.append(q_ref)
    quaternion = np.stack(q_list, axis=0)

    # The C++ file has a TODO and hard-codes [0, -omega, 0].
    cpp_body_rates = np.tile(
        np.array([0.0, -omega, 0.0], dtype=np.float64),
        (len(phi), 1),
    )

    # Dynamics-consistent reference for dual_stage_rl:
    # q_dot = 0.5 Xi(q) omega_body.
    q_dot = np.gradient(quaternion, time, axis=0, edge_order=2)
    body_rates = np.stack(
        [2.0 * xi_matrix(q).T @ qd for q, qd in zip(quaternion, q_dot)],
        axis=0,
    )

    return PowerLoopReference(
        time=time.astype(np.float32),
        position=position.astype(np.float32),
        velocity=velocity.astype(np.float32),
        acceleration=acceleration.astype(np.float32),
        jerk=jerk.astype(np.float32),
        snap=snap.astype(np.float32),
        quaternion=quaternion.astype(np.float32),
        body_rates=body_rates.astype(np.float32),
        cpp_body_rates=cpp_body_rates.astype(np.float32),
        phi=phi.astype(np.float32),
        path_omega=float(omega),
    )

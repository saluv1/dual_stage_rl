"""PX4 (NED / FRD) <-> PS2-RL (ENU / FLU) frame transforms.

PS2-RL's quadrotor model lives in an ENU world frame (z up, gravity acting as
-9.81 on vz) with body rates expressed in an FLU body frame whose +z axis is
the thrust axis.  PX4 publishes and consumes NED world / FRD body.

All quaternions are ``[w, x, y, z]`` — the same layout PX4 and PS2-RL both use.

The rotation composition used below is

    R_ENU_FLU = R_ENU_NED @ R_NED_FRD @ R_FRD_FLU

with ``R_ENU_NED = [[0,1,0],[1,0,0],[0,0,-1]]`` (a pi rotation about
(1,1,0)/sqrt(2)) and ``R_FRD_FLU = diag(1,-1,-1)`` (a pi rotation about x).
Both are proper rotations, so the whole thing is a clean quaternion product.
This module is verified against the matrix form in ``test_frame_transforms``.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT_HALF = math.sqrt(0.5)

# q_ENU_NED: rotates a vector expressed in NED into ENU.
Q_ENU_NED = np.array([0.0, _SQRT_HALF, _SQRT_HALF, 0.0], dtype=np.float64)
# q_FLU_FRD: rotates a vector expressed in FRD into FLU. Its own inverse.
Q_FLU_FRD = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_normalize(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Vector conversions                                                           #
# --------------------------------------------------------------------------- #


def ned_to_enu(v: np.ndarray) -> np.ndarray:
    """World vector (position, velocity, acceleration): NED -> ENU."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[1], v[0], -v[2]], dtype=np.float64)


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    """World vector: ENU -> NED. (Self-inverse pattern, same permutation.)"""
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[1], v[0], -v[2]], dtype=np.float64)


def frd_to_flu(v: np.ndarray) -> np.ndarray:
    """Body vector (e.g. angular velocity): FRD -> FLU."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float64)


def flu_to_frd(v: np.ndarray) -> np.ndarray:
    """Body vector: FLU -> FRD."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Attitude conversions                                                         #
# --------------------------------------------------------------------------- #


def px4_quat_to_enu_flu(q_ned_frd: np.ndarray) -> np.ndarray:
    """PX4 attitude (FRD body -> NED world) into PS2-RL's (FLU body -> ENU world)."""
    q = quat_normalize(q_ned_frd)
    return quat_normalize(quat_multiply(quat_multiply(Q_ENU_NED, q), quat_conjugate(Q_FLU_FRD)))


def enu_flu_to_px4_quat(q_enu_flu: np.ndarray) -> np.ndarray:
    """Inverse of :func:`px4_quat_to_enu_flu`."""
    q = quat_normalize(q_enu_flu)
    return quat_normalize(quat_multiply(quat_multiply(quat_conjugate(Q_ENU_NED), q), Q_FLU_FRD))


def body_frd_from_world_frd(q_ned_frd: np.ndarray, v_ned: np.ndarray) -> np.ndarray:
    """Rotate an NED world vector into the FRD body frame."""
    return rotation_matrix(q_ned_frd).T @ np.asarray(v_ned, dtype=np.float64)


def yaw_enu_to_px4(yaw_enu: float) -> float:
    """ENU yaw (CCW from East) -> PX4/NED yaw (CW from North)."""
    return float(math.pi / 2.0 - yaw_enu)


# --------------------------------------------------------------------------- #
# Odometry helper                                                              #
# --------------------------------------------------------------------------- #


def odometry_to_ps2rl_state(
    position_ned: np.ndarray,
    velocity_ned: np.ndarray,
    q_ned_frd: np.ndarray,
    origin_offset_enu: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the 10D PS2-RL state ``[p(3), v(3), q(4)]`` from PX4 odometry.

    ``origin_offset_enu`` shifts the PX4 local origin so the policy's reference
    trajectory can be placed anywhere in the world.
    """
    p = ned_to_enu(position_ned)
    v = ned_to_enu(velocity_ned)
    q = px4_quat_to_enu_flu(q_ned_frd)
    if origin_offset_enu is not None:
        p = p - np.asarray(origin_offset_enu, dtype=np.float64)
    return np.concatenate([p, v, q]).astype(np.float64)


def test_frame_transforms(rtol: float = 1e-9) -> None:
    """Self-check: quaternion path must equal the matrix path. Raises on failure."""
    rng = np.random.default_rng(0)
    r_enu_ned = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    r_frd_flu = np.diag([1.0, -1.0, -1.0])
    for _ in range(64):
        q = quat_normalize(rng.normal(size=4))
        lhs = rotation_matrix(px4_quat_to_enu_flu(q))
        rhs = r_enu_ned @ rotation_matrix(q) @ r_frd_flu
        assert np.allclose(lhs, rhs, atol=1e-9, rtol=rtol), "quaternion transform mismatch"
        assert np.allclose(enu_flu_to_px4_quat(px4_quat_to_enu_flu(q)), q, atol=1e-9) or np.allclose(
            enu_flu_to_px4_quat(px4_quat_to_enu_flu(q)), -q, atol=1e-9
        ), "round-trip mismatch"
        v = rng.normal(size=3)
        assert np.allclose(ned_to_enu(enu_to_ned(v)), v, atol=1e-12)
        assert np.allclose(frd_to_flu(flu_to_frd(v)), v, atol=1e-12)


if __name__ == "__main__":
    test_frame_transforms()
    print("frame transforms OK")

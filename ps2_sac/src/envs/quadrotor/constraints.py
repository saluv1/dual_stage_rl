from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.envs.quadrotor.dynamics import quat_normalize

Array = jax.Array


class QuadrotorConstraintParams(NamedTuple):
    g: float = 9.81
    z_ceiling: float = 3.0

    # Hover/base-set values from PS2-RL quadrotor setup.
    z_des: float = 2.0
    c_base: float = 8.0


def action_bounds(
    params: QuadrotorConstraintParams = QuadrotorConstraintParams(),
) -> tuple[Array, Array]:
    """
    Quadrotor input bounds:

        a_cmd in [0, 4g]
        omega_i in [-18, 18] rad/s
    """
    u_min = jnp.array([0.0, -18.0, -18.0, -18.0])
    u_max = jnp.array([4.0 * params.g, 18.0, 18.0, 18.0])
    return u_min, u_max


def clip_action(
    u: Array,
    params: QuadrotorConstraintParams = QuadrotorConstraintParams(),
) -> Array:
    u_min, u_max = action_bounds(params)
    return jnp.clip(u, u_min, u_max)


def h_ceiling(
    x: Array,
    params: QuadrotorConstraintParams = QuadrotorConstraintParams(),
) -> Array:
    """
    Safe set:

        S = {x : pz <= z_ceiling}

    Written as h_S(x) >= 0:

        h_S(x) = z_ceiling - pz.
    """
    pz = x[2]
    return params.z_ceiling - pz


def is_safe(
    x: Array,
    params: QuadrotorConstraintParams = QuadrotorConstraintParams(),
) -> Array:
    return h_ceiling(x, params) >= 0.0


def safety_violation(
    x: Array,
    params: QuadrotorConstraintParams = QuadrotorConstraintParams(),
) -> Array:
    """
    Positive value means ceiling violation.
    """
    return jnp.maximum(0.0, -h_ceiling(x, params))


def quat_conjugate(q: Array) -> Array:
    q = quat_normalize(q)
    return jnp.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_multiply(q1: Array, q2: Array) -> Array:
    """
    Hamilton product q1 ⊗ q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=q1.dtype,
    )


def sign_correct_quat(q: Array) -> Array:
    """
    Use equivalent quaternion with nonnegative scalar part.
    """
    q = quat_normalize(q)
    return jnp.where(q[0] < 0.0, -q, q)


def hover_quat_error(q: Array) -> Array:
    """
    Quaternion error relative to hover attitude q_ref = [1, 0, 0, 0].

    For hover q_ref, this reduces to sign-corrected q or conjugate(q)
    depending on convention. We use sign-corrected conjugate(q), matching
    an error-from-current-to-reference convention.

    If your LQR K/P are generated with the opposite convention, flip the
    sign of the vector part or change this function consistently.
    """
    q_ref = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    q_err = quat_multiply(q_ref, quat_conjugate(q))
    return sign_correct_quat(q_err)


def reduced_hover_error(
    x: Array,
    params: QuadrotorConstraintParams = QuadrotorConstraintParams(),
) -> Array:
    """
    Reduced hover-error state used for the quadrotor base set:

        x_e = [
            pz - z_des,
            vx, vy, vz,
            2 q_err_x, 2 q_err_y, 2 q_err_z
        ] in R^7.

    The PS2-RL appendix uses this reduced state for the LQR base set.
    """
    pz = x[2]
    vx, vy, vz = x[3:6]
    q = x[6:10]

    q_err = hover_quat_error(q)

    return jnp.array(
        [
            pz - params.z_des,
            vx,
            vy,
            vz,
            2.0 * q_err[1],
            2.0 * q_err[2],
            2.0 * q_err[3],
        ],
        dtype=x.dtype,
    )


class BaseSetParams(NamedTuple):
    """
    Base set:

        B = {x : x_e^T P x_e <= c_base}
        h_B(x) = c_base - x_e^T P x_e

    P is not numerically specified in the paper. It should come from
    the discrete-time LQR/Riccati construction used by your code.
    """
    P: Array
    c_base: float = 8.0


def h_base(
    x: Array,
    base_params: BaseSetParams,
    constraint_params: QuadrotorConstraintParams = QuadrotorConstraintParams(),
) -> Array:
    x_e = reduced_hover_error(x, constraint_params)
    return base_params.c_base - x_e @ base_params.P @ x_e


def is_in_base(
    x: Array,
    base_params: BaseSetParams,
    constraint_params: QuadrotorConstraintParams = QuadrotorConstraintParams(),
) -> Array:
    return h_base(x, base_params, constraint_params) >= 0.0


h_ceiling_batch = jax.vmap(h_ceiling, in_axes=(0, None))
is_safe_batch = jax.vmap(is_safe, in_axes=(0, None))
safety_violation_batch = jax.vmap(safety_violation, in_axes=(0, None))
reduced_hover_error_batch = jax.vmap(reduced_hover_error, in_axes=(0, None))
h_base_batch = jax.vmap(h_base, in_axes=(0, None, None))
is_in_base_batch = jax.vmap(is_in_base, in_axes=(0, None, None))
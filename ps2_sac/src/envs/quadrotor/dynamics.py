from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class QuadrotorParams(NamedTuple):
    g: float = 9.81
    dt: float = 0.02


def quat_normalize(q: Array, eps: float = 1e-8) -> Array:
    return q / (jnp.linalg.norm(q) + eps)


def split_state(x: Array) -> tuple[Array, Array, Array]:
    """
    State convention:
        x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
    """
    p = x[0:3]
    v = x[3:6]
    q = quat_normalize(x[6:10])
    return p, v, q


def split_action(u: Array) -> tuple[Array, Array]:
    """
    Action convention:
        u = [a_cmd, wx, wy, wz]
    """
    a_cmd = u[0]
    omega_cmd = u[1:4]
    return a_cmd, omega_cmd


def quat_to_rotmat(q: Array) -> Array:
    """
    Rotation matrix R(q).

    Interpreted as mapping body-frame vectors into inertial/world frame.
    This matches dynamics of the form

        v_dot = -g e3 + R(q) (a_cmd e3).
    """
    q = quat_normalize(q)
    qw, qx, qy, qz = q

    return jnp.array(
        [
            [
                1.0 - 2.0 * (qy**2 + qz**2),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx**2 + qz**2),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx**2 + qy**2),
            ],
        ],
        dtype=q.dtype,
    )


def xi_matrix(q: Array) -> Array:
    """
    Quaternion kinematic matrix Xi(q) such that

        q_dot = 0.5 * Xi(q) @ omega_cmd

    with q = [qw, qx, qy, qz].
    """
    q = quat_normalize(q)
    qw, qx, qy, qz = q

    return jnp.array(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ],
        dtype=q.dtype,
    )


def continuous_dynamics(
    x: Array,
    u: Array,
    params: QuadrotorParams = QuadrotorParams(),
) -> Array:
    """
    Continuous-time quadrotor dynamics:

        p_dot = v
        v_dot = -g e3 + R(q) (a_cmd e3)
        q_dot = 0.5 Xi(q) omega_cmd

    State:
        x = [p, v, q] in R^10

    Action:
        u = [a_cmd, wx, wy, wz] in R^4
    """
    _, v, q = split_state(x)
    a_cmd, omega_cmd = split_action(u)

    e3 = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)
    R = quat_to_rotmat(q)

    p_dot = v
    v_dot = -params.g * e3 + R @ (a_cmd * e3)
    q_dot = 0.5 * xi_matrix(q) @ omega_cmd

    return jnp.concatenate([p_dot, v_dot, q_dot], axis=0)


def control_affine_terms(
    x: Array,
    params: QuadrotorParams = QuadrotorParams(),
) -> tuple[Array, Array]:
    """
    Return f(x), g(x) such that

        x_dot = f(x) + g(x) @ u

    where u = [a_cmd, wx, wy, wz].

    This is useful for CBF/BCBF row construction.
    """
    _, v, q = split_state(x)

    dtype = x.dtype
    e3 = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    R = quat_to_rotmat(q)
    Xi = xi_matrix(q)

    f_p = v
    f_v = -params.g * e3
    f_q = jnp.zeros((4,), dtype=dtype)
    f = jnp.concatenate([f_p, f_v, f_q], axis=0)

    g_mat = jnp.zeros((10, 4), dtype=dtype)

    # a_cmd column affects v_dot through R(q) e3.
    g_mat = g_mat.at[3:6, 0].set(R @ e3)

    # omega_cmd columns affect q_dot through 0.5 Xi(q).
    g_mat = g_mat.at[6:10, 1:4].set(0.5 * Xi)

    return f, g_mat


def euler_step(
    x: Array,
    u: Array,
    params: QuadrotorParams = QuadrotorParams(),
) -> Array:
    x_next = x + params.dt * continuous_dynamics(x, u, params)
    q_next = quat_normalize(x_next[6:10])
    return x_next.at[6:10].set(q_next)


def rk4_step(
    x: Array,
    u: Array,
    params: QuadrotorParams = QuadrotorParams(),
) -> Array:
    """
    RK4 integration for better numerical stability.

    For initial testing, euler_step is simpler.
    For training/evaluation, rk4_step is usually safer.
    """
    dt = params.dt

    k1 = continuous_dynamics(x, u, params)
    k2 = continuous_dynamics(x + 0.5 * dt * k1, u, params)
    k3 = continuous_dynamics(x + 0.5 * dt * k2, u, params)
    k4 = continuous_dynamics(x + dt * k3, u, params)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    q_next = quat_normalize(x_next[6:10])
    return x_next.at[6:10].set(q_next)


continuous_dynamics_batch = jax.vmap(continuous_dynamics, in_axes=(0, 0, None))
control_affine_terms_batch = jax.vmap(control_affine_terms, in_axes=(0, None))
euler_step_batch = jax.vmap(euler_step, in_axes=(0, 0, None))
rk4_step_batch = jax.vmap(rk4_step, in_axes=(0, 0, None))
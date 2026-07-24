"""
Control-affine quadrotor dynamics in JAX.

    x_dot = f(x) + g(x) u

State  (n = 10):  x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
Input  (m = 4):   u = [a_cmd, wx, wy, wz]

This mirrors env/dynamics.py (Dynamics.equationsOfMotion) exactly, but is
written as an explicit f/g split because the BCBF rows are affine in u and
therefore need g(x) separately.
"""

import jax
import jax.numpy as jnp

N_STATE = 10
N_INPUT = 4

G_ACC = 9.807
E3 = jnp.array([0.0, 0.0, 1.0])


def normalize_quat(q):
    """Normalize quaternion, guarding against the zero vector."""
    nrm = jnp.linalg.norm(q)
    return jnp.where(nrm < 1e-12, jnp.array([1.0, 0.0, 0.0, 0.0]), q / nrm)


def R_of_q(q):
    """Body -> world rotation matrix. q = [qw, qx, qy, qz]."""
    q = normalize_quat(q)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    return jnp.array([
        [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx*qy - qw*qz),       2.0 * (qx*qz + qw*qy)],
        [2.0 * (qx*qy + qw*qz),       1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy*qz - qw*qx)],
        [2.0 * (qx*qz - qw*qy),       2.0 * (qy*qz + qw*qx),       1.0 - 2.0 * (qx**2 + qy**2)],
    ])


def Xi_of_q(q):
    """
    Quaternion kinematic matrix, q_dot = 0.5 * Xi(q) @ omega.  Shape (4, 3).

    NOTE: built from the *raw* (unnormalized) q on purpose.  With raw q we have
    Xi(q)^T q = 0 exactly, so d||q||^2/dt = 0 and the continuous-time flow is
    norm-preserving.  Normalizing here would break that identity.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    return jnp.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
    ])


def f_x(x, g_acc=G_ACC):
    """Drift term f(x), shape (10,)."""
    v = x[3:6]
    return jnp.concatenate([v, -g_acc * E3, jnp.zeros(4)])


def g_x(x):
    """Control matrix g(x), shape (10, 4)."""
    q = x[6:10]

    b3 = R_of_q(q) @ E3          # thrust direction in world frame
    Xi = Xi_of_q(q)

    Gmat = jnp.zeros((N_STATE, N_INPUT))
    Gmat = Gmat.at[3:6, 0].set(b3)
    Gmat = Gmat.at[6:10, 1:4].set(0.5 * Xi)

    return Gmat


def x_dot(x, u, g_acc=G_ACC):
    """Open-loop vector field, for sanity checks against env/dynamics.py."""
    return f_x(x, g_acc) + g_x(x) @ u

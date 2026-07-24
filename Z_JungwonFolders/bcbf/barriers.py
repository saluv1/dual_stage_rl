"""
Barrier / set functions for the BCBF construction, in JAX.

Safe set      S = { x : h_S(x) >= 0 },   h_S(x) = z_ceil - p_z
Base set      B = { x : h_B(x) >= 0 },   h_B(x) = c_b - e(x)^T P e(x)

The reduced error state e(x) in R^7 and the LQR matrix P are the same objects
used in bcbf/lqrgain.py and bcbf/set_indicator.py:

    e = [ p_z - z_des, v_x, v_y, v_z, 2*q_x, 2*q_y, 2*q_z ]

with the quaternion sign correction (flip (qx,qy,qz) when qw < 0) so that the
error is measured against the *shortest* rotation to level attitude.
"""

import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------
# reduced error state
# ----------------------------------------------------------------------

def reduced_state(x, z_des=2.0):
    """
    JAX version of train.compute_reduced_state.  Shape (7,).

    CAVEAT: the sign correction makes e(x) discontinuous across qw = 0, so h_B
    is discontinuous there too.  qw = 0 is a 180 deg attitude error, which is
    far outside B (h_B < 0 on both sides), so this never affects the terminal
    constraint in practice.  It is still worth knowing when reading gradients.
    """
    pz = x[2]
    v = x[3:6]
    qw = x[6]
    qv = x[7:10]

    s = jnp.where(qw < 0.0, -1.0, 1.0)

    return jnp.concatenate([
        jnp.array([pz - z_des]),
        v,
        2.0 * s * qv,
    ])


# ----------------------------------------------------------------------
# safe set
# ----------------------------------------------------------------------

def h_S(x, z_ceil=3.0):
    """Ceiling constraint.  n_S = 1."""
    return z_ceil - x[2]


def grad_h_S(x, z_ceil=3.0):
    return jax.grad(h_S)(x, z_ceil)


# ----------------------------------------------------------------------
# base set
# ----------------------------------------------------------------------

def h_B(x, P, c_b=8.0, z_des=2.0):
    """LQR ellipsoid around hover.  n_B = 1."""
    e = reduced_state(x, z_des)
    return c_b - e @ P @ e


def grad_h_B(x, P, c_b=8.0, z_des=2.0):
    return jax.grad(h_B)(x, P, c_b, z_des)


# ----------------------------------------------------------------------
# extended class-K functions
# ----------------------------------------------------------------------

def make_linear_alpha(gain):
    """alpha(s) = gain * s.  Paper defaults: alpha_S = 4.0, alpha_B = 2.0."""
    def alpha(s):
        return gain * s
    return alpha

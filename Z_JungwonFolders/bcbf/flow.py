"""
Backup flow Phi_pi_b(x, t') and its sensitivity Psi_pi_b(x, t') = dPhi/dx.

Same idea as DR-bCBF's `integrateStateBackup` (src/spacecraft/dynamics.py):
augment the state with the flattened STM and integrate the variational
equation alongside the dynamics,

    d/dt' Phi  = f_pi_b(Phi),                      Phi(x, 0) = x
    d/dt' Psi  = (d f_pi_b / dx)(Phi) @ Psi,       Psi(x, 0) = I

The one difference from DR-bCBF: there the closed-loop Jacobian is written by
hand (`computeJacobianSTM`) because the backup law is analytic.  Our pi_b
contains a neural network, so the Jacobian comes from jax.jacfwd - this is the
whole reason the module is in JAX.

Fixed-step RK4 is used instead of solve_ivp so the whole thing is jittable and
differentiable (Phase II needs gradients through the rollout).
"""

import jax
import jax.numpy as jnp

from .dynamics_jax import N_STATE


def make_rollout(f_cl, N, dt, n_sub=1):
    """
    Build a jitted rollout function.

    Args
    ----
    f_cl  : closed-loop vector field, R^n -> R^n
    N     : number of backup steps (T = N * dt); mesh has N + 1 nodes
    dt    : mesh spacing, should match the control period (0.02 s)
    n_sub : RK4 substeps per mesh interval (raise if the flow is stiff)

    Returns
    -------
    rollout(x0) -> (X, Psi) with X: (N+1, n), Psi: (N+1, n, n)
    """
    n = N_STATE
    J_cl = jax.jacfwd(f_cl)

    def aug_rhs(z):
        x = z[:n]
        Psi = z[n:].reshape(n, n)
        dx = f_cl(x)
        dPsi = J_cl(x) @ Psi
        return jnp.concatenate([dx, dPsi.reshape(-1)])

    h = dt / n_sub

    def rk4_substep(z, _):
        k1 = aug_rhs(z)
        k2 = aug_rhs(z + 0.5 * h * k1)
        k3 = aug_rhs(z + 0.5 * h * k2)
        k4 = aug_rhs(z + h * k3)
        return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    def mesh_step(z, _):
        z, _ = jax.lax.scan(rk4_substep, z, None, length=n_sub)
        return z, z

    def rollout(x0):
        z0 = jnp.concatenate([x0, jnp.eye(n).reshape(-1)])
        _, zs = jax.lax.scan(mesh_step, z0, None, length=N)

        Z = jnp.concatenate([z0[None, :], zs], axis=0)      # (N+1, n + n^2)
        X = Z[:, :n]
        Psi = Z[:, n:].reshape(-1, n, n)
        return X, Psi

    return jax.jit(rollout)


def flow_only(f_cl, N, dt, n_sub=1):
    """Cheap variant without the STM, e.g. for checking membership in C_T."""
    h = dt / n_sub

    def rk4_substep(x, _):
        k1 = f_cl(x)
        k2 = f_cl(x + 0.5 * h * k1)
        k3 = f_cl(x + 0.5 * h * k2)
        k4 = f_cl(x + h * k3)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    def mesh_step(x, _):
        x, _ = jax.lax.scan(rk4_substep, x, None, length=n_sub)
        return x, x

    def rollout(x0):
        _, xs = jax.lax.scan(mesh_step, x0, None, length=N)
        return jnp.concatenate([x0[None, :], xs], axis=0)

    return jax.jit(rollout)

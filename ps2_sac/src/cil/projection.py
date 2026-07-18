from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import qpax


Array = jax.Array


class ProjectionDiagnostics(NamedTuple):
    max_cil_violation: Array
    max_bound_violation: Array
    correction_norm: Array


def build_box_constraints(
    u_min: Array,
    u_max: Array,
) -> tuple[Array, Array]:
    """
    Convert box constraints

        u_min <= u <= u_max

    into inequality form

        G_box @ u <= h_box.
    """
    action_dim = u_min.shape[-1]
    eye = jnp.eye(action_dim, dtype=u_min.dtype)

    G_box = jnp.concatenate(
        [
            eye,   # u <= u_max
            -eye,  # -u <= -u_min
        ],
        axis=0,
    )
    h_box = jnp.concatenate([u_max, -u_min], axis=0)
    return G_box, h_box


def build_projection_qp(
    u_nom: Array,
    A_cil: Array,
    b_cil: Array,
    u_min: Array,
    u_max: Array,
    reg: float = 1e-6,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    Build the QP

        min_u 0.5 ||u - u_nom||_2^2
        s.t.  A_cil @ u <= b_cil
              u_min <= u <= u_max

    in qpax standard form

        min_u 0.5 u.T @ Q @ u + q.T @ u
        s.t.  Aeq @ u = beq
              G @ u <= h.
    """
    action_dim = u_nom.shape[-1]

    Q = jnp.eye(action_dim, dtype=u_nom.dtype)
    Q = Q + reg * jnp.eye(action_dim, dtype=u_nom.dtype)

    q = -u_nom

    # No equality constraints.
    Aeq = jnp.zeros((0, action_dim), dtype=u_nom.dtype)
    beq = jnp.zeros((0,), dtype=u_nom.dtype)

    G_box, h_box = build_box_constraints(u_min, u_max)

    G = jnp.concatenate([A_cil, G_box], axis=0)
    h = jnp.concatenate([b_cil, h_box], axis=0)

    return Q, q, Aeq, beq, G, h


def project_action(
    u_nom: Array,
    A_cil: Array,
    b_cil: Array,
    u_min: Array,
    u_max: Array,
    reg: float = 1e-6,
) -> Array:
    """
    Project one nominal action onto the CIL feasible set.

    Shapes:
        u_nom:  (action_dim,)
        A_cil:  (num_constraints, action_dim)
        b_cil:  (num_constraints,)
        u_min:  (action_dim,)
        u_max:  (action_dim,)

    Returns:
        u_safe: (action_dim,)
    """
    Q, q, Aeq, beq, G, h = build_projection_qp(
        u_nom=u_nom,
        A_cil=A_cil,
        b_cil=b_cil,
        u_min=u_min,
        u_max=u_max,
        reg=reg,
    )

    u_safe = qpax.solve_qp_primal(Q, q, Aeq, beq, G, h)
    return u_safe



def projection_diagnostics(
    u_nom: Array,
    u_safe: Array,
    A_cil: Array,
    b_cil: Array,
    u_min: Array,
    u_max: Array,
) -> ProjectionDiagnostics:
    """
    Useful for debugging. Positive violation means constraint violation.
    """
    cil_violation = A_cil @ u_safe - b_cil
    upper_violation = u_safe - u_max
    lower_violation = u_min - u_safe

    max_cil_violation = jnp.max(cil_violation)
    max_bound_violation = jnp.maximum(
        jnp.max(upper_violation),
        jnp.max(lower_violation),
    )
    correction_norm = jnp.linalg.norm(u_safe - u_nom)

    return ProjectionDiagnostics(
        max_cil_violation=max_cil_violation,
        max_bound_violation=max_bound_violation,
        correction_norm=correction_norm,
    )


def project_action_batch(
    u_nom_batch: Array,
    A_cil_batch: Array,
    b_cil_batch: Array,
    u_min: Array,
    u_max: Array,
    reg: float = 1e-6,
) -> Array:
    """
    Batched version of project_action.

    Shapes:
        u_nom_batch:   (batch_size, action_dim)
        A_cil_batch:   (batch_size, num_constraints, action_dim)
        b_cil_batch:   (batch_size, num_constraints)
        u_min:         (action_dim,)
        u_max:         (action_dim,)

    Returns:
        u_safe_batch:  (batch_size, action_dim)
    """

    def _single_project(u_nom, A_cil, b_cil):
        return project_action(
            u_nom=u_nom,
            A_cil=A_cil,
            b_cil=b_cil,
            u_min=u_min,
            u_max=u_max,
            reg=reg,
        )

    return jax.vmap(_single_project, in_axes=(0, 0, 0))(
        u_nom_batch,
        A_cil_batch,
        b_cil_batch,
    )
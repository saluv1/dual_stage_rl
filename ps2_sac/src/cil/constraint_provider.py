from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class CILConstraints(NamedTuple):
    """
    Affine CIL constraints in the form

        A @ u <= b.

    For a single state:
        A: (num_constraints, action_dim)
        b: (num_constraints,)

    For a batch:
        A: (batch_size, num_constraints, action_dim)
        b: (batch_size, num_constraints)
    """
    A: Array
    b: Array


class ConstantConstraintParams(NamedTuple):
    """
    Simple test provider.

    Use this before plugging in the real BCBF/pi_safe module.
    """
    A: Array
    b: Array


def constant_constraint_provider(
    params: ConstantConstraintParams,
    obs: Array,
) -> CILConstraints:
    """
    Return the same A,b for every observation.

    Useful for testing the projection layer independently.
    """
    del obs
    return CILConstraints(A=params.A, b=params.b)


constant_constraint_provider_batch = jax.vmap(
    constant_constraint_provider,
    in_axes=(None, 0),
)


def pad_constraints(
    A: Array,
    b: Array,
    max_constraints: int,
    inactive_bound: float = 1e6,
) -> CILConstraints:
    """
    Pad variable-length constraints to a fixed size.

    This is important for jax.jit and jax.vmap, because the number of
    constraints should be static.

    Existing constraints:
        A @ u <= b

    Padded inactive constraints:
        0 @ u <= inactive_bound
    """
    num_constraints, action_dim = A.shape
    pad_n = max_constraints - num_constraints

    A_pad = jnp.zeros((pad_n, action_dim), dtype=A.dtype)
    b_pad = inactive_bound * jnp.ones((pad_n,), dtype=b.dtype)

    A_out = jnp.concatenate([A, A_pad], axis=0)
    b_out = jnp.concatenate([b, b_pad], axis=0)

    return CILConstraints(A=A_out, b=b_out)


def make_constraint_provider(
    provider_fn: Callable[[Any, Array], CILConstraints],
) -> Callable[[Any, Array], CILConstraints]:
    """
    Thin wrapper to make the expected interface explicit.

    Expected signature:
        constraints = provider_fn(params, obs)

    where:
        constraints.A @ u <= constraints.b
    """
    return provider_fn
from __future__ import annotations

from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp

from src.cil.constraint_provider import CILConstraints
from src.cil.projection import project_action, project_action_batch


Array = jax.Array


class SafeActionOutput(NamedTuple):
    u_safe: Array
    u_nom: Array
    A: Array
    b: Array


def get_safe_action(
    u_nom: Array,
    obs: Array,
    provider_params: Any,
    constraint_provider: Callable[[Any, Array], CILConstraints],
    u_min: Array,
    u_max: Array,
    reg: float = 1e-6,
) -> SafeActionOutput:
    """
    Single-state safe action wrapper.

    Inputs:
        u_nom: actor output, shape (action_dim,)
        obs: state/observation, shape (obs_dim,)

    Output:
        u_safe = projection of u_nom onto A(obs) u <= b(obs).
    """
    constraints = constraint_provider(provider_params, obs)

    u_safe = project_action(
        u_nom=u_nom,
        A_cil=constraints.A,
        b_cil=constraints.b,
        u_min=u_min,
        u_max=u_max,
        reg=reg,
    )

    return SafeActionOutput(
        u_safe=u_safe,
        u_nom=u_nom,
        A=constraints.A,
        b=constraints.b,
    )


def get_safe_action_batch(
    u_nom_batch,
    obs_batch,
    provider_params,
    constraint_provider,
    u_min,
    u_max,
    reg=1e-6,
):
    """
    Batched CIL action filter.

    u_nom_batch: shape (batch_size, action_dim)
    obs_batch:   shape (batch_size, obs_dim)

    This function vmaps the single-sample get_safe_action.
    Therefore, even if constraint_provider returns constant A, b,
    each sample gets its own A, b with batch dimension.
    """

    def _single_get_safe_action(u_nom, obs):
        return get_safe_action(
            u_nom=u_nom,
            obs=obs,
            provider_params=provider_params,
            constraint_provider=constraint_provider,
            u_min=u_min,
            u_max=u_max,
            reg=reg,
        )

    return jax.vmap(_single_get_safe_action, in_axes=(0, 0))(
        u_nom_batch,
        obs_batch,
    )
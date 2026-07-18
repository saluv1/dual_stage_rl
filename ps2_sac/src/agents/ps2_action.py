from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from src.cil.constraint_provider import CILConstraints
from src.cil.action_filter import get_safe_action, get_safe_action_batch


Array = jax.Array


class ActorOutput(NamedTuple):
    """
    Output of nominal SAC actor before CIL.

    u_nom:
        nominal action sampled from policy or deterministic mean action.
    log_prob:
        log pi(u_nom | obs). For deterministic eval, this can be None.
    """
    u_nom: Array
    log_prob: Array | None


class SafeActorOutput(NamedTuple):
    """
    Output after applying CIL to nominal actor output.
    """
    u_safe: Array
    u_nom: Array
    log_prob: Array | None
    A: Array
    b: Array


def select_safe_action(
    actor_apply_fn: Callable[..., ActorOutput],
    actor_params: Any,
    obs: Array,
    key: Array,
    provider_params: Any,
    constraint_provider: Callable[[Any, Array], CILConstraints],
    u_min: Array,
    u_max: Array,
    deterministic: bool = False,
    reg: float = 1e-6,
) -> SafeActorOutput:
    """
    Single-state action selection.

    This is the function that will later replace SAC get_action.

    Flow:
        obs -> actor -> u_nom -> CIL -> u_safe
    """
    actor_out = actor_apply_fn(
        actor_params,
        obs,
        key,
        deterministic=deterministic,
    )

    safe_out = get_safe_action(
        u_nom=actor_out.u_nom,
        obs=obs,
        provider_params=provider_params,
        constraint_provider=constraint_provider,
        u_min=u_min,
        u_max=u_max,
        reg=reg,
    )

    return SafeActorOutput(
        u_safe=safe_out.u_safe,
        u_nom=safe_out.u_nom,
        log_prob=actor_out.log_prob,
        A=safe_out.A,
        b=safe_out.b,
    )


def select_safe_action_batch(
    actor_apply_fn: Callable[..., ActorOutput],
    actor_params: Any,
    obs_batch: Array,
    key: Array,
    provider_params: Any,
    constraint_provider_batch: Callable[[Any, Array], CILConstraints],
    u_min: Array,
    u_max: Array,
    deterministic: bool = False,
    reg: float = 1e-6,
) -> SafeActorOutput:
    """
    Batched action selection.

    This is what actor_loss and value_loss should eventually use.
    """
    batch_size = obs_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    def _actor_one(obs, subkey):
        return actor_apply_fn(
            actor_params,
            obs,
            subkey,
            deterministic=deterministic,
        )

    actor_out_batch = jax.vmap(_actor_one)(obs_batch, keys)

    safe_out_batch = get_safe_action_batch(
        u_nom_batch=actor_out_batch.u_nom,
        obs_batch=obs_batch,
        provider_params=provider_params,
        constraint_provider_batch=constraint_provider_batch,
        u_min=u_min,
        u_max=u_max,
        reg=reg,
    )

    return SafeActorOutput(
        u_safe=safe_out_batch.u_safe,
        u_nom=safe_out_batch.u_nom,
        log_prob=actor_out_batch.log_prob,
        A=safe_out_batch.A,
        b=safe_out_batch.b,
    )
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.agents.ps2_action import ActorOutput


Array = jax.Array


class ConstantMockActorParams(NamedTuple):
    """
    Mock actor parameters.

    u_nom:
        Constant nominal action returned by the actor.
    log_prob:
        Dummy log probability. For SAC-style stochastic actor tests.
    """
    u_nom: Array
    log_prob: Array


def constant_mock_actor_apply(
    params: ConstantMockActorParams,
    obs: Array,
    key: Array,
    deterministic: bool = False,
) -> ActorOutput:
    """
    Mock actor used to test

        obs -> u_nom -> CIL -> u_safe

    without connecting the real SAC actor yet.

    This ignores obs/key and always returns the same nominal action.
    """
    del obs, key, deterministic

    return ActorOutput(
        u_nom=params.u_nom,
        log_prob=params.log_prob,
    )


class LinearMockActorParams(NamedTuple):
    """
    Slightly more realistic mock actor.

    u_nom = W @ obs + bias

    Useful after the constant actor test passes.
    """
    W: Array
    bias: Array
    log_prob: Array


def linear_mock_actor_apply(
    params: LinearMockActorParams,
    obs: Array,
    key: Array,
    deterministic: bool = False,
) -> ActorOutput:
    del key, deterministic

    u_nom = params.W @ obs + params.bias

    return ActorOutput(
        u_nom=u_nom,
        log_prob=params.log_prob,
    )
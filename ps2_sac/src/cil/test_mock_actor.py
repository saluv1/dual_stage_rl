from __future__ import annotations

import jax
import jax.numpy as jnp

from src.agents.mock_actor import (
    ConstantMockActorParams,
    constant_mock_actor_apply,
)
from src.agents.ps2_action import select_safe_action_batch
from src.cil.constraint_provider import (
    ConstantConstraintParams,
    constant_constraint_provider_batch,
)


def main():
    key = jax.random.PRNGKey(0)

    # Action bounds.
    u_min = jnp.array([-1.0, -1.0])
    u_max = jnp.array([1.0, 1.0])

    # Constraint:
    #   u0 + u1 <= 0.5
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([0.5])

    provider_params = ConstantConstraintParams(A=A, b=b)

    # Mock actor intentionally outputs unsafe nominal action.
    actor_params = ConstantMockActorParams(
        u_nom=jnp.array([1.2, 1.2]),
        log_prob=jnp.array(0.0),
    )

    obs_batch = jnp.zeros((4, 3))

    out = select_safe_action_batch(
        actor_apply_fn=constant_mock_actor_apply,
        actor_params=actor_params,
        obs_batch=obs_batch,
        key=key,
        provider_params=provider_params,
        constraint_provider_batch=constant_constraint_provider_batch,
        u_min=u_min,
        u_max=u_max,
        deterministic=False,
    )

    print("u_nom:")
    print(out.u_nom)

    print("u_safe:")
    print(out.u_safe)

    print("A u_safe - b:")
    print(jnp.einsum("bij,bj->bi", out.A, out.u_safe) - out.b)


if __name__ == "__main__":
    main()
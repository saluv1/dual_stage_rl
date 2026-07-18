import jax
import jax.numpy as jnp

from src.cil.projection import project_action, projection_diagnostics
from src.cil.constraint_provider import ConstantConstraintParams, constant_constraint_provider


def main():
    # action_dim = 2
    u_min = jnp.array([-1.0, -1.0])
    u_max = jnp.array([1.0, 1.0])

    # Extra CIL constraint:
    #   u_0 + u_1 <= 0.5
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([0.5])

    params = ConstantConstraintParams(A=A, b=b)

    obs = jnp.array([0.0, 0.0, 0.0])
    constraints = constant_constraint_provider(params, obs)

    # This violates u_0 + u_1 <= 0.5 and box upper bound.
    u_nom = jnp.array([1.2, 1.2])

    u_safe = project_action(
        u_nom=u_nom,
        A_cil=constraints.A,
        b_cil=constraints.b,
        u_min=u_min,
        u_max=u_max,
    )

    diag = projection_diagnostics(
        u_nom=u_nom,
        u_safe=u_safe,
        A_cil=constraints.A,
        b_cil=constraints.b,
        u_min=u_min,
        u_max=u_max,
    )

    print("u_nom:", u_nom)
    print("u_safe:", u_safe)
    print("A u_safe - b:", constraints.A @ u_safe - constraints.b)
    print("diagnostics:", diag)
    project_action_jit = jax.jit(project_action)

    u_safe_jit = project_action_jit(
        u_nom,
        constraints.A,
        constraints.b,
        u_min,
        u_max,
    )

    print("u_safe_jit:", u_safe_jit)

    def test_loss(u_nom):
        u_safe = project_action(
            u_nom=u_nom,
            A_cil=constraints.A,
            b_cil=constraints.b,
            u_min=u_min,
            u_max=u_max,
        )
        return jnp.sum(u_safe ** 2)

    grad_u_nom = jax.grad(test_loss)(u_nom)

    print("grad_u_nom:", grad_u_nom)
    print("is finite:", jnp.all(jnp.isfinite(grad_u_nom)))

    from src.cil.projection import project_action_batch

    u_nom_batch = jnp.array([
        [1.2, 1.2],
        [0.1, 0.1],
        [-2.0, 0.0],
    ])

    A_batch = jnp.repeat(constraints.A[None, :, :], u_nom_batch.shape[0], axis=0)
    b_batch = jnp.repeat(constraints.b[None, :], u_nom_batch.shape[0], axis=0)

    u_safe_batch = project_action_batch(
        u_nom_batch,
        A_batch,
        b_batch,
        u_min,
        u_max,
    )

    print("u_safe_batch:", u_safe_batch)
    print("A u_safe - b:", jnp.einsum("bij,bj->bi", A_batch, u_safe_batch) - b_batch)


if __name__ == "__main__":
    main()
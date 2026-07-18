from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.envs.quadrotor.env import QuadrotorEnv
from src.envs.quadrotor.mujoco_playback import playback_trajectory
from src.envs.quadrotor.constraints import action_bounds

from src.cil.constraint_provider import (
    ConstantConstraintParams,
    constant_constraint_provider,
)
from src.cil.action_filter import get_safe_action
from src.envs.quadrotor.mujoco_playback import playback_trajectory_with_thrust_bars

def make_thrust_band_constraints(
    g: float = 9.81,
    thrust_margin: float = 1.0,
) -> ConstantConstraintParams:
    """
    Mock CIL constraint:

        g - thrust_margin <= thrust <= g + thrust_margin

    Action convention:
        u = [thrust, wx, wy, wz]

    Written as A u <= b:

        thrust <= g + margin
       -thrust <= -(g - margin)
    """
    A = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    b = jnp.array(
        [
            g + thrust_margin,
            -(g - thrust_margin),
        ],
        dtype=jnp.float32,
    )

    return ConstantConstraintParams(A=A, b=b)


def main():
    env = QuadrotorEnv(
        for_evaluation=True,
        seed=0,
        horizon=600,
    )

    timestep = env.reset()

    dt = 0.02
    g = 9.81

    # CIL mock constraint: thrust is only allowed in [g - 1, g + 1].
    provider_params = make_thrust_band_constraints(
        g=g,
        thrust_margin=1.0,
    )

    u_min, u_max = action_bounds()

    u_nom_log = []
    u_safe_log = []

    for k in range(600):
        t = k * dt

        # Intentionally aggressive nominal action.
        # Without CIL, this can push the drone upward too hard.
        thrust_nom = g + 6.0 * np.sin(1.0 * t) + 3.0
        omega_nom = np.array(
            [
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        u_nom = jnp.array(
            [
                thrust_nom,
                omega_nom[0],
                omega_nom[1],
                omega_nom[2],
            ],
            dtype=jnp.float32,
        )

        obs = jnp.asarray(env._state, dtype=jnp.float32)

        safe_out = get_safe_action(
            u_nom=u_nom,
            obs=obs,
            provider_params=provider_params,
            constraint_provider=constant_constraint_provider,
            u_min=u_min,
            u_max=u_max,
        )

        u_safe = np.asarray(safe_out.u_safe, dtype=np.float32)

        timestep = env.step(u_safe)

        u_nom_log.append(np.asarray(u_nom, dtype=np.float32))
        u_safe_log.append(u_safe)

        if timestep.last():
            print("Episode ended at step:", k)
            break

    trajectory = np.asarray(env.trajectory, dtype=np.float32)
    u_nom_log = np.asarray(u_nom_log, dtype=np.float32)
    u_safe_log = np.asarray(u_safe_log, dtype=np.float32)

    print("trajectory shape:", trajectory.shape)
    print("u_nom thrust min/max:", u_nom_log[:, 0].min(), u_nom_log[:, 0].max())
    print("u_safe thrust min/max:", u_safe_log[:, 0].min(), u_safe_log[:, 0].max())
    print("z min/max:", trajectory[:, 2].min(), trajectory[:, 2].max())

    # Check mock CIL constraint.
    A = np.asarray(provider_params.A)
    b = np.asarray(provider_params.b)
    violation = u_safe_log @ A.T - b
    print("max CIL violation:", violation.max())

    playback_trajectory_with_thrust_bars(
        trajectory=trajectory,
        u_nom_log=u_nom_log,
        u_safe_log=u_safe_log,
        dt=dt,
    )

if __name__ == "__main__":
    main()
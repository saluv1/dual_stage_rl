import numpy as np

from src.envs.quadrotor.env_powerloop import (
    QuadrotorPowerLoopEnv,
)


env = QuadrotorPowerLoopEnv(
    for_evaluation=True,
    seed=0,
    horizon=106,
    include_progress=True,
    initial_position_noise=0.0,
    perturb_evaluation=False,
    terminate_on_ceiling=False,
    use_cpp_body_rate_reference=False,
)

timestep = env.reset()

position_errors = []
velocity_errors = []
attitude_errors = []
actions = []
rewards = []

for k in range(106):
    specific_force = (
        env.reference.acceleration[k]
        + np.array(
            [0.0, 0.0, 9.81],
            dtype=np.float32,
        )
    )

    a_cmd = np.linalg.norm(
        specific_force
    )

    omega_cmd = (
        env.reference.body_rates[k]
    )

    action = np.concatenate(
        [
            np.array(
                [a_cmd],
                dtype=np.float32,
            ),
            omega_cmd.astype(np.float32),
        ]
    )

    state = np.asarray(
        timestep.observation[:10],
        dtype=np.float32,
    )

    p_error = np.linalg.norm(
        state[0:3]
        - env.reference.position[k]
    )

    v_error = np.linalg.norm(
        state[3:6]
        - env.reference.velocity[k]
    )

    q = state[6:10]
    q_ref = env.reference.quaternion[k]

    q = q / (
        np.linalg.norm(q) + 1e-8
    )

    q_ref = q_ref / (
        np.linalg.norm(q_ref) + 1e-8
    )

    attitude_error = 1.0 - abs(
        np.dot(q, q_ref)
    )

    position_errors.append(p_error)
    velocity_errors.append(v_error)
    attitude_errors.append(attitude_error)
    actions.append(action)

    timestep = env.step(action)
    rewards.append(float(timestep.reward))

    if timestep.last():
        break

print(
    "rollout length:",
    len(actions),
)

print(
    "return:",
    float(np.sum(rewards)),
)

print(
    "position RMSE:",
    float(
        np.sqrt(
            np.mean(
                np.square(position_errors)
            )
        )
    ),
)

print(
    "velocity RMSE:",
    float(
        np.sqrt(
            np.mean(
                np.square(velocity_errors)
            )
        )
    ),
)

print(
    "max position error:",
    float(np.max(position_errors)),
)

print(
    "max velocity error:",
    float(np.max(velocity_errors)),
)

print(
    "max quaternion error:",
    float(np.max(attitude_errors)),
)

print(
    "action min:",
    np.min(actions, axis=0),
)

print(
    "action max:",
    np.max(actions, axis=0),
)

print(
    "termination reason:",
    env.termination_reason,
)
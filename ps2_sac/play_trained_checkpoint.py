from __future__ import annotations

import os
import pickle

import acme
import jax
import numpy as np

from src.agents.sac import SAC
from src.envs.quadrotor.env_powerloop import QuadrotorPowerLoopEnv
from src.envs.quadrotor.mujoco_playback import playback_trajectory


CHECKPOINT_PATH = (
    "results/quad_vanilla_ps2rl/"
    "quad_vanilla_ps2rl_mm.pickle"
)


def main() -> None:
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(CHECKPOINT_PATH)

    with open(CHECKPOINT_PATH, "rb") as file:
        checkpoint = pickle.load(file)

    config = checkpoint["model"]["config"]
    learner_state = checkpoint["model"]["learner_state"]

    env = QuadrotorPowerLoopEnv(
        for_evaluation=True,
        seed=101,
        horizon=int(getattr(config, "horizon", 111)),
        include_time_features=bool(
            getattr(config, "include_time_features", True)
        ),
        initial_position_noise=float(
            getattr(config, "initial_position_noise", 0.1)
        ),
        perturb_evaluation=bool(
            getattr(config, "perturb_evaluation", True)
        ),
        terminate_on_ceiling=bool(
            getattr(config, "terminate_on_ceiling", False)
        ),
        z_max=float(getattr(config, "z_max", 15.0)),
        use_cpp_body_rate_reference=bool(
            getattr(config, "use_cpp_body_rate_reference", False)
        ),
        dt=float(getattr(config, "dt", 0.02)),
        w_pos_xy=float(getattr(config, "w_pos_xy", 2.5)),
        w_pos_z=float(getattr(config, "w_pos_z", 2.0)),
        w_vel=float(getattr(config, "w_vel", 4.0)),
        w_att=float(getattr(config, "w_att", 16.0)),
        w_ref_omega_x=float(
            getattr(config, "w_ref_omega_x", 0.10)
        ),
        w_ref_omega_y=float(
            getattr(config, "w_ref_omega_y", 0.20)
        ),
        w_ref_omega_z=float(
            getattr(config, "w_ref_omega_z", 0.05)
        ),
        w_control_a=float(
            getattr(config, "w_control_a", 0.01)
        ),
        w_control_omega=float(
            getattr(config, "w_control_omega", 0.01)
        ),
    )

    environment_spec = acme.make_environment_spec(env)

    # 네트워크 함수만 복원하기 위한 SAC 객체다.
    agent = SAC(
        jax.random.PRNGKey(0),
        environment_spec,
        config,
        use_cil=False,
        cil_provider_params=None,
        constraint_provider=None,
    )

    timestep = env.reset()
    trajectory = [
        np.asarray(timestep.observation[:10], dtype=np.float32)
    ]
    actions = []
    rewards = []

    rng = jax.random.PRNGKey(12345)

    while not timestep.last():
        rng, action_key = jax.random.split(rng)

        action = agent.get_action(
            action_key,
            learner_state.params.policy,
            timestep.observation,
            True,  # deterministic
        )
        action = np.asarray(action, dtype=np.float32)

        timestep = env.step(action)

        trajectory.append(
            np.asarray(timestep.observation[:10], dtype=np.float32)
        )
        actions.append(action)
        rewards.append(float(timestep.reward))

    trajectory = np.asarray(trajectory, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)

    rollout_path = os.path.join(
        os.path.dirname(CHECKPOINT_PATH),
        "quadrotor_trained_rollout.npz",
    )
    np.savez(
        rollout_path,
        trajectory=trajectory,
        actions=actions,
        rewards=rewards,
    )

    print("Rollout return:", float(np.sum(rewards)))
    print("Trajectory shape:", trajectory.shape)
    print("Actions shape:", actions.shape)
    print("Saved rollout:", rollout_path)

    playback_trajectory(
        trajectory,
        dt=float(getattr(config, "dt", 0.02)),
        realtime=True,
    )


if __name__ == "__main__":
    main()
import numpy as np

from src.envs.quadrotor.env import QuadrotorEnv
from src.envs.quadrotor.mujoco_playback import playback_trajectory


def main():
    env = QuadrotorEnv(
        for_evaluation=True,
        seed=0,
        horizon=500,
    )

    timestep = env.reset()

    trajectory = [np.asarray(timestep.observation, dtype=np.float32)]
    actions = []
    rewards = []

    hover_action = np.array([9.81, 0.0, 0.0, 0.0], dtype=np.float32)

    for _ in range(500):
        timestep = env.step(hover_action)

        trajectory.append(np.asarray(timestep.observation, dtype=np.float32))
        actions.append(hover_action.copy())
        rewards.append(float(timestep.reward))

        if timestep.last():
            break

    trajectory = np.asarray(trajectory, dtype=np.float32)

    print("return:", np.sum(rewards))
    print("final state:", trajectory[-1])
    print("final position:", trajectory[-1, 0:3])
    print("final velocity:", trajectory[-1, 3:6])
    print("final quat:", trajectory[-1, 6:10])

    playback_trajectory(
        trajectory,
        dt=0.02,
        realtime=True,
    )


if __name__ == "__main__":
    main()
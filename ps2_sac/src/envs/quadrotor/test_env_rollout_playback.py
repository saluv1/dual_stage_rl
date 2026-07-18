from __future__ import annotations

import numpy as np

from src.envs.quadrotor.env import QuadrotorEnv
from src.envs.quadrotor.mujoco_playback import playback_trajectory


def main():
    env = QuadrotorEnv(for_evaluation=True, seed=0, horizon=600)
    timestep = env.reset()

    dt = 0.02
    g = 9.81

    z_center = 2.0
    amp = 0.4
    omega = 1.0

    kp_z = 8.0
    kd_z = 5.0

    for k in range(600):
        t = k * dt

        x = env._state
        z = x[2]
        vz = x[5]

        z_ref = z_center + amp * np.sin(omega * t)
        vz_ref = amp * omega * np.cos(omega * t)
        az_ref = -amp * omega**2 * np.sin(omega * t)

        # vertical PD + feedforward
        thrust = g + az_ref + kp_z * (z_ref - z) + kd_z * (vz_ref - vz)

        action = np.array(
            [
                thrust,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        timestep = env.step(action)

        if timestep.last():
            print("Episode ended at step:", k)
            break

    trajectory = np.asarray(env.trajectory, dtype=np.float32)

    print("trajectory shape:", trajectory.shape)
    print("z min/max:", trajectory[:, 2].min(), trajectory[:, 2].max())

    playback_trajectory(trajectory, dt=dt)


if __name__ == "__main__":
    main()
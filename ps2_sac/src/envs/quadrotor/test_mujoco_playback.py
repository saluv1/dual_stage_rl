"""
Run from src:

    python -m envs.quadrotor.test_mujoco_playback
"""

import numpy as np

from src.envs.quadrotor.mujoco_playback import playback_trajectory


def main():
    T = 300
    dt = 0.02

    traj = []
    for k in range(T):
        t = k * dt

        px = 0.0
        py = 0.0
        pz = 2.0 + 0.5 * np.sin(2.0 * np.pi * t)

        vx = 0.0
        vy = 0.0
        vz = 0.0

        # identity quaternion
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        traj.append([px, py, pz, vx, vy, vz, qw, qx, qy, qz])

    trajectory = np.asarray(traj, dtype=np.float32)
    playback_trajectory(trajectory, dt=dt)


if __name__ == "__main__":
    main()
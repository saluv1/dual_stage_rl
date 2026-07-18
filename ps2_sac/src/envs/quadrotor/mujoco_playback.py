from __future__ import annotations

import time
import numpy as np
import mujoco
import mujoco.viewer

from src.envs.quadrotor.mjcf import QUADROTOR_MJCF


def state_to_mujoco_qpos(x: np.ndarray) -> np.ndarray:
    p = x[0:3]
    q = x[6:10]
    q = q / (np.linalg.norm(q) + 1e-8)
    return np.concatenate([p, q], axis=0)


def playback_trajectory(
    trajectory: np.ndarray,
    dt: float = 0.02,
    realtime: bool = True,
) -> None:
    model = mujoco.MjModel.from_xml_string(QUADROTOR_MJCF)
    data = mujoco.MjData(model)

    print("Loaded MuJoCo model")
    print("nq:", model.nq, "nv:", model.nv)
    print("trajectory shape:", trajectory.shape)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Camera setup
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 2.0])
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0

        # Give the viewer time to open.
        for _ in range(30):
            viewer.sync()
            time.sleep(0.02)

        k = 0
        while viewer.is_running():
            x = trajectory[k % len(trajectory)]

            data.qpos[:] = state_to_mujoco_qpos(np.asarray(x))
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            k += 1

            if realtime:
                time.sleep(dt)

def playback_trajectory_with_thrust_bars(
    trajectory: np.ndarray,
    u_nom_log: np.ndarray,
    u_safe_log: np.ndarray,
    dt: float = 0.02,
    realtime: bool = True,
    g: float = 9.81,
) -> None:
    """
    Play trajectory and visualize nominal/safe thrust as two vertical bars.

    Blue-ish bar:
        u_nom thrust

    Green-ish bar:
        u_safe thrust
    """
    import time
    import numpy as np
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_string(QUADROTOR_MJCF)
    data = mujoco.MjData(model)

    print("Loaded MuJoCo model")
    print("trajectory shape:", trajectory.shape)
    print("u_nom_log shape:", u_nom_log.shape)
    print("u_safe_log shape:", u_safe_log.shape)

    thrust_scale = 0.08

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 2.0])
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0

        for _ in range(30):
            viewer.sync()
            time.sleep(0.02)

        k = 0
        while viewer.is_running():
            idx = k % min(len(trajectory), len(u_nom_log), len(u_safe_log))
            x = trajectory[idx]

            data.qpos[:] = state_to_mujoco_qpos(np.asarray(x))
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)

            # This clears user-added geoms from the previous frame.
            viewer.user_scn.ngeom = 0

            nom_thrust = float(u_nom_log[idx, 0])
            safe_thrust = float(u_safe_log[idx, 0])

            nom_height = max(0.01, nom_thrust * thrust_scale)
            safe_height = max(0.01, safe_thrust * thrust_scale)

            # Nominal thrust bar.
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([0.08, 0.08, 0.5 * nom_height]),
                np.array([-1.0, -1.2, 0.5 * nom_height]),
                np.eye(3).reshape(-1),
                np.array([0.1, 0.2, 1.0, 0.7]),
            )
            viewer.user_scn.ngeom += 1

            # Safe thrust bar.
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([0.08, 0.08, 0.5 * safe_height]),
                np.array([-0.7, -1.2, 0.5 * safe_height]),
                np.eye(3).reshape(-1),
                np.array([0.1, 1.0, 0.2, 0.7]),
            )
            viewer.user_scn.ngeom += 1

            viewer.sync()

            if k % 30 == 0:
                print(
                    f"k={idx:04d} "
                    f"z={x[2]:.3f} "
                    f"u_nom_thrust={nom_thrust:.3f} "
                    f"u_safe_thrust={safe_thrust:.3f}"
                )

            k += 1

            if realtime:
                time.sleep(dt)
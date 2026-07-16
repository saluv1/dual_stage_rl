import numpy as np
from ds_rl.Z_JungwonFolders.env.dynamics import Dynamics
from ds_rl.Z_JungwonFolders.bcbf.lqrgain import LQRGain
from ds_rl.Z_JungwonFolders.desired_trajectory.desired_trajectory import DesiredTrajectory
import matplotlib.pyplot as plt


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def compute_lqr_state(state, desired_state):
    """
    Computes the 7D LQR tracking state.

    state:
    [px, py, pz, vx, vy, vz, qw, qx, qy, qz]

    desired_state:
    [px_ref, py_ref, pz_ref, vx_ref, vy_ref, vz_ref,
     qw_ref, qx_ref, qy_ref, qz_ref]

    output:
    [
        pz - pz_ref,
        vx - vx_ref,
        vy - vy_ref,
        vz - vz_ref,
        2*q_err_x,
        2*q_err_y,
        2*q_err_z
    ]
    """

    # Current state
    pz = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    q = state[6:10]

    # Desired state
    pz_ref = desired_state[2]
    vx_ref = desired_state[3]
    vy_ref = desired_state[4]
    vz_ref = desired_state[5]
    q_ref = desired_state[6:10]

    # Quaternion error: q_err = q_ref^{-1} ⊗ q
    q_ref_inv = quat_conjugate(q_ref)
    q_err = quat_multiply(q_ref_inv, q)

    # Sign correction so we use the shortest attitude error
    if q_err[0] < 0:
        q_err = -q_err

    # 7D LQR state
    x_lqr = np.array([
        pz - pz_ref,
        vx - vx_ref,
        vy - vy_ref,
        vz - vz_ref,
        2.0 * q_err[1],
        2.0 * q_err[2],
        2.0 * q_err[3],
    ])

    return x_lqr


# Compute dynamics
dyn = Dynamics()
state = dyn.reset()

# Compute feedforward reference input for trajectory tracking
trajectory = DesiredTrajectory(traj_type=1)
desired_trajectory, ref_input = trajectory.compute_trajectory()

# Compute LQR gain
lqr = LQRGain()
K, P = lqr.gain()

# Constants
g = 9.81
num_steps = 500

for i in range(num_steps):
    
    # Current desired state and input
    desired_state = desired_trajectory[i]
    u_ref = ref_input[i]

    # Convert 10D state error into 7D LQR state
    x_lqr = compute_lqr_state(state, desired_state)

    # Feedforward input + LQR feedback
    u = u_ref - K @ x_lqr

    # Clip input
    u[0] = np.clip(u[0], 0.0, 4.0 * g)
    u[1:] = np.clip(u[1:], -18.0, 18.0)

    # Step dynamics
    state = dyn.step(u)

print(state)

# Convert stored lists to arrays
xlist = np.array(dyn.xlist)
vlist = np.array(dyn.vlist)
qlist = np.array(dyn.qlist)

# Time array
t = np.arange(len(xlist)) * dyn.del_t

# Plot position
plt.figure()
plt.plot(t, xlist[:, 0], label="x")
plt.plot(t, xlist[:, 1], label="y")
plt.plot(t, xlist[:, 2], label="z")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Position")
plt.legend()
plt.grid()
plt.show()

# Plot velocity
plt.figure()
plt.plot(t, vlist[:, 0], label="vx")
plt.plot(t, vlist[:, 1], label="vy")
plt.plot(t, vlist[:, 2], label="vz")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Velocity")
plt.legend()
plt.grid()
plt.show()

# Plot quaternion
plt.figure()
plt.plot(t, qlist[:, 0], label="qw")
plt.plot(t, qlist[:, 1], label="qx")
plt.plot(t, qlist[:, 2], label="qy")
plt.plot(t, qlist[:, 3], label="qz")
plt.xlabel("Time [s]")
plt.ylabel("Quaternion")
plt.title("Quaternion")
plt.legend()
plt.grid()
plt.show()

# Plot 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(xlist[:, 0], xlist[:, 1], xlist[:, 2], label="actual")

desired_trajectory = np.array(desired_trajectory)
ax.plot(
    desired_trajectory[:num_steps, 0],
    desired_trajectory[:num_steps, 1],
    desired_trajectory[:num_steps, 2],
    "--",
    label="desired"
)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("3D Trajectory")
ax.legend()

plt.show()
import numpy as np
from quadrotor import Dynamics
import matplotlib.pyplot as plt

dyn = Dynamics()

state = dyn.reset()

for i in range(100):
    u = np.array([9.807, 0.0, 0.0, 0.0])
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

ax.plot(xlist[:, 0], xlist[:, 1], xlist[:, 2])
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("3D Trajectory")

plt.show()
"""State representation, set indicators, action scaling, and dynamics reset."""
import numpy as np

def compute_reduced_state(full_state):

    z_des = 2.0

    pz = full_state[2]
    vx = full_state[3]
    vy = full_state[4]
    vz = full_state[5]

    qw = full_state[6]
    qx = full_state[7]
    qy = full_state[8]
    qz = full_state[9]

    if qw < 0.0:
        qx = -qx
        qy = -qy
        qz = -qz

    reduced_state = np.array([
        pz - z_des,
        vx,
        vy,
        vz,
        2.0 * qx,
        2.0 * qy,
        2.0 * qz
    ])

    return reduced_state

def compute_bfc(sets, full_state):

    reduced_state = compute_reduced_state(full_state)

    indicator = sets.compute_indicator(full_state, reduced_state)

    if indicator == 0:
        b = 1.0
        f = 0.0
        c = 0.0
    elif indicator == 2:
        b = 0.0
        f = 1.0
        c = 0.0
    else:
        b = 0.0
        f = 0.0
        c = 1.0

    return b, f, c

def scale_action(action_norm, g):

    action_norm = np.array(action_norm, dtype=float)
    action_norm = np.clip(action_norm, -1.0, 1.0)

    a_cmd = 2.0 * g * (action_norm[0] + 1.0)
    wx = 18.0 * action_norm[1]
    wy = 18.0 * action_norm[2]
    wz = 18.0 * action_norm[3]

    return np.array([a_cmd, wx, wy, wz])

def unscale_action(action_phys, g):
    """Inverse of `scale_action`: physical command -> normalized action."""

    a_cmd = action_phys[0]
    w = action_phys[1:4]

    a0 = a_cmd / (2.0 * g) - 1.0
    a1 = w[0] / 18.0
    a2 = w[1] / 18.0
    a3 = w[2] / 18.0

    return np.clip(np.array([a0, a1, a2, a3]), -1.0, 1.0)

def reset_dynamics_state(dyn, state):

    dyn.state = state.copy()

    if hasattr(dyn, "curr_step"):
        dyn.curr_step = 0

    if hasattr(dyn, "xlist"):
        dyn.xlist = []

    if hasattr(dyn, "vlist"):
        dyn.vlist = []

    if hasattr(dyn, "qlist"):
        dyn.qlist = []

def normalize_quat(q):

    q = np.array(q, dtype=float)
    q_norm = np.linalg.norm(q)

    if q_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / q_norm

def quat_from_axis_angle(axis, angle):

    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)

    half = angle / 2.0

    w = np.cos(half)
    xyz = axis * np.sin(half)

    return np.array([w, xyz[0], xyz[1], xyz[2]])

def quat_mult(q1, q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

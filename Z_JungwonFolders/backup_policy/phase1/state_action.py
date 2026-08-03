"""State, set-indicator, and action utilities for Phase-I."""
from __future__ import annotations

import numpy as np


def normalize_quat(q):
    q = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    return np.array([1.0, 0.0, 0.0, 0.0]) if norm < 1e-12 else q / norm


def compute_reduced_state(full_state, z_des=2.0):
    state = np.asarray(full_state, dtype=float).reshape(10)
    q = normalize_quat(state[6:10])
    # Official error uses conjugate(q); select the shortest quaternion branch.
    q_err = np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
    sign = 1.0 if q_err[0] >= 0.0 else -1.0
    theta = 2.0 * sign * q_err[1:4]
    return np.array([
        state[2] - float(z_des),
        state[3], state[4], state[5],
        theta[0], theta[1], theta[2],
    ], dtype=float)


def compute_bfc(sets, full_state):
    reduced_state = compute_reduced_state(full_state)
    indicator = sets.compute_indicator(full_state, reduced_state)
    if indicator == 0:
        return 1.0, 0.0, 0.0
    if indicator == 2:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def scale_action(action_norm, g=9.81):
    a = np.clip(np.asarray(action_norm, dtype=float).reshape(4), -1.0, 1.0)
    return np.array([2.0 * g * (a[0] + 1.0), 18.0*a[1], 18.0*a[2], 18.0*a[3]], dtype=float)


def unscale_action(action_phys, g=9.81):
    u = np.asarray(action_phys, dtype=float).reshape(4)
    return np.clip(np.array([u[0]/(2.0*g)-1.0, u[1]/18.0, u[2]/18.0, u[3]/18.0]), -1.0, 1.0)



def lqr_action(full_state, K, g=9.81):
    """Official hover-DLQR action in physical units."""
    error = compute_reduced_state(full_state)
    u_star = np.array([float(g), 0.0, 0.0, 0.0], dtype=float)
    action = u_star - np.asarray(K, dtype=float) @ error
    return np.array([
        np.clip(action[0], 0.0, 4.0 * float(g)),
        np.clip(action[1], -18.0, 18.0),
        np.clip(action[2], -18.0, 18.0),
        np.clip(action[3], -18.0, 18.0),
    ], dtype=float)

def reset_dynamics_state(dyn, state):
    if hasattr(dyn, "setState"):
        dyn.setState(state)
    else:
        dyn.state = np.asarray(state, dtype=float).copy()
        dyn.curr_step = 0
        dyn.xlist, dyn.vlist, dyn.qlist = [], [], []


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * float(angle)
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def quat_mult(q1, q2):
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)

"""10-D quadrotor dynamics used by the local PS2-RL implementation.

The class keeps the original repository API while matching the released
PS2-RL state/action convention.  Training uses the official forward-Euler
step.  RK45 remains available for the separate numerical evaluation command.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


class Dynamics:
    def __init__(self, integrator: str = "euler", dt: float = 0.02, gravity: float = 9.81) -> None:
        if integrator not in {"euler", "rk45"}:
            raise ValueError("integrator must be 'euler' or 'rk45'")
        self.integrator = integrator
        self.del_t = float(dt)
        self.total_steps = int(round(10.0 / self.del_t))
        self.curr_step = 0
        self.g = float(gravity)
        self.a_cmd_min = 0.0
        self.a_cmd_max = 4.0 * self.g
        self.omega_max = 18.0

        self.x0 = np.array([0.0, 0.0, 0.0], dtype=float)
        self.v0 = np.array([0.0, 0.0, 0.0], dtype=float)
        self.q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.state = self.packState(self.x0, self.v0, self.q0)
        self.xlist: list[np.ndarray] = []
        self.vlist: list[np.ndarray] = []
        self.qlist: list[np.ndarray] = []

    def clipAction(self, u):
        u = np.asarray(u, dtype=float).reshape(4)
        return np.array([
            np.clip(u[0], self.a_cmd_min, self.a_cmd_max),
            np.clip(u[1], -self.omega_max, self.omega_max),
            np.clip(u[2], -self.omega_max, self.omega_max),
            np.clip(u[3], -self.omega_max, self.omega_max),
        ], dtype=float)

    def equationsOfMotion(self, t, state, u):
        del t
        state = np.asarray(state, dtype=float).reshape(10)
        u = self.clipAction(u)
        v = state[3:6]
        q = self.normalizeQuaternion(state[6:10])
        a_cmd = u[0]
        w_cmd = u[1:4]

        x_dot = v
        v_dot = -self.g * np.array([0.0, 0.0, 1.0]) + self.R(q) @ np.array([0.0, 0.0, a_cmd])

        qw, qx, qy, qz = q
        xi = np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw],
        ], dtype=float)
        q_dot = 0.5 * xi @ w_cmd
        return np.concatenate([x_dot, v_dot, q_dot])

    def integrate(self, state, u):
        state = np.asarray(state, dtype=float).reshape(10)
        u = self.clipAction(u)
        if self.integrator == "euler":
            next_state = state + self.del_t * self.equationsOfMotion(0.0, state, u)
        else:
            sol = solve_ivp(
                fun=lambda t, y: self.equationsOfMotion(t, y, u),
                t_span=(0.0, self.del_t),
                y0=state,
                method="RK45",
                rtol=1e-7,
                atol=1e-9,
            )
            if not sol.success:
                raise RuntimeError(f"RK45 integration failed: {sol.message}")
            next_state = sol.y[:, -1]
        next_state = np.asarray(next_state, dtype=float)
        next_state[6:10] = self.normalizeQuaternion(next_state[6:10])
        return next_state

    def step(self, u):
        self.state = self.integrate(self.state, u)
        x, v, q = self.unpackState(self.state)
        self.xlist.append(x.copy())
        self.vlist.append(v.copy())
        self.qlist.append(q.copy())
        self.curr_step += 1
        return self.state

    def setState(self, state):
        self.state = np.asarray(state, dtype=float).reshape(10).copy()
        self.state[6:10] = self.normalizeQuaternion(self.state[6:10])
        self.curr_step = 0
        self.xlist = []
        self.vlist = []
        self.qlist = []
        return self.state

    def reset(self):
        return self.setState(self.packState(self.x0, self.v0, self.q0))

    def packState(self, x, v, q):
        return np.concatenate([np.asarray(x, dtype=float), np.asarray(v, dtype=float), self.normalizeQuaternion(q)])

    def normalizeQuaternion(self, q):
        q = np.asarray(q, dtype=float).reshape(4)
        norm_q = float(np.linalg.norm(q))
        return np.array([1.0, 0.0, 0.0, 0.0]) if norm_q < 1e-12 else q / norm_q

    def unpackState(self, state):
        state = np.asarray(state, dtype=float)
        return state[0:3], state[3:6], state[6:10]

    def R(self, q):
        qw, qx, qy, qz = self.normalizeQuaternion(q)
        return np.array([
            [1.0 - 2.0 * (qy*qy + qz*qz), 2.0 * (qx*qy - qw*qz), 2.0 * (qx*qz + qw*qy)],
            [2.0 * (qx*qy + qw*qz), 1.0 - 2.0 * (qx*qx + qz*qz), 2.0 * (qy*qz - qw*qx)],
            [2.0 * (qx*qz - qw*qy), 2.0 * (qy*qz + qw*qx), 1.0 - 2.0 * (qx*qx + qy*qy)],
        ], dtype=float)

"""
Given linearized system matrices (A, B), solve the discretized LQR problem.

1. Use exact ZOH discretization for the reduced linearized quadrotor model.
2. Given Qd and Rd, solve the discrete algebraic Riccati equation.
3. Output the LQR gain K and Riccati matrix P.

State order:
x = [
    p_z - z_des,
    v_x,
    v_y,
    v_z,
    2*q_err_x,
    2*q_err_y,
    2*q_err_z,
]

Input order:
u = [
    a_cmd,
    omega_x,
    omega_y,
    omega_z,
]

Equilibrium points:
x* = [0, 0, 0, 0, 0, 0, 0]^T
u* = [g, 0, 0, 0]^T.
"""

import numpy as np
from scipy.linalg import solve_discrete_are


class LQRGain:

    def __init__(self, dt=0.02, g=9.807):
        self.dt = dt
        self.g = g

        dt = self.dt
        g = self.g

        self.u_star = np.array([g, 0.0, 0.0, 0.0])

        self.Ad = np.array([
            [1, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, g * dt, 0],
            [0, 0, 1, 0, -g * dt, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        self.Bd = np.array([
            [0.5 * dt**2, 0, 0, 0],
            [0, 0, 0.5 * g * dt**2, 0],
            [0, -0.5 * g * dt**2, 0, 0],
            [dt, 0, 0, 0],
            [0, dt, 0, 0],
            [0, 0, dt, 0],
            [0, 0, 0, dt],
        ], dtype=float)

        self.Qd = np.diag([1.0, 0.16, 0.16, 0.4, 0.8, 0.8, 0.16])

        self.Rd = np.diag([0.02, 0.012, 0.012, 0.004])

        self.P = None
        self.K = None

    def gain(self):
        self.P = solve_discrete_are(self.Ad, self.Bd, self.Qd, self.Rd)

        self.K = np.linalg.solve(
            self.Rd + self.Bd.T @ self.P @ self.Bd,
            self.Bd.T @ self.P @ self.Ad
        )

        return self.K, self.P
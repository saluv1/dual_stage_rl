"""Official PS2-RL forward-Euler hover DLQR, kept in the local class style."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


class LQRGain:
    def __init__(self, dt=0.02, g=9.81):
        self.dt = float(dt)
        self.g = float(g)
        self.u_star = np.array([self.g, 0.0, 0.0, 0.0], dtype=float)

        # e = [z-z_des, vx, vy, vz, theta_x, theta_y, theta_z].
        # theta is the sign-corrected vector part of conjugate(q), multiplied by 2.
        A = np.zeros((7, 7), dtype=float)
        A[0, 3] = 1.0
        A[1, 5] = -self.g
        A[2, 4] = self.g

        B = np.zeros((7, 4), dtype=float)
        B[3, 0] = 1.0
        B[4, 1] = -1.0
        B[5, 2] = -1.0
        B[6, 3] = -1.0

        self.Ad = np.eye(7) + self.dt * A
        self.Bd = self.dt * B
        self.Qd = np.diag([1.0, 0.16, 0.16, 0.4, 0.8, 0.8, 0.16])
        self.Rd = np.diag([0.02, 0.012, 0.012, 0.004])
        self.P = None
        self.K = None

    def gain(self):
        self.P = solve_discrete_are(self.Ad, self.Bd, self.Qd, self.Rd)
        self.P = 0.5 * (np.real(self.P) + np.real(self.P).T)
        self.K = np.linalg.solve(
            self.Rd + self.Bd.T @ self.P @ self.Bd,
            self.Bd.T @ self.P @ self.Ad,
        )
        return self.K, self.P

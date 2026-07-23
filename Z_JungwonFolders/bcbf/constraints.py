"""
Computes BCBF constraints:

    A u <= b

where u is the physical quadrotor action:

    u = [a_cmd, wx, wy, wz]

The constraints include:

    1. BCBF safety constraints along the backup rollout
    2. terminal base-set constraint
    3. input upper/lower bound constraints

This implementation uses finite differences to compute:

    d f_pi_b / d x

for the variational equation:

    d Psi / dt = (d f_pi_b / d x) Psi

State:
    x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]

Input:
    u = [a_cmd, wx, wy, wz]
"""

import numpy as np


class Constraints:

    def __init__(
            self,
            sets,
            backup_policy,
            P,
            rollout_steps=100,
            dt=0.02,
            g=9.81,
            umax=None,
            umin=None,
            alpha_s_gain=1.0,
            alpha_b_gain=1.0,
            finite_diff_eps=1e-5
    ):

        self.sets = sets
        self.backup_policy = backup_policy
        self.P = P

        self.rollout_steps = rollout_steps
        self.dt = dt
        self.g = g

        self.state_dim = 10
        self.action_dim = 4

        if umax is None:
            self.umax = np.array([4.0 * g, 18.0, 18.0, 18.0])
        else:
            self.umax = np.array(umax, dtype=float)

        if umin is None:
            self.umin = np.array([0.0, -18.0, -18.0, -18.0])
        else:
            self.umin = np.array(umin, dtype=float)

        self.alpha_s_gain = alpha_s_gain
        self.alpha_b_gain = alpha_b_gain
        self.finite_diff_eps = finite_diff_eps

    # ----------------------------------------------------------------------
    # Quaternion and dynamics utilities
    # ----------------------------------------------------------------------

    def normalize_quat_state(self, state):

        state = np.array(state, dtype=float).copy()

        q = state[6:10]
        q_norm = np.linalg.norm(q)

        if q_norm < 1e-9:
            state[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            state[6:10] = q / q_norm

        return state

    def quat_to_rot(self, q):

        q = np.array(q, dtype=float)
        q = q / (np.linalg.norm(q) + 1e-9)

        qw, qx, qy, qz = q

        R = np.array([
            [
                1.0 - 2.0 * (qy**2 + qz**2),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy)
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx**2 + qz**2),
                2.0 * (qy * qz - qw * qx)
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx**2 + qy**2)
            ]
        ])

        return R

    def drift_dynamics(self, state):

        state = self.normalize_quat_state(state)

        v = state[3:6]

        f = np.zeros(self.state_dim)

        # p_dot = v
        f[0:3] = v

        # v_dot = -g e3
        f[3:6] = np.array([0.0, 0.0, -self.g])

        # q_dot drift = 0 because angular velocity is an input
        f[6:10] = np.zeros(4)

        return f

    def control_matrix(self, state):

        state = self.normalize_quat_state(state)

        q = state[6:10]
        qw, qx, qy, qz = q

        R = self.quat_to_rot(q)
        e3 = np.array([0.0, 0.0, 1.0])

        G = np.zeros((self.state_dim, self.action_dim))

        # a_cmd affects velocity through R e3
        G[3:6, 0] = R @ e3

        # omega affects quaternion dynamics:
        #
        # q_dot = 0.5 * q ⊗ [0, omega]
        #
        # q_dot = 0.5 * E(q) omega
        E = np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw]
        ])

        G[6:10, 1:4] = 0.5 * E

        return G

    def continuous_dynamics(self, state, action):

        state = self.normalize_quat_state(state)
        action = np.array(action, dtype=float)

        f = self.drift_dynamics(state)
        G = self.control_matrix(state)

        x_dot = f + G @ action

        return x_dot

    def integrate_one_step(self, state, action):

        state = self.normalize_quat_state(state)
        action = np.array(action, dtype=float)

        # RK4 integration for continuous dynamics
        k1 = self.continuous_dynamics(state, action)
        k2 = self.continuous_dynamics(state + 0.5 * self.dt * k1, action)
        k3 = self.continuous_dynamics(state + 0.5 * self.dt * k2, action)
        k4 = self.continuous_dynamics(state + self.dt * k3, action)

        next_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = self.normalize_quat_state(next_state)

        return next_state

    # ----------------------------------------------------------------------
    # Backup closed-loop dynamics and sensitivity
    # ----------------------------------------------------------------------

    def backup_vector_field(self, state):

        state = self.normalize_quat_state(state)

        action = self.backup_policy(state)
        action = np.array(action, dtype=float)
        action = np.clip(action, self.umin, self.umax)

        x_dot = self.continuous_dynamics(state, action)

        return x_dot

    def closed_loop_jacobian(self, state):

        state = self.normalize_quat_state(state)

        eps = self.finite_diff_eps

        A_cl = np.zeros((self.state_dim, self.state_dim))

        for j in range(self.state_dim):

            dx = np.zeros(self.state_dim)
            dx[j] = eps

            state_plus = self.normalize_quat_state(state + dx)
            state_minus = self.normalize_quat_state(state - dx)

            f_plus = self.backup_vector_field(state_plus)
            f_minus = self.backup_vector_field(state_minus)

            A_cl[:, j] = (f_plus - f_minus) / (2.0 * eps)

        return A_cl

    def integrate_rollout_one_step(self, phi, psi):

        phi = self.normalize_quat_state(phi)

        f_backup = self.backup_vector_field(phi)
        A_cl = self.closed_loop_jacobian(phi)

        # Euler integration for phi and psi in the variational system.
        # phi is also normalized afterward because it contains quaternion state.
        phi_next = phi + self.dt * f_backup
        phi_next = self.normalize_quat_state(phi_next)

        psi_next = psi + self.dt * (A_cl @ psi)

        return phi_next, psi_next

    def rollout_backup_and_sensitivity(self, x0):

        x0 = self.normalize_quat_state(x0)

        phi_list = []
        psi_list = []

        phi = x0.copy()
        psi = np.eye(self.state_dim)

        phi_list.append(phi.copy())
        psi_list.append(psi.copy())

        for _ in range(self.rollout_steps):

            phi, psi = self.integrate_rollout_one_step(phi, psi)

            phi_list.append(phi.copy())
            psi_list.append(psi.copy())

        phi_array = np.array(phi_list)
        psi_array = np.array(psi_list)

        return phi_array, psi_array

    # ----------------------------------------------------------------------
    # Barrier functions and gradients
    # ----------------------------------------------------------------------

    def compute_reduced_state(self, full_state):

        z_des = 2.0

        full_state = self.normalize_quat_state(full_state)

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

    def h_s(self, state):

        state = self.normalize_quat_state(state)

        pz = state[2]

        hs = self.sets.zceil - pz

        return hs

    def grad_h_s(self, state):

        grad = np.zeros(self.state_dim)

        # h_S = zceil - pz
        grad[2] = -1.0

        return grad

    def h_b(self, state):

        xe = self.compute_reduced_state(state)

        hb = self.sets.c_b - xe.T @ self.P @ xe

        return hb

    def grad_h_b(self, state):

        state = self.normalize_quat_state(state)

        xe = self.compute_reduced_state(state)

        # h_B = c_B - xe^T P xe
        # d h_B / d xe = -2 P xe
        r = -2.0 * self.P @ xe

        grad = np.zeros(self.state_dim)

        # xe = [
        #   pz - z_des,
        #   vx,
        #   vy,
        #   vz,
        #   2 qx,
        #   2 qy,
        #   2 qz
        # ]

        grad[2] = r[0]        # pz
        grad[3] = r[1]        # vx
        grad[4] = r[2]        # vy
        grad[5] = r[3]        # vz

        # quaternion sign convention
        qw = state[6]
        sign = 1.0

        if qw < 0.0:
            sign = -1.0

        grad[7] = 2.0 * sign * r[4]   # qx
        grad[8] = 2.0 * sign * r[5]   # qy
        grad[9] = 2.0 * sign * r[6]   # qz

        return grad

    def alpha_s(self, h):

        return self.alpha_s_gain * h

    def alpha_b(self, h):

        return self.alpha_b_gain * h

    # ----------------------------------------------------------------------
    # Construct A and b
    # ----------------------------------------------------------------------

    def compute_bcbf_constraints(self, x0):

        x0 = self.normalize_quat_state(x0)

        phi_list, psi_list = self.rollout_backup_and_sensitivity(x0)

        f0 = self.drift_dynamics(x0)
        g0 = self.control_matrix(x0)

        A_rows = []
        b_rows = []

        # --------------------------------------------------------------
        # Safety constraints for every backup rollout node
        # --------------------------------------------------------------
        for i in range(1, self.rollout_steps + 1):

            phi_i = phi_list[i]
            psi_i = psi_list[i]

            hs_i = self.h_s(phi_i)
            grad_hs_i = self.grad_h_s(phi_i)

            f_backup_i = self.backup_vector_field(phi_i)

            # a_s^T = grad_hS(phi_i)^T Psi_i g(x0)
            a_s = grad_hs_i @ psi_i @ g0

            # b_s = grad_hS(phi_i)^T [Psi_i f(x0) - f_backup(phi_i)]
            #       + alpha_S(hS(phi_i))
            b_s = (
                grad_hs_i @ (psi_i @ f0 - f_backup_i)
                + self.alpha_s(hs_i)
            )

            # a_s u + b_s >= 0
            # => -a_s u <= b_s
            A_rows.append(-a_s)
            b_rows.append(b_s)

        # --------------------------------------------------------------
        # Terminal base-set constraint at final backup node
        # --------------------------------------------------------------
        phi_N = phi_list[-1]
        psi_N = psi_list[-1]

        hb_N = self.h_b(phi_N)
        grad_hb_N = self.grad_h_b(phi_N)

        # a_b^T = grad_hB(phi_N)^T Psi_N g(x0)
        a_b = grad_hb_N @ psi_N @ g0

        # b_b = grad_hB(phi_N)^T Psi_N f(x0)
        #       + alpha_B(hB(phi_N))
        b_b = (
            grad_hb_N @ (psi_N @ f0)
            + self.alpha_b(hb_N)
        )

        # a_b u + b_b >= 0
        # => -a_b u <= b_b
        A_rows.append(-a_b)
        b_rows.append(b_b)

        A_bcbf = np.vstack(A_rows)
        b_bcbf = np.array(b_rows)

        return A_bcbf, b_bcbf, phi_list, psi_list

    def compute_input_constraints(self):

        # u <= umax
        A_upper = np.eye(self.action_dim)
        b_upper = self.umax.copy()

        # -u <= -umin
        A_lower = -np.eye(self.action_dim)
        b_lower = -self.umin.copy()

        A_u = np.vstack([
            A_upper,
            A_lower
        ])

        b_u = np.concatenate([
            b_upper,
            b_lower
        ])

        return A_u, b_u

    def compute_constraints(self, x0):

        A_bcbf, b_bcbf, phi_list, psi_list = self.compute_bcbf_constraints(x0)

        A_u, b_u = self.compute_input_constraints()

        A = np.vstack([
            A_bcbf,
            A_u
        ])

        b = np.concatenate([
            b_bcbf,
            b_u
        ])

        return A, b, A_bcbf, b_bcbf, A_u, b_u, phi_list, psi_list
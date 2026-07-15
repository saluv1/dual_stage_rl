"""
Dynamics model for quadrotor.
"""

import numpy as np
from scipy.integrate import solve_ivp


class Dynamics:

    def __init__(self) -> None:

        # Timestep settings
        self.del_t = 0.01
        self.total_steps = int(10 / self.del_t)
        self.curr_step = 0

        # Current state
        self.x0 = np.array([0.0, 0.0, 0.0])
        self.v0 = np.array([0.0, 0.0, 0.0])
        self.q0 = np.array([1, 0, 0, 0])

        # Current state
        self.state = self.packState(self.x0, self.v0, self.q0)

        # Lists to store full state
        self.xlist = []
        self.vlist = []
        self.qlist = []

        # Constants
        self.g = 9.807

    def equationsOfMotion(self, t, state, u):
        """
        Equations of motion for quadrotor.
        """

        # Unpack states
        x = state[0:3]
        v = state[3:6]
        q = state[6:10]

        # Unpack actions
        a_cmd = u[0]
        w_cmd = u[1:4]

        # Translational kinematics
        x_dot = v

        # Translational dynamics
        gravity_term = -self.g * np.array([0,0,1])
        rotational_term = np.matmul(self.R(q), a_cmd*np.array([0,0,1]))
        v_dot = gravity_term + rotational_term

        # Rotational kinematics
        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        q_dot_matrix = np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw]
        ])

        q_dot = 0.5*np.matmul(q_dot_matrix, w_cmd)

        # Return state_dot
        state_dot = np.concatenate([x_dot, v_dot, q_dot])

        return state_dot
    
    def integrate(self, state, u):
        """
        Integrate dynamics
        """

        sol = solve_ivp(
            fun = lambda t, y: self.equationsOfMotion(t,y,u),
            t_span = (0.0, self.del_t),
            y0=state,
            method="RK45",
            rtol=1e-6,
            atol=1e-9
        )

        next_state = sol.y[:,-1]

        next_state[6:10] = self.normalizeQuaternion(next_state[6:10])

        return next_state

    def step(self, u):
        """
        Given the current state and action, compute the next state
        """

        # Get ready the state and actions. Make them into numpy arrays just in case.
        u = np.array(u, dtype = float)
        state = self.state

        # Integrate to get next step
        self.state = self.integrate(state, u)
        x,v,q = self.unpackState(self.state)

        self.xlist.append(x.copy())
        self.vlist.append(v.copy())
        self.qlist.append(q.copy())

        self.curr_step += 1

        return self.state
    
    def reset(self):
        """
        Reset everything to initial conditions
        """

        self.curr_step = 0

        self.xlist = []
        self.vlist = []
        self.qlist = []

        self.state = self.packState(self.x0, self.v0, self.q0)

        return self.state
    
    def packState(self, x, v, q):
        """
        Pack position, velocity, and quaternion into one state vector.
        """

        state = np.concatenate([x, v, q])

        return state

    def normalizeQuaternion(self, q):
        """
        Normalize quaternion to avoid numerical drift.
        """

        norm_q = np.linalg.norm(q)

        if norm_q < 1e-12:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q = q / norm_q

        return q

    def unpackState(self, state):
        """
        Unpack state vector into position, velocity, and quaternion.
        """

        x = state[0:3]
        v = state[3:6]
        q = state[6:10]

        return x, v, q

    def R(self, q):
        """
        Compute rotation matrix corresponding to quaternion.

        Quaternion convention:
            q = [qw, qx, qy, qz]

        This rotation matrix maps a vector from body frame to world frame.
        """

        q = self.normalizeQuaternion(q)

        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        R = np.array([
            [
                1.0 - 2.0 * (qy**2 + qz**2),
                2.0 * (qx*qy - qw*qz),
                2.0 * (qx*qz + qw*qy)
            ],
            [
                2.0 * (qx*qy + qw*qz),
                1.0 - 2.0 * (qx**2 + qz**2),
                2.0 * (qy*qz - qw*qx)
            ],
            [
                2.0 * (qx*qz - qw*qy),
                2.0 * (qy*qz + qw*qx),
                1.0 - 2.0 * (qx**2 + qy**2)
            ]
        ])

        return R
"""
Generates the desired trajectory for quadrotor trajectory tracking
"""

import numpy as np


class DesiredTrajectory:

    def __init__(self, dt=0.02, traj_type=0):

        # Constants
        self.dt = dt
        self.traj_type = traj_type      # 0: Takeoff, 1: Circular

        # List to store trajectory points and command inputs
        self.traj_points = []
        self.command_inputs = []

    def compute_trajectory(self):

        # Reset trajectory lists
        self.traj_points = []
        self.command_inputs = []

        if self.traj_type == 0:
            
            # Takeoff trajectory
            velocity = 0.3
            num_steps = 1000
            g = 9.81

            for i in range(num_steps):

                t = self.dt * i

                # Desired position
                px = 0.0
                py = 0.0
                pz = velocity * t

                # Desired velocity
                vx = 0.0
                vy = 0.0
                vz = velocity

                # Desired attitude: level hover
                qw = 1.0
                qx = 0.0
                qy = 0.0
                qz = 0.0

                # Desired state: [p, v, q]
                self.traj_points.append(
                    np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz])
                )

                # Feedforward command input
                a_cmd = g
                wx = 0.0
                wy = 0.0
                wz = 0.0

                # Desired input: [a_cmd, omega_x, omega_y, omega_z]
                self.command_inputs.append(
                    np.array([a_cmd, wx, wy, wz])
                )
        
        else:
            
            # Circular trajectory
            radius = 0.3
            period = 10.0
            height = 0.0
            num_steps = 1000
            g = 9.81

            omega = 2.0 * np.pi / period

            for i in range(num_steps):

                t = self.dt * i

                # Desired position
                # Shifted circle so that trajectory starts from the origin.
                px = radius * (np.cos(omega * t) - 1.0)
                py = radius * np.sin(omega * t)
                pz = height

                # Desired velocity
                vx = -radius * omega * np.sin(omega * t)
                vy = radius * omega * np.cos(omega * t)
                vz = 0.0

                # Desired yaw: fixed yaw
                yaw = 0.0

                # Simple desired attitude: level hover
                qw = 1.0
                qx = 0.0
                qy = 0.0
                qz = 0.0

                # Desired state: [p, v, q]
                self.traj_points.append(
                    np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz])
                )

                # Feedforward command input
                # Small circle: use hover thrust and zero yaw rate as a simple first test.
                a_cmd = g
                wx = 0.0
                wy = 0.0
                wz = 0.0

                # Desired input: [a_cmd, omega_x, omega_y, omega_z]
                self.command_inputs.append(
                    np.array([a_cmd, wx, wy, wz])
                )

        return self.traj_points, self.command_inputs
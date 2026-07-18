"""
Quadrotor powerloop environment.

State:
    x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]

Action:
    u = [a_cmd, wx, wy, wz]
"""

from __future__ import annotations

from acme import specs
import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np

from src.envs.quadrotor.dynamics import QuadrotorParams, rk4_step
from src.envs.quadrotor.constraints import (
    QuadrotorConstraintParams,
    action_bounds,
    clip_action,
    safety_violation,
    is_safe,
)


class QuadrotorEnv(dm_env.Environment):
    def __init__(
        self,
        for_evaluation: bool = False,
        seed: int = 0,
        horizon: int = 106,
    ) -> None:
        self._for_evaluation = for_evaluation
        self._horizon = horizon
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

        self._dyn_params = QuadrotorParams(dt=0.02)
        self._constraint_params = QuadrotorConstraintParams()
        self._state = None

        if self._for_evaluation:
            self.trajectory = []
            self.actions = []
            self.rewards = []
    
    def _safety_violation(self, x):
        x = jnp.asarray(x, dtype=jnp.float32)
        return float(safety_violation(x, self._constraint_params))
    def reset(self) -> dm_env.TimeStep:
        self._step_count = 0

        # Powerloop start 근처. 일단 hover 근처로 시작.
        x0 = np.array(
            [
                0.0, 0.0, 2.0,      # p
                0.0, 0.0, 0.0,      # v
                1.0, 0.0, 0.0, 0.0  # q
            ],
            dtype=np.float32,
        )

        self._state = x0

        if self._for_evaluation:
            self.trajectory = [x0.copy()]
            self.actions = []
            self.rewards = []

        return dm_env.restart(self._state)

    def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
        if self._state is None:
            return self.reset()

        u_min, u_max = action_bounds()
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, np.asarray(u_min), np.asarray(u_max))

        x = jnp.asarray(self._state)
        u = jnp.asarray(action)

        x_next = rk4_step(x, u, self._dyn_params)
        x_next_np = np.asarray(x_next, dtype=np.float32)

        reward = self._reward(x_next, action)

        self._state = x_next_np
        self._step_count += 1

        safe = bool(is_safe(jnp.asarray(self._state)))
        timeout = self._step_count >= self._horizon
        done = timeout or not safe

        if self._for_evaluation:
            self.trajectory.append(self._state.copy())
            self.actions.append(action.copy())
            self.rewards.append(float(reward))

        if done:
            return dm_env.termination(reward, self._state)

        return dm_env.transition(reward, self._state)

    def _reward(self, x, u):
        """
        Hover tracking reward for quadrotor.

        State:
            x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]

        Action:
            u = [a_cmd, wx, wy, wz]

        Hover target:
            p_des = [0, 0, 2]
            v_des = [0, 0, 0]
            q_des = [1, 0, 0, 0]
            u_des = [g, 0, 0, 0]
        """
        x = np.asarray(x, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)

        p = x[0:3]
        v = x[3:6]
        q = x[6:10]

        a_cmd = u[0]
        omega_cmd = u[1:4]

        g = 9.81

        # Normalize quaternion.
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-6:
            q = q / q_norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Use shortest quaternion representation.
        # q and -q represent the same attitude.
        if q[0] < 0.0:
            q = -q

        # Desired hover state.
        p_des = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        v_des = np.zeros(3, dtype=np.float32)

        pos_err = p - p_des
        vel_err = v - v_des

        xy_err = pos_err[0:2]
        z_err = pos_err[2]

        # For q_des = identity, small-angle attitude error is approximately 2*q_vec.
        att_err = 2.0 * q[1:4]

        thrust_err = (a_cmd - g) / g

        # Normalize body rates by paper action limit.
        omega_err = omega_cmd / 18.0

        # Optional: ceiling violation penalty.
        # PS2-RL itself enforces safety through CIL, not reward penalty.
        # But for debugging toy CIL / vanilla SAC, this helps avoid bad trajectories.
        z_ceil = 3.0
        ceiling_violation = max(0.0, p[2] - z_ceil)

        # Weighted negative quadratic reward.
        # Inspired by paper's tracking reward structure, adapted to hover.
        cost = 0.0
        cost += 2.5 * np.sum(np.square(np.clip(xy_err, -5.0, 5.0)))
        cost += 2.0 * np.square(np.clip(z_err, -5.0, 5.0))
        cost += 4.0 * np.sum(np.square(np.clip(vel_err, -10.0, 10.0)))
        cost += 16.0 * np.sum(np.square(np.clip(att_err, -2.0, 2.0)))

        # Small control regularization.
        # This does NOT enforce safety; it just discourages unnecessarily violent actions.
        cost += 0.01 * np.square(np.clip(thrust_err, -4.0, 4.0))
        cost += 0.01 * np.sum(np.square(np.clip(omega_err, -1.0, 1.0)))

        # Debugging-only safety shaping.
        cost += 100.0 * np.square(np.clip(ceiling_violation, 0.0, 5.0))

        reward = -cost

        if not np.isfinite(reward):
            reward = -1000.0

        return float(reward)


    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(10,),
            minimum=-float("inf"),
            maximum=float("inf"),
            dtype=np.float32,
        )

    def action_spec(self) -> specs.BoundedArray:
        u_min, u_max = action_bounds()
        return specs.BoundedArray(
            shape=(4,),
            minimum=np.asarray(u_min, dtype=np.float32),
            maximum=np.asarray(u_max, dtype=np.float32),
            dtype=np.float32,
        )

    def close(self) -> None:
        pass
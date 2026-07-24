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
        self._termination_reason = None
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

        safe = bool(
            is_safe(
                jnp.asarray(self._state, dtype=jnp.float32),
                self._constraint_params,
            )
        )
        ground_collision = bool(self._state[2] <= 0.0)
        timeout = self._step_count >= self._horizon
        crash_penalty = 0.0
        xy_radius = float(
            np.linalg.norm(self._state[0:2])
        )

        lateral_escape = bool(
            xy_radius >= 5.0
        )
        if not safe:
            reward = float(reward) - crash_penalty
            self._termination_reason = "ceiling_violation"

        elif ground_collision:
            reward = float(reward) - crash_penalty
            self._termination_reason = "ground_collision"

        elif timeout:
            self._termination_reason = "timeout"
        elif lateral_escape:
            self._termination_reason= "lateral_escape"
        else:
            self._termination_reason = None
        if self._for_evaluation:
            self.trajectory.append(self._state.copy())
            self.actions.append(action.copy())
            self.rewards.append(float(reward))

        # 실제 safety constraint 위반:
        # LAST이며 discount=0 -> true terminal
        if not safe:
            return dm_env.termination(
                reward=float(reward),
                observation=self._state,
            )
        if ground_collision:
            # z <= 0
            return dm_env.termination( reward=float(reward),
                observation=self._state,
            )
        # 단순한 시간 제한:
        # LAST이지만 discount=1 -> bootstrap 가능한 truncation
        if lateral_escape:
            self._termination_reason = "lateral_escape"

            return dm_env.termination(
                reward=float(reward),
                observation=self._state,
            )
        if timeout:
            return dm_env.truncation(
                reward=float(reward),
                observation=self._state,
                discount=1.0,
            )

        return dm_env.transition(
            reward=float(reward),
            observation=self._state,
        )
    def _reward(self, x, u):
        """
        Reward adapted directly from the provided MATLAB reward.

        State:
            x = [px, py, pz,
                vx, vy, vz,
                qw, qx, qy, qz]

        Control:
            u = [a_cmd, omega_x_cmd, omega_y_cmd, omega_z_cmd]
        """
        x = np.asarray(x, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)

        # ==========================================
        # TUNABLE PARAMETERS
        # ==========================================
        tau_pos = 0.4
        tau_vel = 1.0
        tau_R = 1.0
        tau_omega = 5.0

        w_pos = 1.0
        w_vel = 0.2
        w_R = 0.2
        w_omega = 0.2

        w_act_effort = 0.0

        # ==========================================
        # STATE / CONTROL EXTRACTION
        # ==========================================
        obs_pos = x[0:3]
        obs_vel = x[3:6]
        obs_q = x[6:10]

        a_cmd = float(u[0])
        omega_cmd = u[1:4]

        des_pos = np.array(
            [0.0, 0.0, 2.0],
            dtype=np.float32,
        )

        des_vel = np.zeros(
            3,
            dtype=np.float32,
        )

        des_R_mat = np.eye(
            3,
            dtype=np.float32,
        )

        des_omega = np.zeros(
            3,
            dtype=np.float32,
        )

        # ==========================================
        # QUATERNION NORMALIZATION
        # ==========================================
        q_norm = np.linalg.norm(obs_q)

        if q_norm > 1e-6:
            obs_q = obs_q / q_norm
        else:
            obs_q = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        qw, qx, qy, qz = obs_q

        # ==========================================
        # QUATERNION -> ROTATION MATRIX
        # ==========================================
        obs_R_mat = np.array(
            [
                [
                    1.0 - 2.0 * (qy**2 + qz**2),
                    2.0 * (qx * qy - qw * qz),
                    2.0 * (qx * qz + qw * qy),
                ],
                [
                    2.0 * (qx * qy + qw * qz),
                    1.0 - 2.0 * (qx**2 + qz**2),
                    2.0 * (qy * qz - qw * qx),
                ],
                [
                    2.0 * (qx * qz - qw * qy),
                    2.0 * (qy * qz + qw * qx),
                    1.0 - 2.0 * (qx**2 + qy**2),
                ],
            ],
            dtype=np.float32,
        )

        # ==========================================
        # ERROR CALCULATIONS
        # ==========================================
        e_pos = np.linalg.norm(
            des_pos - obs_pos,
            ord=2,
        )

        e_vel = np.linalg.norm(
            des_vel - obs_vel,
            ord=2,
        )

        # This reduced-order model has no separate omega state.
        # omega_cmd is used directly in q_dot.
        e_omega = np.linalg.norm(
            des_omega - omega_cmd,
            ord=2,
        )

        e_R = 0.5 * np.trace(
            np.eye(3, dtype=np.float32)
            - des_R_mat.T @ obs_R_mat
        )

        # ==========================================
        # REWARD CALCULATIONS
        # ==========================================
        r_pos = np.exp(
            -((e_pos / tau_pos) ** 2)
        )

        r_vel = np.exp(
            -((e_vel / tau_vel) ** 2)
        )

        r_R = np.exp(
            -((e_R / tau_R) ** 2)
        )

        r_omega = np.exp(
            -((e_omega / tau_omega) ** 2)
        )

        tracking_reward_raw = (
            w_pos * r_pos
            + w_vel * r_vel
            + w_R * r_R
            + w_omega * r_omega
        )

        sum_weights = (
            w_pos
            + w_vel
            + w_R
            + w_omega
        )

        tracking_reward_normalized = (
            tracking_reward_raw
            / sum_weights
        )

        rl_effort_penalty = (
            w_act_effort
            * np.linalg.norm(
                u,
                ord=2,
            )
        )

        # ==========================================
        # FINAL REWARD
        # ==========================================
        reward = (
            tracking_reward_normalized
            - rl_effort_penalty
        )

        if not np.isfinite(reward):
            reward = -10.0

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
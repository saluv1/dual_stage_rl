"""Quadrotor power-loop tracking environment.

Physical state:
    x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
Action:
    u = [a_cmd, wx_cmd, wy_cmd, wz_cmd]
Policy observation:
    [x, normalized_phase] by default.
"""

from __future__ import annotations

from acme import specs
import chex
import dm_env
import jax.numpy as jnp
import numpy as np

from src.envs.quadrotor.constraints import (
    QuadrotorConstraintParams,
    action_bounds,
    is_safe,
)
from src.envs.quadrotor.dynamics import QuadrotorParams, rk4_step
from src.envs.quadrotor.powerloop_reference import (
    generate_powerloop_reference,
    quat_conjugate,
    quat_multiply,
)


class QuadrotorPowerLoopEnv(dm_env.Environment):
    def __init__(
        self,
        for_evaluation: bool = False,
        seed: int = 0,
        horizon: int = 106,
        include_progress: bool = True,
        initial_position_noise: float = 0.1,
        perturb_evaluation: bool = False,
        terminate_on_ceiling: bool = False,
        use_cpp_body_rate_reference: bool = False,
    ) -> None:
        self._for_evaluation = bool(for_evaluation)
        self._include_progress = bool(include_progress)
        self._initial_position_noise = float(initial_position_noise)
        self._perturb_evaluation = bool(perturb_evaluation)
        self._terminate_on_ceiling = bool(terminate_on_ceiling)
        self._use_cpp_body_rate_reference = bool(use_cpp_body_rate_reference)

        self._rng = np.random.default_rng(seed)
        self._dyn_params = QuadrotorParams(dt=0.02)
        self._constraint_params = QuadrotorConstraintParams()
        self.reference = generate_powerloop_reference()
        self._horizon = min(int(horizon), len(self.reference))
        if self._horizon <= 0:
            raise ValueError("horizon must be positive.")

        self._step_count = 0
        self._termination_reason = None
        self._state = None

        if self._for_evaluation:
            self.trajectory = []
            self.actions = []
            self.rewards = []
            self.reference_indices = []

    @property
    def termination_reason(self):
        return self._termination_reason

    def _reference_index(self) -> int:
        return min(self._step_count, self._horizon - 1)

    def _observation(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Environment has not been reset.")
        if not self._include_progress:
            return self._state.copy()
        phase = self._reference_index() / max(self._horizon - 1, 1)
        return np.concatenate(
            [self._state, np.array([phase], dtype=np.float32)], axis=0
        ).astype(np.float32)

    def reset(self) -> dm_env.TimeStep:
        self._step_count = 0
        self._termination_reason = None

        # Bottom of the loop: p=[0,0,0.5], v=[4.5,0,0].
        x0 = self.reference.state[0].copy()
        should_perturb = (not self._for_evaluation) or self._perturb_evaluation
        if should_perturb and self._initial_position_noise > 0.0:
            x0[0:3] += self._rng.uniform(
                -self._initial_position_noise,
                self._initial_position_noise,
                size=3,
            ).astype(np.float32)
        x0[6:10] /= np.linalg.norm(x0[6:10]) + 1e-8
        self._state = x0.astype(np.float32)

        if self._for_evaluation:
            self.trajectory = [self._state.copy()]
            self.actions = []
            self.rewards = []
            self.reference_indices = []

        return dm_env.restart(self._observation())

    def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
        if self._state is None:
            return self.reset()

        u_min, u_max = action_bounds()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError(f"Expected action shape (4,), got {action.shape}.")
        action = np.clip(
            action,
            np.asarray(u_min, dtype=np.float32),
            np.asarray(u_max, dtype=np.float32),
        )

        k = self._reference_index()
        # The paper defines r_k from x_k and u_k, not from x_{k+1}.
        reward = self._reward(self._state, action, k)

        x_next = rk4_step(
            jnp.asarray(self._state, dtype=jnp.float32),
            jnp.asarray(action, dtype=jnp.float32),
            self._dyn_params,
        )
        self._state = np.asarray(x_next, dtype=np.float32)
        self._step_count += 1

        ceiling_safe = bool(
            is_safe(
                jnp.asarray(self._state, dtype=jnp.float32),
                self._constraint_params,
            )
        )
        timeout = self._step_count >= self._horizon

        if not ceiling_safe:
            self._termination_reason = "ceiling_violation"
        elif timeout:
            self._termination_reason = "timeout"
        else:
            self._termination_reason = None

        if self._for_evaluation:
            self.trajectory.append(self._state.copy())
            self.actions.append(action.copy())
            self.rewards.append(float(reward))
            self.reference_indices.append(k)

        # Vanilla warm-start must be able to follow the intentionally unsafe
        # reference, so use terminate_on_ceiling=False for that stage.
        if (not ceiling_safe) and self._terminate_on_ceiling:
            return dm_env.termination(
                reward=float(reward),
                observation=self._observation(),
            )

        if timeout:
            return dm_env.truncation(
                reward=float(reward),
                observation=self._observation(),
                discount=1.0,
            )

        return dm_env.transition(
            reward=float(reward),
            observation=self._observation(),
        )

    def _reward(self, x: np.ndarray, u: np.ndarray, reference_index: int) -> float:
        """Exact negative weighted power-loop cost reported in PS2-RL."""
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        k = int(reference_index)

        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        q /= np.linalg.norm(q) + 1e-8

        a_cmd = float(u[0])
        omega_cmd = u[1:4]

        p_ref = self.reference.position[k].astype(np.float64)
        v_ref = self.reference.velocity[k].astype(np.float64)
        q_ref = self.reference.quaternion[k].astype(np.float64)
        omega_ref = (
            self.reference.cpp_body_rates[k]
            if self._use_cpp_body_rate_reference
            else self.reference.body_rates[k]
        ).astype(np.float64)

        # q_e = q_ref tensor-product q_conjugate.
        q_e = quat_multiply(q_ref, quat_conjugate(q))
        sign = 1.0 if q_e[0] >= 0.0 else -1.0
        e_att = sign * q_e[1:4]

        e_p_xy = p[0:2] - p_ref[0:2]
        e_p_z = float(p[2] - p_ref[2])
        e_v = v - v_ref
        e_omega = omega_cmd - omega_ref

        w_omega = np.array([0.10, 0.20, 0.05], dtype=np.float64)
        cost = (
            2.5 * float(e_p_xy @ e_p_xy)
            + 2.0 * e_p_z**2
            + 4.0 * float(e_v @ e_v)
            + 16.0 * float(e_att @ e_att)
            + float(w_omega @ (e_omega**2))
            + 0.01 * a_cmd**2
            + 0.01 * float(omega_cmd @ omega_cmd)
        )
        reward = -cost
        return float(reward) if np.isfinite(reward) else -1e6

    def observation_spec(self) -> specs.BoundedArray:
        dim = 11 if self._include_progress else 10
        minimum = np.full(dim, -np.inf, dtype=np.float32)
        maximum = np.full(dim, np.inf, dtype=np.float32)
        if self._include_progress:
            minimum[-1] = 0.0
            maximum[-1] = 1.0
        return specs.BoundedArray(
            shape=(dim,),
            minimum=minimum,
            maximum=maximum,
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

"""Quadrotor power-loop tracking environment.

Physical state:
    x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]

Action:
    u = [a_cmd, wx_cmd, wy_cmd, wz_cmd]

Observation:
    [physical_state,
     normalized_position_error,
     normalized_velocity_error,
     attitude_error,
     normalized_reference_body_rate,
     normalized_phase]

The first 10 observation entries are always the physical state so that the
CIL/BCBF code can consume them directly. The actor and critic can be
configured to consume only the tracking features after index 10.
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
    """Fixed-horizon power-loop tracking environment."""

    PHYSICAL_STATE_DIM = 10
    TRACKING_FEATURE_DIM_WITHOUT_PROGRESS = 12

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
        self._use_cpp_body_rate_reference = bool(
            use_cpp_body_rate_reference
        )

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

    @property
    def observation_dim(self) -> int:
        tracking_dim = self.TRACKING_FEATURE_DIM_WITHOUT_PROGRESS
        if self._include_progress:
            tracking_dim += 1
        return self.PHYSICAL_STATE_DIM + tracking_dim

    def _reference_index(self) -> int:
        return min(self._step_count, self._horizon - 1)

    def _reference_body_rate(self, reference_index: int) -> np.ndarray:
        if self._use_cpp_body_rate_reference:
            omega_ref = self.reference.cpp_body_rates[reference_index]
        else:
            omega_ref = self.reference.body_rates[reference_index]

        return np.asarray(omega_ref, dtype=np.float64)

    def _reference_values(
        self,
        reference_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p_ref = np.asarray(
            self.reference.position[reference_index],
            dtype=np.float64,
        )
        v_ref = np.asarray(
            self.reference.velocity[reference_index],
            dtype=np.float64,
        )
        q_ref = np.asarray(
            self.reference.quaternion[reference_index],
            dtype=np.float64,
        ).copy()
        q_ref /= np.linalg.norm(q_ref) + 1e-8

        omega_ref = self._reference_body_rate(reference_index)
        return p_ref, v_ref, q_ref, omega_ref

    def _tracking_errors(
        self,
        state: np.ndarray,
        reference_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64)

        p = state[0:3]
        v = state[3:6]
        q = state[6:10].copy()
        q /= np.linalg.norm(q) + 1e-8

        p_ref, v_ref, q_ref, omega_ref = self._reference_values(
            reference_index
        )

        e_p = p - p_ref
        e_v = v - v_ref

        # Paper convention: q_e = q_ref tensor-product q_conjugate.
        q_error = quat_multiply(q_ref, quat_conjugate(q))
        sign = 1.0 if q_error[0] >= 0.0 else -1.0
        e_att = sign * q_error[1:4]

        return e_p, e_v, e_att, omega_ref

    def _observation(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Environment has not been reset.")

        reference_index = self._reference_index()
        e_p, e_v, e_att, omega_ref = self._tracking_errors(
            self._state,
            reference_index,
        )

        # Natural scales of the reference trajectory.
        e_p_normalized = e_p / 1.5
        e_v_normalized = e_v / 4.5
        omega_ref_normalized = omega_ref / 3.0

        tracking_parts = [
            e_p_normalized,
            e_v_normalized,
            e_att,
            omega_ref_normalized,
        ]

        if self._include_progress:
            phase = (
                float(reference_index)
                / float(max(self._horizon - 1, 1))
            )
            tracking_parts.append(
                np.asarray([phase], dtype=np.float64)
            )

        tracking_features = np.concatenate(
            tracking_parts,
            axis=0,
        )
        physical_state = np.asarray(
            self._state,
            dtype=np.float64,
        )

        observation = np.concatenate(
            [physical_state, tracking_features],
            axis=0,
        ).astype(np.float32)

        expected_shape = (self.observation_dim,)
        if observation.shape != expected_shape:
            raise RuntimeError(
                "Unexpected observation shape: "
                f"{observation.shape}; expected {expected_shape}."
            )

        return observation

    def reset(self) -> dm_env.TimeStep:
        self._step_count = 0
        self._termination_reason = None

        # Bottom of the loop: p=[0,0,0.5], v=[4.5,0,0].
        x0 = self.reference.state[0].copy()
        should_perturb = (
            (not self._for_evaluation)
            or self._perturb_evaluation
        )
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
            raise ValueError(
                f"Expected action shape (4,), got {action.shape}."
            )

        action = np.clip(
            action,
            np.asarray(u_min, dtype=np.float32),
            np.asarray(u_max, dtype=np.float32),
        )

        reference_index = self._reference_index()

        # The reward is r_k = r(x_k, u_k, reference_k).
        reward = self._reward(
            self._state,
            action,
            reference_index,
        )

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

        if self._for_evaluation:
            self.trajectory.append(self._state.copy())
            self.actions.append(action.copy())
            self.rewards.append(float(reward))
            self.reference_indices.append(reference_index)

        # Vanilla training should use terminate_on_ceiling=False because the
        # reference intentionally exceeds the ceiling. Phase-II safety logic
        # can set it to True when an environment-side terminal is desired.
        if (not ceiling_safe) and self._terminate_on_ceiling:
            self._termination_reason = "ceiling_violation"
            return dm_env.termination(
                reward=float(reward),
                observation=self._observation(),
            )

        # This is a fixed finite-horizon tracking problem. Do not bootstrap
        # beyond the end of the reference trajectory.
        if timeout:
            self._termination_reason = "timeout"
            return dm_env.termination(
                reward=float(reward),
                observation=self._observation(),
            )

        self._termination_reason = None
        return dm_env.transition(
            reward=float(reward),
            observation=self._observation(),
        )

    def _reward(
        self,
        x: np.ndarray,
        u: np.ndarray,
        reference_index: int,
    ) -> float:
        """Negative weighted power-loop tracking cost."""
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        e_p, e_v, e_att, omega_ref = self._tracking_errors(
            x,
            int(reference_index),
        )

        a_cmd = float(u[0])
        omega_cmd = u[1:4]
        e_omega = omega_cmd - omega_ref

        w_omega = np.asarray(
            [0.10, 0.20, 0.05],
            dtype=np.float64,
        )

        cost = (
            2.5 * float(e_p[0:2] @ e_p[0:2])
            + 2.0 * float(e_p[2] ** 2)
            + 4.0 * float(e_v @ e_v)
            + 16.0 * float(e_att @ e_att)
            + float(w_omega @ np.square(e_omega))
            + 0.01 * float(a_cmd ** 2)
            + 0.01 * float(omega_cmd @ omega_cmd)
        )

        reward = -cost
        return float(reward) if np.isfinite(reward) else -1e6

    def observation_spec(self) -> specs.BoundedArray:
        dim = self.observation_dim
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

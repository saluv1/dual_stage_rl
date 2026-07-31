"""Quadrotor vanilla power-loop tracking environment.

This file keeps dual_stage_rl's ``dm_env.Environment`` interface and existing
state/action conventions, but aligns the vanilla tracking task with the public
PS2-RL implementation:

* state: ``[p(3), v(3), q_wxyz(4)]``
* action: ``[a_cmd, omega_x, omega_y, omega_z]``
* explicit Euler integration at 50 Hz
* post-step reward against the post-step reference
* 26-D observation ``[x, x_ref, omega_ref, t, sin(phase), cos(phase)]``
* reference-length episode plus a 0.1 s terminal tail
* no ground terminal and no active ceiling terminal for vanilla training
"""

from __future__ import annotations

from acme import specs
import chex
import dm_env
import jax.numpy as jnp
import numpy as np

from src.envs.quadrotor.constraints import action_bounds
from src.envs.quadrotor.dynamics import QuadrotorParams, euler_step
from src.envs.quadrotor.powerloop_reference import (
    generate_powerloop_reference,
    quat_conjugate,
    quat_multiply,
)


class QuadrotorPowerLoopEnv(dm_env.Environment):
    """Finite-horizon vanilla power-loop tracking task."""

    PHYSICAL_STATE_DIM = 10
    REFERENCE_STATE_DIM = 10
    REFERENCE_RATE_DIM = 3
    TIME_FEATURE_DIM = 3

    def __init__(
        self,
        for_evaluation: bool = False,
        seed: int = 0,
        horizon: int | None = None,
        include_time_features: bool = True,
        # Backward-compatible alias used by the old config/main.py.
        include_progress: bool | None = None,
        max_steps_extra_sec: float = 0.1,
        initial_position_noise: float = 0.1,
        perturb_evaluation: bool = True,
        terminate_on_ceiling: bool = False,
        z_max: float = 15.0,
        use_cpp_body_rate_reference: bool = False,
        dt: float = 0.02,
        w_pos_xy: float = 2.5,
        w_pos_z: float = 2.0,
        w_vel: float = 4.0,
        w_att: float = 16.0,
        w_ref_omega_x: float = 0.10,
        w_ref_omega_y: float = 0.20,
        w_ref_omega_z: float = 0.05,
        w_control_a: float = 0.01,
        w_control_omega: float = 0.01,
    ) -> None:
        if include_progress is not None:
            include_time_features = bool(include_progress)

        self._for_evaluation = bool(for_evaluation)
        self._include_time_features = bool(include_time_features)
        self._initial_position_noise = float(initial_position_noise)
        self._perturb_evaluation = bool(perturb_evaluation)
        self._terminate_on_ceiling = bool(terminate_on_ceiling)
        self._z_max = float(z_max)
        self._use_cpp_body_rate_reference = bool(
            use_cpp_body_rate_reference
        )

        self._rng = np.random.default_rng(int(seed))
        self._dyn_params = QuadrotorParams(dt=float(dt))
        self.reference = generate_powerloop_reference(
            sampling_frequency=1.0 / float(dt)
        )

        if horizon is None or int(horizon) <= 0:
            tail_steps = int(
                np.ceil(float(max_steps_extra_sec) / float(dt))
            )
            self._horizon = len(self.reference) + tail_steps
        else:
            self._horizon = int(horizon)
        if self._horizon <= 0:
            raise ValueError("horizon must be positive.")

        self._weights = {
            "pos_xy": float(w_pos_xy),
            "pos_z": float(w_pos_z),
            "vel": float(w_vel),
            "att": float(w_att),
            "omega": np.asarray(
                [w_ref_omega_x, w_ref_omega_y, w_ref_omega_z],
                dtype=np.float64,
            ),
            "control_a": float(w_control_a),
            "control_omega": float(w_control_omega),
        }

        self._step_count = 0
        self._termination_reason = None
        self._state: np.ndarray | None = None

        if self._for_evaluation:
            self.trajectory: list[np.ndarray] = []
            self.actions: list[np.ndarray] = []
            self.rewards: list[float] = []
            self.reference_indices: list[int] = []

    @property
    def termination_reason(self):
        return self._termination_reason

    @property
    def observation_dim(self) -> int:
        base = (
            self.PHYSICAL_STATE_DIM
            + self.REFERENCE_STATE_DIM
            + self.REFERENCE_RATE_DIM
        )
        return base + (self.TIME_FEATURE_DIM if self._include_time_features else 0)

    def _reference_index(self, step_count: int | None = None) -> int:
        if step_count is None:
            step_count = self._step_count
        return min(max(int(step_count), 0), len(self.reference) - 1)

    def _reference_body_rate(self, reference_index: int) -> np.ndarray:
        source = (
            self.reference.cpp_body_rates
            if self._use_cpp_body_rate_reference
            else self.reference.body_rates
        )
        return np.asarray(source[reference_index], dtype=np.float64)

    def _reference_values(
        self,
        step_count: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, int]:
        """Return clamped reference, rate, progress and time features."""
        index = self._reference_index(step_count)
        ref_state = np.asarray(
            self.reference.state[index], dtype=np.float64
        ).copy()
        ref_state[6:10] /= np.linalg.norm(ref_state[6:10]) + 1e-8
        ref_omega = self._reference_body_rate(index)

        progress = float(index) / float(max(len(self.reference) - 1, 1))
        phase = 2.0 * np.pi * progress
        time_sec = float(step_count) * float(self._dyn_params.dt)
        return (
            ref_state,
            ref_omega,
            time_sec,
            float(np.sin(phase)),
            float(np.cos(phase)),
            index,
        )

    def _observation(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Environment has not been reset.")

        ref_state, ref_omega, time_sec, phase_sin, phase_cos, _ = (
            self._reference_values(self._step_count)
        )
        pieces = [
            np.asarray(self._state, dtype=np.float64),
            ref_state,
            ref_omega,
        ]
        if self._include_time_features:
            pieces.append(
                np.asarray(
                    [time_sec, phase_sin, phase_cos], dtype=np.float64
                )
            )

        observation = np.concatenate(pieces, axis=0).astype(np.float32)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                f"Unexpected observation shape {observation.shape}; "
                f"expected {(self.observation_dim,)}."
            )
        return observation

    def reset(self) -> dm_env.TimeStep:
        self._step_count = 0
        self._termination_reason = None

        x0 = np.asarray(self.reference.state[0], dtype=np.float64).copy()
        should_perturb = (
            not self._for_evaluation or self._perturb_evaluation
        )
        if should_perturb and self._initial_position_noise > 0.0:
            x0[0:3] += self._rng.uniform(
                -self._initial_position_noise,
                self._initial_position_noise,
                size=3,
            )

        # PS2-RL clips the initial altitude below the inactive hard deck.
        x0[2] = min(x0[2], self._z_max - 0.05)
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

        previous_step = self._step_count
        x_next = euler_step(
            jnp.asarray(self._state, dtype=jnp.float32),
            jnp.asarray(action, dtype=jnp.float32),
            self._dyn_params,
        )
        self._state = np.asarray(x_next, dtype=np.float32)
        self._step_count += 1

        ref_state_next, ref_omega_next, _, _, _, next_ref_index = (
            self._reference_values(self._step_count)
        )
        _, ref_omega_previous, _, _, _, _ = self._reference_values(
            previous_step
        )
        reward_ref_omega = 0.5 * (
            ref_omega_previous + ref_omega_next
        )
        reward = self._reward(
            self._state,
            action,
            ref_state_next,
            reward_ref_omega,
        )

        ceiling_safe = bool(self._state[2] <= self._z_max)
        timeout = self._step_count >= self._horizon

        if self._for_evaluation:
            self.trajectory.append(self._state.copy())
            self.actions.append(action.copy())
            self.rewards.append(float(reward))
            self.reference_indices.append(next_ref_index)

        if (not ceiling_safe) and self._terminate_on_ceiling:
            self._termination_reason = "ceiling_violation"
            return dm_env.termination(
                reward=float(reward), observation=self._observation()
            )

        # PS2-RL treats the finite task horizon as done, so the Bellman target
        # does not bootstrap past the terminal tail.
        if timeout:
            self._termination_reason = "timeout"
            return dm_env.termination(
                reward=float(reward), observation=self._observation()
            )

        self._termination_reason = None
        return dm_env.transition(
            reward=float(reward), observation=self._observation()
        )

    def _reward(
        self,
        x_next: np.ndarray,
        u: np.ndarray,
        ref_state: np.ndarray,
        reward_ref_omega: np.ndarray,
    ) -> float:
        """Negative PS2-RL trajectory-following cost at the post-step state."""
        x_next = np.asarray(x_next, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        ref_state = np.asarray(ref_state, dtype=np.float64)
        reward_ref_omega = np.asarray(
            reward_ref_omega, dtype=np.float64
        )

        pos_error = x_next[0:3] - ref_state[0:3]
        vel_error = x_next[3:6] - ref_state[3:6]

        q_ref = ref_state[6:10].copy()
        q_now = x_next[6:10].copy()
        q_ref /= np.linalg.norm(q_ref) + 1e-8
        q_now /= np.linalg.norm(q_now) + 1e-8
        q_error = quat_multiply(q_ref, quat_conjugate(q_now))
        sign = 1.0 if q_error[0] >= 0.0 else -1.0
        attitude_error = sign * q_error[1:4]

        omega_error = u[1:4] - reward_ref_omega
        w = self._weights
        cost = (
            w["pos_xy"]
            * float(pos_error[0] ** 2 + pos_error[1] ** 2)
            + w["pos_z"] * float(pos_error[2] ** 2)
            + w["vel"] * float(vel_error @ vel_error)
            + w["att"] * float(attitude_error @ attitude_error)
            + float(w["omega"] @ np.square(omega_error))
            + w["control_a"] * float(u[0] ** 2)
            + w["control_omega"] * float(u[1:4] @ u[1:4])
        )
        reward = -cost
        return float(reward) if np.isfinite(reward) else -1e6

    def observation_spec(self) -> specs.BoundedArray:
        dim = self.observation_dim
        return specs.BoundedArray(
            shape=(dim,),
            minimum=np.full((dim,), -np.inf, dtype=np.float32),
            maximum=np.full((dim,), np.inf, dtype=np.float32),
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

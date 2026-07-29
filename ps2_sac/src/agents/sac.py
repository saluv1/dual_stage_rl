"""Modern Soft Actor-Critic agent.

Features:
    - Twin online Q networks and twin Polyak target Q networks.
    - Automatic entropy-temperature tuning.
    - Optional CIL projection for actor and target actions.
    - Separate physical-state and network-observation slices.
    - Optional normalization of physical actions before the Q networks.
    - Diagnostics for Q variance and state/action sensitivity.

This implementation does not use a separate V or V-target network.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from acme import specs, types
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import optax
import rlax

from src.agents.networks import PolicyNetwork, SoftQNetwork
from src.cil.action_filter import get_safe_action, get_safe_action_batch
from src.replay_buffers.buffer import ReplayBuffer
from src.utils.training_utils import LearnerState, OptState, ParamState, Transitions


DEFAULT_LOG_STD_MIN = -5.0
DEFAULT_LOG_STD_MAX = 2.0


class SAC:
    """Twin-target-Q SAC with learned alpha and optional CIL."""

    def __init__(
        self,
        rng: chex.PRNGKey,
        environment_spec: specs.EnvironmentSpec,
        config: ConfigDict,
        use_cil: bool = False,
        cil_provider_params=None,
        constraint_provider=None,
    ) -> None:
        self.environment_spec = environment_spec
        self.config = config

        self._use_cil = bool(use_cil)
        self._cil_provider_params = cil_provider_params
        self._constraint_provider = constraint_provider

        self.action_dim = int(environment_spec.actions.shape[-1])
        self.observation_dim = int(environment_spec.observations.shape[-1])

        self._u_min = jnp.asarray(
            environment_spec.actions.minimum,
            dtype=jnp.float32,
        )
        self._u_max = jnp.asarray(
            environment_spec.actions.maximum,
            dtype=jnp.float32,
        )

        # The first cil_state_dim entries must remain the physical state.
        self._cil_state_dim = int(
            getattr(config, "cil_state_dim", self.observation_dim)
        )
        if not 1 <= self._cil_state_dim <= self.observation_dim:
            raise ValueError(
                "config.cil_state_dim must lie in [1, observation_dim]."
            )

        # By default, actor and critic use the full observation. For the
        # power-loop environment set network_obs_start=10 so they use only
        # the tracking-error features after the physical state.
        self._network_obs_start = int(
            getattr(config, "network_obs_start", 0)
        )
        if not 0 <= self._network_obs_start < self.observation_dim:
            raise ValueError(
                "config.network_obs_start must lie in "
                "[0, observation_dim)."
            )

        self._network_observation_dim = (
            self.observation_dim - self._network_obs_start
        )
        expected_network_dim = int(
            getattr(
                config,
                "network_observation_dim",
                self._network_observation_dim,
            )
        )
        if expected_network_dim != self._network_observation_dim:
            raise ValueError(
                "Unexpected actor/critic observation dimension: "
                f"got {self._network_observation_dim}, "
                f"expected {expected_network_dim}."
            )

        self._normalize_q_actions = bool(
            getattr(config, "normalize_q_actions", False)
        )

        self._hidden_sizes = tuple(
            int(size)
            for size in getattr(config, "hidden_sizes", (256, 256))
        )
        if not self._hidden_sizes:
            raise ValueError("config.hidden_sizes must not be empty.")

        self._auto_alpha = bool(getattr(config, "auto_alpha", True))
        self._alpha_init = float(getattr(config, "alpha", 0.2))
        self._alpha_min = float(getattr(config, "alpha_min", 0.01))
        self._target_entropy = float(
            getattr(config, "target_entropy", -float(self.action_dim))
        )

        # Smoothly bound the policy log standard deviation. Unlike a hard
        # clip, this keeps a nonzero gradient when the raw network output
        # moves outside the desired numerical range.
        self._log_std_min = float(
            getattr(config, "log_std_min", DEFAULT_LOG_STD_MIN)
        )
        self._log_std_max = float(
            getattr(config, "log_std_max", DEFAULT_LOG_STD_MAX)
        )
        if self._log_std_min >= self._log_std_max:
            raise ValueError(
                "config.log_std_min must be smaller than "
                "config.log_std_max."
            )

        if self._alpha_init <= 0.0:
            raise ValueError("config.alpha must be positive.")
        if self._alpha_min <= 0.0:
            raise ValueError("config.alpha_min must be positive.")
        if self._alpha_init < self._alpha_min:
            raise ValueError(
                "config.alpha must be greater than or equal to "
                "config.alpha_min."
            )

        self._log_alpha_min = math.log(self._alpha_min)

        self._q_clip = float(getattr(config, "q_clip", 5e6))
        self._grad_clip_norm = float(
            getattr(config, "grad_clip_norm", 5.0)
        )
        self._reward_scale = float(getattr(config, "scale_reward", 1.0))
        self._discount = float(getattr(config, "gamma", 0.99))
        self._tau = float(getattr(config, "tau", 0.005))

        if self._q_clip <= 0.0:
            raise ValueError("config.q_clip must be positive.")
        if self._grad_clip_norm <= 0.0:
            raise ValueError("config.grad_clip_norm must be positive.")
        if self._reward_scale <= 0.0:
            raise ValueError("config.scale_reward must be positive.")
        if not 0.0 <= self._discount <= 1.0:
            raise ValueError("config.gamma must lie in [0, 1].")
        if not 0.0 < self._tau <= 1.0:
            raise ValueError("config.tau must lie in (0, 1].")

        self._rng = rng

        self._init_policy, self._apply_policy = hk.without_apply_rng(
            hk.transform(self._hk_apply_policy)
        )
        self._init_q, self._apply_q = hk.without_apply_rng(
            hk.transform(self._hk_apply_q)
        )

        self.optimizer_q = optax.chain(
            optax.clip_by_global_norm(self._grad_clip_norm),
            optax.adam(float(self.config.q_lr)),
        )
        self.optimizer_p = optax.chain(
            optax.clip_by_global_norm(self._grad_clip_norm),
            optax.adam(float(self.config.p_lr)),
        )
        self.optimizer_alpha = optax.chain(
            optax.clip_by_global_norm(self._grad_clip_norm),
            optax.adam(float(self.config.alpha_lr)),
        )

        self._grad_q = jax.value_and_grad(
            self._loss_fn_q,
            argnums=(0, 1),
            has_aux=True,
        )
        self._grad_pi = jax.value_and_grad(
            self._loss_fn_pi,
            argnums=0,
            has_aux=True,
        )
        self._grad_alpha = jax.value_and_grad(
            self._loss_fn_alpha,
            argnums=0,
        )

        self.init_fn = jax.jit(self._init_fn)
        self.update_fn = jax.jit(self._update_fn)
        self.apply_policy = jax.jit(self._apply_policy)
        self.apply_q = jax.jit(self._apply_q)
        self.get_action = jax.jit(self._get_action, static_argnums=3)

        self.buffer = ReplayBuffer(
            size_=int(self.config.replay_buffer_capacity),
            featuredim_=self.observation_dim,
            actiondim_=self.action_dim,
        )

    def initialize(self) -> LearnerState:
        self._rng, init_key = jax.random.split(self._rng, 2)

        dummy_obs = jnp.expand_dims(
            jnp.zeros(
                self.environment_spec.observations.shape,
                dtype=jnp.float32,
            ),
            axis=0,
        )
        dummy_actions = jnp.expand_dims(
            jnp.zeros(
                self.environment_spec.actions.shape,
                dtype=jnp.float32,
            ),
            axis=0,
        )

        return self.init_fn(init_key, dummy_obs, dummy_actions)

    def _init_fn(
        self,
        rng: chex.PRNGKey,
        dummy_obs: types.NestedArray,
        dummy_actions: types.NestedArray,
    ) -> LearnerState:
        key_q1, key_q2, key_policy = jax.random.split(rng, 3)

        q1_params = self._init_q(key_q1, dummy_obs, dummy_actions)
        q2_params = self._init_q(key_q2, dummy_obs, dummy_actions)
        policy_params = self._init_policy(key_policy, dummy_obs)

        q1_target_params = jax.tree_util.tree_map(jnp.array, q1_params)
        q2_target_params = jax.tree_util.tree_map(jnp.array, q2_params)

        log_alpha = jnp.asarray(
            math.log(self._alpha_init),
            dtype=jnp.float32,
        )

        return LearnerState(
            params=ParamState(
                policy=policy_params,
                q1=q1_params,
                q2=q2_params,
                q1_target=q1_target_params,
                q2_target=q2_target_params,
                log_alpha=log_alpha,
            ),
            opt_state=OptState(
                policy=self.optimizer_p.init(policy_params),
                q1=self.optimizer_q.init(q1_params),
                q2=self.optimizer_q.init(q2_params),
                alpha=self.optimizer_alpha.init(log_alpha),
            ),
        )

    def _current_alpha(
        self,
        log_alpha: chex.ArrayNumpy,
    ) -> chex.ArrayNumpy:
        if not self._auto_alpha:
            return jnp.asarray(self._alpha_init, dtype=jnp.float32)
        return jnp.maximum(jnp.exp(log_alpha), self._alpha_min)

    def _clip_q_value(
        self,
        q_value: chex.ArrayNumpy,
    ) -> chex.ArrayNumpy:
        return jnp.clip(q_value, -self._q_clip, self._q_clip)

    def _constraint_observations(
        self,
        observations: types.NestedArray,
    ) -> types.NestedArray:
        return observations[..., : self._cil_state_dim]

    def _network_observations(
        self,
        observations: types.NestedArray,
    ) -> types.NestedArray:
        return observations[..., self._network_obs_start :]

    def _normalize_actions_for_q(
        self,
        actions: types.NestedArray,
    ) -> types.NestedArray:
        if not self._normalize_q_actions:
            return actions

        midpoint = 0.5 * (self._u_max + self._u_min)
        half_range = 0.5 * (self._u_max - self._u_min)
        normalized_actions = (
            actions - midpoint
        ) / jnp.maximum(half_range, 1e-6)

        return jnp.clip(normalized_actions, -1.0, 1.0)

    def _bound_log_sigma(
        self,
        raw_log_sigma: types.NestedArray,
    ) -> types.NestedArray:
        """Smoothly map an unconstrained output into [min, max]."""
        unit_interval = 0.5 * (jnp.tanh(raw_log_sigma) + 1.0)
        return (
            self._log_std_min
            + unit_interval * (self._log_std_max - self._log_std_min)
        )

    def _loss_fn_q(
        self,
        q1_params: types.NestedArray,
        q2_params: types.NestedArray,
        policy_params: types.NestedArray,
        q1_target_params: types.NestedArray,
        q2_target_params: types.NestedArray,
        log_alpha: chex.ArrayNumpy,
        transitions: Transitions,
        key: chex.PRNGKey,
    ) -> Tuple[chex.ArrayNumpy, Dict[str, chex.ArrayNumpy]]:
        next_mu, next_log_sigma = self.apply_policy(
            policy_params,
            transitions.next_observations,
        )
        next_nominal_actions, next_action_log_probs = self._sample_action(
            key,
            next_mu,
            next_log_sigma,
        )

        next_q_actions = self._project_actions_for_q(
            transitions.next_observations,
            next_nominal_actions,
        )

        target_q1 = self.apply_q(
            q1_target_params,
            transitions.next_observations,
            next_q_actions,
        )
        target_q2 = self.apply_q(
            q2_target_params,
            transitions.next_observations,
            next_q_actions,
        )

        alpha = jax.lax.stop_gradient(
            self._current_alpha(log_alpha)
        )
        target_soft_q = (
            jnp.minimum(target_q1, target_q2)
            - alpha * next_action_log_probs
        )

        rewards = transitions.rewards.astype(target_soft_q.dtype)
        dones = transitions.dones.astype(target_soft_q.dtype)
        chex.assert_equal_shape([rewards, dones, target_soft_q])

        target_q_value = (
            self._reward_scale * rewards
            + self._discount * (1.0 - dones) * target_soft_q
        )
        target_q_value = self._clip_q_value(target_q_value)
        target_q_value = jax.lax.stop_gradient(target_q_value)

        predicted_q1 = self.apply_q(
            q1_params,
            transitions.observations,
            transitions.actions,
        )
        predicted_q2 = self.apply_q(
            q2_params,
            transitions.observations,
            transitions.actions,
        )
        chex.assert_equal_shape(
            [target_q_value, predicted_q1, predicted_q2]
        )

        td_error_q1 = target_q_value - predicted_q1
        td_error_q2 = target_q_value - predicted_q2

        q1_loss = 0.5 * jnp.mean(jnp.square(td_error_q1))
        q2_loss = 0.5 * jnp.mean(jnp.square(td_error_q2))
        total_q_loss = q1_loss + q2_loss

        shuffled_actions = jnp.roll(
            transitions.actions,
            shift=1,
            axis=0,
        )
        q1_shuffled_actions = self.apply_q(
            q1_params,
            transitions.observations,
            shuffled_actions,
        )

        shuffled_observations = jnp.roll(
            transitions.observations,
            shift=1,
            axis=0,
        )
        q1_shuffled_observations = self.apply_q(
            q1_params,
            shuffled_observations,
            transitions.actions,
        )

        diagnostics = {
            "loss_q1": q1_loss,
            "loss_q2": q2_loss,
            "reward_mean": jnp.mean(rewards),
            "reward_min": jnp.min(rewards),
            "reward_max": jnp.max(rewards),
            "reward_std": jnp.std(rewards),
            "q1_mean": jnp.mean(predicted_q1),
            "q1_min": jnp.min(predicted_q1),
            "q1_max": jnp.max(predicted_q1),
            "q1_std": jnp.std(predicted_q1),
            "q2_mean": jnp.mean(predicted_q2),
            "q2_std": jnp.std(predicted_q2),
            "target_q_mean": jnp.mean(target_q_value),
            "target_q_min": jnp.min(target_q_value),
            "target_q_max": jnp.max(target_q_value),
            "target_q_std": jnp.std(target_q_value),
            "td1_abs_mean": jnp.mean(jnp.abs(td_error_q1)),
            "td1_abs_max": jnp.max(jnp.abs(td_error_q1)),
            "td2_abs_mean": jnp.mean(jnp.abs(td_error_q2)),
            "td2_abs_max": jnp.max(jnp.abs(td_error_q2)),
            "q1_action_sensitivity": jnp.mean(
                jnp.abs(predicted_q1 - q1_shuffled_actions)
            ),
            "q1_state_sensitivity": jnp.mean(
                jnp.abs(predicted_q1 - q1_shuffled_observations)
            ),
            "network_obs_std": jnp.mean(
                jnp.std(
                    self._network_observations(
                        transitions.observations
                    ),
                    axis=0,
                )
            ),
            "physical_action_std": jnp.mean(
                jnp.std(transitions.actions, axis=0)
            ),
        }
        return total_q_loss, diagnostics

    def _loss_fn_pi(
        self,
        policy_params: types.NestedArray,
        q1_params: types.NestedArray,
        q2_params: types.NestedArray,
        log_alpha: chex.ArrayNumpy,
        transitions: Transitions,
        key: chex.PRNGKey,
    ) -> Tuple[
        chex.ArrayNumpy,
        Tuple[
            chex.ArrayNumpy,
            chex.ArrayNumpy,
            Dict[str, chex.ArrayNumpy],
        ],
    ]:
        mu, raw_log_sigma = self.apply_policy(
            policy_params,
            transitions.observations,
        )
        bounded_log_sigma = self._bound_log_sigma(raw_log_sigma)
        nominal_actions, action_log_probs = self._sample_action(
            key,
            mu,
            raw_log_sigma,
        )

        q_actions = self._project_actions_for_q(
            transitions.observations,
            nominal_actions,
        )
        q1_pi = self.apply_q(
            q1_params,
            transitions.observations,
            q_actions,
        )
        q2_pi = self.apply_q(
            q2_params,
            transitions.observations,
            q_actions,
        )

        min_q_pi = self._clip_q_value(jnp.minimum(q1_pi, q2_pi))
        alpha = jax.lax.stop_gradient(
            self._current_alpha(log_alpha)
        )
        policy_loss = jnp.mean(
            alpha * action_log_probs - min_q_pi
        )

        l2_coef = float(getattr(self.config, "policy_l2_coef", 0.0))
        if l2_coef != 0.0:
            l2_loss = 0.5 * sum(
                jnp.sum(jnp.square(parameter))
                for parameter in jax.tree_util.tree_leaves(
                    policy_params
                )
            )
            policy_loss = policy_loss + l2_coef * l2_loss

        sampled_entropy = -jnp.mean(action_log_probs)

        normalized_actions = (
            2.0
            * (nominal_actions - self._u_min)
            / jnp.maximum(self._u_max - self._u_min, 1e-6)
            - 1.0
        )

        pi_diagnostics = {
            "q1_pi_mean": jnp.mean(q1_pi),
            "q1_pi_std": jnp.std(q1_pi),
            "q2_pi_mean": jnp.mean(q2_pi),
            "q2_pi_std": jnp.std(q2_pi),
            "min_q_pi_mean": jnp.mean(min_q_pi),
            "mu_abs_mean": jnp.mean(jnp.abs(mu)),
            "mu_abs_max": jnp.max(jnp.abs(mu)),
            "raw_log_sigma_mean": jnp.mean(raw_log_sigma),
            "bounded_log_sigma_mean": jnp.mean(bounded_log_sigma),
            "bounded_log_sigma_min": jnp.min(bounded_log_sigma),
            "bounded_log_sigma_max": jnp.max(bounded_log_sigma),
            "action_saturation": jnp.mean(
                (jnp.abs(normalized_actions) > 0.98).astype(jnp.float32)
            ),
        }

        return policy_loss, (
            sampled_entropy,
            action_log_probs,
            pi_diagnostics,
        )

    def _loss_fn_alpha(
        self,
        log_alpha: chex.ArrayNumpy,
        action_log_probs: chex.ArrayNumpy,
    ) -> chex.ArrayNumpy:
        entropy_residual = jax.lax.stop_gradient(
            action_log_probs + self._target_entropy
        )
        return -jnp.mean(log_alpha * entropy_residual)

    def _update_fn(
        self,
        curr_ls: LearnerState,
        transitions: Transitions,
        update_key: chex.PRNGKey,
    ) -> Tuple[LearnerState, Dict[str, chex.ArrayNumpy]]:
        key_q, key_pi = jax.random.split(update_key, 2)

        (loss_q, q_diagnostics), (grad_q1, grad_q2) = self._grad_q(
            curr_ls.params.q1,
            curr_ls.params.q2,
            curr_ls.params.policy,
            curr_ls.params.q1_target,
            curr_ls.params.q2_target,
            curr_ls.params.log_alpha,
            transitions,
            key_q,
        )

        q1_updates, q1_opt_state = self.optimizer_q.update(
            grad_q1,
            curr_ls.opt_state.q1,
            curr_ls.params.q1,
        )
        q2_updates, q2_opt_state = self.optimizer_q.update(
            grad_q2,
            curr_ls.opt_state.q2,
            curr_ls.params.q2,
        )

        q1_update_norm = optax.global_norm(q1_updates)
        q2_update_norm = optax.global_norm(q2_updates)
        q1_param_norm = optax.global_norm(curr_ls.params.q1)
        q2_param_norm = optax.global_norm(curr_ls.params.q2)

        new_q1_params = optax.apply_updates(
            curr_ls.params.q1,
            q1_updates,
        )
        new_q2_params = optax.apply_updates(
            curr_ls.params.q2,
            q2_updates,
        )

        (
            loss_pi,
            (entropy, action_log_probs, pi_diagnostics),
        ), grad_pi = self._grad_pi(
            curr_ls.params.policy,
            new_q1_params,
            new_q2_params,
            curr_ls.params.log_alpha,
            transitions,
            key_pi,
        )

        policy_updates, policy_opt_state = self.optimizer_p.update(
            grad_pi,
            curr_ls.opt_state.policy,
            curr_ls.params.policy,
        )
        new_policy_params = optax.apply_updates(
            curr_ls.params.policy,
            policy_updates,
        )

        if self._auto_alpha:
            loss_alpha, grad_alpha = self._grad_alpha(
                curr_ls.params.log_alpha,
                action_log_probs,
            )
            alpha_updates, alpha_opt_state = self.optimizer_alpha.update(
                grad_alpha,
                curr_ls.opt_state.alpha,
                curr_ls.params.log_alpha,
            )
            updated_log_alpha = optax.apply_updates(
                curr_ls.params.log_alpha,
                alpha_updates,
            )
            new_log_alpha = jnp.maximum(
                updated_log_alpha,
                self._log_alpha_min,
            )
        else:
            loss_alpha = jnp.asarray(0.0, dtype=jnp.float32)
            grad_alpha = jnp.zeros_like(curr_ls.params.log_alpha)
            alpha_opt_state = curr_ls.opt_state.alpha
            new_log_alpha = curr_ls.params.log_alpha

        new_q1_target_params = jax.tree_util.tree_map(
            lambda target, online: (
                target + self._tau * (online - target)
            ),
            curr_ls.params.q1_target,
            new_q1_params,
        )
        new_q2_target_params = jax.tree_util.tree_map(
            lambda target, online: (
                target + self._tau * (online - target)
            ),
            curr_ls.params.q2_target,
            new_q2_params,
        )

        new_learner_state = LearnerState(
            params=ParamState(
                policy=new_policy_params,
                q1=new_q1_params,
                q2=new_q2_params,
                q1_target=new_q1_target_params,
                q2_target=new_q2_target_params,
                log_alpha=new_log_alpha,
            ),
            opt_state=OptState(
                policy=policy_opt_state,
                q1=q1_opt_state,
                q2=q2_opt_state,
                alpha=alpha_opt_state,
            ),
        )

        alpha = self._current_alpha(new_log_alpha)
        logs = {
            "loss_q": loss_q,
            "loss_pi": loss_pi,
            "loss_alpha": loss_alpha,
            "alpha": alpha,
            "log_alpha": new_log_alpha,
            "entropy": entropy,
            "mean_log_prob": jnp.mean(action_log_probs),
            "grad_q1": optax.global_norm(grad_q1),
            "grad_q2": optax.global_norm(grad_q2),
            "grad_pi": optax.global_norm(grad_pi),
            "grad_alpha": jnp.abs(grad_alpha),
            "update_q1": q1_update_norm,
            "update_q2": q2_update_norm,
            "param_q1": q1_param_norm,
            "param_q2": q2_param_norm,
            "relative_update_q1": (
                q1_update_norm / (q1_param_norm + 1e-8)
            ),
            "relative_update_q2": (
                q2_update_norm / (q2_param_norm + 1e-8)
            ),
            **q_diagnostics,
            **pi_diagnostics,
        }
        return new_learner_state, logs

    def _hk_apply_policy(
        self,
        observations: types.NestedArray,
    ) -> types.NestedArray:
        network_observations = self._network_observations(observations)
        return PolicyNetwork(
            output_sizes=self._hidden_sizes,
            action_spec=self.environment_spec.actions,
        )(network_observations)

    def _hk_apply_q(
        self,
        observations: types.NestedArray,
        actions: types.NestedArray,
    ) -> types.NestedArray:
        network_observations = self._network_observations(observations)
        q_actions = self._normalize_actions_for_q(actions)
        return SoftQNetwork(
            output_sizes=self._hidden_sizes,
        )(network_observations, q_actions)

    def _project_actions_for_q(
        self,
        observations: types.NestedArray,
        actions: types.NestedArray,
    ) -> types.NestedArray:
        if not self._use_cil:
            return actions

        constraint_observations = self._constraint_observations(
            observations
        )
        safe_out = get_safe_action_batch(
            actions,
            constraint_observations,
            self._cil_provider_params,
            self._constraint_provider,
            self._u_min,
            self._u_max,
        )
        return safe_out.u_safe

    def _get_action(
        self,
        key: chex.PRNGKey,
        policy_params: types.NestedArray,
        observations: types.NestedArray,
        deterministic: bool = False,
    ) -> types.NestedArray:
        obs = jnp.reshape(observations, (1, -1))
        mu, log_sigma = self.apply_policy(policy_params, obs)

        if deterministic:
            nominal_action_batched = self._transform_action_to_env_spec(mu)
        else:
            nominal_action_batched, _ = self._sample_action(
                key,
                mu,
                log_sigma,
            )

        nominal_action = jnp.ravel(nominal_action_batched)
        if not self._use_cil:
            return nominal_action

        constraint_observation = self._constraint_observations(
            jnp.ravel(observations)
        )
        safe_out = get_safe_action(
            u_nom=nominal_action,
            obs=constraint_observation,
            provider_params=self._cil_provider_params,
            constraint_provider=self._constraint_provider,
            u_min=self._u_min,
            u_max=self._u_max,
        )
        return safe_out.u_safe

    def _sample_action(
        self,
        key: chex.PRNGKey,
        mu: types.NestedArray,
        raw_log_sigma: types.NestedArray,
    ) -> Tuple[types.NestedArray, types.NestedArray]:
        bounded_log_sigma = self._bound_log_sigma(raw_log_sigma)
        standard_deviation = jnp.exp(bounded_log_sigma)

        pre_tanh_actions = rlax.gaussian_diagonal().sample(
            key,
            mu,
            standard_deviation,
        )
        log_prob = rlax.gaussian_diagonal().logprob(
            pre_tanh_actions,
            mu,
            standard_deviation,
        )

        # Stable tanh Jacobian correction. The affine conversion from
        # normalized to physical actions is intentionally excluded so that
        # target_entropy remains in normalized action coordinates.
        log_prob -= (
            2.0
            * (
                jnp.log(2.0)
                - pre_tanh_actions
                - jax.nn.softplus(-2.0 * pre_tanh_actions)
            )
        ).sum(axis=-1)

        actions = self._transform_action_to_env_spec(pre_tanh_actions)
        return actions, log_prob

    def _transform_action_to_env_spec(
        self,
        actions: types.NestedArray,
    ) -> types.NestedArray:
        normalized_actions = jnp.tanh(actions)
        return (
            (self._u_max - self._u_min)
            * (normalized_actions + 1.0)
            / 2.0
            + self._u_min
        )
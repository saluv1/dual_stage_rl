"""
Soft Actor-Critic agent with:
  * twin Q networks,
  * online/target V networks,
  * automatic entropy-temperature tuning,
  * alpha floor,
  * Q-value clipping,
  * optional differentiable CIL action projection.
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

from src.agents.networks import PolicyNetwork, SoftQNetwork, ValueNetwork
from src.cil.action_filter import get_safe_action, get_safe_action_batch
from src.replay_buffers.buffer import ReplayBuffer
from src.utils.training_utils import (
    LearnerState,
    OptState,
    ParamState,
    Transitions,
)


SIGMA_MIN = -20.0
SIGMA_MAX = 2.0


class SAC:
    """Legacy SAC with V/V-target, learned alpha, and optional CIL."""

    def __init__(
        self,
        rng: chex.PRNGKey,
        environment_spec: specs.EnvironmentSpec,
        config: ConfigDict,
        use_cil: bool = False,
        cil_provider_params=None,
        constraint_provider=None,
    ):
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

        # --------------------------------------------------------------
        # Entropy-temperature configuration
        # --------------------------------------------------------------
        self._auto_alpha = bool(
            getattr(config, "auto_alpha", True)
        )
        self._alpha_init = float(
            getattr(config, "alpha", 0.2)
        )
        self._alpha_min = float(
            getattr(config, "alpha_min", 0.01)
        )
        self._target_entropy = float(
            getattr(
                config,
                "target_entropy",
                -float(self.action_dim),
            )
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

        # --------------------------------------------------------------
        # Numerical-stability configuration
        # --------------------------------------------------------------
        self._q_clip = float(
            getattr(config, "q_clip", 5e6)
        )
        self._grad_clip_norm = float(
            getattr(config, "grad_clip_norm", 5.0)
        )

        if self._q_clip <= 0.0:
            raise ValueError("config.q_clip must be positive.")
        if self._grad_clip_norm <= 0.0:
            raise ValueError(
                "config.grad_clip_norm must be positive."
            )

        self._rng = rng

        # --------------------------------------------------------------
        # Haiku networks
        # --------------------------------------------------------------
        self._init_policy, self._apply_policy = (
            hk.without_apply_rng(
                hk.transform(self._hk_apply_policy)
            )
        )
        self._init_value, self._apply_value = (
            hk.without_apply_rng(
                hk.transform(self._hk_apply_value)
            )
        )
        self._init_q, self._apply_q = (
            hk.without_apply_rng(
                hk.transform(self._hk_apply_q)
            )
        )

        # --------------------------------------------------------------
        # Optimizers
        # --------------------------------------------------------------
        self.optimizer_q = optax.chain(
            optax.clip_by_global_norm(
                self._grad_clip_norm
            ),
            optax.adam(self.config.q_lr),
        )
        self.optimizer_v = optax.chain(
            optax.clip_by_global_norm(
                self._grad_clip_norm
            ),
            optax.adam(self.config.v_lr),
        )
        self.optimizer_p = optax.chain(
            optax.clip_by_global_norm(
                self._grad_clip_norm
            ),
            optax.adam(self.config.p_lr),
        )
        self.optimizer_alpha = optax.chain(
            optax.clip_by_global_norm(
                self._grad_clip_norm
            ),
            optax.adam(self.config.alpha_lr),
        )

        # --------------------------------------------------------------
        # Loss gradients
        # --------------------------------------------------------------
        self._grad_q1 = jax.value_and_grad(
            self._loss_fn_q,
            argnums=0,
        )
        self._grad_q2 = jax.value_and_grad(
            self._loss_fn_q,
            argnums=1,
        )
        self._grad_v = jax.value_and_grad(
            self._loss_fn_v
        )
        self._grad_pi = jax.value_and_grad(
            self._loss_fn_pi,
            has_aux=True,
        )
        self._grad_alpha = jax.value_and_grad(
            self._loss_fn_alpha
        )

        # --------------------------------------------------------------
        # JIT wrappers
        # --------------------------------------------------------------
        self.init_fn = jax.jit(self._init_fn)
        self.update_fn = jax.jit(self._update_fn)
        self.apply_policy = jax.jit(self._apply_policy)
        self.apply_value = jax.jit(self._apply_value)
        self.apply_q = jax.jit(self._apply_q)
        self.get_action = jax.jit(
            self._get_action,
            static_argnums=3,
        )

        self.buffer = ReplayBuffer(
            size_=self.config.replay_buffer_capacity,
            featuredim_=self.observation_dim,
            actiondim_=self.action_dim,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> LearnerState:
        self._rng, key = jax.random.split(self._rng, 2)

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

        return self.init_fn(
            key,
            dummy_obs,
            dummy_actions,
        )

    def _init_fn(
        self,
        rng: chex.PRNGKey,
        dummy_obs: types.NestedArray,
        dummy_actions: types.NestedArray,
    ) -> LearnerState:
        key_q1, key_q2, key_value, key_policy = (
            jax.random.split(rng, 4)
        )

        q1_params = self._init_q(
            key_q1,
            dummy_obs,
            dummy_actions,
        )
        q2_params = self._init_q(
            key_q2,
            dummy_obs,
            dummy_actions,
        )
        v_params = self._init_value(
            key_value,
            dummy_obs,
        )
        policy_params = self._init_policy(
            key_policy,
            dummy_obs,
        )

        v_target_params = jax.tree_util.tree_map(
            lambda value: value.copy(),
            v_params,
        )

        log_alpha = jnp.asarray(
            math.log(self._alpha_init),
            dtype=jnp.float32,
        )

        return LearnerState(
            params=ParamState(
                policy=policy_params,
                v=v_params,
                q1=q1_params,
                q2=q2_params,
                v_target=v_target_params,
                log_alpha=log_alpha,
            ),
            opt_state=OptState(
                policy=self.optimizer_p.init(
                    policy_params
                ),
                v=self.optimizer_v.init(v_params),
                q1=self.optimizer_q.init(q1_params),
                q2=self.optimizer_q.init(q2_params),
                alpha=self.optimizer_alpha.init(
                    log_alpha
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _current_alpha(
        self,
        log_alpha: chex.ArrayNumpy,
    ) -> chex.ArrayNumpy:
        if not self._auto_alpha:
            return jnp.asarray(
                self._alpha_init,
                dtype=jnp.float32,
            )

        return jnp.maximum(
            jnp.exp(log_alpha),
            self._alpha_min,
        )

    def _clip_q_value(
        self,
        q_value: chex.ArrayNumpy,
    ) -> chex.ArrayNumpy:
        return jnp.clip(
            q_value,
            -self._q_clip,
            self._q_clip,
        )

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def _loss_fn_q(
        self,
        q1_params: types.NestedArray,
        q2_params: types.NestedArray,
        v_target_params: types.NestedArray,
        transitions: Transitions,
    ) -> chex.ArrayNumpy:
        target_v = self.apply_value(
            v_target_params,
            transitions.next_observations,
        )

        rewards = transitions.rewards.astype(
            target_v.dtype
        )
        dones = transitions.dones.astype(
            target_v.dtype
        )

        chex.assert_equal_shape(
            [rewards, dones, target_v]
        )

        target_q_value = (
            rewards * self.config.scale_reward
            + self.config.gamma
            * (1.0 - dones)
            * target_v
        )

        # Clip the Bellman target. The critic prediction itself is not
        # clipped before the MSE, so an out-of-range critic still receives
        # a gradient that pulls it back toward the clipped target.
        target_q_value = self._clip_q_value(
            target_q_value
        )
        target_q_value = jax.lax.stop_gradient(
            target_q_value
        )

        predicted_q1_value = self.apply_q(
            q1_params,
            transitions.observations,
            transitions.actions,
        )
        predicted_q2_value = self.apply_q(
            q2_params,
            transitions.observations,
            transitions.actions,
        )

        chex.assert_equal_shape(
            [
                target_q_value,
                predicted_q1_value,
                predicted_q2_value,
            ]
        )

        td_error_q1 = (
            target_q_value - predicted_q1_value
        )
        td_error_q2 = (
            target_q_value - predicted_q2_value
        )

        q1_loss = 0.5 * jnp.mean(
            jnp.square(td_error_q1)
        )
        q2_loss = 0.5 * jnp.mean(
            jnp.square(td_error_q2)
        )

        return q1_loss + q2_loss

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
        Tuple[chex.ArrayNumpy, chex.ArrayNumpy],
    ]:
        mu, sigma = self.apply_policy(
            policy_params,
            transitions.observations,
        )

        new_actions, action_log_probs = (
            self._sample_action(
                key,
                mu,
                sigma,
            )
        )

        # During Phase II, Q is evaluated on the CIL-projected action.
        q_actions = self._project_actions_for_q(
            transitions.observations,
            new_actions,
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

        predicted_new_q_value = self._clip_q_value(
            jnp.minimum(q1_pi, q2_pi)
        )

        alpha = jax.lax.stop_gradient(
            self._current_alpha(log_alpha)
        )

        policy_loss = jnp.mean(
            alpha * action_log_probs
            - predicted_new_q_value
        )

        l2_loss = 0.5 * sum(
            jnp.sum(jnp.square(parameter))
            for parameter in jax.tree_util.tree_leaves(
                policy_params
            )
        )
        l2_coef = float(
            getattr(
                self.config,
                "policy_l2_coef",
                0.0,
            )
        )

        total_policy_loss = (
            policy_loss + l2_coef * l2_loss
        )

        sampled_entropy = -jnp.mean(
            action_log_probs
        )

        return total_policy_loss, (
            sampled_entropy,
            action_log_probs,
        )

    def _loss_fn_v(
        self,
        v_params: types.NestedArray,
        policy_params: types.NestedArray,
        q1_params: types.NestedArray,
        q2_params: types.NestedArray,
        log_alpha: chex.ArrayNumpy,
        transitions: Transitions,
        key: chex.PRNGKey,
    ) -> chex.ArrayNumpy:
        mu, sigma = self.apply_policy(
            policy_params,
            transitions.observations,
        )

        new_actions, action_log_probs = (
            self._sample_action(
                key,
                mu,
                sigma,
            )
        )

        q_actions = self._project_actions_for_q(
            transitions.observations,
            new_actions,
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

        q_pi = self._clip_q_value(
            jnp.minimum(q1_pi, q2_pi)
        )

        alpha = jax.lax.stop_gradient(
            self._current_alpha(log_alpha)
        )

        target_value = (
            q_pi - alpha * action_log_probs
        )
        target_value = self._clip_q_value(
            target_value
        )
        target_value = jax.lax.stop_gradient(
            target_value
        )

        predicted_value = self.apply_value(
            v_params,
            transitions.observations,
        )

        error = predicted_value - target_value
        return 0.5 * jnp.mean(jnp.square(error))

    def _loss_fn_alpha(
        self,
        log_alpha: chex.ArrayNumpy,
        action_log_probs: chex.ArrayNumpy,
    ) -> chex.ArrayNumpy:
        """Automatic entropy-temperature objective."""
        entropy_residual = jax.lax.stop_gradient(
            action_log_probs
            + self._target_entropy
        )

        # Optimize log_alpha, but use alpha=exp(log_alpha) in the
        # Lagrangian objective. This matches the automatic-temperature
        # formulation while keeping alpha strictly positive.
        alpha = jnp.exp(log_alpha)
        return -jnp.mean(
            alpha * entropy_residual
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _update_fn(
        self,
        curr_ls: LearnerState,
        transitions: Transitions,
        update_key: chex.PRNGKey,
    ) -> Tuple[LearnerState, Dict[str, chex.ArrayNumpy]]:
        key_pi, key_v = jax.random.split(
            update_key,
            2,
        )

        # --------------------------------------------------------------
        # Q update
        # --------------------------------------------------------------
        loss_q, grad_q1 = self._grad_q1(
            curr_ls.params.q1,
            curr_ls.params.q2,
            curr_ls.params.v_target,
            transitions,
        )
        _, grad_q2 = self._grad_q2(
            curr_ls.params.q1,
            curr_ls.params.q2,
            curr_ls.params.v_target,
            transitions,
        )

        q1_updates, q1_opt_state = (
            self.optimizer_q.update(
                grad_q1,
                curr_ls.opt_state.q1,
            )
        )
        q2_updates, q2_opt_state = (
            self.optimizer_q.update(
                grad_q2,
                curr_ls.opt_state.q2,
            )
        )

        curr_ls.params.q1 = optax.apply_updates(
            curr_ls.params.q1,
            q1_updates,
        )
        curr_ls.params.q2 = optax.apply_updates(
            curr_ls.params.q2,
            q2_updates,
        )
        curr_ls.opt_state.q1 = q1_opt_state
        curr_ls.opt_state.q2 = q2_opt_state

        # --------------------------------------------------------------
        # Policy update
        # --------------------------------------------------------------
        (
            loss_pi,
            (entropy, action_log_probs),
        ), grad_pi = self._grad_pi(
            curr_ls.params.policy,
            curr_ls.params.q1,
            curr_ls.params.q2,
            curr_ls.params.log_alpha,
            transitions,
            key_pi,
        )

        policy_updates, policy_opt_state = (
            self.optimizer_p.update(
                grad_pi,
                curr_ls.opt_state.policy,
            )
        )
        curr_ls.params.policy = optax.apply_updates(
            curr_ls.params.policy,
            policy_updates,
        )
        curr_ls.opt_state.policy = policy_opt_state

        # --------------------------------------------------------------
        # Value update
        # --------------------------------------------------------------
        loss_v, grad_v = self._grad_v(
            curr_ls.params.v,
            curr_ls.params.policy,
            curr_ls.params.q1,
            curr_ls.params.q2,
            curr_ls.params.log_alpha,
            transitions,
            key_v,
        )

        v_updates, v_opt_state = (
            self.optimizer_v.update(
                grad_v,
                curr_ls.opt_state.v,
            )
        )
        curr_ls.params.v = optax.apply_updates(
            curr_ls.params.v,
            v_updates,
        )
        curr_ls.opt_state.v = v_opt_state

        # --------------------------------------------------------------
        # Temperature update
        # --------------------------------------------------------------
        if self._auto_alpha:
            loss_alpha, grad_alpha = (
                self._grad_alpha(
                    curr_ls.params.log_alpha,
                    action_log_probs,
                )
            )

            alpha_updates, alpha_opt_state = (
                self.optimizer_alpha.update(
                    grad_alpha,
                    curr_ls.opt_state.alpha,
                )
            )

            new_log_alpha = optax.apply_updates(
                curr_ls.params.log_alpha,
                alpha_updates,
            )

            # Hard floor: alpha >= alpha_min.
            curr_ls.params.log_alpha = jnp.maximum(
                new_log_alpha,
                self._log_alpha_min,
            )
            curr_ls.opt_state.alpha = (
                alpha_opt_state
            )
        else:
            loss_alpha = jnp.asarray(
                0.0,
                dtype=jnp.float32,
            )
            grad_alpha = jnp.zeros_like(
                curr_ls.params.log_alpha
            )

        # --------------------------------------------------------------
        # Target V Polyak update
        # --------------------------------------------------------------
        curr_ls.params.v_target = (
            jax.tree_util.tree_map(
                lambda target, online: (
                    target
                    + self.config.tau
                    * (online - target)
                ),
                curr_ls.params.v_target,
                curr_ls.params.v,
            )
        )

        alpha = self._current_alpha(
            curr_ls.params.log_alpha
        )

        logs = {
            "loss_q": loss_q,
            "loss_pi": loss_pi,
            "loss_v": loss_v,
            "loss_alpha": loss_alpha,
            "alpha": alpha,
            "log_alpha": curr_ls.params.log_alpha,
            "entropy": entropy,
            "mean_log_prob": jnp.mean(
                action_log_probs
            ),
            "grad_q1": self._tree_squared_norm(
                grad_q1
            ),
            "grad_q2": self._tree_squared_norm(
                grad_q2
            ),
            "grad_pi": self._tree_squared_norm(
                grad_pi
            ),
            "grad_v": self._tree_squared_norm(
                grad_v
            ),
            "grad_alpha": jnp.sum(
                jnp.square(grad_alpha)
            ),
        }

        return curr_ls, logs

    @staticmethod
    def _tree_squared_norm(tree) -> chex.ArrayNumpy:
        return sum(
            jnp.sum(jnp.square(leaf))
            for leaf in jax.tree_util.tree_leaves(tree)
        )

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------
    def _hk_apply_policy(
        self,
        observations: types.NestedArray,
    ) -> types.NestedArray:
        return PolicyNetwork(
            [256, 256],
            self.environment_spec.actions,
        )(observations)

    def _hk_apply_q(
        self,
        observations: types.NestedArray,
        actions: types.NestedArray,
    ) -> types.NestedArray:
        return SoftQNetwork(
            [256, 256]
        )(observations, actions)

    def _hk_apply_value(
        self,
        observations: types.NestedArray,
    ) -> types.NestedArray:
        return ValueNetwork(
            [256, 256]
        )(observations)

    # ------------------------------------------------------------------
    # CIL and action sampling
    # ------------------------------------------------------------------
    def _project_actions_for_q(
        self,
        observations: types.NestedArray,
        actions: types.NestedArray,
    ) -> types.NestedArray:
        if not self._use_cil:
            return actions

        safe_out = get_safe_action_batch(
            actions,
            observations,
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
        mu, sigma = self.apply_policy(
            policy_params,
            obs,
        )

        if deterministic:
            u_nom_batched = (
                self._transform_action_to_env_spec(mu)
            )
        else:
            u_nom_batched, _ = self._sample_action(
                key,
                mu,
                sigma,
            )

        u_nom = jnp.ravel(u_nom_batched)

        if not self._use_cil:
            return u_nom

        safe_out = get_safe_action(
            u_nom=u_nom,
            obs=jnp.ravel(observations),
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
        sigma: types.NestedArray,
    ) -> Tuple[types.NestedArray, types.NestedArray]:
        standard_deviation = jnp.exp(
            jnp.clip(
                sigma,
                SIGMA_MIN,
                SIGMA_MAX,
            )
        )

        pre_tanh_actions = (
            rlax.gaussian_diagonal().sample(
                key,
                mu,
                standard_deviation,
            )
        )

        log_prob = rlax.gaussian_diagonal().logprob(
            pre_tanh_actions,
            mu,
            standard_deviation,
        )

        # Tanh-squashing correction. The subsequent affine map into the
        # physical action bounds is intentionally omitted from log_prob so
        # target_entropy=-action_dim remains in normalized-action units.
        log_prob -= (
            2.0
            * (
                jnp.log(2.0)
                - pre_tanh_actions
                - jax.nn.softplus(
                    -2.0 * pre_tanh_actions
                )
            )
        ).sum(axis=1)

        actions = self._transform_action_to_env_spec(
            pre_tanh_actions
        )

        return actions, log_prob

    def _transform_action_to_env_spec(
        self,
        actions: types.NestedArray,
    ) -> types.NestedArray:
        action_max = jnp.asarray(
            self.environment_spec.actions.maximum,
            dtype=jnp.float32,
        )
        action_min = jnp.asarray(
            self.environment_spec.actions.minimum,
            dtype=jnp.float32,
        )

        normalized_actions = jnp.tanh(actions)

        return (
            (action_max - action_min)
            * (normalized_actions + 1.0)
            / 2.0
            + action_min
        )

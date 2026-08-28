from types import SimpleNamespace
import unittest

import numpy as np

from ps2rl_px4_bridge.policy_runner import DelayObservationConfig, PS2RLPolicy


class _FakeJax:
    @staticmethod
    def device_get(value):
        return value


class _FakeJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value)


def _policy(delay_cfg: DelayObservationConfig, actor_obs_dim: int) -> PS2RLPolicy:
    policy = PS2RLPolicy.__new__(PS2RLPolicy)
    policy.delay_obs_cfg = delay_cfg
    policy.include_time_features = True
    policy.env_cfg = SimpleNamespace(gravity=9.81)
    policy.dt = 0.02
    policy._ref_dt = 0.02
    policy._ref_last = 1
    x0 = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    x1 = x0.copy()
    x1[0] += 0.02
    policy._ref_states = np.stack([x0, x1])
    policy._ref_omega = np.zeros((2, 3), dtype=np.float64)
    policy._actor_obs_dim = actor_obs_dim
    policy.entry_state = x0.copy()
    policy.reset_history(x0)
    return policy


class ObservationLayoutTests(unittest.TestCase):
    def test_nominal_observation_is_unchanged_26d(self):
        policy = _policy(DelayObservationConfig(), 26)
        obs = policy.build_obs(policy.entry_state, 0.0)

        self.assertEqual(obs.shape, (26,))
        np.testing.assert_allclose(obs[:10], policy.entry_state)
        np.testing.assert_allclose(obs[-3:], np.array([0.0, 0.0, 1.0]))

    def test_delay_policy_starts_with_five_hover_commands(self):
        cfg = DelayObservationConfig(hist_action_len=5, act_delay_steps=5, act_delay_jitter=2)
        policy = _policy(cfg, 46)
        obs = policy.build_obs(policy.entry_state, 0.0)

        self.assertEqual(obs.shape, (46,))
        expected = np.tile(np.array([9.81, 0.0, 0.0, 0.0]), (5, 1))
        np.testing.assert_allclose(obs[26:46].reshape(5, 4), expected)

    def test_current_command_is_visible_only_on_next_observation(self):
        cfg = DelayObservationConfig(hist_action_len=5, act_delay_steps=5, act_delay_jitter=2)
        policy = _policy(cfg, 46)
        policy._jax = _FakeJax()
        policy._jnp = _FakeJnp()
        policy._params = object()
        selected = np.array([12.0, 1.0, 2.0, 3.0])
        captured = []

        def eval_action(_params, obs):
            captured.append(np.asarray(obs).copy())
            return selected.copy(), selected.copy(), np.array(0.0)

        policy._eval_action = eval_action
        policy.act(policy.entry_state, 0.0)
        policy.act(policy.entry_state, 0.02)

        hover = np.array([9.81, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(captured[0][26:46].reshape(5, 4), np.tile(hover, (5, 1)))
        expected_next = np.vstack([np.tile(hover, (4, 1)), selected])
        np.testing.assert_allclose(captured[1][26:46].reshape(5, 4), expected_next)

    def test_saved_delay_dimension_is_46(self):
        cfg = DelayObservationConfig.from_dict(
            {
                "act_delay_steps": 5,
                "act_delay_jitter": 2,
                "hist_action_len": 5,
                "hist_state_len": 0,
                "ref_preview_len": 0,
            }
        )
        self.assertEqual(26 + cfg.extra_obs_dim(), 46)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from backup_policy.replay_buffer import ReplayBuffer
from backup_policy.td3 import TD3, infer_checkpoint_obs_dim, safe_arrival_obs
from bcbf.official_ab import ABConfig, ABConstraintBuilder


class Phase1CoreTests(unittest.TestCase):
    def test_terminal_wrapper(self):
        q = torch.tensor([[0.2], [0.7], [0.4]])
        goal = torch.tensor([[1.0], [0.0], [0.0]])
        fail = torch.tensor([[0.0], [1.0], [0.0]])
        full = TD3.full_value_from_flags(q, goal, fail)
        self.assertTrue(torch.allclose(full, torch.tensor([[1.0], [0.0], [0.4]])))

    def test_replay_and_update_are_finite(self):
        rng = np.random.default_rng(0)
        replay = ReplayBuffer(10, 4, max_size=256)
        for i in range(192):
            state = rng.normal(size=10).astype(np.float32)
            state[6:10] /= max(np.linalg.norm(state[6:10]), 1e-6)
            next_state = state.copy()
            next_state[:6] += 0.01 * rng.normal(size=6)
            action = np.array([
                rng.uniform(0.0, 4.0 * 9.81),
                *rng.uniform(-18.0, 18.0, size=3),
            ])
            goal_next = float(i % 17 == 0)
            fail_next = float(i % 19 == 0 and not goal_next)
            replay.add(state, action, next_state, 0.0, 1.0, goal_next, fail_next)
        policy = TD3(10, 4, 1.0)
        metrics = policy.train(replay, batch_size=64)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_checkpoint_round_trip_and_legacy_encoder(self):
        state = np.array([0, 0, 2, 0, 0, 0, 1, 0, 0, 0], dtype=float)
        self.assertEqual(safe_arrival_obs(state, obs_dim=10).shape, (10,))
        self.assertEqual(safe_arrival_obs(state, obs_dim=8).shape, (8,))
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "model"
            policy = TD3(10, 4, 1.0, obs_dim=10)
            path = policy.save(prefix)
            self.assertEqual(infer_checkpoint_obs_dim(path), 10)
            loaded = TD3.from_checkpoint(path)
            self.assertEqual(loaded.obs_dim, 10)
            self.assertEqual(loaded.select_action(state).shape, (4,))

    def test_ab_shapes_and_identity(self):
        cfg = ABConfig(num_steps=4)
        builder = ABConstraintBuilder.for_smoke_test(cfg)
        state = np.array([0, 0, 2.2, 0.2, 0, -0.1, 1, 0, 0, 0], dtype=float)
        a, b, info = builder.compute_bcbf_rows(state)
        u = np.asarray(info["backup_action"])
        direct = builder.direct_derivative_margins(state, u)
        self.assertEqual(a.shape, (cfg.num_bcbf_rows, 4))
        self.assertEqual(b.shape, (cfg.num_bcbf_rows,))
        abs_error = float(np.max(np.abs((b - a @ u) - direct)))
        scale = max(1.0, float(np.max(np.abs(direct))), float(np.max(np.abs(b - a @ u))))
        rel_error = abs_error / scale
        self.assertTrue(abs_error <= 3e-3 or rel_error <= 3e-4, f"abs={abs_error:.3e}, rel={rel_error:.3e}")
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all(np.isfinite(b)))


if __name__ == "__main__":
    unittest.main()

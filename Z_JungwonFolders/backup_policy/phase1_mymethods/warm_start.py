"""LQR warm-start / exploration-prior for the my-methods Phase-I trainer.

The clipped hover-LQR base controller is the SAME controller that certifies the
base set B. Running whole episodes under it early in training injects successful
safe-arrival trajectories into the replay buffer, so the critic sees positive
(b=1) targets from hard states that a random policy would essentially never
reach. The per-episode use probability is annealed from lqr_prob_start down to
lqr_prob_end over lqr_anneal_steps.

Ported from the original monolithic train.py; the controller logic is unchanged,
only the imports point at the current modular helpers.
"""
from __future__ import annotations

import numpy as np

from backup_policy.phase1.state_action import compute_reduced_state, unscale_action


class LQRController:
    """Clipped LQR base controller around hover, usable as a behavior policy."""

    def __init__(self, K, g, z_des=2.0):
        self.K = K
        self.g = g
        self.z_des = z_des
        self.u_star = np.array([g, 0.0, 0.0, 0.0])

    def act_phys(self, state):
        xe = compute_reduced_state(state)
        u = self.u_star - self.K @ xe
        u[0] = np.clip(u[0], 0.0, 4.0 * self.g)
        u[1:4] = np.clip(u[1:4], -18.0, 18.0)
        return u

    def act_norm(self, state):
        return unscale_action(self.act_phys(state), self.g)


def lqr_fraction(t: int, prob_start: float, prob_end: float, anneal_steps: int) -> float:
    """Annealed per-episode probability of using the LQR behavior policy."""
    anneal = min(1.0, t / max(1, anneal_steps))
    return prob_start + (prob_end - prob_start) * anneal
"""Optional DLQR replay prefill for local experiments.

The official-aligned default is zero warm-start transitions.  When explicitly
enabled, this module stores only real environment transitions and successor
terminal flags; it never inserts terminal self-loops.
"""
from __future__ import annotations

import numpy as np

from env.dynamics import Dynamics
from .sampling import sample_initial_state
from .state_action import (
    compute_bfc,
    compute_reduced_state,
    reset_dynamics_state,
    scale_action,
    unscale_action,
)


class LQRController:
    def __init__(self, K, g: float, z_des: float = 2.0):
        self.K = np.asarray(K, dtype=float)
        self.g = float(g)
        self.z_des = float(z_des)
        self.u_star = np.array([self.g, 0.0, 0.0, 0.0], dtype=float)

    def act_phys(self, state: np.ndarray) -> np.ndarray:
        xe = compute_reduced_state(state)
        return np.clip(
            self.u_star - self.K @ xe,
            np.array([0.0, -18.0, -18.0, -18.0]),
            np.array([4.0 * self.g, 18.0, 18.0, 18.0]),
        )

    def act_norm(self, state: np.ndarray) -> np.ndarray:
        return unscale_action(self.act_phys(state), self.g)


def warm_start_buffer(
    replay_buffer,
    lqr: LQRController,
    sets,
    regions,
    rng: np.random.Generator,
    n_transitions: int = 0,
    max_episode_steps: int = 100,
) -> None:
    n_transitions = int(n_transitions)
    if n_transitions <= 0:
        print("Warm start disabled (official-aligned default).")
        return

    dyn = Dynamics()
    added = successes = failures = episodes = 0
    while added < n_transitions:
        state, _ = sample_initial_state(
            sets=sets, regions=regions, s=1.0, rng=rng, return_region=True
        )
        reset_dynamics_state(dyn, state)
        episodes += 1

        for _ in range(max_episode_steps):
            action_norm = lqr.act_norm(state)
            action_phys = scale_action(action_norm, dyn.g)
            b_cur, _, c_cur = compute_bfc(sets, state)
            next_state = dyn.step(action_phys).copy()
            b_next, f_next, _ = compute_bfc(sets, next_state)
            replay_buffer.add(
                state, action_phys, next_state, b_cur, c_cur,
                goal_next=b_next, fail_next=f_next,
            )
            added += 1
            state = next_state
            if b_next > 0.5:
                successes += 1
                break
            if f_next > 0.5:
                failures += 1
                break
            if added >= n_transitions:
                break

    print(
        f"Warm start: {added} real transitions from {episodes} DLQR episodes "
        f"(success={successes}, failure={failures})."
    )

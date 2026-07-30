"""Certified LQR controller and replay-buffer warm start."""
import numpy as np
from env.dynamics import Dynamics
from .state_action import compute_reduced_state, compute_bfc, scale_action, unscale_action, reset_dynamics_state
from .sampling import sample_initial_state

class LQRController:
    """
    Clipped LQR base controller around hover -- the SAME controller that
    certifies the base set B.

    Used only to PREFILL the replay buffer before training starts. The
    safe-arrival reward is an indicator b(x) in {0,1} with no shaping, so if
    the buffer contains almost no successes the critic regresses toward zero
    everywhere and the actor gradient is flat -- there is no signal to ascend,
    not merely a weak one. Measured random-policy arrival was ~35% from easy
    states but only ~3.5% from the hardest ones.

    A one-time prefill is deliberately preferred over mixing LQR episodes into
    ongoing collection: it keeps the behavior distribution stationary, adds no
    annealing schedule to tune, and -- since LQR data stops arriving once
    training begins -- creates no pull toward imitating the analytic backup,
    which the paper's whole contribution is about beating (35.5% -> 69.3%).
    """

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

def warm_start_buffer(
        replay_buffer,
        lqr,
        sets,
        regions,
        rng,
        n_transitions=100000,
        max_episode_steps=300,
        s_values=(0.0, 0.25, 0.5, 0.75, 1.0),
        action_noise=0.05
):
    """
    Prefill the replay buffer with LQR rollouts spanning ALL curriculum regions.

    Sweeping s across its range (and forcing each region in turn) is the point:
    the cold-start problem is worst in near_ceiling / bridge, so prefilling only
    easy states would leave exactly the gap we are trying to close.

    A little action noise is added so the critic sees more than a single action
    per state -- otherwise Q is trained on a measure-zero slice of the action
    space and extrapolates freely everywhere else.

    Note the buffer is FIFO with capacity 4e5, so this data is evicted after
    roughly 4e5 steps. It is a starting signal, not a permanent one -- by then
    the policy should be generating its own successes.
    """

    dyn = Dynamics()

    eval_regions = [
        "synthetic_capture", "synthetic_mid",
        "trace_general", "near_ceiling", "bridge"
    ]
    eval_regions = [
        r for r in eval_regions
        if r in ["synthetic_capture", "synthetic_mid"]
        or (r in regions and len(regions[r]) > 0)
    ]

    added = 0
    n_success = 0
    n_failure = 0
    n_episodes = 0
    per_region_success = {r: [0, 0] for r in eval_regions}

    while added < n_transitions:

        region = eval_regions[n_episodes % len(eval_regions)]
        s_val = s_values[(n_episodes // len(eval_regions)) % len(s_values)]

        try:
            state, region_name = sample_initial_state(
                sets=sets,
                regions=regions,
                s=s_val,
                rng=rng,
                return_region=True,
                force_region=region
            )
        except RuntimeError:
            n_episodes += 1
            continue

        reset_dynamics_state(dyn, state)
        n_episodes += 1
        per_region_success[region_name][1] += 1

        for step in range(max_episode_steps):

            b_cur, f_cur, c_cur = compute_bfc(sets, state)

            action_norm = lqr.act_norm(state)
            action_norm = np.clip(
                action_norm + rng.normal(0.0, action_noise, size=4),
                -1.0, 1.0
            )

            next_state = dyn.step(scale_action(action_norm, dyn.g)).copy()

            if not np.all(np.isfinite(next_state)):
                break

            b_next, f_next, c_next = compute_bfc(sets, next_state)

            replay_buffer.add(state, action_norm, next_state, b_cur, c_cur)
            added += 1

            success = b_next == 1.0
            failure = f_next == 1.0

            # Same terminal-anchor logic as the training loop (FIX #1).
            if success or failure:
                b_term, f_term, c_term = compute_bfc(sets, next_state)
                replay_buffer.add(
                    next_state, action_norm, next_state, b_term, c_term
                )
                added += 1

            state = next_state.copy()

            if success:
                n_success += 1
                per_region_success[region_name][0] += 1
                break

            if failure:
                n_failure += 1
                break

            if added >= n_transitions:
                break

    print("---------------------------------------")
    print(f"Warm start: {added} transitions from {n_episodes} LQR episodes")
    print(f"  successes: {n_success}, failures: {n_failure}")
    print("  per-region LQR arrival rate:")
    for r, (succ, tot) in per_region_success.items():
        if tot > 0:
            print(f"    {r}: {succ}/{tot} = {succ / tot:.3f}")
    print("---------------------------------------")

    dyn.state = state.copy()

    if hasattr(dyn, "curr_step"):
        dyn.curr_step = 0

    if hasattr(dyn, "xlist"):
        dyn.xlist = []

    if hasattr(dyn, "vlist"):
        dyn.vlist = []

    if hasattr(dyn, "qlist"):
        dyn.qlist = []

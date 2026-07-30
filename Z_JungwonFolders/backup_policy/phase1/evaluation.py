"""Stratified safe-arrival evaluation and mu_SA estimation."""
import numpy as np
from env.dynamics import Dynamics
from .state_action import compute_reduced_state, compute_bfc, scale_action, reset_dynamics_state
from .sampling import sample_initial_state, MU_SA_WEIGHTS

def eval_policy(
        policy,
        sets,
        regions,
        s,
        rng,
        eval_episodes=150,
        max_episode_steps=300,
        success_horizon_steps=100,
        stratified=True
):
    """
    Stratified evaluation producing a mu_SA estimate.

    mu_SA (paper Sec. 4.1.2, Eq. 9) is the fraction of the design region Omega
    from which the policy safely arrives at B within the backup horizon T.
    Here T = success_horizon_steps = 100 steps = 2.0 s, matching Table 8.

    FIX #3: the previous evaluator sampled from the CURRICULUM mixture, so the
    number of near_ceiling episodes was random and often tiny -- and the mixture
    itself changed as the curriculum advanced, making successive evaluations
    non-comparable. We now run a FIXED number of episodes PER REGION and combine
    them with FIXED weights (MU_SA_WEIGHTS), so the reported mu_SA is a stable,
    comparable estimate of the same quantity throughout training.
    """

    eval_dyn = Dynamics()

    if stratified:
        eval_regions = [
            "synthetic_capture", "synthetic_mid",
            "trace_general", "near_ceiling", "bridge"
        ]
        eval_regions = [
            r for r in eval_regions
            if r in ["synthetic_capture", "synthetic_mid"]
            or (r in regions and len(regions[r]) > 0)
        ]
        per_region = max(1, eval_episodes // len(eval_regions))
        plan = [(r, per_region) for r in eval_regions]
    else:
        plan = [(None, eval_episodes)]

    success_count = 0
    success_horizon_count = 0
    failure_count = 0
    timeout_count = 0
    total_episodes = 0

    steps_list = []
    min_hs_list = []
    final_hb_list = []

    region_stats = {}

    for forced_region, n_eps in plan:

        for _ in range(n_eps):

            state, region_name = sample_initial_state(
                sets=sets,
                regions=regions,
                s=s,
                rng=rng,
                return_region=True,
                force_region=forced_region
            )

            total_episodes += 1

            if region_name not in region_stats:
                region_stats[region_name] = {
                    "n": 0,
                    "success": 0,
                    "success_horizon": 0,
                    "failure": 0,
                    "timeout": 0,
                    "steps": [],
                    "final_hb": []
                }

            region_stats[region_name]["n"] += 1

            reset_dynamics_state(eval_dyn, state)

            min_hs = 1e9
            final_hb = None
            outcome = "timeout"
            terminal_step = max_episode_steps

            for step in range(max_episode_steps):

                action_norm = policy.select_action(np.array(state))
                action_norm = np.clip(action_norm, -1.0, 1.0)

                action = scale_action(action_norm, eval_dyn.g)
                next_state = eval_dyn.step(action).copy()

                b_next, f_next, c_next = compute_bfc(sets, next_state)

                min_hs = min(min_hs, sets.hs)

                reduced_next = compute_reduced_state(next_state)
                sets.compute_hb(reduced_next)
                final_hb = sets.hb

                state = next_state.copy()

                if b_next == 1.0:
                    outcome = "success"
                    terminal_step = step + 1
                    break

                if f_next == 1.0:
                    outcome = "failure"
                    terminal_step = step + 1
                    break

            if outcome == "success":

                success_count += 1
                region_stats[region_name]["success"] += 1

                # Only arrivals within the backup horizon T count toward mu_SA.
                if terminal_step <= success_horizon_steps:
                    success_horizon_count += 1
                    region_stats[region_name]["success_horizon"] += 1

            elif outcome == "failure":

                failure_count += 1
                region_stats[region_name]["failure"] += 1

            else:

                timeout_count += 1
                region_stats[region_name]["timeout"] += 1

            steps_list.append(terminal_step)
            min_hs_list.append(min_hs)

            # Clip h_B for REPORTING only. Divergent rollouts produce values on
            # the order of -1e6, which made the reported average meaningless.
            final_hb_list.append(max(final_hb, -1e3))

            region_stats[region_name]["steps"].append(terminal_step)
            region_stats[region_name]["final_hb"].append(max(final_hb, -1e3))

    success_rate = success_count / total_episodes
    success_horizon_rate = success_horizon_count / total_episodes
    failure_rate = failure_count / total_episodes
    timeout_rate = timeout_count / total_episodes
    avg_steps = np.mean(steps_list)
    avg_min_hs = np.mean(min_hs_list)
    avg_final_hb = np.mean(final_hb_list)
    median_final_hb = np.median(final_hb_list)

    # ---- mu_SA estimate: fixed-weight combination of per-region rates ----
    mu_sa = 0.0
    weight_total = 0.0

    for name, stats in region_stats.items():
        w = MU_SA_WEIGHTS.get(name, 0.0)
        if w <= 0.0 or stats["n"] == 0:
            continue
        mu_sa += w * (stats["success_horizon"] / stats["n"])
        weight_total += w

    if weight_total > 0.0:
        mu_sa = mu_sa / weight_total

    worst_region_success = min(
        stats["success_horizon"] / stats["n"]
        for stats in region_stats.values()
    )

    near_ceiling_rate = None
    if "near_ceiling" in region_stats and region_stats["near_ceiling"]["n"] > 0:
        near_ceiling_rate = (
            region_stats["near_ceiling"]["success_horizon"]
            / region_stats["near_ceiling"]["n"]
        )

    print("---------------------------------------")
    print(f"Evaluation over {total_episodes} episodes")
    print(f"Difficulty s: {s:.3f}")
    print(f"mu_SA (weighted, T={success_horizon_steps} steps): {mu_sa:.3f}")
    if near_ceiling_rate is not None:
        print(f"  near_ceiling rate: {near_ceiling_rate:.3f}   (paper: 0.693)")
    print(f"Success rate (any horizon): {success_rate:.3f}")
    print(f"Success <= {success_horizon_steps} steps: {success_horizon_rate:.3f}")
    print(f"Failure rate: {failure_rate:.3f}")
    print(f"Timeout rate: {timeout_rate:.3f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average min h_S: {avg_min_hs:.3f}")
    print(f"Average final h_B (clipped): {avg_final_hb:.3f}")
    print(f"Median final h_B: {median_final_hb:.3f}")
    print(f"Worst-region success_horizon: {worst_region_success:.3f}")
    print("")
    print("Per-region evaluation:")

    for name, stats in region_stats.items():

        n = stats["n"]
        s_rate = stats["success"] / n
        sh_rate = stats["success_horizon"] / n
        f_rate = stats["failure"] / n
        t_rate = stats["timeout"] / n
        avg_region_steps = np.mean(stats["steps"])
        med_region_hb = np.median(stats["final_hb"])

        print(
            f"  {name}: "
            f"n={n}, "
            f"success={s_rate:.3f}, "
            f"success_horizon={sh_rate:.3f}, "
            f"failure={f_rate:.3f}, "
            f"timeout={t_rate:.3f}, "
            f"avg_steps={avg_region_steps:.1f}, "
            f"median_final_hB={med_region_hb:.1f}"
        )

    print("---------------------------------------")

    return (
        success_rate,
        success_horizon_rate,
        failure_rate,
        timeout_rate,
        avg_steps,
        avg_min_hs,
        avg_final_hb,
        median_final_hb,
        worst_region_success,
        mu_sa
    )

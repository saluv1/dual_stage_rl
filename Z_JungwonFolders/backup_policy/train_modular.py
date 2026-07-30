"""Modular Phase-I safe-arrival training with indexed experiment folders."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.dynamics import Dynamics
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from backup_policy.td3 import TD3
from backup_policy.replay_buffer import ReplayBuffer
from backup_policy.phase1.state_action import compute_bfc, reset_dynamics_state, scale_action
from backup_policy.phase1.sampling import (
    classify_trace_states,
    generate_reference_trace,
    inspect_sampler,
    sample_initial_state,
)
from backup_policy.phase1.warm_start import LQRController, warm_start_buffer
from backup_policy.phase1.evaluation_detailed import evaluate_detailed
from backup_policy.phase1.experiment_io import (
    append_csv,
    checkpoint_prefix,
    create_run_directory,
    save_json,
    update_metadata,
)
from backup_policy.phase1.plotting import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PS2-RL Phase-I safe-arrival policy.")
    parser.add_argument("--experiment-root", default="experiments/safe_arrival")
    parser.add_argument("--run-index", type=int, default=None, help="Optional explicit index; otherwise the next index is used.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=5_000_000)
    parser.add_argument("--start-timesteps", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes-per-region", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--success-horizon-steps", type=int, default=100)
    parser.add_argument("--warm-start-transitions", type=int, default=100_000)
    parser.add_argument("--expl-noise", type=float, default=0.10)
    parser.add_argument("--checkpoint-every-evals", type=int, default=10)
    parser.add_argument("--skip-sampler-inspection", action="store_true")
    return parser.parse_args()


def build_eval_row(timestep, difficulty, summary):
    overall = summary["overall"]
    row = {
        "timestep": timestep,
        "difficulty": difficulty,
        "mu_sa": overall["weighted_mu_sa"],
        "success_rate": overall["safe_arrival_rate"],
        "success_horizon_rate": overall["safe_arrival_within_horizon_rate"],
        "failure_rate": overall["failure_rate"],
        "timeout_rate": overall["timeout_rate"],
        "mean_arrival_time_s": overall["mean_arrival_time_s_success_only"],
        "median_arrival_time_s": overall["median_arrival_time_s_success_only"],
        "worst_region_success_horizon_rate": overall["worst_region_within_horizon_rate"],
        "mean_min_hs": overall["mean_min_hs"],
        "worst_min_hs": overall["worst_min_hs"],
    }
    for region, metrics in summary["per_region"].items():
        row[f"{region}_success_rate"] = metrics["safe_arrival_rate"]
        row[f"{region}_success_horizon_rate"] = metrics["safe_arrival_within_horizon_rate"]
        row[f"{region}_mean_arrival_time_s"] = metrics["mean_arrival_time_s_success_only"]
        row[f"{region}_failure_rate"] = metrics["failure_rate"]
    return row


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    paths = create_run_directory(args.experiment_root, args.run_index)
    save_json(paths["configs"] / "training_config.json", vars(args))
    update_metadata(paths["root"], status="initializing", seed=args.seed)
    print(f"Run directory: {paths['root']}")

    dyn = Dynamics()
    K, P = LQRGain(dt=dyn.del_t, g=dyn.g).gain()
    sets = SetIndicator(P=P, c_b=8.0, zceil=3.0)
    sets.P = P
    lqr = LQRController(K=K, g=dyn.g, z_des=2.0)

    trace_states = generate_reference_trace(n_points=200, n_variants=20, seed=0)
    regions = classify_trace_states(sets, trace_states, P, 8.0, near_ceiling_margin=0.25)
    save_json(paths["configs"] / "region_sizes.json", {name: len(values) for name, values in regions.items()})

    print("Trace region sizes:")
    for name, states in regions.items():
        print(f"  {name}: {len(states)}")
    if not args.skip_sampler_inspection:
        for probe in [0.0, 0.25, 0.5, 0.75, 1.0]:
            inspect_sampler(sets, regions, probe, rng, n_samples=500)

    policy = TD3(
        state_dim=10,
        action_dim=4,
        max_action=1.0,
        discount=0.99,
        tau=0.0025,
        policy_noise=0.10,
        noise_clip=0.10,
        policy_freq=2,
        actor_lr=1e-4,
        critic_lr=3e-4,
    )
    replay_buffer = ReplayBuffer(10, 4)

    difficulty = 0.0
    difficulty_step = 0.10
    difficulty_backoff = 0.05
    mu_sa_threshold = 0.80
    mu_sa_backoff_threshold = 0.40
    curriculum_window = 3
    min_evals_between_updates = 3
    stall_patience = 12
    stall_min_mu_sa = 0.55
    evals_since_update = 0
    mu_history = []
    best_mu_sa = -1.0
    eval_index = 0

    update_metadata(paths["root"], status="warm_start")
    warm_start_buffer(
        replay_buffer=replay_buffer,
        lqr=lqr,
        sets=sets,
        regions=regions,
        rng=rng,
        n_transitions=args.warm_start_transitions,
        max_episode_steps=args.max_episode_steps,
    )

    state = sample_initial_state(sets, regions, difficulty, rng)
    reset_dynamics_state(dyn, state)
    episode_timesteps = 0
    episode_num = 0
    episode_success = episode_failure = episode_timeout = 0
    update_metadata(paths["root"], status="training")

    try:
        for t in range(args.max_timesteps):
            episode_timesteps += 1
            if t < args.start_timesteps:
                action_norm = np.random.uniform(-1.0, 1.0, size=4)
            else:
                action_norm = policy.select_action(np.asarray(state))
                action_norm = np.clip(
                    action_norm + np.random.normal(0.0, args.expl_noise, size=4),
                    -1.0,
                    1.0,
                )

            action = scale_action(action_norm, dyn.g)
            b_cur, _, c_cur = compute_bfc(sets, state)
            next_state = dyn.step(action).copy()
            b_next, f_next, _ = compute_bfc(sets, next_state)
            replay_buffer.add(state, action_norm, next_state, b_cur, c_cur)

            success = b_next == 1.0
            failure = f_next == 1.0
            if success or failure:
                b_term, _, c_term = compute_bfc(sets, next_state)
                replay_buffer.add(next_state, action_norm, next_state, b_term, c_term)

            state = next_state.copy()
            if t >= args.start_timesteps and replay_buffer.size >= args.batch_size and t % 8 == 0:
                policy.train(replay_buffer, args.batch_size)

            timeout = episode_timesteps >= args.max_episode_steps
            if success or failure or timeout:
                if success:
                    episode_success += 1
                elif failure:
                    episode_failure += 1
                else:
                    episode_timeout += 1
                episode_num += 1
                append_csv(paths["logs"] / "training_episodes.csv", {
                    "episode": episode_num,
                    "timestep": t + 1,
                    "episode_steps": episode_timesteps,
                    "difficulty": difficulty,
                    "outcome": "success" if success else "failure" if failure else "timeout",
                })
                if episode_num % 50 == 0:
                    print(
                        f"T={t + 1} Episode={episode_num} EpisodeT={episode_timesteps} "
                        f"s={difficulty:.3f} success={episode_success} "
                        f"failure={episode_failure} timeout={episode_timeout}"
                    )
                state = sample_initial_state(sets, regions, difficulty, rng)
                reset_dynamics_state(dyn, state)
                episode_timesteps = 0

            if (t + 1) % args.eval_freq == 0:
                eval_index += 1
                summary, _, _ = evaluate_detailed(
                    policy=policy,
                    sets=sets,
                    regions=regions,
                    rng=rng,
                    difficulty=difficulty,
                    episodes_per_region=args.eval_episodes_per_region,
                    trajectories_per_region=0,
                    max_episode_steps=args.max_episode_steps,
                    success_horizon_steps=args.success_horizon_steps,
                )
                mu_sa = summary["overall"]["weighted_mu_sa"]
                append_csv(
                    paths["logs"] / "training_evaluations.csv",
                    build_eval_row(t + 1, difficulty, summary),
                )
                save_json(paths["logs"] / "latest_training_evaluation.json", summary)
                policy.save(checkpoint_prefix(paths["checkpoints"], "latest"))

                if eval_index % args.checkpoint_every_evals == 0:
                    label = f"eval_{eval_index:06d}_step_{t + 1:09d}"
                    policy.save(checkpoint_prefix(paths["checkpoints"], label))

                if mu_sa > best_mu_sa:
                    best_mu_sa = mu_sa
                    policy.save(checkpoint_prefix(paths["checkpoints"], "best"))
                    save_json(paths["logs"] / "best_evaluation.json", {
                        "eval_index": eval_index,
                        "timestep": t + 1,
                        "difficulty": difficulty,
                        "summary": summary,
                    })
                    print(f"New best weighted mu_SA={mu_sa:.3f}")

                print(
                    f"Evaluation {eval_index}: step={t + 1}, s={difficulty:.2f}, "
                    f"mu_SA={mu_sa:.3f}, total={summary['overall']['safe_arrival_rate']:.3f}, "
                    f"within_T={summary['overall']['safe_arrival_within_horizon_rate']:.3f}"
                )
                plot_training_history(paths["logs"] / "training_evaluations.csv", paths["training_plots"])

                evals_since_update += 1
                mu_history.append(mu_sa)
                if len(mu_history) > curriculum_window:
                    mu_history.pop(0)
                if evals_since_update >= min_evals_between_updates and len(mu_history) >= curriculum_window:
                    windowed = float(np.mean(mu_history))
                    stalled = evals_since_update >= stall_patience and windowed >= stall_min_mu_sa
                    if (windowed >= mu_sa_threshold or stalled) and difficulty < 1.0:
                        step = difficulty_step if windowed >= mu_sa_threshold else 0.5 * difficulty_step
                        difficulty = min(1.0, difficulty + step)
                        evals_since_update = 0
                        mu_history = []
                        print(f"Curriculum increased to s={difficulty:.3f}")
                    elif windowed < mu_sa_backoff_threshold and difficulty > 0.0:
                        difficulty = max(0.0, difficulty - difficulty_backoff)
                        evals_since_update = 0
                        mu_history = []
                        print(f"Curriculum backed off to s={difficulty:.3f}")

                update_metadata(
                    paths["root"],
                    status="training",
                    current_timestep=t + 1,
                    current_episode=episode_num,
                    current_difficulty=difficulty,
                    best_mu_sa=best_mu_sa,
                    latest_evaluation=summary["overall"],
                )

    except KeyboardInterrupt:
        print("Training interrupted; saving an interrupted checkpoint.")
        policy.save(checkpoint_prefix(paths["checkpoints"], "interrupted"))
        update_metadata(paths["root"], status="interrupted")
        raise
    else:
        policy.save(checkpoint_prefix(paths["checkpoints"], "final"))
        policy.save(checkpoint_prefix(paths["checkpoints"], "latest"))
        update_metadata(
            paths["root"],
            status="completed",
            final_timestep=args.max_timesteps,
            final_episode=episode_num,
            best_mu_sa=best_mu_sa,
        )
        print(f"Training completed. Outputs: {paths['root']}")


if __name__ == "__main__":
    main()

"""Train Phase-I safe arrival with MY methods (own sampler + LQR warm-start).

This is the variant trainer. It differs from ``train_modular.py`` in exactly
three ways:

  1. Initial-state sampling uses the hand-designed continuous-curriculum
     sampler (backup_policy.phase1_mymethods.sampling_mine), including the
     tracker-trace regions, instead of the official reset_library.
  2. Optional LQR warm-start: whole episodes are driven by the certified hover
     LQR with annealed probability, seeding the buffer with safe arrivals.
  3. The safe-arrival policy uses the 8-D reduced observation
     (pz - z_des, v, attitude-error), dropping xy position.

EVERYTHING about evaluation is deliberately shared with the current method:
the SAME official reset_library val split, the SAME Euler dynamics, the SAME
evaluate_detailed and metrics. Only training-time behavior and the saved obs
width differ, so the two runs are directly comparable. Model/log folders are
separate so the runs never collide.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.dynamics import Dynamics
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from backup_policy.td3 import TD3, LEGACY_OBS_DIM, OFFICIAL_OBS_DIM
from backup_policy.replay_buffer import ReplayBuffer
from backup_policy.phase1.state_action import compute_bfc, lqr_action, reset_dynamics_state, scale_action
from backup_policy.phase1.sampling import load_official_sampler
from backup_policy.phase1.evaluation_detailed import build_fixed_evaluation_set, evaluate_detailed, save_evaluation_set
from backup_policy.phase1.experiment_io import append_csv, checkpoint_prefix, create_run_directory, save_json, update_metadata
from backup_policy.phase1.plotting import plot_training_history

# My methods:
from backup_policy.phase1_mymethods import sampling_mine as MINE
from backup_policy.phase1_mymethods.warm_start import LQRController, lqr_fraction


def parse_args():
    p = argparse.ArgumentParser(description="Train Phase-I safe arrival with my methods.")
    p.add_argument("--experiment-root", default="Trained Models (My Methods)")
    p.add_argument("--evaluation-root", default="evaluation")
    p.add_argument("--reset-library", default="official_phase1_evaluation/assets/reset_library.pkl")
    p.add_argument("--run-index", type=int, default=None)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--max-timesteps", type=int, default=5_000_000)
    p.add_argument("--start-timesteps", type=int, default=5_000)
    p.add_argument("--update-after", type=int, default=2_000)
    p.add_argument("--update-every", type=int, default=8)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--eval-episodes-per-region", type=int, default=64)
    p.add_argument("--validation-seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--replay-size", type=int, default=400_000)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--success-horizon-steps", type=int, default=100)
    p.add_argument("--expl-noise", type=float, default=0.10)
    p.add_argument("--expl-noise-clip", type=float, default=0.10)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.0025)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--policy-freq", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--action-penalty", type=float, default=0.05)
    # curriculum (advance on windowed success, like the current trainer)
    p.add_argument("--curriculum-increment", type=float, default=0.10)
    p.add_argument("--curriculum-threshold", type=float, default=0.80)
    p.add_argument("--curriculum-window", type=int, default=50)
    p.add_argument("--curriculum-min-episodes", type=int, default=100)
    # my-methods knobs
    p.add_argument("--obs-dim", type=int, default=LEGACY_OBS_DIM, choices=[LEGACY_OBS_DIM, OFFICIAL_OBS_DIM],
                   help="8 = reduced (drop xy, my old design); 10 = full official state")
    p.add_argument("--trace-mode", default="tracker", choices=["tracker", "analytic"])
    p.add_argument("--trace-path", default="")
    p.add_argument("--use-warm-start", action="store_true", help="enable annealed LQR behavior policy")
    p.add_argument("--lqr-prob-start", type=float, default=0.50)
    p.add_argument("--lqr-prob-end", type=float, default=0.05)
    p.add_argument("--lqr-anneal-steps", type=int, default=400_000)
    p.add_argument("--skip-sampler-inspection", action="store_true")
    return p.parse_args()


def evaluation_rank(summary):
    p = summary["per_region"]; o = summary["overall"]
    arrival = o["mean_arrival_time_s_success_only"]
    return (
        float(o["weighted_mu_sa"]), -float(o["failure_rate"]),
        float(p.get("near_ceiling", {}).get("safe_arrival_rate", 0.0)),
        float(p.get("bridge", {}).get("safe_arrival_rate", 0.0)),
        float(o.get("terminal_at_horizon_rate", 0.0)),
        float(o.get("safe_rollout_rate", 0.0)),
        float(o.get("invariance_after_entry_rate", 0.0)),
        float(o.get("mean_discounted_safe_arrival_score", 0.0)),
        -float(arrival if arrival is not None else 1e9),
    )


def build_eval_row(timestep, difficulty, summary):
    o = summary["overall"]
    row = {
        "timestep": timestep, "difficulty": difficulty,
        "mu_sa": o["weighted_mu_sa"], "success_rate": o["safe_arrival_rate"],
        "success_horizon_rate": o["safe_arrival_within_horizon_rate"],
        "failure_rate": o["failure_rate"], "timeout_rate": o["timeout_rate"],
        "mean_arrival_time_s": o["mean_arrival_time_s_success_only"],
        "worst_region_success_horizon_rate": o["worst_region_within_horizon_rate"],
    }
    for region, values in summary["per_region"].items():
        row[f"{region}_success_horizon_rate"] = values["safe_arrival_within_horizon_rate"]
        row[f"{region}_failure_rate"] = values["failure_rate"]
    return row


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    paths = create_run_directory(args.experiment_root, args.run_index, args.evaluation_root)
    save_json(paths["configs"] / "training_config.json", vars(args))
    update_metadata(paths["root"], status="initializing", seed=args.seed, method="my_methods")
    print(f"Run directory: {paths['root']}")

    dyn = Dynamics(integrator="euler", dt=0.02, gravity=9.81)
    K, P = LQRGain(dt=dyn.del_t, g=dyn.g).gain()
    sets = SetIndicator(P=P, c_b=8.0, zceil=3.0)
    sets.P = P

    # --- MY sampler: build tracker-trace regions, then curriculum-sample ---
    trace_states = MINE.load_or_generate_trace(
        trace_path=args.trace_path, n_points=200, n_variants=20,
        seed=args.seed, dyn_factory=Dynamics, mode=args.trace_mode,
    )
    regions = MINE.classify_trace_states(
        sets=sets, trace_states=trace_states, P=P, c_b=8.0, near_ceiling_margin=0.25,
    )
    print("Trace region sizes:", {k: len(v) for k, v in regions.items()})
    if not args.skip_sampler_inspection:
        for probe in (0.0, 0.5, 1.0):
            MINE.inspect_sampler(sets, regions, probe, np.random.default_rng(args.seed + 70_000), n_samples=200)

    # --- SHARED evaluation: official reset_library val split + Euler env ---
    val_sampler = load_official_sampler(args.reset_library, split="val")
    validation_set = build_fixed_evaluation_set(
        sets, val_sampler, seed=args.validation_seed,
        episodes_per_region=args.eval_episodes_per_region, difficulty=1.0,
    )
    validation_path = paths["configs"] / "fixed_validation_states.npz"
    save_evaluation_set(validation_path, validation_set)

    # --- policy with MY obs width (8-D reduced by default) ---
    policy = TD3(
        state_dim=10, action_dim=4, max_action=1.0,
        discount=args.discount, tau=args.tau,
        policy_noise=0.0, noise_clip=0.0, policy_freq=args.policy_freq,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        obs_dim=args.obs_dim, max_grad_norm=args.max_grad_norm,
        huber_delta=args.huber_delta, action_penalty=args.action_penalty,
    )
    replay_buffer = ReplayBuffer(10, 4, max_size=args.replay_size)
    lqr = LQRController(K=K, g=dyn.g, z_des=2.0)

    difficulty = 0.0
    success_window = deque(maxlen=args.curriculum_window)
    completed_episodes = 0
    best_rank = None
    best_mu_sa = -1.0
    eval_index = 0
    last_update_metrics = {}

    state = MINE.sample_initial_state(sets=sets, regions=regions, s=difficulty, rng=rng)
    reset_dynamics_state(dyn, state)
    episode_steps = 0
    episode_success = episode_failure = episode_timeout = 0
    episode_uses_lqr = bool(args.use_warm_start and rng.random() < args.lqr_prob_start)
    update_metadata(paths["root"], status="training")

    try:
        for t in range(args.max_timesteps):
            episode_steps += 1

            if args.use_warm_start and episode_uses_lqr:
                # Whole-episode LQR behavior policy (warm start).
                base = lqr.act_norm(state)
                action_norm = np.clip(base + rng.normal(0.0, 0.05, size=4), -1.0, 1.0)
            elif t < args.start_timesteps:
                action_norm = rng.uniform(-1.0, 1.0, size=4)
            else:
                actor_action = policy.select_action(state)
                noise = np.clip(rng.normal(0.0, args.expl_noise, size=4),
                                -args.expl_noise_clip, args.expl_noise_clip)
                action_norm = np.clip(actor_action + noise, -1.0, 1.0)

            b_cur, _, c_cur = compute_bfc(sets, state)
            raw_actor_action = scale_action(action_norm, dyn.g)
            applied_action = lqr_action(state, K, dyn.g) if b_cur > 0.5 else raw_actor_action
            next_state = dyn.step(applied_action).copy()
            b_next, f_next, _ = compute_bfc(sets, next_state)

            replay_buffer.add(state, raw_actor_action, next_state, b_cur, c_cur,
                              goal_next=b_next, fail_next=f_next)

            success = b_next > 0.5
            failure = f_next > 0.5
            state = next_state

            if ((t + 1) >= args.update_after and replay_buffer.size >= args.batch_size
                    and (t + 1) % args.update_every == 0):
                last_update_metrics = policy.train(replay_buffer, args.batch_size)

            timeout = episode_steps >= args.max_episode_steps
            if success or failure or timeout:
                completed_episodes += 1
                success_window.append(1.0 if success else 0.0)
                episode_success += int(success)
                episode_failure += int(failure)
                episode_timeout += int(timeout and not success and not failure)
                append_csv(paths["logs"] / "training_episodes.csv", {
                    "episode": completed_episodes, "timestep": t + 1,
                    "episode_steps": episode_steps, "difficulty": difficulty,
                    "used_lqr": int(episode_uses_lqr),
                    "outcome": "success" if success else "failure" if failure else "timeout",
                })

                ready = (completed_episodes >= args.curriculum_min_episodes
                         and len(success_window) >= args.curriculum_window)
                if ready and np.mean(success_window) >= args.curriculum_threshold and difficulty < 1.0:
                    difficulty = min(1.0, difficulty + args.curriculum_increment)
                    success_window.clear()
                    print(f"Curriculum increased to s={difficulty:.3f}")

                if completed_episodes % 50 == 0:
                    print(f"T={t + 1} Episode={completed_episodes} EpisodeT={episode_steps} "
                          f"s={difficulty:.3f} success={episode_success} "
                          f"failure={episode_failure} timeout={episode_timeout}")

                state = MINE.sample_initial_state(sets=sets, regions=regions, s=difficulty, rng=rng)
                reset_dynamics_state(dyn, state)
                episode_steps = 0
                if args.use_warm_start:
                    frac = lqr_fraction(t + 1, args.lqr_prob_start, args.lqr_prob_end, args.lqr_anneal_steps)
                    episode_uses_lqr = bool(rng.random() < frac)

            if (t + 1) % args.eval_freq == 0:
                eval_index += 1
                summary, _, _ = evaluate_detailed(
                    policy, sets, val_sampler, evaluation_set=validation_set,
                    trajectories_per_region=0, max_episode_steps=args.max_episode_steps,
                    success_horizon_steps=args.success_horizon_steps,
                    integrator="euler", beta=args.discount,
                )
                mu_sa = summary["overall"]["weighted_mu_sa"]
                rank = evaluation_rank(summary)
                append_csv(paths["logs"] / "training_evaluations.csv", build_eval_row(t + 1, difficulty, summary))
                save_json(paths["logs"] / "latest_training_evaluation.json", summary)
                policy.save(checkpoint_prefix(paths["checkpoints"], "last"))
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_mu_sa = mu_sa
                    policy.save(checkpoint_prefix(paths["checkpoints"], "best"))
                    save_json(paths["logs"] / "best_evaluation.json", {
                        "eval_index": eval_index, "timestep": t + 1,
                        "training_difficulty": difficulty, "rank": list(rank), "summary": summary,
                    })
                    print(f"New best fixed-validation weighted mu_SA={mu_sa:.3f}")
                print(f"Evaluation {eval_index}: step={t + 1}, s={difficulty:.2f}, "
                      f"mu_SA={mu_sa:.3f}, total={summary['overall']['safe_arrival_rate']:.3f}, "
                      f"within_T={summary['overall']['safe_arrival_within_horizon_rate']:.3f}")
                plot_training_history(paths["logs"] / "training_evaluations.csv", paths["training_plots"])
                update_metadata(paths["root"], status="training", current_timestep=t + 1,
                                current_episode=completed_episodes, current_difficulty=difficulty,
                                best_mu_sa=best_mu_sa, latest_update_metrics=last_update_metrics,
                                fixed_validation_states=str(validation_path),
                                method="my_methods", obs_dim=args.obs_dim,
                                use_warm_start=bool(args.use_warm_start))

    except KeyboardInterrupt:
        print("Training interrupted; saving interrupted and last checkpoints.")
        policy.save(checkpoint_prefix(paths["checkpoints"], "interrupted"))
        policy.save(checkpoint_prefix(paths["checkpoints"], "last"))
        update_metadata(paths["root"], status="interrupted")
        return

    policy.save(checkpoint_prefix(paths["checkpoints"], "last"))
    best_path = Path(checkpoint_prefix(paths["checkpoints"], "best")).with_suffix(".pt")
    if not best_path.exists():
        policy.save(checkpoint_prefix(paths["checkpoints"], "best"))
    update_metadata(paths["root"], status="completed", final_timestep=args.max_timesteps,
                    final_episode=completed_episodes, best_mu_sa=best_mu_sa)
    print(f"Training completed. Outputs: {paths['root']}")


if __name__ == "__main__":
    main()
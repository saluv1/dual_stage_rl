import os
import sys
import argparse
import csv
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sac.sac import SAC
from sac.replay_buffer import ReplayBuffer
from sac.powerloop_env import PowerLoopEnv


def evaluate_policy(agent, seed, episodes=10, ref_mode="flat"):

    # Evaluation runs the FULL horizon with early termination effectively
    # disabled (thresholds set to infinity). Rationale: if eval terminated on
    # divergence, a policy that flies off at step 10 would report its mean error
    # over only those 10 near-reference steps and look deceptively good. Running
    # the whole loop gives a tracking metric that is comparable across
    # checkpoints and reflects true quality. We separately report how many steps
    # the policy stays within the training divergence bounds, as a robustness
    # signal.
    term_pos = 1.0
    term_att = 90.0

    env = PowerLoopEnv(
        seed=seed + 1000,
        term_pos_err=float("inf"),
        term_att_deg=180.0,
        ref_mode=ref_mode
    )
    term_att_err = 2.0 * np.sin(np.deg2rad(term_att) / 2.0)

    rewards = []
    pos_errors = []
    v_errors = []
    att_errors = []
    unsafe_counts = []
    survived_steps = []

    for _ in range(episodes):

        state = env.reset()
        done = False

        ep_reward = 0.0
        ep_pos = []
        ep_v = []
        ep_att = []
        ep_unsafe = 0
        ep_survived = 0
        still_tracking = True

        while not done:

            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            ep_pos.append(info["tracking_error_pos"])
            ep_v.append(info["tracking_error_v"])
            ep_att.append(info["tracking_error_att"])
            ep_unsafe += int(info["unsafe"] > 0.5)

            # Count steps before the policy first breaches the training bounds.
            if still_tracking:
                if (info["tracking_error_pos"] > term_pos
                        or info["tracking_error_att"] > term_att_err):
                    still_tracking = False
                else:
                    ep_survived += 1

            state = next_state

        rewards.append(ep_reward)
        pos_errors.append(np.mean(ep_pos))
        v_errors.append(np.mean(ep_v))
        att_errors.append(np.mean(ep_att))
        unsafe_counts.append(ep_unsafe)
        survived_steps.append(ep_survived)

    return {
        "eval_reward": float(np.mean(rewards)),
        "eval_pos_error": float(np.mean(pos_errors)),
        "eval_v_error": float(np.mean(v_errors)),
        "eval_att_error": float(np.mean(att_errors)),
        "eval_unsafe_steps": float(np.mean(unsafe_counts)),
        "eval_survived_steps": float(np.mean(survived_steps)),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=1500000)
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--replay_size", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--updates_per_env_step", type=float, default=1.0 / 8.0)
    parser.add_argument("--hidden_size", type=int, default=256)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=5e-5)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--alpha_lr", type=float, default=5e-5)
    parser.add_argument("--alpha_init", type=float, default=0.2)
    parser.add_argument("--alpha_min", type=float, default=0.01)
    parser.add_argument("--target_entropy", type=float, default=-4.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--q_clip", type=float, default=5e6)

    # Early-termination thresholds (fix C). Terminate an episode as a real
    # failure if position error exceeds term_pos_err (m) OR attitude error
    # exceeds term_att_deg (degrees). Tunable so you can loosen them if early
    # training terminates too aggressively.
    parser.add_argument("--term_pos_err", type=float, default=1.0)
    parser.add_argument("--term_att_deg", type=float, default=90.0)

    # If set, disable early termination entirely (thresholds -> infinity). Use
    # on FEASIBLE references, where errors stay bounded so the reward explosion
    # that termination was added to prevent does not occur. Removing termination
    # closes the "escape to value 0" door that lets a negative-reward agent
    # crash on purpose.
    parser.add_argument("--no_termination", action="store_true")

    # Survival bonus added to every step's reward. Makes a well-tracked step
    # net-positive so surviving is worth more than terminating (value 0),
    # removing the suicidal-agent incentive. ~8 is enough to cover the loop's
    # hardest (inversion) step; the near-hover circle needs only ~4.
    parser.add_argument("--survival_bonus", type=float, default=0.0)

    # Reference selector:
    #   "circular" = fixed-yaw feasible horizontal circle, easiest task
    #   "flat"     = feasible vertical flatness loop (full flip), harder
    #   "flip"     = infeasible circle+360-flip target, hardest
    parser.add_argument("--ref_mode", type=str, default="circular",
                        choices=["circular", "flat", "flip"])

    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=50000)

    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--result_dir", type=str, default="./results/sac_powerloop")
    parser.add_argument("--model_name", type=str, default="sac_powerloop_vanilla")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    env = PowerLoopEnv(
        seed=args.seed,
        term_pos_err=float("inf") if args.no_termination else args.term_pos_err,
        term_att_deg=180.0 if args.no_termination else args.term_att_deg,
        ref_mode=args.ref_mode,
        survival_bonus=args.survival_bonus
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dim=args.hidden_size,
        gamma=args.gamma,
        tau=args.tau,
        alpha_init=args.alpha_init,
        alpha_min=args.alpha_min,
        target_entropy=args.target_entropy,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        grad_clip=args.grad_clip,
        q_clip=args.q_clip
    )

    replay_buffer = ReplayBuffer(
        capacity=args.replay_size,
        seed=args.seed
    )

    log_path = os.path.join(args.result_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "total_step",
            "episode",
            "episode_step",
            "episode_reward",
            "episode_pos_error",
            "episode_v_error",
            "episode_att_error",
            "episode_unsafe_steps",
            "critic_loss",
            "policy_loss",
            "alpha_loss",
            "alpha",
            "q_mean",
            "log_pi_mean"
        ])

    eval_log_path = os.path.join(args.result_dir, "eval_log.csv")

    with open(eval_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "total_step",
            "eval_reward",
            "eval_pos_error",
            "eval_v_error",
            "eval_att_error",
            "eval_unsafe_steps",
            "eval_survived_steps"
        ])

    total_numsteps = 0
    updates = 0
    updates_accumulator = 0.0
    episode_num = 0

    last_update_info = {
        "critic_loss": np.nan,
        "policy_loss": np.nan,
        "alpha_loss": np.nan,
        "alpha": args.alpha_init,
        "q_mean": np.nan,
        "log_pi_mean": np.nan,
    }

    while total_numsteps < args.num_steps:

        state = env.reset()
        done = False

        episode_reward = 0.0
        episode_steps = 0
        episode_pos_errors = []
        episode_v_errors = []
        episode_att_errors = []
        episode_unsafe_steps = 0

        while not done and total_numsteps < args.num_steps:

            if total_numsteps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, info = env.step(action)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            episode_pos_errors.append(info["tracking_error_pos"])
            episode_v_errors.append(info["tracking_error_v"])
            episode_att_errors.append(info["tracking_error_att"])
            episode_unsafe_steps += int(info["unsafe"] > 0.5)

            # FIX #3: distinguish truncation from termination.
            #
            # This task has NO true terminal -- the episode ends at the 106-step
            # horizon purely because of a time limit, not because the world
            # ended. The vehicle is still flying and the loop continues. So the
            # value of the final state is its genuine continuation value, so we
            # keep bootstrapping through a time-limit truncation.
            #
            # Two distinct end conditions now exist (fix C):
            #   - divergence  -> terminated = True  -> mask 0 (a real failure;
            #                    the far-field state has no valid continuation
            #                    value we want to bootstrap toward)
            #   - horizon     -> truncated  = True  -> mask 1 (time limit only)
            # The env sets these explicitly in info; read them directly rather
            # than re-deriving, so the trainer stays correct if the env's end
            # conditions change again.
            terminated = bool(info.get("terminated", False))

            mask = 0.0 if terminated else 1.0

            replay_buffer.push(state, action, reward, next_state, mask)

            state = next_state

            if len(replay_buffer) >= args.batch_size and total_numsteps >= args.start_steps:

                updates_accumulator += args.updates_per_env_step

                while updates_accumulator >= 1.0:
                    last_update_info = agent.update_parameters(
                        replay_buffer,
                        args.batch_size
                    )
                    updates += 1
                    updates_accumulator -= 1.0

            if total_numsteps % args.eval_freq == 0:

                eval_info = evaluate_policy(
                    agent,
                    seed=args.seed,
                    episodes=args.eval_episodes,
                    ref_mode=args.ref_mode
                )

                print("---------------------------------------")
                print(f"Evaluation at step {total_numsteps}")
                print(f"Reward: {eval_info['eval_reward']:.2f}")
                print(f"Position error: {eval_info['eval_pos_error']:.3f}")
                print(f"Velocity error: {eval_info['eval_v_error']:.3f}")
                print(f"Attitude error: {eval_info['eval_att_error']:.3f}")
                print(f"Unsafe steps: {eval_info['eval_unsafe_steps']:.1f}")
                print(f"Survived steps (of {env.horizon}): {eval_info['eval_survived_steps']:.1f}")
                print("---------------------------------------")

                with open(eval_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        total_numsteps,
                        eval_info["eval_reward"],
                        eval_info["eval_pos_error"],
                        eval_info["eval_v_error"],
                        eval_info["eval_att_error"],
                        eval_info["eval_unsafe_steps"],
                        eval_info["eval_survived_steps"]
                    ])

            if total_numsteps % args.save_freq == 0:
                save_path = os.path.join(args.model_dir, args.model_name + ".pt")
                agent.save(save_path)
                print(f"Saved SAC checkpoint to {save_path}")

        episode_num += 1

        mean_pos = float(np.mean(episode_pos_errors)) if len(episode_pos_errors) > 0 else np.nan
        mean_v = float(np.mean(episode_v_errors)) if len(episode_v_errors) > 0 else np.nan
        mean_att = float(np.mean(episode_att_errors)) if len(episode_att_errors) > 0 else np.nan

        print(
            f"Episode: {episode_num} "
            f"Total steps: {total_numsteps} "
            f"Episode steps: {episode_steps} "
            f"Reward: {episode_reward:.2f} "
            f"Pos err: {mean_pos:.3f} "
            f"Vel err: {mean_v:.3f} "
            f"Att err: {mean_att:.3f} "
            f"Unsafe steps: {episode_unsafe_steps} "
            f"Alpha: {last_update_info['alpha']:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                total_numsteps,
                episode_num,
                episode_steps,
                episode_reward,
                mean_pos,
                mean_v,
                mean_att,
                episode_unsafe_steps,
                last_update_info["critic_loss"],
                last_update_info["policy_loss"],
                last_update_info["alpha_loss"],
                last_update_info["alpha"],
                last_update_info["q_mean"],
                last_update_info["log_pi_mean"]
            ])

    save_path = os.path.join(args.model_dir, args.model_name + ".pt")
    agent.save(save_path)
    print(f"Finished training. Saved SAC checkpoint to {save_path}")


if __name__ == "__main__":
    main()
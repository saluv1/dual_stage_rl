import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.dynamics import Dynamics
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from backup_policy.td3 import TD3
from backup_policy.replay_buffer import ReplayBuffer


def compute_reduced_state(full_state):

    z_des = 2.0

    pz = full_state[2]
    vx = full_state[3]
    vy = full_state[4]
    vz = full_state[5]

    qw = full_state[6]
    qx = full_state[7]
    qy = full_state[8]
    qz = full_state[9]

    # Quaternion sign correction
    if qw < 0:
        qx = -qx
        qy = -qy
        qz = -qz

    reduced_state = np.array([
        pz - z_des,
        vx,
        vy,
        vz,
        2.0 * qx,
        2.0 * qy,
        2.0 * qz
    ])

    return reduced_state


def compute_bfc(sets, full_state):

    reduced_state = compute_reduced_state(full_state)

    indicator = sets.compute_indicator(full_state, reduced_state)

    # indicator:
    #   0: Base B
    #   1: Continuation H
    #   2: Failure F
    if indicator == 0:
        b = 1.0
        f = 0.0
        c = 0.0
    elif indicator == 2:
        b = 0.0
        f = 1.0
        c = 0.0
    else:
        b = 0.0
        f = 0.0
        c = 1.0

    return b, f, c


def scale_action(action_norm, g):

    action_norm = np.array(action_norm, dtype=float)
    action_norm = np.clip(action_norm, -1.0, 1.0)

    # Actor outputs normalized action in [-1, 1]^4.
    # Convert to physical quadrotor input:
    #
    # a_cmd in [0, 4g]
    # omega_x, omega_y, omega_z in [-18, 18]
    a_cmd = 2.0 * g * (action_norm[0] + 1.0)
    wx = 18.0 * action_norm[1]
    wy = 18.0 * action_norm[2]
    wz = 18.0 * action_norm[3]

    action = np.array([a_cmd, wx, wy, wz])

    return action


def reset_dynamics_state(dyn, state):

    dyn.state = state.copy()

    if hasattr(dyn, "curr_step"):
        dyn.curr_step = 0

    if hasattr(dyn, "xlist"):
        dyn.xlist = []

    if hasattr(dyn, "vlist"):
        dyn.vlist = []

    if hasattr(dyn, "qlist"):
        dyn.qlist = []


def get_curriculum_ranges(curriculum_level):

    # Level 0: easiest
    if curriculum_level == 0:
        pz_range = (1.8, 2.4)
        v_range = 0.35
        q_range = 0.035

    # Level 1: slightly wider
    elif curriculum_level == 1:
        pz_range = (1.6, 2.6)
        v_range = 0.60
        q_range = 0.060

    # Level 2: close to your previous sampler
    elif curriculum_level == 2:
        pz_range = (1.4, 2.8)
        v_range = 1.00
        q_range = 0.100

    # Level 3: harder, includes more near-ceiling states
    else:
        pz_range = (1.2, 2.95)
        v_range = 1.40
        q_range = 0.140

    return pz_range, v_range, q_range


def sample_initial_state(sets, curriculum_level=0):

    max_tries = 1000

    pz_range, v_range, q_range = get_curriculum_ranges(curriculum_level)

    for _ in range(max_tries):

        px = 0.0
        py = 0.0

        pz = np.random.uniform(pz_range[0], pz_range[1])

        vx = np.random.uniform(-v_range, v_range)
        vy = np.random.uniform(-v_range, v_range)
        vz = np.random.uniform(-v_range, v_range)

        qw = 1.0
        qx = np.random.uniform(-q_range, q_range)
        qy = np.random.uniform(-q_range, q_range)
        qz = np.random.uniform(-q_range, q_range)

        q = np.array([qw, qx, qy, qz])
        q = q / np.linalg.norm(q)

        state = np.array([
            px, py, pz,
            vx, vy, vz,
            q[0], q[1], q[2], q[3]
        ])

        b, f, c = compute_bfc(sets, state)

        # We want initial states in continuation region:
        # safe but not already inside base set.
        if c == 1.0:
            return state

    return state


def eval_policy(policy, sets, curriculum_level, eval_episodes=20, max_episode_steps=300):

    eval_dyn = Dynamics()

    success_count = 0
    failure_count = 0
    timeout_count = 0
    steps_list = []
    min_hs_list = []

    for _ in range(eval_episodes):

        state = sample_initial_state(sets, curriculum_level)
        reset_dynamics_state(eval_dyn, state)

        min_hs = 1e9

        for step in range(max_episode_steps):

            action_norm = policy.select_action(np.array(state))
            action_norm = np.clip(action_norm, -1.0, 1.0)

            action = scale_action(action_norm, eval_dyn.g)
            next_state = eval_dyn.step(action).copy()

            b_next, f_next, c_next = compute_bfc(sets, next_state)

            min_hs = min(min_hs, sets.hs)

            state = next_state.copy()

            if b_next == 1.0:
                success_count += 1
                steps_list.append(step + 1)
                break

            if f_next == 1.0:
                failure_count += 1
                steps_list.append(step + 1)
                break

            if step == max_episode_steps - 1:
                timeout_count += 1
                steps_list.append(max_episode_steps)

        min_hs_list.append(min_hs)

    success_rate = success_count / eval_episodes
    failure_rate = failure_count / eval_episodes
    timeout_rate = timeout_count / eval_episodes
    avg_steps = np.mean(steps_list)
    avg_min_hs = np.mean(min_hs_list)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes")
    print(f"Curriculum level: {curriculum_level}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Failure rate: {failure_rate:.3f}")
    print(f"Timeout rate: {timeout_rate:.3f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average min h_S: {avg_min_hs:.3f}")
    print("---------------------------------------")

    return success_rate, failure_rate, timeout_rate, avg_steps, avg_min_hs


if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    dyn = Dynamics()

    gains = LQRGain(dt=dyn.del_t, g=dyn.g)
    K, P = gains.gain()

    sets = SetIndicator(
        P=P,
        c_b=8.0,
        zceil=3.0
    )

    state_dim = 10
    action_dim = 4

    # Actor outputs normalized actions in [-1, 1].
    max_action = 1.0

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=0.99,
        tau=0.0025,
        policy_noise=0.10,
        noise_clip=0.10,
        policy_freq=2,
        actor_lr=1e-4,
        critic_lr=3e-4
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    max_timesteps = 500000
    start_timesteps = 5000
    eval_freq = 5000
    batch_size = 128
    max_episode_steps = 300
    expl_noise = 0.10

    curriculum_level = 0
    max_curriculum_level = 3
    curriculum_success_threshold = 0.80
    min_evals_between_updates = 2
    evals_since_curriculum_update = 0

    evaluations = []

    state = sample_initial_state(sets, curriculum_level)
    reset_dynamics_state(dyn, state)

    episode_timesteps = 0
    episode_num = 0
    episode_success = 0
    episode_failure = 0
    episode_timeout = 0

    for t in range(max_timesteps):

        episode_timesteps += 1

        # Select normalized action
        if t < start_timesteps:
            action_norm = np.random.uniform(-1.0, 1.0, size=action_dim)
        else:
            action_norm = policy.select_action(np.array(state))
            action_norm = (
                action_norm + np.random.normal(0.0, expl_noise, size=action_dim)
            ).clip(-1.0, 1.0)

        # Convert normalized action to physical action
        action = scale_action(action_norm, dyn.g)

        # Step dynamics
        next_state = dyn.step(action).copy()

        # Compute safe-arrival indicators on next state
        b_next, f_next, c_next = compute_bfc(sets, next_state)

        # Store normalized action in replay buffer
        replay_buffer.add(state, action_norm, next_state, b_next, c_next)

        state = next_state.copy()

        # Train after collecting enough data
        if t >= start_timesteps and replay_buffer.size >= batch_size:

            # Paper-style update schedule:
            # one gradient update per 8 environment steps.
            if t % 8 == 0:
                policy.train(replay_buffer, batch_size)

        success = b_next == 1.0
        failure = f_next == 1.0
        timeout = episode_timesteps >= max_episode_steps

        done = success or failure or timeout

        if done:

            if success:
                episode_success += 1
            elif failure:
                episode_failure += 1
            else:
                episode_timeout += 1

            print(
                f"Total T: {t + 1} "
                f"Episode Num: {episode_num + 1} "
                f"Episode T: {episode_timesteps} "
                f"Curriculum: {curriculum_level} "
                f"Success: {episode_success} "
                f"Failure: {episode_failure} "
                f"Timeout: {episode_timeout}"
            )

            state = sample_initial_state(sets, curriculum_level)
            reset_dynamics_state(dyn, state)

            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % eval_freq == 0:

            eval_result = eval_policy(
                policy=policy,
                sets=sets,
                curriculum_level=curriculum_level,
                eval_episodes=50,
                max_episode_steps=max_episode_steps
            )

            success_rate = eval_result[0]
            evaluations.append((curriculum_level, *eval_result))
            np.save("./results/td3_safe_arrival_eval.npy", evaluations)

            policy.save("./models/td3_safe_arrival")

            evals_since_curriculum_update += 1

            if (
                success_rate >= curriculum_success_threshold
                and curriculum_level < max_curriculum_level
                and evals_since_curriculum_update >= min_evals_between_updates
            ):
                curriculum_level += 1
                evals_since_curriculum_update = 0

                print("=======================================")
                print(f"Curriculum increased to level {curriculum_level}")
                print("=======================================")
"""
Script for training and evaluating the agent.
"""

from collections import defaultdict

import jax
import numpy as np


def train(
    environment,
    eval_environment,
    agent,
    rng,
    min_buffer_capacity=50,
    number_updates=5,
    batch_size=10,
    nb_updated_transitions=2,
    exploratory_policy_steps=200,
    nb_training_steps=None,
    eval_frequency=10000,
    eval_episodes=5,
    verbose=True,
    verbose_frequency=100,
):
    """
    Perform the environment interaction and learning loop.

    Args:
        environment:
            Training dm_env environment.

        eval_environment:
            Separate dm_env environment used only for deterministic evaluation.

        agent:
            Agent providing initialize(), get_action(), update_fn(), and buffer.

        rng:
            JAX PRNG key.

        min_buffer_capacity:
            Minimum number of transitions required before gradient updates begin.

        number_updates:
            Number of gradient updates performed at each update point.

        batch_size:
            Replay-buffer minibatch size.

        nb_updated_transitions:
            Number of new environment transitions between update points.

            For example:
                nb_updated_transitions=8
                number_updates=1

            means one gradient update per eight environment transitions.

        exploratory_policy_steps:
            Number of initial steps using uniformly sampled actions.

            When this is zero, actions are sampled from the stochastic policy
            from the beginning.

        nb_training_steps:
            Total number of environment interactions.

        eval_frequency:
            Evaluate every this many environment steps.

        eval_episodes:
            Number of deterministic evaluation episodes.

        verbose:
            Whether to print training metrics.

        verbose_frequency:
            Number of environment steps between metric prints.

    Returns:
        eval_rewards:
            Mean evaluation returns.

        all_logs:
            Dictionary containing loss, gradient, and entropy histories.

        num_total_steps:
            Number of completed environment interactions.

        learner_state:
            Final learner state.
    """

    if nb_training_steps is None:
        raise ValueError("nb_training_steps must be specified.")

    if min_buffer_capacity <= 0:
        raise ValueError("min_buffer_capacity must be positive.")

    if nb_updated_transitions <= 0:
        raise ValueError("nb_updated_transitions must be positive.")

    if number_updates <= 0:
        raise ValueError("number_updates must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    all_logs = defaultdict(list)
    eval_rewards = []

    num_total_steps = 0
    nb_up_transitions = 0

    # ------------------------------------------------------------------
    # Initialize agent and training environment
    # ------------------------------------------------------------------
    learner_state = agent.initialize()

    # This variable is used only for the training environment.
    train_timestep = environment.reset()

    while num_total_steps < nb_training_steps:
        train_observation = train_timestep.observation

        # --------------------------------------------------------------
        # Select training action
        # --------------------------------------------------------------
        if num_total_steps < exploratory_policy_steps:
            action_spec = environment.action_spec()

            action = np.random.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
            ).astype(action_spec.dtype)

        else:
            rng, action_key = jax.random.split(rng, 2)

            action = agent.get_action(
                action_key,
                learner_state.params.policy,
                train_observation,
                deterministic=False,
            )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Optional CIL debugging
        # --------------------------------------------------------------
        if (
            getattr(agent, "_use_cil", False)
            and num_total_steps < 20
        ):
            import jax.numpy as jnp

            constraints = agent._constraint_provider(
                agent._cil_provider_params,
                jnp.asarray(
                    train_observation,
                    dtype=jnp.float32,
                ),
            )

            action_jax = jnp.asarray(
                action,
                dtype=jnp.float32,
            )

            violation = (
                constraints.A @ action_jax
                - constraints.b
            )

            print("====== CIL debug ======")
            print("step:", num_total_steps)
            print("action:", action)
            print(
                "A action - b:",
                np.asarray(violation),
            )
            print(
                "max violation:",
                float(jnp.max(violation)),
            )

            if action.shape[0] >= 1:
                print(
                    "thrust:",
                    float(action[0]),
                )

        # --------------------------------------------------------------
        # Step training environment
        # --------------------------------------------------------------
        next_train_timestep = environment.step(action)
        num_total_steps += 1

        # --------------------------------------------------------------
        # Correct terminal/truncation handling
        # --------------------------------------------------------------
        #
        # dm_env semantics used by the current QuadrotorEnv:
        #
        #   ground/ceiling termination:
        #       last=True, discount=0
        #
        #   time-limit truncation:
        #       last=True, discount=1
        #
        # Q bootstrapping must stop only for true physical termination.
        #
        timestep_discount = float(
            np.asarray(next_train_timestep.discount)
        )

        true_terminal = bool(
            timestep_discount == 0.0
        )

        # --------------------------------------------------------------
        # Store transition
        # --------------------------------------------------------------
        agent.buffer.store(
            np.asarray(
                train_observation,
                dtype=np.float32,
            ),
            action,
            float(next_train_timestep.reward),
            np.asarray(
                next_train_timestep.observation,
                dtype=np.float32,
            ),
            true_terminal,
        )

        nb_up_transitions += 1

        # --------------------------------------------------------------
        # Agent update
        # --------------------------------------------------------------
        buffer_ready = (
            agent.buffer.__len__()
            >= min_buffer_capacity
        )

        update_ready = (
            nb_up_transitions
            >= nb_updated_transitions
        )

        if buffer_ready and update_ready:
            nb_up_transitions = 0

            for _ in range(number_updates):
                transitions = agent.buffer.sample(
                    batch_size
                )

                rng, update_key = jax.random.split(
                    rng,
                    2,
                )

                learner_state, logs = agent.update_fn(
                    learner_state,
                    transitions,
                    update_key,
                )

                for metric_name, metric_value in logs.items():
                    # Keep the existing values usable by np.mean().
                    all_logs[metric_name].append(
                        np.asarray(metric_value)
                    )

        # --------------------------------------------------------------
        # Advance or reset training environment
        # --------------------------------------------------------------
        #
        # This uses last(), not true_terminal, because both timeout and
        # collision require a new environment episode.
        if next_train_timestep.last():
            train_timestep = environment.reset()
        else:
            train_timestep = next_train_timestep

        # --------------------------------------------------------------
        # Training metric output
        # --------------------------------------------------------------
        if (
            verbose
            and num_total_steps % verbose_frequency == 0
        ):
            if buffer_ready and len(all_logs) > 0:
                for metric_name, metric_history in all_logs.items():
                    if len(metric_history) == 0:
                        continue

                    recent_values = metric_history[
                        -verbose_frequency:
                    ]

                    mean_metric = float(
                        np.mean(recent_values)
                    )

                    print(
                        f"Mean value in last "
                        f"{verbose_frequency} logged updates "
                        f"for {metric_name}: "
                        f"{mean_metric}"
                    )
            else:
                print(
                    f"Filling buffer: "
                    f"{agent.buffer.__len__()}/"
                    f"{min_buffer_capacity}"
                )

            print(
                f"nb of steps: {num_total_steps}\n"
            )

        # --------------------------------------------------------------
        # Deterministic evaluation
        # --------------------------------------------------------------
        if (
            eval_frequency > 0
            and num_total_steps % eval_frequency == 0
        ):
            episode_returns = []

            for episode_idx in range(eval_episodes):
                # IMPORTANT:
                # Never assign this to train_timestep.
                eval_timestep = eval_environment.reset()

                episode_return = 0.0
                episode_length = 0

                episode_actions = []
                episode_states = [
                    np.asarray(
                        eval_timestep.observation,
                        dtype=np.float32,
                    )
                ]

                while not eval_timestep.last():
                    # The deterministic branch does not sample noise.
                    # A valid key is still supplied for consistent typing.
                    rng, eval_action_key = jax.random.split(
                        rng,
                        2,
                    )

                    eval_action = agent.get_action(
                        eval_action_key,
                        learner_state.params.policy,
                        eval_timestep.observation,
                        deterministic=True,
                    )

                    eval_action = np.asarray(
                        eval_action,
                        dtype=np.float32,
                    )

                    eval_timestep = (
                        eval_environment.step(
                            eval_action
                        )
                    )

                    episode_return += float(
                        eval_timestep.reward
                    )

                    episode_length += 1

                    episode_actions.append(
                        eval_action.copy()
                    )

                    episode_states.append(
                        np.asarray(
                            eval_timestep.observation,
                            dtype=np.float32,
                        )
                    )

                episode_returns.append(
                    episode_return
                )

                episode_states = np.asarray(
                    episode_states,
                    dtype=np.float32,
                )

                if len(episode_actions) > 0:
                    episode_actions = np.asarray(
                        episode_actions,
                        dtype=np.float32,
                    )

                    initial_action = (
                        episode_actions[0]
                    )

                    mean_action = np.mean(
                        episode_actions,
                        axis=0,
                    )

                    min_action = np.min(
                        episode_actions,
                        axis=0,
                    )

                    max_action = np.max(
                        episode_actions,
                        axis=0,
                    )
                else:
                    initial_action = None
                    mean_action = None
                    min_action = None
                    max_action = None

                mean_reward_per_step = (
                    episode_return
                    / max(episode_length, 1)
                )

                final_state = episode_states[-1]
                final_z = float(final_state[2])

                z_min = float(
                    np.min(episode_states[:, 2])
                )

                z_max = float(
                    np.max(episode_states[:, 2])
                )

                velocity_norms = np.linalg.norm(
                    episode_states[:, 3:6],
                    axis=1,
                )

                max_velocity_norm = float(
                    np.max(velocity_norms)
                )

                termination_reason = getattr(
                    eval_environment,
                    "_termination_reason",
                    None,
                )

                print(
                    f"Eval episode {episode_idx}: "
                    f"return={episode_return:.6f}, "
                    f"length={episode_length}, "
                    f"mean_reward="
                    f"{mean_reward_per_step:.6f}, "
                    f"final_z={final_z:.6f}, "
                    f"reason={termination_reason}"
                )

                print(
                    f"  z min/max: "
                    f"{z_min:.6f} / {z_max:.6f}"
                )

                print(
                    f"  max velocity norm: "
                    f"{max_velocity_norm:.6f}"
                )

                print(
                    f"  final state: "
                    f"{final_state}"
                )

                if initial_action is not None:
                    print(
                        f"  initial action: "
                        f"{initial_action}"
                    )

                    print(
                        f"  mean action: "
                        f"{mean_action}"
                    )

                    print(
                        f"  action min: "
                        f"{min_action}"
                    )

                    print(
                        f"  action max: "
                        f"{max_action}"
                    )

            mean_evaluation_return = float(
                np.mean(episode_returns)
            )

            eval_rewards.append(
                mean_evaluation_return
            )

            print(
                f"Evaluation after "
                f"{num_total_steps} steps: "
                f"{mean_evaluation_return}"
            )

            print(
                "All rewards:",
                episode_returns,
            )

    return (
        eval_rewards,
        all_logs,
        num_total_steps,
        learner_state,
    )
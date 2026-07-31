"""Training and evaluation loop for the existing dual_stage_rl SAC agent.

The function keeps its original public signature.  ``environment`` may now be
one dm_env environment or a sequence of environments.  The quadrotor vanilla
config passes 32 environments and counts total *transitions* exactly as the
PS2-RL reproduction code does.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence

import jax
import numpy as np


def _as_environment_list(environment):
    if isinstance(environment, Sequence) and not isinstance(
        environment, (str, bytes)
    ):
        environments = list(environment)
    else:
        environments = [environment]
    if not environments:
        raise ValueError("At least one training environment is required.")
    return environments


def _extract_physical_states(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.shape[-1] < 10:
        raise ValueError("Quadrotor observations must contain a 10-D state.")
    return observations[..., :10]


def _evaluate(
    *,
    eval_environment,
    agent,
    learner_state,
    rng,
    eval_episodes: int,
    step_label: int,
):
    episode_returns = []

    for episode_idx in range(eval_episodes):
        timestep = eval_environment.reset()
        episode_return = 0.0
        episode_actions = []
        episode_observations = [
            np.asarray(timestep.observation, dtype=np.float32)
        ]

        while not timestep.last():
            rng, action_key = jax.random.split(rng, 2)
            action = agent.get_action(
                action_key,
                learner_state.params.policy,
                timestep.observation,
                deterministic=True,
            )
            action = np.asarray(action, dtype=np.float32)
            timestep = eval_environment.step(action)
            episode_return += float(timestep.reward)
            episode_actions.append(action.copy())
            episode_observations.append(
                np.asarray(timestep.observation, dtype=np.float32)
            )

        observations = np.asarray(episode_observations, dtype=np.float32)
        states = _extract_physical_states(observations)
        actions = np.asarray(episode_actions, dtype=np.float32)
        length = int(actions.shape[0])
        episode_returns.append(episode_return)

        if length:
            initial_action = actions[0]
            mean_action = np.mean(actions, axis=0)
            min_action = np.min(actions, axis=0)
            max_action = np.max(actions, axis=0)
        else:
            initial_action = mean_action = min_action = max_action = None

        velocity_norms = np.linalg.norm(states[:, 3:6], axis=1)
        reason = getattr(
            eval_environment,
            "termination_reason",
            getattr(eval_environment, "_termination_reason", None),
        )
        print(
            f"Eval episode {episode_idx}: return={episode_return:.6f}, "
            f"length={length}, "
            f"mean_reward={episode_return / max(length, 1):.6f}, "
            f"final_z={float(states[-1, 2]):.6f}, reason={reason}"
        )
        print(
            "  z min/max: "
            f"{float(np.min(states[:, 2])):.6f} / "
            f"{float(np.max(states[:, 2])):.6f}"
        )
        print(
            "  max velocity norm: "
            f"{float(np.max(velocity_norms)):.6f}"
        )
        print(f"  final physical state: {states[-1]}")
        if initial_action is not None:
            print(f"  initial action: {initial_action}")
            print(f"  mean action: {mean_action}")
            print(f"  action min: {min_action}")
            print(f"  action max: {max_action}")

    mean_return = float(np.mean(episode_returns))
    print(f"Evaluation after {step_label} transitions: {mean_return}")
    print("All rewards:", episode_returns)
    return mean_return, rng


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
    initial_learner_state=None,
    numpy_seed=0,
):
    """Interact, update SAC, and periodically evaluate.

    ``nb_training_steps``, ``exploratory_policy_steps``, evaluation intervals,
    and update intervals are measured in transitions across all parallel
    environments, not vector-environment rounds.
    """
    if nb_training_steps is None:
        raise ValueError("nb_training_steps must be specified.")
    for name, value in (
        ("min_buffer_capacity", min_buffer_capacity),
        ("number_updates", number_updates),
        ("batch_size", batch_size),
        ("nb_updated_transitions", nb_updated_transitions),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")

    environments = _as_environment_list(environment)
    num_envs = len(environments)
    np_rng = np.random.default_rng(int(numpy_seed))

    learner_state = (
        agent.initialize()
        if initial_learner_state is None
        else initial_learner_state
    )
    timesteps = [env.reset() for env in environments]

    recent_logs = defaultdict(lambda: deque(maxlen=100))
    all_logs = defaultdict(list)
    eval_rewards = []
    num_total_steps = 0
    next_eval_step = int(eval_frequency) if eval_frequency > 0 else None
    next_verbose_step = (
        int(verbose_frequency) if verbose and verbose_frequency > 0 else None
    )

    while num_total_steps < int(nb_training_steps):
        active_count = min(
            num_envs,
            int(nb_training_steps) - num_total_steps,
        )
        active_envs = environments[:active_count]
        active_timesteps = timesteps[:active_count]
        observations = np.stack(
            [
                np.asarray(ts.observation, dtype=np.float32)
                for ts in active_timesteps
            ],
            axis=0,
        )

        # One batched actor call replaces 32 Python/JAX calls. The random
        # warm-up decision is made once per vector step, as in PS2-RL.
        rng, action_key = jax.random.split(rng, 2)
        policy_actions = np.asarray(
            agent.get_actions(
                action_key,
                learner_state.params.policy,
                observations,
                deterministic=False,
            ),
            dtype=np.float32,
        )

        action_spec = active_envs[0].action_spec()
        random_actions = np_rng.uniform(
            low=np.asarray(action_spec.minimum, dtype=np.float32),
            high=np.asarray(action_spec.maximum, dtype=np.float32),
            size=(active_count, int(action_spec.shape[0])),
        ).astype(np.float32)
        # Match PS2-RL's vector collector: the whole vector step is random
        # while the pre-step global transition count is below start_steps.
        actions = (
            random_actions
            if num_total_steps < int(exploratory_policy_steps)
            else policy_actions
        )

        next_timesteps = []
        rewards = np.empty((active_count,), dtype=np.float32)
        next_observations = np.empty_like(observations)
        dones = np.empty((active_count,), dtype=np.float32)

        for i, (env, timestep, action) in enumerate(
            zip(active_envs, active_timesteps, actions)
        ):
            next_timestep = env.step(action)
            rewards[i] = np.float32(next_timestep.reward)
            next_observations[i] = np.asarray(
                next_timestep.observation, dtype=np.float32
            )
            # PS2-RL marks the finite horizon as done.  dm_env termination
            # has discount=0, while a true time-limit truncation would have 1.
            dones[i] = np.float32(
                float(np.asarray(next_timestep.discount)) == 0.0
            )
            next_timesteps.append(next_timestep)

        agent.buffer.store_batch(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

        for i, (env, next_timestep) in enumerate(
            zip(active_envs, next_timesteps)
        ):
            timesteps[i] = env.reset() if next_timestep.last() else next_timestep

        previous_total_steps = num_total_steps
        num_total_steps += active_count

        # Count only update boundaries crossed after update_after.  This is
        # the vectorized equivalent of PS2-RL's integer-boundary formula and
        # avoids incorrectly replaying all intervals accumulated before the
        # learner was allowed to start.
        update_after = int(min_buffer_capacity)
        interval = int(nb_updated_transitions)
        lower = max(previous_total_steps + 1, update_after)
        if len(agent.buffer) >= int(batch_size) and num_total_steps >= lower:
            due_updates = (
                num_total_steps // interval
                - (lower - 1) // interval
            )
        else:
            due_updates = 0

        for _ in range(int(due_updates)):
            for _ in range(int(number_updates)):
                transitions = agent.buffer.sample(int(batch_size))
                rng, update_key = jax.random.split(rng, 2)
                learner_state, logs = agent.update_fn(
                    learner_state,
                    transitions,
                    update_key,
                )
                for metric_name, metric_value in logs.items():
                    value = np.asarray(metric_value)

                    # Store only a scalar diagnostic, never a batch/vector array.
                    if value.size == 0:
                        scalar_value = float("nan")
                    else:
                        scalar_value = float(np.mean(value))

                    recent_logs[metric_name].append(scalar_value)

        if next_verbose_step is not None and num_total_steps >= next_verbose_step:
            if recent_logs:
                for metric_name, history in recent_logs.items():
                    mean_value = float(np.mean(history))

                    print(
                        "Mean value in last 100 logged updates for "
                        f"{metric_name}: {mean_value}"
                    )

                    # Save one aggregated point per logging interval,
                    # instead of every learner update.
                    all_logs[metric_name].append(mean_value)
            else:
                print(
                    f"Filling buffer: {len(agent.buffer)}/"
                    f"{min_buffer_capacity}"
                )
            print(f"nb of transitions: {num_total_steps}\n")
            while next_verbose_step <= num_total_steps:
                next_verbose_step += int(verbose_frequency)

        if next_eval_step is not None and num_total_steps >= next_eval_step:
            mean_return, rng = _evaluate(
                eval_environment=eval_environment,
                agent=agent,
                learner_state=learner_state,
                rng=rng,
                eval_episodes=int(eval_episodes),
                step_label=num_total_steps,
            )
            eval_rewards.append(mean_return)
            while next_eval_step <= num_total_steps:
                next_eval_step += int(eval_frequency)

    return eval_rewards, all_logs, num_total_steps, learner_state

"""
Main file for passing the parameters and calling training.
"""

from absl import flags
import numpy as np
from ml_collections import config_flags
from src.train_agent import train
from src.agents.sac import SAC
from src.utils.training_utils import environments, env_names
import tensorflow as tf
from absl import app
import jax
import acme
import pickle
import os
import shutil

import jax.numpy as jnp
from src.envs.quadrotor.env import QuadrotorEnv
from src.envs.quadrotor.mujoco_playback import playback_trajectory
from src.cil.constraint_provider import (
    ConstantConstraintParams,
    constant_constraint_provider,
)
def rollout_trained_quadrotor_policy(
    agent,
    learner_state,
    seed: int = 0,
    horizon: int = 500,
):
    """
    Roll out the trained SAC policy in QuadrotorEnv and collect trajectory.

    This uses agent.get_action(..., deterministic=True), so if CIL is enabled,
    the action returned here is already the projected safe action.
    """
    env = QuadrotorEnv(
        for_evaluation=True,
        seed=seed,
        horizon=horizon,
    )

    timestep = env.reset()

    trajectory = []
    actions = []
    rewards = []

    trajectory.append(np.asarray(timestep.observation, dtype=np.float32))

    rng = jax.random.PRNGKey(seed + 12345)

    for _ in range(horizon):
        rng, key = jax.random.split(rng, 2)

        action = agent.get_action(
            key,
            learner_state.params.policy,
            timestep.observation,
            True,  # deterministic=True
        )

        action = np.asarray(action, dtype=np.float32)

        timestep = env.step(action)

        trajectory.append(np.asarray(timestep.observation, dtype=np.float32))
        actions.append(action)
        rewards.append(float(timestep.reward))

        if timestep.last():
            break

    trajectory = np.asarray(trajectory, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)

    print("Visualization rollout return:", float(np.sum(rewards)))
    print("Trajectory shape:", trajectory.shape)
    print("Actions shape:", actions.shape)

    return trajectory, actions, rewards
def make_thrust_band_constraints(
    g: float = 9.81,
    thrust_margin: float = 1.0,
) -> ConstantConstraintParams:
    """
    Mock CIL constraint for quadrotor:

        g - margin <= thrust <= g + margin

    action = [thrust, wx, wy, wz]

    Written as A u <= b:
        thrust <= g + margin
       -thrust <= -(g - margin)
    """
    A = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    b = jnp.array(
        [
            g + thrust_margin,
            -(g - thrust_margin),
        ],
        dtype=jnp.float32,
    )

    return ConstantConstraintParams(A=A, b=b)
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    "src/configs/pendulum.py",
    'File path to the default configuration file.',
    lock_config=True)
flags.DEFINE_string('save_pth', 'results', 'Path to folder where to save the model')
flags.DEFINE_string('experiment', 'experiment_0', 'Name of the experiment')
flags.DEFINE_integer('seed', 42, 'Seed for experiment')
flags.DEFINE_boolean('verbose', False, 'Verbose for showing losses, grads and entropy.')


def main(argv):
    print(f'Running SAC on {env_names[FLAGS.config.env_idx]}')
    print(f'Model will be saved in {FLAGS.save_pth}/{FLAGS.experiment}')

    # Create folder for saving model
    full_path = os.path.join(FLAGS.save_pth, FLAGS.experiment)
    
    if not os.path.exists(FLAGS.save_pth):
      os.mkdir(FLAGS.save_pth)

    if os.path.exists(full_path):
      answer = input("Saving the model will overwrite folder named {}. Continue (y/n)?".format(full_path))
      if answer.lower() in ["y", "yes"]:
        shutil.rmtree(full_path)
      else:
        print("Change experiment name please and repeat.")
        return
    
    os.mkdir(full_path)

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    config = FLAGS.config
    rng = jax.random.PRNGKey(FLAGS.seed)
    environment = environments[config.env_idx]

    if config.env_idx == 3:
        horizon = int(getattr(config, "horizon", 106))

        env = environment(
            for_evaluation=False,
            seed=FLAGS.seed,
            horizon=horizon,
        )

        eval_env = environment(
            for_evaluation=True,
            seed=FLAGS.seed + 1,
            horizon=horizon,
        )
    else:
        try:
            env = environment(
                for_evaluation=False,
                seed=FLAGS.seed,
            )
        except TypeError:
            env = environment(for_evaluation=False)

        try:
            eval_env = environment(
                for_evaluation=True,
                seed=FLAGS.seed + 1,
            )
        except TypeError:
            eval_env = environment(for_evaluation=True)

        if hasattr(env, "_env"):
            env._env.seed(seed=FLAGS.seed)

        if hasattr(eval_env, "_env"):
            eval_env._env.seed(seed=FLAGS.seed + 1)

    environment_spec = acme.make_environment_spec(env)

    rng, key = jax.random.split(rng, 2)

    # Mock CIL provider.
    # Quadrotor action dimension이 4일 때만 사용해야 합니다.
    use_cil = bool(getattr(config, "use_cil", False))

    if use_cil:
        thrust_margin = float(
            getattr(config, "cil_thrust_margin", 1.0)
        )

        cil_provider_params = make_thrust_band_constraints(
            g=9.81,
            thrust_margin=thrust_margin,
        )
        constraint_provider = constant_constraint_provider
    else:
        thrust_margin = None
        cil_provider_params = None
        constraint_provider = None
    print("====== Actual run configuration ======")
    print("environment:", env_names[config.env_idx])
    print("training horizon:", getattr(env, "_horizon", None))
    print("evaluation horizon:", getattr(eval_env, "_horizon", None))
    print("use_cil:", use_cil)
    print("cil_thrust_margin:", thrust_margin)
    print("scale_reward:", config.scale_reward)
    print("actor lr:", config.p_lr)
    print("critic lr:", config.q_lr)
    print("value lr:", config.v_lr)
    model = SAC(
        key,
        environment_spec,
        config,
        use_cil=use_cil,
        cil_provider_params=cil_provider_params,
        constraint_provider=constraint_provider,
    )

    # Call training of SAC agent
    eval_rewards, all_logs, num_total_steps, learner_state = train( environment = env,
                      eval_environment=eval_env,
                      agent = model,
                      rng = rng,
                      min_buffer_capacity=config.min_buffer_capacity,
                      number_updates=config.number_updates,
                      batch_size=config.batch_size,
                      nb_updated_transitions=config.nb_updated_transitions,
                      exploratory_policy_steps=config.exp_policy_steps,
                      nb_training_steps=config.num_total_steps,
                      verbose=FLAGS.verbose,
                      verbose_frequency=100,
                      eval_frequency=config.eval_frequency,
                      eval_episodes=config.eval_episodes,
                      )
    if getattr(config, "visualize_after_training", False):
        trajectory, actions, rewards = rollout_trained_quadrotor_policy(
            agent=model,
            learner_state=learner_state,
            seed=FLAGS.seed + 100,
            horizon=int(getattr(config, "horizon", 500)),
        )

    np.savez(
        "quadrotor_trained_rollout.npz",
        trajectory=trajectory,
        actions=actions,
        rewards=rewards,
    )

    playback_trajectory(
        trajectory,
        dt=0.02,
          realtime=True,
    )
    metrics = {
      'eval_rewards': eval_rewards,
      'all_logs': all_logs,
      'num_total_steps': num_total_steps,
    }

    model = {
      'config': config,
      'learner_state': learner_state,
    }

    mm = {
      'metrics': metrics,
      'model': model,
    }

    # Save model and metrics
    with open(os.path.join(full_path, FLAGS.experiment + "_mm.pickle"), 'wb') as f:
      pickle.dump(mm, f)

    print('done')

if __name__ == '__main__':
  app.run(main)
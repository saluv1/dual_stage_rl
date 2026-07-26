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
from src.utils.training_utils import (
    LearnerState,
    ParamState,
)
import jax.numpy as jnp
from src.envs.quadrotor.env import QuadrotorEnv
from src.envs.quadrotor.mujoco_playback import playback_trajectory
from src.cil.constraint_provider import (
    ConstantConstraintParams,
    constant_constraint_provider,
)


def assert_parameter_compatibility(
    fresh_params,
    saved_params,
) -> None:
    """
    Check that a checkpoint has the same parameter-tree structure
    and leaf shapes as the newly initialized model.
    """
    fresh_structure = jax.tree_util.tree_structure(
        fresh_params
    )
    saved_structure = jax.tree_util.tree_structure(
        saved_params
    )

    if fresh_structure != saved_structure:
        raise ValueError(
            "Warm-start checkpoint parameter structure does not "
            "match the current SAC model. Check observation dimension, "
            "action dimension, and network architecture."
        )

    fresh_leaves = jax.tree_util.tree_leaves(
        fresh_params
    )
    saved_leaves = jax.tree_util.tree_leaves(
        saved_params
    )

    for index, (fresh_leaf, saved_leaf) in enumerate(
        zip(fresh_leaves, saved_leaves)
    ):
        fresh_shape = np.shape(fresh_leaf)
        saved_shape = np.shape(saved_leaf)

        if fresh_shape != saved_shape:
            raise ValueError(
                "Warm-start checkpoint shape mismatch at parameter "
                f"leaf {index}: current={fresh_shape}, "
                f"checkpoint={saved_shape}. "
                "Vanilla and Phase II must use the same observation "
                "dimension and network architecture."
            )
def rollout_trained_quadrotor_policy(
    agent,
    learner_state,
    environment_cls,
    environment_kwargs=None,
    seed: int = 0,
):
    """
    Deterministically roll out a trained quadrotor policy.

    Parameters
    ----------
    agent:
        Trained SAC agent.

    learner_state:
        Final learner state containing the policy parameters.

    environment_cls:
        Environment class to instantiate.

        Examples:
            QuadrotorEnv
            QuadrotorPowerLoopEnv

    environment_kwargs:
        Keyword arguments passed to the environment constructor.

        Example for power-loop:
            {
                "horizon": 106,
                "include_progress": True,
                "initial_position_noise": 0.1,
                "perturb_evaluation": False,
                "terminate_on_ceiling": False,
                "use_cpp_body_rate_reference": False,
            }

    seed:
        Evaluation environment and action RNG seed.

    Returns
    -------
    trajectory:
        Physical quadrotor trajectory with shape (T + 1, 10).

        Even if the policy observation is 11-dimensional
        [physical_state, phase], only the physical 10-dimensional
        state is stored here.

    actions:
        Executed actions with shape (T, 4).

        When CIL is enabled, these are already projected safe actions.

    rewards:
        Per-step rewards with shape (T,).
    """
    if environment_kwargs is None:
        environment_kwargs = {}

    # Make a copy so the caller's dictionary is not modified.
    environment_kwargs = dict(environment_kwargs)

    # These are supplied explicitly below.
    environment_kwargs.pop("for_evaluation", None)
    environment_kwargs.pop("seed", None)

    env = environment_cls(
        for_evaluation=True,
        seed=seed,
        **environment_kwargs,
    )

    timestep = env.reset()

    trajectory = []
    actions = []
    rewards = []

    def extract_physical_state(observation):
        """
        Hover observation:
            [x] -> shape (10,)

        Power-loop observation:
            [x, normalized_phase] -> shape (11,)

        In both cases the first 10 entries are the physical state.
        """
        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).reshape(-1)

        if observation.shape[0] < 10:
            raise ValueError(
                "Quadrotor observation must contain at least "
                f"10 physical-state elements, got "
                f"shape={observation.shape}."
            )

        return observation[:10].copy()

    # Store the initial physical state.
    trajectory.append(
        extract_physical_state(
            timestep.observation
        )
    )

    rng = jax.random.PRNGKey(
        seed + 12345
    )

    # Use the environment horizon rather than a hard-coded value.
    max_steps = int(
        environment_kwargs.get(
            "horizon",
            getattr(env, "_horizon", 500),
        )
    )

    for _ in range(max_steps):
        rng, action_key = jax.random.split(
            rng,
            2,
        )

        # The policy receives the complete observation.
        #
        # Hover:
        #     shape (10,)
        #
        # Power-loop:
        #     shape (11,), including normalized phase.
        action = agent.get_action(
            action_key,
            learner_state.params.policy,
            timestep.observation,
            True,  # deterministic=True
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        timestep = env.step(action)

        # Store only the 10-dimensional physical state.
        trajectory.append(
            extract_physical_state(
                timestep.observation
            )
        )

        # The action returned by get_action is:
        #
        # vanilla:
        #     nominal deterministic SAC action
        #
        # Phase II:
        #     CIL-projected deterministic action
        actions.append(
            action.copy()
        )

        rewards.append(
            float(timestep.reward)
        )

        if timestep.last():
            break

    trajectory = np.asarray(
        trajectory,
        dtype=np.float32,
    )

    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    print(
        "Visualization rollout return:",
        float(np.sum(rewards)),
    )
    print(
        "Visualization rollout length:",
        len(actions),
    )
    print(
        "Trajectory shape:",
        trajectory.shape,
    )
    print(
        "Actions shape:",
        actions.shape,
    )
    print(
        "Termination reason:",
        getattr(
            env,
            "termination_reason",
            getattr(
                env,
                "_termination_reason",
                None,
            ),
        ),
    )

    return (
        trajectory,
        actions,
        rewards,
    )
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
    environment_kwargs = {}

    if config.env_idx == 3:
        environment_kwargs = {
            "horizon": int(
                getattr(config, "horizon", 200)
            ),
        }

        env = environment(
            for_evaluation=False,
            seed=FLAGS.seed,
            **environment_kwargs,
        )

        eval_env = environment(
            for_evaluation=True,
            seed=FLAGS.seed + 1,
            **environment_kwargs,
        )

    elif config.env_idx == 4:
        environment_kwargs = {
            "horizon": int(
                getattr(config, "horizon", 106)
            ),
            "include_progress": bool(
                getattr(config, "include_progress", True)
            ),
            "initial_position_noise": float(
                getattr(config, "initial_position_noise", 0.1)
            ),
            "perturb_evaluation": bool(
                getattr(config, "perturb_evaluation", False)
            ),
            "terminate_on_ceiling": bool(
                getattr(config, "terminate_on_ceiling", False)
            ),
            "use_cpp_body_rate_reference": bool(
                getattr(
                    config,
                    "use_cpp_body_rate_reference",
                    False,
                )
            ),
        }

        env = environment(
            for_evaluation=False,
            seed=FLAGS.seed,
            **environment_kwargs,
        )

        eval_env = environment(
            for_evaluation=True,
            seed=FLAGS.seed + 1,
            **environment_kwargs,
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
    fresh_learner_state = model.initialize()
    initial_learner_state = fresh_learner_state

    warm_start_value = getattr(
        config,
        "warm_start_path",
        "",
    )

    if warm_start_value is None:
        warm_start_path = ""
    else:
        warm_start_path = os.path.expanduser(
            str(warm_start_value).strip()
        )

    if warm_start_path:
        if not os.path.isfile(warm_start_path):
            raise FileNotFoundError(
                "Warm-start checkpoint not found: "
                f"{warm_start_path}"
            )

        print(
            "Loading warm-start checkpoint:",
            warm_start_path,
        )

        with open(warm_start_path, "rb") as file:
            checkpoint = pickle.load(file)

        if "model" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain a 'model' entry."
            )

        if "learner_state" not in checkpoint["model"]:
            raise KeyError(
                "Checkpoint does not contain "
                "checkpoint['model']['learner_state']."
            )

        saved_learner_state = (
            checkpoint["model"]["learner_state"]
        )

        # Check all actor, Q, V, and target-V parameter shapes.
        assert_parameter_compatibility(
            fresh_learner_state.params,
            saved_learner_state.params,
        )

        # Warm-start network parameters, but reset Adam states.
        initial_learner_state = LearnerState(
            params=ParamState(
                q1=saved_learner_state.params.q1,
                q2=saved_learner_state.params.q2,
                v=saved_learner_state.params.v,
                policy=saved_learner_state.params.policy,
                v_target=saved_learner_state.params.v_target,
            ),
            opt_state=fresh_learner_state.opt_state,
        )

        print(
            "Warm-started network parameters from:",
            warm_start_path,
        )
        print(
            "Optimizer states were freshly initialized."
        )
    else:
        print(
            "No warm-start checkpoint specified; "
            "training from fresh parameters."
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
                      initial_learner_state=initial_learner_state,
                      )


    # --------------------------------------------------------------
    # Save checkpoint immediately after training
    # --------------------------------------------------------------
    metrics_payload = {
        "eval_rewards": eval_rewards,
        "all_logs": all_logs,
        "num_total_steps": num_total_steps,
    }

    model_payload = {
        "config": config,
        "learner_state": learner_state,
    }

    checkpoint = {
        "metrics": metrics_payload,
        "model": model_payload,
    }

    checkpoint_path = os.path.join(
        full_path,
        FLAGS.experiment + "_mm.pickle",
    )

    with open(checkpoint_path, "wb") as file:
        pickle.dump(
            checkpoint,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        "Saved checkpoint:",
        checkpoint_path,
    )

    # --------------------------------------------------------------
    # Optional deterministic rollout and visualization
    # --------------------------------------------------------------
    if getattr(
        config,
        "visualize_after_training",
        False,
    ):
        if config.env_idx not in (3, 4):
            raise ValueError(
                "Quadrotor visualization is only "
                "supported for env_idx 3 or 4."
            )

        trajectory, actions, rewards = (
            rollout_trained_quadrotor_policy(
                agent=model,
                learner_state=learner_state,
                environment_cls=environment,
                environment_kwargs=environment_kwargs,
                seed=FLAGS.seed + 100,
            )
        )

        rollout_path = os.path.join(
            full_path,
            "quadrotor_trained_rollout.npz",
        )

        np.savez(
            rollout_path,
            trajectory=trajectory,
            actions=actions,
            rewards=rewards,
        )

        print(
            "Saved rollout:",
            rollout_path,
        )

        playback_trajectory(
            trajectory,
            dt=0.02,
            realtime=True,
        )

    print("done")
if __name__ == '__main__':
  app.run(main)
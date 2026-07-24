"""
Default hyperparameters for QuadrotorEnv.

This config is for testing PS2-SAC + CIL integration first.
It is intentionally small so we can verify rollout, replay buffer,
and CIL action filtering before running a long training job.
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Batch size
    config.batch_size = 128

    # Reward scaling
    config.scale_reward = 1.0

    # Learning rates
    config.lr = 5e-5
    config.p_lr = config.lr
    config.v_lr = 1e-4
    config.q_lr = 1e-4
    config.alpha = 0.02
    # Environment index
    # Make sure src/utils/training_utils.py maps 3 -> QuadrotorEnv.
    config.env_idx = 3
    config.policy_l2_coef = 0
    # Buffer settings
    config.min_buffer_capacity = 10000

    # For CIL debugging, set exploration to 0 so every action goes through agent.get_action().
    # Later, if you want random exploration, we should also CIL-filter random actions.
    config.exp_policy_steps = 0

    # Updates
    config.number_updates = 1
    config.nb_updated_transitions = 2

    # Total environment steps for initial debug
    config.num_total_steps = int(300_000)

    # Discount
    config.gamma = 0.99

    # Replay buffer
    config.replay_buffer_capacity = int(3e5)

    # Target network update
    config.tau = 0.005

    # Evaluation
    config.eval_frequency = 2500
    config.eval_episodes = 5

    #visualization
    config.visualize_after_training = True


    config.use_cil = True
    config.cil_thrust_margin = 2.0

    config.horizon=200
    return config
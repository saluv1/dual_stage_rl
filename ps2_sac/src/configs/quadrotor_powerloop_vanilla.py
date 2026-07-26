"""Vanilla SAC warm-start for power-loop tracking."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.env_idx = 4
    config.batch_size = 64
    config.scale_reward = 1.0

    config.p_lr = 5e-5
    config.q_lr = 1e-4
    config.v_lr = 1e-4

    # The paper uses learned temperature: alpha0=0.2, alpha_min=0.01,
    # target entropy=-4, and temperature LR=5e-5. The current legacy SAC
    # has fixed alpha, so this is only the initial/fallback value.
    config.alpha = 0.2
    config.auto_alpha = False
    config.alpha_min = 0.01
    config.target_entropy = -4.0
    config.alpha_lr = 5e-5

    config.policy_l2_coef = 0.0
    config.grad_clip_norm = 5.0
    config.q_clip = 5e6

    config.min_buffer_capacity = 10_000
    config.replay_buffer_capacity = int(3e5)
    config.exp_policy_steps = 0

    config.number_updates = 1
    config.nb_updated_transitions = 8
    config.num_total_steps = int(1_500_000)

    config.gamma = 0.99
    config.tau = 0.005

    config.eval_frequency = 25_000
    config.eval_episodes = 5

    config.horizon = 106
    config.include_progress = True
    config.initial_position_noise = 0.1
    config.perturb_evaluation = False
    config.use_cpp_body_rate_reference = False

    # The vanilla reference itself reaches z=3.5, so do not terminate at z=3.
    config.terminate_on_ceiling = False
    config.use_cil = False
    config.cil_mode = "none"

    config.visualize_after_training = True
    return config

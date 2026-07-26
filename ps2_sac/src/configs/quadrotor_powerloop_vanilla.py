"""Vanilla SAC for power-loop tracking with learned temperature."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    config.env_idx = 4
    config.horizon = 106
    config.include_progress = True
    config.initial_position_noise = 0.1
    config.perturb_evaluation = False
    config.use_cpp_body_rate_reference = False

    # The unconstrained reference reaches z=3.5 m.
    config.terminate_on_ceiling = False

    # ------------------------------------------------------------------
    # SAC networks and optimization
    # ------------------------------------------------------------------
    config.batch_size = 64
    config.scale_reward = 1.0

    config.p_lr = 5e-5
    config.q_lr = 1e-4
    config.v_lr = 1e-4

    config.gamma = 0.99
    config.tau = 0.005

    config.policy_l2_coef = 0.0
    config.grad_clip_norm = 5.0
    config.q_clip = 5e6

    # ------------------------------------------------------------------
    # Learned entropy temperature
    # ------------------------------------------------------------------
    config.alpha = 0.2
    config.auto_alpha = True
    config.alpha_min = 0.01
    config.target_entropy = -4.0
    config.alpha_lr = 5e-5

    # ------------------------------------------------------------------
    # Replay and update schedule
    # ------------------------------------------------------------------
    config.min_buffer_capacity = 10_000
    config.replay_buffer_capacity = int(3e5)
    config.exp_policy_steps = 0

    # One gradient update per eight environment transitions.
    config.number_updates = 1
    config.nb_updated_transitions = 8

    config.num_total_steps = int(1_500_000)

    # ------------------------------------------------------------------
    # Evaluation and saving
    # ------------------------------------------------------------------
    config.eval_frequency = 25_000
    config.eval_episodes = 5

    # Vanilla training starts from fresh parameters.
    config.warm_start_path = ""
    config.use_cil = False
    config.cil_mode = "none"

    # Keep disabled for a long/headless run. The checkpoint is still saved.
    config.visualize_after_training = False

    return config

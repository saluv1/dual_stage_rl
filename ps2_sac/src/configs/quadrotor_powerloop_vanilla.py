"""PS2-RL-aligned vanilla SAC configuration for power-loop tracking."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------------
    # Existing dual_stage_rl environment selection and task definition.
    # ------------------------------------------------------------------
    config.env_idx = 4
    config.dt = 0.02

    # 106 reference points plus the 0.1 s (5-step) vanilla terminal tail.
    # Set to 0/None only if the caller supports automatic horizon resolution.
    config.horizon = 111
    config.max_steps_extra_sec = 0.1
    config.include_time_features = True
    # Backward-compatible field consumed by old helper code.
    config.include_progress = True

    config.initial_position_noise = 0.1
    config.perturb_evaluation = True
    config.use_cpp_body_rate_reference = False

    # Vanilla tracking has no active safety termination. PS2-RL places the
    # upper constraint far above the 3.5 m loop for this stage.
    config.terminate_on_ceiling = False
    config.z_max = 15.0

    # Paper/reproduction reward member contained in the 32-config sweep.
    config.w_pos_xy = 2.5
    config.w_pos_z = 2.0
    config.w_vel = 4.0
    config.w_att = 16.0
    config.w_ref_omega_x = 0.10
    config.w_ref_omega_y = 0.20
    config.w_ref_omega_z = 0.05
    config.w_control_a = 0.01
    config.w_control_omega = 0.01

    # ------------------------------------------------------------------
    # SAC architecture and objective.
    # ------------------------------------------------------------------
    config.hidden_sizes = (256, 256)
    config.batch_size = 64

    # Official vanilla SAC uses raw rewards and physical actions in Q(s,a).
    config.scale_reward = 1.0
    config.normalize_q_actions = False

    # Actor/critic receive the complete 26-D PS2-RL observation.
    config.cil_state_dim = 10
    config.network_obs_start = 0
    config.network_observation_dim = 26

    config.p_lr = 1e-4
    config.q_lr = 5e-4
    # Retained only so older print/config code does not fail.
    config.v_lr = 5e-4
    config.alpha_lr = 1e-4

    config.gamma = 0.99
    config.tau = 0.005
    config.grad_clip_norm = 5.0
    config.q_clip = 5e6
    config.policy_l2_coef = 0.0
    config.log_std_min = -5.0
    config.log_std_max = 2.0

    config.auto_alpha = True
    config.alpha = 0.2
    config.alpha_min = 0.1
    config.target_entropy = -4.0

    # ------------------------------------------------------------------
    # PS2-RL vanilla data collection and update schedule.
    # Values are counted in transitions across all environments.
    # ------------------------------------------------------------------
    config.num_envs = 32
    config.replay_seed = 1
    config.replay_buffer_capacity = 300_000
    config.min_buffer_capacity = 2_000
    config.exp_policy_steps = 4_000

    config.number_updates = 1
    config.nb_updated_transitions = 8
    config.num_total_steps = 5_000_000

    # ------------------------------------------------------------------
    # Evaluation/checkpointing.
    # ------------------------------------------------------------------
    config.eval_frequency = 5_000
    config.eval_episodes = 10
    config.warm_start_path = ""
    config.use_cil = False
    config.cil_mode = "none"
    config.visualize_after_training = False

    return config

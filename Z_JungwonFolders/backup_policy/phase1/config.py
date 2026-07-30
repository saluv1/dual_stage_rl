"""Configuration values for Phase-I TD3 training."""
from dataclasses import dataclass

@dataclass
class Phase1Config:
    seed: int = 0
    model_name: str = "./models/td3_safe_arrival_v4"
    results_file: str = "./results/td3_safe_arrival_v4_eval.npy"
    max_timesteps: int = 5_000_000
    start_timesteps: int = 5_000
    eval_freq: int = 10_000
    batch_size: int = 128
    max_episode_steps: int = 300
    success_horizon_steps: int = 100
    eval_episodes: int = 150
    expl_noise: float = 0.10
    warm_start_transitions: int = 100_000
    initial_difficulty: float = 0.0
    difficulty_step: float = 0.10
    difficulty_backoff: float = 0.05
    mu_sa_threshold: float = 0.80
    mu_sa_backoff_threshold: float = 0.40
    curriculum_window: int = 3
    min_evals_between_updates: int = 3
    stall_patience: int = 12
    stall_min_mu_sa: float = 0.55

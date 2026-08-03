"""Canonical official-aligned defaults for the local PyTorch Phase-I trainer."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase1Config:
    seed: int = 2
    max_timesteps: int = 5_000_000
    start_timesteps: int = 5_000
    update_after: int = 2_000
    update_every: int = 8
    eval_freq: int = 5_000
    batch_size: int = 128
    replay_size: int = 400_000
    max_episode_steps: int = 100
    success_horizon_steps: int = 100
    eval_episodes_per_region: int = 64
    exploration_std: float = 0.10
    exploration_clip: float = 0.10
    warm_start_transitions: int = 0
    discount: float = 0.99
    tau: float = 0.0025
    policy_delay: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 5.0
    huber_delta: float = 1.0
    action_penalty: float = 0.05
    curriculum_increment: float = 0.10
    curriculum_threshold: float = 0.80
    curriculum_window: int = 50
    curriculum_min_episodes: int = 100

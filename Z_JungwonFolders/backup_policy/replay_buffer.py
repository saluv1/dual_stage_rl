"""Replay storage for real Phase-I transitions and physical raw actor actions."""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(4e5)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.b = np.zeros((self.max_size, 1), dtype=np.float32)
        self.c = np.zeros((self.max_size, 1), dtype=np.float32)
        self.goal_next = np.zeros((self.max_size, 1), dtype=np.float32)
        self.fail_next = np.zeros((self.max_size, 1), dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, b, c, goal_next, fail_next) -> None:
        vals = np.asarray([b, c, goal_next, fail_next], dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Non-finite transition flags: {vals}")
        if b > 0.5 and c > 0.5:
            raise ValueError(f"Current-state b and c are mutually exclusive: b={b}, c={c}")
        if goal_next > 0.5 and fail_next > 0.5:
            raise ValueError("A successor cannot be both goal and failure")

        i = self.ptr
        self.state[i] = state
        self.action[i] = action
        self.next_state[i] = next_state
        self.b[i, 0] = b
        self.c[i, 0] = c
        self.goal_next[i, 0] = goal_next
        self.fail_next[i, 0] = fail_next
        self.ptr = (i + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError(f"Replay has {self.size} samples, requested {batch_size}")
        ind = np.random.randint(0, self.size, size=batch_size)
        tensor = lambda array: torch.as_tensor(array[ind], dtype=torch.float32, device=self.device)
        return (
            tensor(self.state), tensor(self.action), tensor(self.next_state),
            tensor(self.b), tensor(self.c), tensor(self.goal_next), tensor(self.fail_next),
        )

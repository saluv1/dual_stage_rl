"""Uniform replay buffer used by the existing SAC agent.

The original buffer stored its full capacity as immutable JAX arrays and used
``array.at[index].set`` for every environment transition.  That is extremely
expensive for the 300k-transition buffer and 32 parallel environments used by
PS2-RL's quadrotor vanilla tracker.  Storage stays on the host as NumPy arrays;
only sampled minibatches are converted to JAX arrays by the jitted learner.
"""

from __future__ import annotations

import chex
import numpy as np

from src.utils.training_utils import Transitions


class ReplayBuffer:
    """Fixed-size circular uniform replay buffer."""

    def __init__(
        self,
        size_: int,
        featuredim_: int,
        actiondim_: int,
        seed: int = 0,
    ) -> None:
        if size_ <= 0:
            raise ValueError("Replay-buffer size must be positive.")
        if featuredim_ <= 0 or actiondim_ <= 0:
            raise ValueError("Feature and action dimensions must be positive.")

        self.__size = int(size_)
        self.__counter = 0
        self.__rng = np.random.default_rng(int(seed))

        self.__states = np.empty(
            (self.__size, int(featuredim_)), dtype=np.float32
        )
        self.__next_states = np.empty_like(self.__states)
        self.__actions = np.empty(
            (self.__size, int(actiondim_)), dtype=np.float32
        )
        self.__rewards = np.empty((self.__size,), dtype=np.float32)
        self.__dones = np.empty((self.__size,), dtype=np.float32)

    def store(
        self,
        state: chex.Array,
        action: chex.Array,
        reward: float,
        next_state: chex.Array,
        done: bool,
    ) -> None:
        """Store one transition."""
        index = self.__counter % self.__size
        self.__states[index] = np.asarray(state, dtype=np.float32)
        self.__actions[index] = np.asarray(action, dtype=np.float32)
        self.__rewards[index] = np.float32(reward)
        self.__next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.__dones[index] = np.float32(bool(done))
        self.__counter += 1

    def store_batch(
        self,
        states: chex.Array,
        actions: chex.Array,
        rewards: chex.Array,
        next_states: chex.Array,
        dones: chex.Array,
    ) -> None:
        """Store a batch while preserving circular-buffer semantics."""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)

        batch_size = int(states.shape[0])
        expected = (
            actions.shape[0],
            rewards.shape[0],
            next_states.shape[0],
            dones.shape[0],
        )
        if any(size != batch_size for size in expected):
            raise ValueError("All replay batch fields must have equal length.")
        if batch_size == 0:
            return

        # A batch larger than the capacity can only retain its newest suffix.
        if batch_size >= self.__size:
            states = states[-self.__size :]
            actions = actions[-self.__size :]
            rewards = rewards[-self.__size :]
            next_states = next_states[-self.__size :]
            dones = dones[-self.__size :]
            batch_size = self.__size

        start = self.__counter % self.__size
        first = min(batch_size, self.__size - start)
        second = batch_size - first

        end = start + first
        self.__states[start:end] = states[:first]
        self.__actions[start:end] = actions[:first]
        self.__rewards[start:end] = rewards[:first]
        self.__next_states[start:end] = next_states[:first]
        self.__dones[start:end] = dones[:first]

        if second:
            self.__states[:second] = states[first:]
            self.__actions[:second] = actions[first:]
            self.__rewards[:second] = rewards[first:]
            self.__next_states[:second] = next_states[first:]
            self.__dones[:second] = dones[first:]

        self.__counter += batch_size

    def sample(self, batch_size: int) -> Transitions:
        """Sample a minibatch uniformly with replacement."""
        memory = len(self)
        if memory == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        indices = self.__rng.integers(
            low=0,
            high=memory,
            size=int(batch_size),
            endpoint=False,
        )
        return Transitions(
            observations=self.__states[indices],
            actions=self.__actions[indices],
            rewards=self.__rewards[indices],
            next_observations=self.__next_states[indices],
            dones=self.__dones[indices],
        )

    def buffer_state(self) -> None:
        print("Buffer full" if len(self) == self.__size else "Buffer not full")

    def __len__(self) -> int:
        return min(self.__counter, self.__size)

import random
import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=int(3e5), seed=0):

        random.seed(seed)
        np.random.seed(seed)

        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array([mask], dtype=np.float32)
        )

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask = map(np.stack, zip(*batch))

        return state, action, reward, next_state, mask

    def __len__(self):

        return len(self.buffer)

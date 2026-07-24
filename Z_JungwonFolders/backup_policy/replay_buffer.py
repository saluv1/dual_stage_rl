import numpy as np
import torch


class ReplayBuffer(object):

    def __init__(self, state_dim, action_dim, max_size=int(4e5)):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))

        # Safe-arrival indicators.
        #
        # FIX #1: these are the indicators of the CURRENT state x, NOT of the
        # successor x'. PS2-RL Eq. (11) is
        #
        #     Q(x,u) = b(x) + beta * c(x) * Q(F(x,u), pi(F(x,u)))
        #
        # and both indicators are evaluated at x. The previous version stored
        # b(x'), c(x'), which shifted the learned value by one step and -- more
        # damagingly -- meant the critic never saw a transition whose current
        # state was in B or F, so the anchors Q|_B = 1 and Q|_F = 0 were never
        # supplied. train.py now stores those anchors explicitly.
        self.b = np.zeros((max_size, 1))   # b(x): base arrival indicator
        self.c = np.zeros((max_size, 1))   # c(x): continuation indicator

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, b, c):

        # b, f, c partition the state space, so b and c are mutually exclusive.
        # b == 1 and c == 1 simultaneously is impossible and indicates the
        # indicators were computed on the wrong state -- catch it immediately
        # rather than silently training on a corrupted target.
        assert not (b > 0.5 and c > 0.5), (
            f"b and c are mutually exclusive, got b={b}, c={c}. "
            "Indicators were likely computed on the wrong state."
        )

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.b[self.ptr] = b
        self.c[self.ptr] = c

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):

        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.b[ind]).to(self.device),
            torch.FloatTensor(self.c[ind]).to(self.device)
        )
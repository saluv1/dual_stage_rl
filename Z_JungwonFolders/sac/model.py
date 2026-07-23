import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


def weights_init_(m):

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.constant_(m.bias, 0.0)


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):

        super(QNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):

        xu = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(xu))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class GaussianPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, action_low, action_high):

        super(GaussianPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        action_low = torch.FloatTensor(action_low)
        action_high = torch.FloatTensor(action_high)

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

        self.apply(weights_init_)

    def forward(self, state):

        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):

        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action

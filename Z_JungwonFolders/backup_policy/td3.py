"""
TD3 for PS2-RL Phase I safe-arrival policy.

Based on:
https://github.com/sfujim/TD3/blob/master/TD3.py

FIX #5 (quaternion sign canonicalization):
    q and -q represent the SAME rotation, but the network was receiving the
    raw quaternion. ~50% of powerloop states have qw < 0, so two physically
    identical states could arrive as opposite input vectors. This is worst
    near the 180-degree flip (qw ~ 0), which is exactly where the near-ceiling
    states live -- the input became discontinuous across a boundary running
    straight through the hardest region.

    `safe_arrival_obs` / `safe_arrival_obs_torch` canonicalize the sign by
    negating ALL FOUR components when qw < 0.

    NOTE: this differs from `compute_reduced_state` in train.py, which flips
    only qx,qy,qz and leaves qw alone. That is correct THERE because it only
    uses the vector part to build 2*q_err. Do not copy that pattern here --
    negating only the vector part gives a DIFFERENT rotation.

FIX #4 (drop px, py from the observation):
    The safe-arrival problem is invariant to horizontal translation. h_B uses
    (pz - z_des, v, 2*q_err) and h_S uses pz only -- neither depends on px, py.
    Translate a state 5 m sideways and the correct recovery action is identical.

    Worse than merely useless, px/py LEAK THE CURRICULUM REGION: synthetic
    states draw px,py ~ U(-0.5, 0.5) while trace states inherit px in
    [-1.5, 1.5] from the powerloop circle. px therefore acts as a near-perfect
    region label, letting the network memorize per-region behavior instead of
    learning recovery -- which then breaks precisely when the mixture shifts.

    We keep the FULL quaternion rather than 2*q_err. Both are lossless once the
    sign is canonicalized (2*q_vec is injective on theta in [0,180] and qw is
    recoverable as sqrt(1-|q_vec|^2)), but the full quaternion is uniformly
    conditioned, whereas d(2 sin(theta/2))/d(theta) = cos(theta/2) compresses
    by ~2x in the 111-137 deg band where the near-ceiling states actually live.

Raw 10-D states are stored in the replay buffer; encoding happens at the
network boundary. That keeps the buffer format stable if the encoder changes.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------
# Output layout (8-D):
#   [pz - z_des, vx, vy, vz, qw, qx, qy, qz]
# with the quaternion sign canonicalized so that qw >= 0, and px, py dropped.

SAFE_ARRIVAL_OBS_DIM = 8
Z_DES = 2.0


def safe_arrival_obs(state, z_des=Z_DES):
    """
    Encode a raw 10-D state (or a batch) into the 8-D safe-arrival observation.

    Accepts shape (10,) or (N, 10); returns (8,) or (N, 8).

    The input array is never mutated: components are recombined via np.stack
    from freshly computed values rather than edited in place. This matters --
    in the training loop the array passed in is the live `state`, which also
    gets written into the replay buffer.
    """

    state = np.asarray(state, dtype=float)
    single = (state.ndim == 1)

    if single:
        state = state[None, :]

    pz = state[:, 2]
    vx = state[:, 3]
    vy = state[:, 4]
    vz = state[:, 5]

    qw = state[:, 6]
    qx = state[:, 7]
    qy = state[:, 8]
    qz = state[:, 9]

    # Negate ALL FOUR components when qw < 0.
    sign = np.where(qw < 0.0, -1.0, 1.0)

    obs = np.stack(
        [
            pz - z_des,
            vx,
            vy,
            vz,
            qw * sign,
            qx * sign,
            qy * sign,
            qz * sign,
        ],
        axis=1
    )

    if single:
        return obs[0]

    return obs


def safe_arrival_obs_torch(state, z_des=Z_DES):
    """
    Torch version of `safe_arrival_obs`, for (N, 10) batches inside training.

    Uses torch.where rather than a Python `if` because qw is a batch.
    """

    pz = state[:, 2]
    vx = state[:, 3]
    vy = state[:, 4]
    vz = state[:, 5]

    qw = state[:, 6]
    qx = state[:, 7]
    qy = state[:, 8]
    qz = state[:, 9]

    sign = torch.where(
        qw < 0.0,
        -torch.ones_like(qw),
        torch.ones_like(qw)
    )

    return torch.stack(
        [
            pz - z_des,
            vx,
            vy,
            vz,
            qw * sign,
            qx * sign,
            qy * sign,
            qz * sign,
        ],
        dim=1
    )


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):

        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)

        self.max_action = max_action

    def forward(self, state):

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        # Normalized action in [-max_action, max_action]
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 128)
        self.l5 = nn.Linear(128, 128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, state, action):

        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):

        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3(object):

    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.0025,
            policy_noise=0.10,
            noise_clip=0.10,
            policy_freq=2,
            actor_lr=1e-4,
            critic_lr=3e-4,
            obs_dim=SAFE_ARRIVAL_OBS_DIM
    ):

        # `state_dim` is the raw state width (10) and is kept in the signature
        # for compatibility, but the networks operate on the ENCODED
        # observation, which drops px, py -> obs_dim = 8.
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        """`state` is the raw 10-D full state; encoding happens here."""

        obs = safe_arrival_obs(np.asarray(state, dtype=float))
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)

        return self.actor(obs).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=128):

        self.total_it += 1

        # Sample replay buffer. `b` and `c` are now the indicators of the
        # CURRENT state x, per PS2-RL Eq. (11). See replay_buffer.py.
        state, action, next_state, b, c = replay_buffer.sample(batch_size)

        # FIX #5: encode ONCE here and reuse. `obs` is used by both the critic
        # loss and the actor loss; `next_obs` by both the target actor and the
        # target critic. Encoding inline at each use site invites missing one,
        # and a half-encoded critic is very hard to debug.
        obs = safe_arrival_obs_torch(state)
        next_obs = safe_arrival_obs_torch(next_state)

        with torch.no_grad():

            # Select action according to target policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_obs) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # PS2-RL safe-arrival target, Eq. (11):
            #   y = b(x) + beta * c(x) * min_i Q_i'(x', pi'(x'))
            #
            # FIX #1: this formula was ALWAYS correct here -- the bug was that
            # train.py fed it b(x'), c(x') instead of b(x), c(x). Now that the
            # buffer holds current-state indicators, this line is right as-is.
            #
            # With x in B  -> b=1, c=0 -> y = 1 exactly (no bootstrap).
            # With x in F  -> b=0, c=0 -> y = 0 exactly (no bootstrap).
            # These are the two ground-truth anchors that pin the value scale;
            # train.py now stores them explicitly as terminal self-loops.
            target_Q = b + c * self.discount * target_Q

            # Safe-arrival values are provably in [0, 1] (App. D.2). With the
            # anchors in place the targets already satisfy this; the clamp is
            # kept as a cheap guard and can be removed once the critic head
            # question (#2) is settled.
            target_Q = torch.clamp(target_Q, 0.0, 1.0)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # Huber critic loss
        critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss (uses the same encoded `obs` as above)
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update frozen target critic
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

            # Update frozen target actor
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    def save(self, filename):

        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
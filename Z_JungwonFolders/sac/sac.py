import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from sac.model import GaussianPolicy, QNetwork


def soft_update(target, source, tau):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SAC:

    def __init__(
            self,
            state_dim,
            action_dim,
            action_low,
            action_high,
            hidden_dim=256,
            gamma=0.99,
            tau=0.005,
            alpha_init=0.2,
            alpha_min=0.01,
            target_entropy=-4.0,
            actor_lr=5e-5,
            critic_lr=1e-4,
            alpha_lr=5e-5,
            grad_clip=5.0,
            q_clip=5e6,
            device=None
    ):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        self.gamma = gamma
        self.tau = tau
        self.alpha_min = alpha_min
        self.target_entropy = target_entropy
        self.grad_clip = grad_clip
        self.q_clip = q_clip

        self.critic = QNetwork(
            state_dim,
            action_dim,
            hidden_dim
        ).to(self.device)

        self.critic_target = QNetwork(
            state_dim,
            action_dim,
            hidden_dim
        ).to(self.device)

        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(
            state_dim,
            action_dim,
            hidden_dim,
            action_low,
            action_high
        ).to(self.device)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=actor_lr)

        self.log_alpha = torch.tensor(
            [torch.log(torch.tensor(alpha_init)).item()],
            requires_grad=True,
            device=self.device
        )

        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):

        return torch.clamp(self.log_alpha.exp(), min=self.alpha_min)

    def select_action(self, state, evaluate=False):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, replay_buffer, batch_size):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        with torch.no_grad():

            next_action, next_log_pi, _ = self.policy.sample(next_state_batch)

            q1_next, q2_next = self.critic_target(next_state_batch, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_pi
            min_q_next = torch.clamp(min_q_next, -self.q_clip, self.q_clip)

            target_q = reward_batch + mask_batch * self.gamma * min_q_next
            target_q = torch.clamp(target_q, -self.q_clip, self.q_clip)

        q1, q2 = self.critic(state_batch, action_batch)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha.detach() * log_pi - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_pi + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if self.log_alpha.exp().item() < self.alpha_min:
            with torch.no_grad():
                self.log_alpha.copy_(torch.log(torch.tensor([self.alpha_min], device=self.device)))

        soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": min_q_pi.mean().item(),
            "log_pi_mean": log_pi.mean().item()
        }

    def save(self, filename):

        directory = os.path.dirname(filename)
        if directory != "":
            os.makedirs(directory, exist_ok=True)

        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, filename)

    def load(self, filename, evaluate=False):

        checkpoint = torch.load(filename, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.log_alpha.requires_grad_(True)

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()

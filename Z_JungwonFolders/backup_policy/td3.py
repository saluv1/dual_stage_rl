"""Official-aligned TD3 backbone for PS2-RL Phase-I safe arrival.

The replay buffer stores real transitions only.  Terminal successor semantics
are imposed exactly from ``goal_next`` / ``fail_next`` flags:

    Q(x', a') = 1  on B,
    Q(x', a') = 0  on F,
    Q(x', a') = sigmoid(q_raw(x', a')) otherwise.

The online critic uses the same full-value wrapper.  The actor is optimized
only on continuation states, because the composed backup policy hands control
to the DLQR inside the base set.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_STATE_DIM = 10
OFFICIAL_OBS_DIM = 10
LEGACY_OBS_DIM = 8
SAFE_ARRIVAL_OBS_DIM = OFFICIAL_OBS_DIM
Z_DES = 2.0


def _normalize_quaternion_np(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-9, norm, 1.0)
    out = q / safe
    bad = (norm[..., 0] <= 1e-9)
    if np.any(bad):
        out = out.copy()
        out[bad] = np.array([1.0, 0.0, 0.0, 0.0])
    return out


def safe_arrival_obs(state: np.ndarray, obs_dim: int = OFFICIAL_OBS_DIM, z_des: float = Z_DES) -> np.ndarray:
    """Encode raw 10-D state(s) for a checkpoint.

    New training uses the official full 10-D state.  ``obs_dim=8`` remains
    supported only so previously trained checkpoints can still be evaluated.
    """
    arr = np.asarray(state, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.shape[-1] != RAW_STATE_DIM:
        raise ValueError(f"Expected raw state width 10, got {arr.shape}")

    if obs_dim == OFFICIAL_OBS_DIM:
        obs = arr.copy()
        obs[:, 6:10] = _normalize_quaternion_np(obs[:, 6:10])
    elif obs_dim == LEGACY_OBS_DIM:
        q = _normalize_quaternion_np(arr[:, 6:10])
        sign = np.where(q[:, :1] < 0.0, -1.0, 1.0)
        q = q * sign
        obs = np.concatenate(
            [arr[:, 2:3] - z_des, arr[:, 3:6], q], axis=1
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported safe-arrival observation width: {obs_dim}")

    return obs[0] if single else obs


def safe_arrival_obs_torch(state: torch.Tensor, obs_dim: int = OFFICIAL_OBS_DIM, z_des: float = Z_DES) -> torch.Tensor:
    if state.ndim != 2 or state.shape[1] != RAW_STATE_DIM:
        raise ValueError(f"Expected tensor shape (N, 10), got {tuple(state.shape)}")

    q = state[:, 6:10]
    qnorm = torch.linalg.vector_norm(q, dim=1, keepdim=True).clamp_min(1e-9)
    q = q / qnorm

    if obs_dim == OFFICIAL_OBS_DIM:
        return torch.cat([state[:, :6], q], dim=1)
    if obs_dim == LEGACY_OBS_DIM:
        sign = torch.where(q[:, :1] < 0.0, -torch.ones_like(q[:, :1]), torch.ones_like(q[:, :1]))
        return torch.cat([state[:, 2:3] - z_des, state[:, 3:6], q * sign], dim=1)
    raise ValueError(f"Unsupported safe-arrival observation width: {obs_dim}")


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = float(max_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 128)
        self.l5 = nn.Linear(128, 128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2


def infer_actor_obs_dim(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ("l1.weight", "actor.l1.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    raise KeyError("Could not infer actor observation width from l1.weight")


def _load_payload_or_actor(path_or_prefix: str | Path) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor], Path]:
    prefix = Path(path_or_prefix)
    compact = prefix if prefix.suffix == ".pt" else prefix.with_suffix(".pt")
    if compact.exists():
        payload = torch.load(compact, map_location=device)
        if not isinstance(payload, dict) or "actor" not in payload:
            raise ValueError(f"Compact checkpoint has no 'actor' state dict: {compact}")
        return payload, payload["actor"], compact

    actor_path = Path(str(prefix) + "_actor")
    if actor_path.exists():
        actor_state = torch.load(actor_path, map_location=device)
        return None, actor_state, actor_path
    raise FileNotFoundError(f"No checkpoint found for {prefix} (.pt or _actor)")


def infer_checkpoint_obs_dim(path_or_prefix: str | Path) -> int:
    payload, actor_state, _ = _load_payload_or_actor(path_or_prefix)
    if payload is not None and "obs_dim" in payload:
        return int(payload["obs_dim"])
    return infer_actor_obs_dim(actor_state)


class TD3:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        tau: float = 0.0025,
        policy_noise: float = 0.0,
        noise_clip: float = 0.0,
        policy_freq: int = 2,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        obs_dim: int = OFFICIAL_OBS_DIM,
        max_grad_norm: float = 5.0,
        huber_delta: float = 1.0,
        action_penalty: float = 0.05,
        gravity: float = 9.81,
        omega_max: float = 18.0,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.max_action = float(max_action)
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_freq = int(policy_freq)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.max_grad_norm = float(max_grad_norm)
        self.huber_delta = float(huber_delta)
        self.action_penalty = float(action_penalty)
        self.gravity = float(gravity)
        self.omega_max = float(omega_max)
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if self.omega_max <= 0.0:
            raise ValueError("omega_max must be positive")
        self.total_it = 0
        self._build_networks()

    def _build_networks(self) -> None:
        self.actor = Actor(self.obs_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(self.obs_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def scale_action_torch(self, action_norm: torch.Tensor) -> torch.Tensor:
        """Map the local [-1, 1] actor coordinates to official physical actions."""
        thrust = 2.0 * self.gravity * (action_norm[:, 0:1] + 1.0)
        omega = self.omega_max * action_norm[:, 1:4]
        return torch.cat([thrust, omega], dim=1)

    def normalized_physical_action(self, action_phys: torch.Tensor) -> torch.Tensor:
        """Return the action normalization used by the official actor penalty."""
        scale = action_phys.new_tensor(
            [4.0 * self.gravity, self.omega_max, self.omega_max, self.omega_max]
        )
        return action_phys / scale

    @staticmethod
    def continuation_value(raw_q: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw_q)

    @staticmethod
    def full_value_from_flags(q_cont: torch.Tensor, goal: torch.Tensor, fail: torch.Tensor) -> torch.Tensor:
        continuation = torch.clamp(1.0 - goal - fail, 0.0, 1.0)
        return goal + continuation * q_cont

    def select_action(self, state: np.ndarray) -> np.ndarray:
        obs_np = safe_arrival_obs(np.asarray(state), obs_dim=self.obs_dim)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).reshape(1, -1)
        with torch.no_grad():
            action = self.actor(obs)
        return action.cpu().numpy().reshape(-1)

    def train(self, replay_buffer, batch_size: int = 128) -> dict[str, float]:
        self.total_it += 1
        state, action, next_state, b, c, goal_next, fail_next = replay_buffer.sample(batch_size)
        obs = safe_arrival_obs_torch(state, obs_dim=self.obs_dim)
        next_obs = safe_arrival_obs_torch(next_state, obs_dim=self.obs_dim)

        with torch.no_grad():
            next_action_norm = self.actor_target(next_obs)
            if self.policy_noise > 0.0:
                # The official noise is specified in physical action-scale units.
                # Because thrust is represented locally by a0 in [-1, 1] with
                # a_cmd = 2 g (a0 + 1), its normalized noise is twice as large.
                std = next_action_norm.new_tensor(
                    [2.0 * self.policy_noise, self.policy_noise, self.policy_noise, self.policy_noise]
                )
                noise = torch.randn_like(next_action_norm) * std
                if self.noise_clip > 0.0:
                    clip = next_action_norm.new_tensor(
                        [2.0 * self.noise_clip, self.noise_clip, self.noise_clip, self.noise_clip]
                    )
                    noise = torch.maximum(torch.minimum(noise, clip), -clip)
                next_action_norm = next_action_norm + noise
            next_action_norm = next_action_norm.clamp(-self.max_action, self.max_action)
            next_action = self.scale_action_torch(next_action_norm)

            raw_t1, raw_t2 = self.critic_target(next_obs, next_action)
            next_q1 = self.full_value_from_flags(self.continuation_value(raw_t1), goal_next, fail_next)
            next_q2 = self.full_value_from_flags(self.continuation_value(raw_t2), goal_next, fail_next)
            next_q = torch.minimum(next_q1, next_q2)
            target_q = torch.clamp(b + self.discount * c * next_q, 0.0, 1.0)

        raw_q1, raw_q2 = self.critic(obs, action)
        # b and c are current-state indicators.  Failure states have b=c=0.
        pred_q1 = b + c * self.continuation_value(raw_q1)
        pred_q2 = b + c * self.continuation_value(raw_q2)

        critic_loss = F.huber_loss(pred_q1, target_q, delta=self.huber_delta) + F.huber_loss(
            pred_q2, target_q, delta=self.huber_delta
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        actor_grad_norm_value = 0.0
        actor_mask_mean = 0.0
        action_penalty_value = 0.0
        if self.total_it % self.policy_freq == 0:
            actor_action_norm = self.actor(obs)
            actor_action = self.scale_action_torch(actor_action_norm)
            raw_pi1, raw_pi2 = self.critic(obs, actor_action)
            q_pi1 = b + c * self.continuation_value(raw_pi1)
            q_pi2 = b + c * self.continuation_value(raw_pi2)
            q_pi = torch.minimum(q_pi1, q_pi2)

            mask = c
            denom = mask.sum().clamp_min(1.0)
            q_mean = (mask * q_pi).sum() / denom
            action_normalized_physical = self.normalized_physical_action(actor_action)
            action_cost = (
                mask * action_normalized_physical.square().mean(dim=1, keepdim=True)
            ).sum() / denom
            actor_loss = -q_mean + self.action_penalty * action_cost

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            for source, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.mul_(1.0 - self.tau).add_(self.tau * source.data)
            for source, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                target.data.mul_(1.0 - self.tau).add_(self.tau * source.data)

            actor_loss_value = float(actor_loss.detach().cpu())
            actor_grad_norm_value = float(torch.as_tensor(actor_grad_norm).detach().cpu())
            actor_mask_mean = float(mask.mean().detach().cpu())
            action_penalty_value = float(action_cost.detach().cpu())

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "critic_grad_norm": float(torch.as_tensor(critic_grad_norm).detach().cpu()),
            "actor_grad_norm": actor_grad_norm_value,
            "actor_mask_mean": actor_mask_mean,
            "action_penalty": action_penalty_value,
            "target_q_mean": float(target_q.mean().detach().cpu()),
        }

    def save(self, filename: str | Path) -> Path:
        path = Path(filename)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "obs_dim": self.obs_dim,
                "max_action": self.max_action,
                "discount": self.discount,
                "tau": self.tau,
                "policy_noise": self.policy_noise,
                "noise_clip": self.noise_clip,
                "policy_freq": self.policy_freq,
                "max_grad_norm": self.max_grad_norm,
                "huber_delta": self.huber_delta,
                "action_penalty": self.action_penalty,
                "gravity": self.gravity,
                "omega_max": self.omega_max,
            },
            path,
        )
        return path

    def load(self, filename: str | Path, load_optimizers: bool = True) -> Path:
        payload, actor_state, resolved = _load_payload_or_actor(filename)
        checkpoint_obs_dim = int(payload.get("obs_dim", infer_actor_obs_dim(actor_state))) if payload else infer_actor_obs_dim(actor_state)
        if checkpoint_obs_dim != self.obs_dim:
            self.obs_dim = checkpoint_obs_dim
            self._build_networks()

        self.actor.load_state_dict(actor_state)
        if payload is not None:
            if "critic" in payload:
                self.critic.load_state_dict(payload["critic"])
            if "actor_target" in payload:
                self.actor_target.load_state_dict(payload["actor_target"])
            else:
                self.actor_target.load_state_dict(self.actor.state_dict())
            if "critic_target" in payload:
                self.critic_target.load_state_dict(payload["critic_target"])
            else:
                self.critic_target.load_state_dict(self.critic.state_dict())
            if load_optimizers:
                if "actor_optimizer" in payload:
                    self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
                if "critic_optimizer" in payload:
                    self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.total_it = int(payload.get("total_it", 0))
            self.gravity = float(payload.get("gravity", self.gravity))
            self.omega_max = float(payload.get("omega_max", self.omega_max))
        else:
            prefix = Path(filename)
            critic_path = Path(str(prefix) + "_critic")
            if critic_path.exists():
                self.critic.load_state_dict(torch.load(critic_path, map_location=device))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        return resolved

    @classmethod
    def from_checkpoint(cls, filename: str | Path, state_dim: int = 10, action_dim: int = 4, max_action: float = 1.0) -> "TD3":
        obs_dim = infer_checkpoint_obs_dim(filename)
        model = cls(state_dim=state_dim, action_dim=action_dim, max_action=max_action, obs_dim=obs_dim)
        model.load(filename, load_optimizers=False)
        return model

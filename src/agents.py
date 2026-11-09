"""Reinforcement Learning Agents for Multi-Hub Replenishment.

Provides unified wrappers for:
- PPO (Proximal Policy Optimization for continuous multi-SKU order quantities)
- DQN (Deep Q-Network for discrete/binned replenishment decisions)

Architected with dual-backend support:
1. Native Stable-Baselines3 integration when installed
2. Standalone PyTorch Actor-Critic & Q-Network fallback (zero external dependencies)
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Try importing Stable-Baselines3
try:
    from stable_baselines3 import PPO as SB3_PPO
    from stable_baselines3 import DQN as SB3_DQN
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class BaseReplenishmentAgent:
    """Base class for replenishment agents."""

    def __init__(self, name: str = "BaseRLAgent"):
        self.name = name

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


# =====================================================================
# Standalone PyTorch Actor-Critic Architecture (PPO)
# =====================================================================

class ActorCriticNetwork(nn.Module):
    """Gaussian Actor-Critic network for continuous action space."""

    def __init__(self, obs_dim: int, action_dim: int, max_action: float = 30.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor head (mean actions)
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid(),  # Bound normalized actions in [0, 1]
        )
        # Trainable log standard deviation
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

        # Critic head (State-Value V(s))
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_net(obs)
        norm_mean = self.actor_mean(features)
        action_mean = norm_mean * self.max_action
        value = self.critic(features)
        return action_mean, value

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples an action, returns (action, log_prob, value)."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.shape[0], device=obs.device)
        else:
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.rsample()
            action = torch.clamp(action, 0.0, self.max_action)
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates batch of actions for PPO loss updates."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy


# =====================================================================
# Standalone PyTorch Deep Q-Network (DQN)
# =====================================================================

class QNetwork(nn.Module):
    """Q-Network for discrete replenishment action bins."""

    def __init__(self, obs_dim: int, num_bins: int = 5, n_units: int = 24):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_bins = num_bins
        self.n_units = n_units

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_units * num_bins),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Returns Q-values of shape [batch, n_units, num_bins]
        q_flat = self.net(obs)
        return q_flat.view(-1, self.n_units, self.num_bins)


# =====================================================================
# High-Level Agent Wrappers
# =====================================================================

class PPOAgent(BaseReplenishmentAgent):
    """Proximal Policy Optimization Replenishment Agent."""

    def __init__(
        self,
        obs_dim: int = 168,
        action_dim: int = 24,
        max_action: float = 30.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        use_sb3: bool = HAS_SB3,
        device: Optional[str] = None,
    ):
        super().__init__(name="PPO-Continuous-ActorCritic")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.use_sb3 = use_sb3 and HAS_SB3
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.sb3_model = None
        self.policy = ActorCriticNetwork(obs_dim, action_dim, max_action).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Generates replenishment action given observation."""
        if self.use_sb3 and self.sb3_model is not None:
            action, _ = self.sb3_model.predict(obs, deterministic=deterministic)
            return np.clip(action, 0.0, self.max_action).astype(np.float32)

        self.policy.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            action_mean, _ = self.policy(obs_t)
            if deterministic:
                action = action_mean.cpu().numpy()[0]
            else:
                std = torch.exp(self.policy.actor_log_std)
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample().clamp(0.0, self.max_action).cpu().numpy()[0]

        return np.clip(action, 0.0, self.max_action).astype(np.float32)

    def save(self, path: str) -> None:
        """Saves model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if self.use_sb3 and self.sb3_model is not None:
            self.sb3_model.save(path)
        else:
            torch.save(
                {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "max_action": self.max_action,
                    "state_dict": self.policy.state_dict(),
                },
                path,
            )

    def load(self, path: str) -> None:
        """Loads model checkpoint."""
        if self.use_sb3 and HAS_SB3:
            try:
                self.sb3_model = SB3_PPO.load(path)
                return
            except Exception:
                pass

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.policy.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            self.policy.load_state_dict(checkpoint)


class DQNAgent(BaseReplenishmentAgent):
    """Deep Q-Network Agent with Discretized Case-Pack Actions."""

    def __init__(
        self,
        obs_dim: int = 168,
        action_dim: int = 24,
        max_action: float = 30.0,
        num_bins: int = 5,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: Optional[str] = None,
    ):
        super().__init__(name="DQN-Discretized-Policy")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_bins = num_bins
        self.bin_values = np.linspace(0.0, max_action, num_bins, dtype=np.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(obs_dim, num_bins, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.q_net.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            q_vals = self.q_net(obs_t)  # [1, action_dim, num_bins]
            best_bin_indices = torch.argmax(q_vals, dim=-1).cpu().numpy()[0]
            actions = self.bin_values[best_bin_indices]
        return actions.astype(np.float32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "num_bins": self.num_bins,
                "state_dict": self.q_net.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.q_net.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            self.q_net.load_state_dict(checkpoint)


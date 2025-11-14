"""Soft Actor-Critic (SAC) implementation.

This module implements the SAC algorithm for continuous control.
Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018
"""

import os
from typing import List, Optional, Tuple, Dict, Any
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

from rl_control.utils.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """Actor network for SAC (Gaussian policy).
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        log_std_min: Minimum log standard deviation
        log_std_max: Maximum log standard deviation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2
    ) -> None:
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        self.mean_linear = nn.Linear(prev_dim, action_dim)
        self.log_std_linear = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (mean, log_std)
        """
        x = self.shared_net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for inference.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
            
        Returns:
            Action tensor
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t)


class Critic(nn.Module):
    """Critic network for SAC (twin Q-functions).
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ) -> None:
        super(Critic, self).__init__()
        
        # Q1 network
        layers_q1 = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers_q1.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers_q1.append(nn.Linear(prev_dim, 1))
        self.q1 = nn.Sequential(*layers_q1)
        
        # Q2 network
        layers_q2 = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers_q2.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers_q2.append(nn.Linear(prev_dim, 1))
        self.q2 = nn.Sequential(*layers_q2)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (q1_value, q2_value)
        """
        x = torch.cat([state, action], dim=1)
        
        q1 = self.q1(x)
        q2 = self.q2(x)
        
        return q1, q2


class SAC:
    """Soft Actor-Critic agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        lr: Learning rate
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Entropy temperature (if None, uses automatic tuning)
        buffer_size: Replay buffer size
        batch_size: Batch size for training
        device: Torch device
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: Optional[float] = None,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        
        # Create networks
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        if alpha is None:
            self.auto_entropy = True
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.auto_entropy = False
            self.alpha = torch.tensor(alpha, device=self.device)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        
        # Training statistics
        self.training_stats: Dict[str, List[float]] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "alpha": []
        }
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from policy.
        
        Args:
            state: Current state
            deterministic: If True, return deterministic action
            
        Returns:
            Action to take
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy()[0]
    
    def update(self) -> Dict[str, float]:
        """Update actor and critic networks.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            target_q = rewards + (1 - dones) * self.gamma * (q_next - self.alpha * next_log_probs)
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_entropy else 0.0,
            "alpha": self.alpha.item()
        }
    
    def train(
        self,
        env: Any,
        total_timesteps: int,
        warmup_steps: int = 1000,
        updates_per_step: int = 1,
        log_interval: int = 1000
    ) -> None:
        """Train the SAC agent.
        
        Args:
            env: Gym environment
            total_timesteps: Total number of timesteps to train
            warmup_steps: Number of random exploration steps
            updates_per_step: Number of gradient updates per environment step
            log_interval: Logging interval
        """
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        episode_reward = 0
        episode_length = 0
        
        pbar = tqdm(range(total_timesteps), desc="Training SAC")
        
        for step in pbar:
            # Select action
            if step < warmup_steps:
                action = env.action_space.sample()
            else:
                action = self.get_action(state, deterministic=False)
            
            # Step environment
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            # Store transition
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            # Episode ended
            if done:
                self.training_stats["episode_rewards"].append(episode_reward)
                self.training_stats["episode_lengths"].append(episode_length)
                
                state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
                episode_reward = 0
                episode_length = 0
            
            # Update networks
            if step >= warmup_steps:
                for _ in range(updates_per_step):
                    update_metrics = self.update()
                    
                    if update_metrics:
                        for key, value in update_metrics.items():
                            self.training_stats[key].append(value)
            
            # Update progress bar
            if step % log_interval == 0 and len(self.training_stats["episode_rewards"]) > 0:
                mean_reward = np.mean(self.training_stats["episode_rewards"][-10:])
                pbar.set_postfix({"mean_reward": f"{mean_reward:.2f}"})
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy else None,
            "training_stats": self.training_stats
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if self.auto_entropy and checkpoint["log_alpha"] is not None:
            self.log_alpha.data = checkpoint["log_alpha"].data
            self.alpha = self.log_alpha.exp()
        self.training_stats = checkpoint["training_stats"]

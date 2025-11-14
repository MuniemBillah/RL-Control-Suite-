"""Proximal Policy Optimization (PPO) implementation.

This module implements the PPO algorithm with clipped surrogate objective.
Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from tqdm import tqdm


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        continuous: Whether actions are continuous
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        continuous: bool = False
    ) -> None:
        super(ActorCritic, self).__init__()
        
        self.continuous = continuous
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(prev_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(prev_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action_distribution, value)
        """
        features = self.feature_extractor(state)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_log_std)
            dist = Normal(mean, std)
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
        
        value = self.critic(features)
        
        return dist, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: Current state
            deterministic: If True, return mean action (for continuous) or argmax (for discrete)
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        dist, value = self.forward(state)
        
        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        if not self.continuous:
            log_prob = log_prob.unsqueeze(-1)
        
        return action, log_prob, value


class PPO:
    """Proximal Policy Optimization agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        lr: Learning rate
        gamma: Discount factor
        lam: GAE lambda parameter
        clip_epsilon: PPO clipping parameter
        value_coef: Value function loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        continuous: Whether actions are continuous
        device: Torch device
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        continuous: bool = False,
        device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create actor-critic network
        self.policy = ActorCritic(
            state_dim, action_dim, hidden_dims, continuous
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Training statistics
        self.training_stats: Dict[str, List[float]] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": []
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
            action, _, _ = self.policy.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy()[0]
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = next_value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Update policy using PPO objective.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of old log probabilities
            advantages: Batch of advantages
            returns: Batch of returns
            epochs: Number of update epochs
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of training metrics
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        metrics = {"policy_loss": [], "value_loss": [], "entropy": []}
        
        for _ in range(epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get current policy outputs
                dist, values = self.policy(states_tensor[batch_indices])
                
                # Calculate new log probabilities
                new_log_probs = dist.log_prob(actions_tensor[batch_indices])
                if not self.policy.continuous:
                    new_log_probs = new_log_probs.unsqueeze(-1)
                
                # Calculate ratio and clipped ratio
                ratio = torch.exp(new_log_probs - old_log_probs_tensor[batch_indices])
                
                # Calculate surrogate losses
                surr1 = ratio * advantages_tensor[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor[batch_indices]
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, returns_tensor[batch_indices].unsqueeze(-1))
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def train(
        self,
        env: Any,
        total_timesteps: int,
        steps_per_update: int = 2048,
        log_interval: int = 10
    ) -> None:
        """Train the PPO agent.
        
        Args:
            env: Gym environment
            total_timesteps: Total number of timesteps to train
            steps_per_update: Number of steps before each update
            log_interval: Logging interval
        """
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        episode_reward = 0
        episode_length = 0
        
        states_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        rewards_buffer = []
        dones_buffer = []
        values_buffer = []
        
        pbar = tqdm(range(total_timesteps), desc="Training PPO")
        
        for step in pbar:
            # Collect trajectory
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_tensor)
            
            action_np = action.cpu().numpy()[0]
            
            # Step environment
            result = env.step(action_np)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            # Store transition
            states_buffer.append(state)
            actions_buffer.append(action_np)
            log_probs_buffer.append(log_prob.cpu().numpy()[0])
            rewards_buffer.append(reward)
            dones_buffer.append(done)
            values_buffer.append(value.cpu().numpy()[0][0])
            
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
            
            # Update policy
            if (step + 1) % steps_per_update == 0:
                # Get next value for GAE
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    _, next_value = self.policy(next_state_tensor)
                    next_value = next_value.cpu().numpy()[0][0]
                
                # Compute advantages and returns
                advantages, returns = self.compute_gae(
                    rewards_buffer, values_buffer, dones_buffer, next_value
                )
                
                # Update policy
                update_metrics = self.update(
                    np.array(states_buffer),
                    np.array(actions_buffer),
                    np.array(log_probs_buffer),
                    advantages,
                    returns
                )
                
                # Track metrics
                for key, value in update_metrics.items():
                    self.training_stats[key].append(value)
                
                # Clear buffers
                states_buffer = []
                actions_buffer = []
                log_probs_buffer = []
                rewards_buffer = []
                dones_buffer = []
                values_buffer = []
                
                # Update progress bar
                if len(self.training_stats["episode_rewards"]) > 0:
                    mean_reward = np.mean(self.training_stats["episode_rewards"][-10:])
                    pbar.set_postfix({"mean_reward": f"{mean_reward:.2f}"})
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]

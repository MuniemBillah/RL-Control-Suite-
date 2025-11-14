"""Replay buffer for off-policy RL algorithms."""

from typing import Tuple
import numpy as np


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms.
    
    Args:
        capacity: Maximum buffer size
        state_dim: Dimension of state space
        action_dim: Dimension of action space
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Preallocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer.
    
    Args:
        capacity: Maximum buffer size
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        alpha: Prioritization exponent
        beta: Importance sampling exponent
        beta_increment: Beta increment per sampling
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ) -> None:
        super().__init__(capacity, state_dim, action_dim)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Priority tree (simplified - using array)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add transition with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        super().add(state, action, reward, next_state, done)
        
        # Set priority to maximum for new transitions
        self.priorities[self.ptr - 1] = self.max_priority
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions
            priorities: New priorities (e.g., TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class EpisodeBuffer:
    """Buffer for storing complete episodes.
    
    Args:
        max_episodes: Maximum number of episodes to store
    """
    
    def __init__(self, max_episodes: int = 1000) -> None:
        self.max_episodes = max_episodes
        self.episodes = []
    
    def add_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray
    ) -> None:
        """Add complete episode.
        
        Args:
            states: Array of states in episode
            actions: Array of actions in episode
            rewards: Array of rewards in episode
        """
        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'return': np.sum(rewards)
        }
        
        self.episodes.append(episode)
        
        # Remove oldest if over capacity
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
    
    def sample_episodes(self, batch_size: int) -> list:
        """Sample random episodes.
        
        Args:
            batch_size: Number of episodes to sample
            
        Returns:
            List of episodes
        """
        indices = np.random.choice(len(self.episodes), size=min(batch_size, len(self.episodes)), replace=False)
        return [self.episodes[i] for i in indices]
    
    def get_best_episodes(self, n: int = 10) -> list:
        """Get episodes with highest returns.
        
        Args:
            n: Number of episodes to return
            
        Returns:
            List of best episodes
        """
        sorted_episodes = sorted(self.episodes, key=lambda x: x['return'], reverse=True)
        return sorted_episodes[:n]
    
    def __len__(self) -> int:
        """Return number of stored episodes."""
        return len(self.episodes)

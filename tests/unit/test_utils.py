"""Unit tests for utils module."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from rl_control.utils import ReplayBuffer, PrioritizedReplayBuffer, EpisodeBuffer


class TestReplayBuffer:
    """Test suite for ReplayBuffer."""
    
    def test_replay_buffer_initialization(self):
        """Test replay buffer initialization."""
        buffer = ReplayBuffer(capacity=1000, state_dim=4, action_dim=2)
        
        assert buffer.capacity == 1000
        assert buffer.state_dim == 4
        assert buffer.action_dim == 2
        assert len(buffer) == 0
    
    def test_replay_buffer_add(self):
        """Test adding transitions to buffer."""
        buffer = ReplayBuffer(capacity=10, state_dim=4, action_dim=2)
        
        state = np.random.randn(4)
        action = np.random.randn(2)
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
    
    def test_replay_buffer_capacity(self):
        """Test buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10, state_dim=4, action_dim=2)
        
        # Add more than capacity
        for _ in range(15):
            state = np.random.randn(4)
            action = np.random.randn(2)
            buffer.add(state, action, 1.0, state, False)
        
        assert len(buffer) == 10  # Should not exceed capacity
    
    def test_replay_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100, state_dim=4, action_dim=2)
        
        # Add some transitions
        for _ in range(50):
            state = np.random.randn(4)
            action = np.random.randn(2)
            buffer.add(state, action, 1.0, state, False)
        
        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape == (batch_size, 4)
        assert actions.shape == (batch_size, 2)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 4)
        assert dones.shape == (batch_size,)
    
    def test_replay_buffer_clear(self):
        """Test clearing buffer."""
        buffer = ReplayBuffer(capacity=100, state_dim=4, action_dim=2)
        
        # Add transitions
        for _ in range(10):
            state = np.random.randn(4)
            action = np.random.randn(2)
            buffer.add(state, action, 1.0, state, False)
        
        assert len(buffer) == 10
        
        buffer.clear()
        
        assert len(buffer) == 0


class TestPrioritizedReplayBuffer:
    """Test suite for PrioritizedReplayBuffer."""
    
    def test_prioritized_buffer_initialization(self):
        """Test prioritized buffer initialization."""
        buffer = PrioritizedReplayBuffer(
            capacity=1000,
            state_dim=4,
            action_dim=2,
            alpha=0.6,
            beta=0.4
        )
        
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert len(buffer) == 0
    
    def test_prioritized_buffer_add(self):
        """Test adding to prioritized buffer."""
        buffer = PrioritizedReplayBuffer(
            capacity=10,
            state_dim=4,
            action_dim=2
        )
        
        state = np.random.randn(4)
        action = np.random.randn(2)
        
        buffer.add(state, action, 1.0, state, False)
        
        assert len(buffer) == 1
        assert buffer.priorities[0] == buffer.max_priority
    
    def test_prioritized_buffer_sample(self):
        """Test sampling from prioritized buffer."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            state_dim=4,
            action_dim=2
        )
        
        # Add transitions
        for _ in range(50):
            state = np.random.randn(4)
            action = np.random.randn(2)
            buffer.add(state, action, 1.0, state, False)
        
        # Sample batch
        batch_size = 32
        result = buffer.sample(batch_size)
        
        states, actions, rewards, next_states, dones, indices, weights = result
        
        assert states.shape == (batch_size, 4)
        assert indices.shape == (batch_size,)
        assert weights.shape == (batch_size,)
    
    def test_prioritized_buffer_update_priorities(self):
        """Test updating priorities."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            state_dim=4,
            action_dim=2
        )
        
        # Add transitions
        for _ in range(10):
            state = np.random.randn(4)
            action = np.random.randn(2)
            buffer.add(state, action, 1.0, state, False)
        
        # Update priorities
        indices = np.array([0, 1, 2])
        new_priorities = np.array([0.5, 1.0, 0.3])
        
        buffer.update_priorities(indices, new_priorities)
        
        assert buffer.priorities[0] == 0.5
        assert buffer.priorities[1] == 1.0
        assert buffer.priorities[2] == 0.3


class TestEpisodeBuffer:
    """Test suite for EpisodeBuffer."""
    
    def test_episode_buffer_initialization(self):
        """Test episode buffer initialization."""
        buffer = EpisodeBuffer(max_episodes=100)
        
        assert buffer.max_episodes == 100
        assert len(buffer) == 0
    
    def test_episode_buffer_add_episode(self):
        """Test adding episode."""
        buffer = EpisodeBuffer(max_episodes=10)
        
        states = np.random.randn(50, 4)
        actions = np.random.randn(50, 2)
        rewards = np.random.randn(50)
        
        buffer.add_episode(states, actions, rewards)
        
        assert len(buffer) == 1
    
    def test_episode_buffer_capacity(self):
        """Test buffer respects episode capacity."""
        buffer = EpisodeBuffer(max_episodes=5)
        
        # Add more than capacity
        for _ in range(10):
            states = np.random.randn(10, 4)
            actions = np.random.randn(10, 2)
            rewards = np.random.randn(10)
            buffer.add_episode(states, actions, rewards)
        
        assert len(buffer) == 5
    
    def test_episode_buffer_sample_episodes(self):
        """Test sampling episodes."""
        buffer = EpisodeBuffer(max_episodes=20)
        
        # Add episodes
        for _ in range(10):
            states = np.random.randn(10, 4)
            actions = np.random.randn(10, 2)
            rewards = np.random.randn(10)
            buffer.add_episode(states, actions, rewards)
        
        # Sample episodes
        sampled = buffer.sample_episodes(batch_size=5)
        
        assert len(sampled) == 5
        assert 'states' in sampled[0]
        assert 'actions' in sampled[0]
        assert 'rewards' in sampled[0]
        assert 'return' in sampled[0]
    
    def test_episode_buffer_get_best_episodes(self):
        """Test getting best episodes."""
        buffer = EpisodeBuffer(max_episodes=20)
        
        # Add episodes with known returns
        returns = [1.0, 5.0, 3.0, 10.0, 2.0]
        for ret in returns:
            states = np.random.randn(10, 4)
            actions = np.random.randn(10, 2)
            rewards = np.ones(10) * (ret / 10)
            buffer.add_episode(states, actions, rewards)
        
        # Get best 3 episodes
        best = buffer.get_best_episodes(n=3)
        
        assert len(best) == 3
        assert best[0]['return'] >= best[1]['return']
        assert best[1]['return'] >= best[2]['return']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

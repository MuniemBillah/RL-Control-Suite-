"""Unit tests for PPO algorithm."""

import pytest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from rl_control.algorithms import PPO


class TestPPO:
    """Test suite for PPO algorithm."""
    
    def test_ppo_initialization(self):
        """Test PPO agent initialization."""
        agent = PPO(
            state_dim=4,
            action_dim=2,
            hidden_dims=[64, 64],
            lr=3e-4,
            continuous=False
        )
        
        assert agent.policy is not None
        assert agent.optimizer is not None
        assert agent.gamma == 0.99
        assert agent.clip_epsilon == 0.2
    
    def test_ppo_forward_pass(self):
        """Test forward pass through PPO policy."""
        agent = PPO(state_dim=4, action_dim=2, continuous=False)
        
        state = torch.randn(1, 4)
        dist, value = agent.policy(state)
        
        assert dist is not None
        assert value.shape == (1, 1)
    
    def test_ppo_get_action(self):
        """Test action selection."""
        agent = PPO(state_dim=4, action_dim=2, continuous=False)
        
        state = np.random.randn(4)
        action = agent.get_action(state, deterministic=False)
        
        assert action in [0, 1]
    
    def test_ppo_continuous_actions(self):
        """Test PPO with continuous actions."""
        agent = PPO(state_dim=4, action_dim=2, continuous=True)
        
        state = np.random.randn(4)
        action = agent.get_action(state)
        
        assert action.shape == (2,)
        assert isinstance(action, np.ndarray)
    
    def test_ppo_gae_computation(self):
        """Test Generalized Advantage Estimation."""
        agent = PPO(state_dim=4, action_dim=2)
        
        rewards = [1.0, 0.5, 0.0, 1.0]
        values = [0.5, 0.3, 0.2, 0.8]
        dones = [False, False, False, True]
        next_value = 0.0
        
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        
        assert advantages.shape == (4,)
        assert returns.shape == (4,)
        assert np.abs(advantages.mean()) < 1e-6  # Should be normalized
    
    def test_ppo_save_load(self, tmp_path):
        """Test model saving and loading."""
        agent = PPO(state_dim=4, action_dim=2)
        
        # Get initial action
        state = np.random.randn(4)
        action_before = agent.get_action(state, deterministic=True)
        
        # Save model
        save_path = tmp_path / "ppo_test.pt"
        agent.save(str(save_path))
        
        # Create new agent and load
        agent_loaded = PPO(state_dim=4, action_dim=2)
        agent_loaded.load(str(save_path))
        
        # Compare actions
        action_after = agent_loaded.get_action(state, deterministic=True)
        assert action_before == action_after


class TestPPOIntegration:
    """Integration tests for PPO."""
    
    def test_ppo_training_episode(self):
        """Test PPO can complete a training episode."""
        # Create simple environment
        class MockEnv:
            def __init__(self):
                self.observation_space = type('obj', (object,), {'shape': (4,)})()
                self.action_space = type('obj', (object,), {'n': 2})()
            
            def reset(self):
                return np.zeros(4), {}
            
            def step(self, action):
                return np.zeros(4), 1.0, True, False, {}
        
        env = MockEnv()
        agent = PPO(state_dim=4, action_dim=2, continuous=False)
        
        # Run one update
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        for _ in range(10):
            state = np.random.randn(4)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = agent.policy.get_action(state_tensor)
            
            states.append(state)
            actions.append(action.cpu().numpy()[0])
            log_probs.append(log_prob.cpu().numpy()[0][0])
            rewards.append(1.0)
            dones.append(False)
            values.append(value.cpu().numpy()[0][0])
        
        # Compute GAE
        advantages, returns = agent.compute_gae(rewards, values, dones, 0.0)
        
        # Perform update
        metrics = agent.update(
            np.array(states),
            np.array(actions),
            np.array(log_probs),
            advantages,
            returns,
            epochs=1,
            batch_size=5
        )
        
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

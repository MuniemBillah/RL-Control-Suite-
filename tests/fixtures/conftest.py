"""Common test fixtures and utilities."""

import pytest
import numpy as np
import torch


@pytest.fixture
def simple_env():
    """Create a simple mock environment for testing."""
    class MockEnv:
        def __init__(self):
            self.observation_space = type('obj', (object,), {'shape': (4,)})()
            self.action_space = type('obj', (object,), {
                'n': 2,
                'shape': (1,),
                'low': np.array([-1.0]),
                'high': np.array([1.0])
            })()
            self.state = None
        
        def reset(self):
            self.state = np.zeros(4)
            return self.state, {}
        
        def step(self, action):
            self.state += np.random.randn(4) * 0.1
            reward = np.random.randn()
            done = np.random.rand() < 0.1
            return self.state, reward, done, False, {}
    
    return MockEnv()


@pytest.fixture
def sample_states():
    """Generate sample states for testing."""
    return np.random.randn(10, 4).astype(np.float32)


@pytest.fixture
def sample_actions():
    """Generate sample actions for testing."""
    return np.random.randn(10, 2).astype(np.float32)


@pytest.fixture
def replay_buffer_data():
    """Generate sample data for replay buffer."""
    batch_size = 100
    state_dim = 4
    action_dim = 2
    
    return {
        'states': np.random.randn(batch_size, state_dim).astype(np.float32),
        'actions': np.random.randn(batch_size, action_dim).astype(np.float32),
        'rewards': np.random.randn(batch_size).astype(np.float32),
        'next_states': np.random.randn(batch_size, state_dim).astype(np.float32),
        'dones': np.random.randint(0, 2, batch_size).astype(np.float32)
    }


@pytest.fixture
def seed_all():
    """Seed all random number generators for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

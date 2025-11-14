"""Unit tests for safety module (Control Barrier Functions)."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from rl_control.safety import (
    ControlBarrierFunction,
    create_position_limit_cbf,
    create_velocity_limit_cbf,
    MultiCBF,
)


class TestControlBarrierFunction:
    """Test suite for Control Barrier Function."""
    
    def test_cbf_initialization(self):
        """Test CBF initialization."""
        def barrier_fn(state):
            return 1.0 - state[0] ** 2
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=1.0)
        
        assert cbf.barrier_fn is not None
        assert cbf.alpha == 1.0
    
    def test_cbf_is_safe(self):
        """Test safety checking."""
        def barrier_fn(state):
            return 1.0 - state[0] ** 2
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=1.0)
        
        # Safe state
        safe_state = np.array([0.5, 0.0])
        assert cbf.is_safe(safe_state)
        
        # Unsafe state
        unsafe_state = np.array([1.5, 0.0])
        assert not cbf.is_safe(unsafe_state)
    
    def test_cbf_safety_margin(self):
        """Test safety margin computation."""
        def barrier_fn(state):
            return 1.0 - state[0] ** 2
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=1.0)
        
        state = np.array([0.5, 0.0])
        margin = cbf.safety_margin(state)
        
        expected = 1.0 - 0.25
        assert np.abs(margin - expected) < 1e-6
    
    def test_cbf_numerical_gradient(self):
        """Test numerical gradient computation."""
        def barrier_fn(state):
            return 1.0 - state[0] ** 2
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=1.0)
        
        state = np.array([0.5, 0.0])
        gradient = cbf._numerical_gradient(state)
        
        # Gradient should be [-2*x, 0] = [-1.0, 0]
        assert gradient.shape == (2,)
        assert np.abs(gradient[0] - (-1.0)) < 1e-4
        assert np.abs(gradient[1]) < 1e-4
    
    def test_cbf_filter_safe_action(self):
        """Test that safe actions pass through unchanged."""
        def barrier_fn(state):
            return 1.0 - np.sum(state ** 2)
        
        def dynamics_fn(state, action):
            return action
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=0.5)
        
        state = np.array([0.1, 0.1])
        action = np.array([0.0, 0.0])  # Zero action is always safe
        
        filtered_action = cbf.filter_action(state, action, dynamics_fn)
        
        assert np.allclose(filtered_action, action, atol=1e-2)
    
    def test_cbf_statistics(self):
        """Test statistics tracking."""
        def barrier_fn(state):
            return 1.0 - state[0] ** 2
        
        def dynamics_fn(state, action):
            return action
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=1.0)
        
        # Filter some actions
        state = np.array([0.8, 0.0])
        unsafe_action = np.array([1.0, 0.0])
        
        cbf.filter_action(state, unsafe_action, dynamics_fn)
        
        stats = cbf.get_statistics()
        
        assert 'num_interventions' in stats
        assert stats['num_interventions'] >= 0


class TestCBFFactoryFunctions:
    """Test CBF factory functions."""
    
    def test_position_limit_cbf(self):
        """Test position limit CBF creation."""
        cbf = create_position_limit_cbf(
            position_index=0,
            max_position=2.0,
            alpha=1.0
        )
        
        # Test safe state
        safe_state = np.array([1.0, 0.0, 0.0])
        assert cbf.is_safe(safe_state)
        
        # Test unsafe state
        unsafe_state = np.array([2.5, 0.0, 0.0])
        assert not cbf.is_safe(unsafe_state)
    
    def test_velocity_limit_cbf(self):
        """Test velocity limit CBF creation."""
        cbf = create_velocity_limit_cbf(
            velocity_index=1,
            max_velocity=8.0,
            alpha=1.0
        )
        
        # Test safe state
        safe_state = np.array([0.0, 5.0, 0.0])
        assert cbf.is_safe(safe_state)
        
        # Test unsafe state
        unsafe_state = np.array([0.0, 10.0, 0.0])
        assert not cbf.is_safe(unsafe_state)


class TestMultiCBF:
    """Test suite for Multiple CBFs."""
    
    def test_multi_cbf_initialization(self):
        """Test MultiCBF initialization."""
        cbf1 = create_position_limit_cbf(0, 2.0)
        cbf2 = create_velocity_limit_cbf(1, 8.0)
        
        multi_cbf = MultiCBF([cbf1, cbf2])
        
        assert len(multi_cbf.cbfs) == 2
    
    def test_multi_cbf_is_safe(self):
        """Test safety checking with multiple constraints."""
        cbf1 = create_position_limit_cbf(0, 2.0)
        cbf2 = create_velocity_limit_cbf(1, 8.0)
        
        multi_cbf = MultiCBF([cbf1, cbf2])
        
        # Both constraints satisfied
        safe_state = np.array([1.0, 5.0, 0.0])
        assert multi_cbf.is_safe(safe_state)
        
        # One constraint violated
        unsafe_state = np.array([1.0, 10.0, 0.0])
        assert not multi_cbf.is_safe(unsafe_state)


class TestSafeAgent:
    """Test safe agent wrapper."""
    
    def test_safe_agent_creation(self):
        """Test safe agent wrapper creation."""
        # Mock agent
        class MockAgent:
            def get_action(self, state, deterministic=False):
                return np.array([0.1, 0.1])
        
        agent = MockAgent()
        
        def barrier_fn(state):
            return 1.0 - np.sum(state ** 2)
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=1.0)
        safe_agent = cbf.wrap_agent(agent)
        
        assert safe_agent.agent == agent
        assert safe_agent.cbf == cbf
    
    def test_safe_agent_get_action(self):
        """Test safe agent action selection."""
        class MockAgent:
            def get_action(self, state, deterministic=False):
                return np.array([0.1, 0.1])
        
        agent = MockAgent()
        
        def barrier_fn(state):
            return 1.0 - np.sum(state ** 2)
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=1.0)
        safe_agent = cbf.wrap_agent(agent)
        
        state = np.array([0.5, 0.5])
        action = safe_agent.get_action(state)
        
        assert action.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

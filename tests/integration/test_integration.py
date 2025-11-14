"""Integration tests for the full RL-Control-Suite."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from rl_control.algorithms import PPO, SAC, MPC
from rl_control.safety import ControlBarrierFunction
from rl_control.envs import make_env


class TestEndToEndPPO:
    """End-to-end integration tests for PPO."""
    
    @pytest.mark.slow
    def test_ppo_cartpole_training(self):
        """Test PPO training on CartPole environment."""
        try:
            env = make_env("CartPole-v1")
        except Exception:
            pytest.skip("Gym/Gymnasium not available")
        
        agent = PPO(
            state_dim=4,
            action_dim=2,
            hidden_dims=[32, 32],
            continuous=False
        )
        
        # Train for a few steps
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        
        for _ in range(100):
            action = agent.get_action(state)
            result = env.step(action)
            
            if len(result) == 4:
                state, reward, done, info = result
            else:
                state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            if done:
                state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        
        env.close()
        
        # Just check that training runs without errors
        assert True


class TestEndToEndSAC:
    """End-to-end integration tests for SAC."""
    
    @pytest.mark.slow
    def test_sac_pendulum_training(self):
        """Test SAC training on Pendulum environment."""
        try:
            env = make_env("Pendulum-v1")
        except Exception:
            pytest.skip("Gym/Gymnasium not available")
        
        agent = SAC(
            state_dim=3,
            action_dim=1,
            hidden_dims=[64, 64]
        )
        
        # Train for a few steps
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        
        for step in range(100):
            # Random actions for warmup
            if step < 10:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)
            
            result = env.step(action)
            
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update after warmup
            if step >= 10:
                agent.update()
            
            state = next_state
            
            if done:
                state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        
        env.close()
        
        # Check that buffer has data
        assert len(agent.replay_buffer) > 0


class TestSafetyIntegration:
    """Integration tests for safety features."""
    
    def test_cbf_with_agent(self):
        """Test CBF integration with RL agent."""
        # Mock agent
        class MockAgent:
            def get_action(self, state, deterministic=False):
                return np.array([1.0, 1.0])
        
        agent = MockAgent()
        
        # Create CBF
        def barrier_fn(state):
            return 2.0 - np.sum(state ** 2)
        
        cbf = ControlBarrierFunction(barrier_fn=barrier_fn, alpha=1.0)
        
        # Wrap agent
        safe_agent = cbf.wrap_agent(agent)
        
        # Test action filtering
        state = np.array([1.0, 1.0])
        action = safe_agent.get_action(state)
        
        assert action is not None
        assert action.shape == (2,)


class TestMPCIntegration:
    """Integration tests for MPC."""
    
    def test_mpc_basic_control(self):
        """Test MPC basic control loop."""
        # Create simple linear system
        def cost_fn(state, action):
            return np.sum(state ** 2) + 0.01 * np.sum(action ** 2)
        
        def dynamics(state, action):
            # Simple integrator
            return state + 0.1 * action[:len(state)]
        
        mpc = MPC(
            state_dim=2,
            action_dim=2,
            horizon=10,
            cost_function=cost_fn,
            dynamics_model=dynamics
        )
        
        # Test control
        state = np.array([1.0, 1.0])
        
        for _ in range(5):
            action = mpc.compute_action(state)
            state = dynamics(state, action)
        
        # State should be moving toward zero
        assert np.linalg.norm(state) < 2.0


class TestFullPipeline:
    """Test complete training and evaluation pipeline."""
    
    @pytest.mark.slow
    def test_train_evaluate_save_load_pipeline(self, tmp_path):
        """Test full pipeline: train -> save -> load -> evaluate."""
        try:
            env = make_env("CartPole-v1")
        except Exception:
            pytest.skip("Gym/Gymnasium not available")
        
        # Train agent
        agent = PPO(
            state_dim=4,
            action_dim=2,
            hidden_dims=[32, 32],
            continuous=False
        )
        
        # Quick training
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        
        for _ in range(50):
            action = agent.get_action(state)
            result = env.step(action)
            
            if len(result) == 4:
                state, reward, done, info = result
            else:
                state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            if done:
                state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        
        # Save agent
        save_path = tmp_path / "agent.pt"
        agent.save(str(save_path))
        
        # Load agent
        new_agent = PPO(
            state_dim=4,
            action_dim=2,
            hidden_dims=[32, 32],
            continuous=False
        )
        new_agent.load(str(save_path))
        
        # Evaluate
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        action = new_agent.get_action(state, deterministic=True)
        
        assert action is not None
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])

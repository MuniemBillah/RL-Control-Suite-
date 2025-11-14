"""Environment abstractions and utilities."""

from typing import Any, Optional, Tuple
import numpy as np


def make_env(env_name: str, **kwargs: Any) -> Any:
    """Create and return a Gym environment.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional arguments for environment
        
    Returns:
        Gym environment
    """
    try:
        import gymnasium as gym
        env = gym.make(env_name, **kwargs)
    except ImportError:
        try:
            import gym
            env = gym.make(env_name, **kwargs)
        except ImportError:
            raise ImportError(
                "Neither gymnasium nor gym is installed. "
                "Install with: pip install gymnasium"
            )
    
    return env


class NormalizeObservation:
    """Wrapper to normalize observations using running statistics.
    
    Args:
        env: Gym environment
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, env: Any, epsilon: float = 1e-8) -> None:
        self.env = env
        self.epsilon = epsilon
        
        # Running statistics
        self.obs_mean = np.zeros(env.observation_space.shape[0])
        self.obs_var = np.ones(env.observation_space.shape[0])
        self.count = 0
    
    def _update_stats(self, obs: np.ndarray) -> None:
        """Update running mean and variance.
        
        Args:
            obs: Observation to include in statistics
        """
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        delta2 = obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.count
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation.
        
        Args:
            obs: Raw observation
            
        Returns:
            Normalized observation
        """
        return (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
    
    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        """Reset environment.
        
        Returns:
            Normalized observation and info
        """
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        
        self._update_stats(obs)
        return self._normalize(obs), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take environment step.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (normalized_obs, reward, terminated, truncated, info)
        """
        result = self.env.step(action)
        
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
        
        self._update_stats(obs)
        return self._normalize(obs), reward, terminated, truncated, info
    
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to wrapped environment."""
        return getattr(self.env, name)


class NormalizeReward:
    """Wrapper to normalize rewards using running statistics.
    
    Args:
        env: Gym environment
        gamma: Discount factor for return normalization
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, env: Any, gamma: float = 0.99, epsilon: float = 1e-8) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Running statistics
        self.return_val = 0.0
        self.return_mean = 0.0
        self.return_var = 1.0
        self.count = 0
    
    def _update_stats(self, reward: float) -> None:
        """Update running return statistics.
        
        Args:
            reward: Reward to include in statistics
        """
        self.return_val = reward + self.gamma * self.return_val
        
        self.count += 1
        delta = self.return_val - self.return_mean
        self.return_mean += delta / self.count
        delta2 = self.return_val - self.return_mean
        self.return_var += (delta * delta2 - self.return_var) / self.count
    
    def _normalize(self, reward: float) -> float:
        """Normalize reward.
        
        Args:
            reward: Raw reward
            
        Returns:
            Normalized reward
        """
        return reward / np.sqrt(self.return_var + self.epsilon)
    
    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        """Reset environment.
        
        Returns:
            Observation and info
        """
        self.return_val = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take environment step.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (obs, normalized_reward, terminated, truncated, info)
        """
        result = self.env.step(action)
        
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
        
        self._update_stats(reward)
        normalized_reward = self._normalize(reward)
        
        if terminated or truncated:
            self.return_val = 0.0
        
        return obs, normalized_reward, terminated, truncated, info
    
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to wrapped environment."""
        return getattr(self.env, name)


class ClipAction:
    """Wrapper to clip continuous actions to valid range.
    
    Args:
        env: Gym environment
    """
    
    def __init__(self, env: Any) -> None:
        self.env = env
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take environment step with clipped action.
        
        Args:
            action: Action to take (will be clipped)
            
        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        clipped_action = np.clip(action, self.action_low, self.action_high)
        return self.env.step(clipped_action)
    
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to wrapped environment."""
        return getattr(self.env, name)


class RecordEpisodeStatistics:
    """Wrapper to record episode statistics.
    
    Args:
        env: Gym environment
    """
    
    def __init__(self, env: Any) -> None:
        self.env = env
        self.episode_returns: list = []
        self.episode_lengths: list = []
        self.current_return = 0.0
        self.current_length = 0
    
    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        """Reset environment.
        
        Returns:
            Observation and info with episode statistics
        """
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        
        self.current_return = 0.0
        self.current_length = 0
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take environment step.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        result = self.env.step(action)
        
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
        
        self.current_return += reward
        self.current_length += 1
        
        if terminated or truncated:
            info["episode"] = {
                "r": self.current_return,
                "l": self.current_length
            }
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
        
        return obs, reward, terminated, truncated, info
    
    def get_statistics(self) -> dict:
        """Get episode statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.episode_returns:
            return {}
        
        return {
            "mean_return": np.mean(self.episode_returns),
            "std_return": np.std(self.episode_returns),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "num_episodes": len(self.episode_returns)
        }
    
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to wrapped environment."""
        return getattr(self.env, name)


def create_wrapped_env(
    env_name: str,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    clip_action: bool = True,
    record_stats: bool = True,
    **kwargs: Any
) -> Any:
    """Create environment with standard wrappers.
    
    Args:
        env_name: Name of environment
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        clip_action: Whether to clip continuous actions
        record_stats: Whether to record episode statistics
        **kwargs: Additional arguments for environment
        
    Returns:
        Wrapped environment
    """
    env = make_env(env_name, **kwargs)
    
    if normalize_obs:
        env = NormalizeObservation(env)
    
    if normalize_reward:
        env = NormalizeReward(env)
    
    if clip_action and hasattr(env.action_space, 'low'):
        env = ClipAction(env)
    
    if record_stats:
        env = RecordEpisodeStatistics(env)
    
    return env

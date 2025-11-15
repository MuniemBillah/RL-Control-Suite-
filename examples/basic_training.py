"""Basic training example with PPO on CartPole."""

from rl_control.algorithms import PPO
from rl_control.envs import make_env
from rl_control.utils import Logger
import numpy as np


def train_ppo_cartpole():
    """Train PPO on CartPole-v1 environment."""
    print("="*60)
    print("Training PPO on CartPole-v1")
    print("="*60)
    
    # Create environment
    env = make_env("CartPole-v1")
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"\nEnvironment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize PPO agent
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        continuous=False
    )
    
    # Initialize logger
    logger = Logger(log_dir="logs", experiment_name="ppo_cartpole")
    
    # Training configuration
    config = {
        "algorithm": "PPO",
        "environment": "CartPole-v1",
        "total_timesteps": 50000,
        "hidden_dims": [64, 64],
        "learning_rate": 3e-4,
        "gamma": 0.99
    }
    logger.save_config(config)
    
    print("\nStarting training...")
    print(f"Total timesteps: {config['total_timesteps']}")
    
    # Train the agent
    agent.train(
        env=env,
        total_timesteps=config["total_timesteps"],
        steps_per_update=2048,
        log_interval=10
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Save the trained model
    model_path = "models/ppo_cartpole.pt"
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Print training statistics
    if agent.training_stats["episode_rewards"]:
        mean_reward = np.mean(agent.training_stats["episode_rewards"][-10:])
        print(f"\nFinal mean reward (last 10 episodes): {mean_reward:.2f}")
        
        # Log final metrics
        logger.log_scalar("final_mean_reward", mean_reward)
        logger.save_metrics()
    
    # Test the trained agent
    print("\n" + "="*60)
    print("Testing trained agent...")
    print("="*60)
    
    test_episodes = 10
    test_rewards = []
    
    for episode in range(test_episodes):
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, deterministic=True)
            result = env.step(action)
            
            if len(result) == 4:
                state, reward, done, info = result
            else:
                state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            episode_reward += reward
        
        test_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    mean_test_reward = np.mean(test_rewards)
    print(f"\nMean test reward: {mean_test_reward:.2f} ± {np.std(test_rewards):.2f}")
    
    env.close()
    
    return agent, mean_test_reward


if __name__ == "__main__":
    train_ppo_cartpole()

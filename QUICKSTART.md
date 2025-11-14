# Quick Start Guide 🚀

Get started with RL-Control-Suite in 5 minutes!

## Installation

```bash
# Install from source
pip install -e .

# Or with all dependencies
pip install -e ".[dev,mpc,jax]"
```

## Your First Agent

### 1. Train PPO on CartPole (2 minutes)

```python
from rl_control.algorithms import PPO
from rl_control.envs import make_env

# Create environment
env = make_env("CartPole-v1")

# Create agent
agent = PPO(
    state_dim=4,
    action_dim=2,
    hidden_dims=[64, 64],
    continuous=False
)

# Train
agent.train(env, total_timesteps=50000)

# Test
state, _ = env.reset()
for _ in range(100):
    action = agent.get_action(state, deterministic=True)
    state, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break
```

### 2. SAC for Continuous Control

```python
from rl_control.algorithms import SAC
from rl_control.envs import make_env

env = make_env("Pendulum-v1")

agent = SAC(
    state_dim=3,
    action_dim=1,
    hidden_dims=[256, 256]
)

agent.train(env, total_timesteps=50000)
```

### 3. Safe Control with CBF

```python
from rl_control.safety import create_position_limit_cbf

# Create safety constraint
cbf = create_position_limit_cbf(
    position_index=0,
    max_position=2.0
)

# Wrap agent for safety
safe_agent = cbf.wrap_agent(agent)

# Now agent respects safety constraints!
action = safe_agent.get_action(state)
```

## Run Examples

```bash
# Basic PPO training
python examples/basic_training.py

# EV charging optimization
python examples/ev_optimization.py

# Safe navigation with CBF
python examples/safe_navigation.py
```

## Test the Installation

```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src/rl_control
```

## Next Steps

- 📚 Read the [full documentation](README.md)
- 🔬 Explore the [examples](examples/)
- 🧪 Read the [testing guide](TESTING_GUIDE.md)
- 🤝 Check out [contributing guidelines](CONTRIBUTING.md)

## Common Commands

```bash
# Install package
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Build docs
cd docs && make html

# Run example
python examples/basic_training.py
```

## Need Help?

- 📖 Check the [README](README.md)
- 🐛 [Report issues](https://github.com/yourusername/rl-control-suite/issues)
- 💬 [Discussions](https://github.com/yourusername/rl-control-suite/discussions)

Happy Learning! 🎓

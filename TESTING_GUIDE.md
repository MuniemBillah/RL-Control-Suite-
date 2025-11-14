# RL-Control-Suite Testing Guide

This guide will help you test the RL-Control-Suite project to ensure everything works correctly.

## Prerequisites

Before testing, make sure you have:
- Python 3.9 or higher installed
- Git (optional, for cloning)
- 2GB+ of free disk space
- Internet connection (for downloading dependencies)

## Installation

### Method 1: Quick Install (Recommended)

```bash
# Navigate to the project directory
cd rl-control-suite

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the package
pip install -e ".[dev]"
```

### Method 2: Docker Install

```bash
# Build the Docker image
docker build -t rl-control-suite .

# Run tests in Docker
docker run rl-control-suite pytest tests/ -v
```

## Running Tests

### 1. Unit Tests

Test individual components in isolation:

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_ppo.py -v

# Run with coverage report
pytest tests/unit/ -v --cov=src/rl_control --cov-report=html
```

Expected output: All tests should pass ✓

### 2. Integration Tests

Test full workflows:

```bash
# Run integration tests
pytest tests/integration/ -v

# Run without slow tests
pytest tests/integration/ -v -m "not slow"
```

Expected output: Integration tests should pass ✓

### 3. Full Test Suite

Run all tests with coverage:

```bash
pytest tests/ -v --cov=src/rl_control --cov-report=html --cov-report=term-missing
```

Expected coverage: >85%

View coverage report: Open `htmlcov/index.html` in your browser

### 4. Code Quality Checks

#### Linting
```bash
flake8 src/ tests/ --max-line-length=100
```

#### Formatting
```bash
black --check src/ tests/
```

#### Type Checking
```bash
mypy src/rl_control --ignore-missing-imports
```

## Testing Examples

### Example 1: Basic PPO Training

```bash
cd examples
python basic_training.py
```

Expected output:
- Training progress bar
- Episode rewards increasing
- Model saved to `models/ppo_cartpole.pt`
- Test results showing >150 average reward

### Example 2: EV Charging Optimization

```bash
cd examples
python ev_optimization.py
```

Expected output:
- Charging schedule printed
- Total cost and energy usage
- Plot saved to `ev_charging_optimization.png`

### Example 3: Safe Navigation

```bash
cd examples
python safe_navigation.py
```

Expected output:
- Training progress for both agents
- Success/collision rates
- Plot saved to `safe_navigation_results.png`

## Manual Testing

### Test 1: Import All Modules

```python
python -c "from rl_control import PPO, SAC, MPC, ControlBarrierFunction, make_env; print('✓ All imports successful')"
```

### Test 2: Create and Train PPO Agent

```python
from rl_control.algorithms import PPO
from rl_control.envs import make_env

env = make_env("CartPole-v1")
agent = PPO(state_dim=4, action_dim=2, continuous=False)

# Quick test
state, _ = env.reset()
action = agent.get_action(state)
print(f"✓ PPO agent created and action generated: {action}")
```

### Test 3: Safety Constraints

```python
import numpy as np
from rl_control.safety import create_position_limit_cbf

cbf = create_position_limit_cbf(position_index=0, max_position=2.0)

safe_state = np.array([1.0, 0.0, 0.0])
unsafe_state = np.array([3.0, 0.0, 0.0])

print(f"✓ Safe state: {cbf.is_safe(safe_state)}")  # Should be True
print(f"✓ Unsafe state: {cbf.is_safe(unsafe_state)}")  # Should be False
```

## Common Issues and Solutions

### Issue 1: Gymnasium/Gym Not Found

**Error:** `ModuleNotFoundError: No module named 'gymnasium'`

**Solution:**
```bash
pip install gymnasium
```

### Issue 2: PyTorch Not Found

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install torch
```

### Issue 3: Tests Fail Due to Missing Dependencies

**Solution:**
```bash
pip install -e ".[dev]"
```

### Issue 4: Coverage Too Low

**Solution:**
- Check which files are not covered: `pytest --cov --cov-report=term-missing`
- Some integration tests may be skipped if gym is not installed

## Performance Benchmarks

Expected performance on standard hardware (CPU):

| Test | Time | Memory |
|------|------|--------|
| Unit Tests | <30s | <500MB |
| Integration Tests | <60s | <1GB |
| Full Suite | <90s | <1GB |
| Basic Training Example | 2-5 min | <1GB |

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- **Tests**: Run on every push and PR
- **Linting**: Check code style
- **Coverage**: Aim for >85%
- **Docs**: Build documentation
- **Publish**: Deploy to PyPI on release

View CI status: Check the badges in README.md

## Testing Online

### Option 1: Google Colab

You can test the project online using Google Colab:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Install the package:
```python
!pip install git+https://github.com/yourusername/rl-control-suite.git
```
4. Run examples:
```python
from rl_control import PPO, make_env

env = make_env("CartPole-v1")
agent = PPO(state_dim=4, action_dim=2, continuous=False)

# Quick training
agent.train(env, total_timesteps=1000)
```

### Option 2: Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/rl-control-suite/main)

Click the badge to launch an interactive environment.

### Option 3: Gitpod

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/yourusername/rl-control-suite)

## Troubleshooting

### Debug Mode

Run tests with verbose output:
```bash
pytest tests/ -vv -s
```

### Test Specific Components

```bash
# Test only PPO
pytest tests/unit/test_ppo.py -v

# Test only safety
pytest tests/unit/test_safety.py -v

# Test only utils
pytest tests/unit/test_utils.py -v
```

### Generate Test Report

```bash
pytest tests/ --html=report.html --self-contained-html
```

Open `report.html` in your browser for detailed results.

## Next Steps

After successful testing:

1. **Explore Examples**: Run all example scripts
2. **Read Documentation**: Check `docs/` folder
3. **Try Custom Experiments**: Modify examples for your use case
4. **Contribute**: See CONTRIBUTING.md for guidelines

## Support

If you encounter issues:

1. Check this guide for solutions
2. Review the [documentation](https://rl-control-suite.readthedocs.io/)
3. Search [existing issues](https://github.com/yourusername/rl-control-suite/issues)
4. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Your environment (OS, Python version)

---

**Happy Testing! 🎉**

For more information, visit the [project repository](https://github.com/yourusername/rl-control-suite).

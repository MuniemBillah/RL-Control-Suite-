# Contributing to RL-Control-Suite

Thank you for your interest in contributing to RL-Control-Suite! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, package versions)
- Code samples if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- A detailed description of the proposed functionality
- Explain why this enhancement would be useful
- List any alternative solutions you've considered

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed
3. **Ensure tests pass**
   ```bash
   pytest tests/ -v --cov
   ```
4. **Run linting and formatting**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/rl_control
   ```
5. **Commit your changes**
   - Use clear and descriptive commit messages
   - Reference related issues (e.g., "Fixes #123")
6. **Push to your fork** and submit a pull request

## Development Setup

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rl-control-suite.git
   cd rl-control-suite
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev,docs]"
   ```

### Docker Development

```bash
docker-compose up dev
```

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 100 characters
- Use type hints for function signatures

### Documentation

- Use Google-style docstrings
- Include examples in docstrings when appropriate
- Update README.md for user-facing changes
- Add tutorials for new features

### Example Docstring

```python
def train_agent(env, total_timesteps: int, log_interval: int = 100) -> Dict[str, Any]:
    """Train RL agent on environment.
    
    Args:
        env: Gym environment
        total_timesteps: Total number of training timesteps
        log_interval: Logging frequency
        
    Returns:
        Dictionary containing training metrics
        
    Example:
        >>> env = make_env("CartPole-v1")
        >>> agent = PPO(state_dim=4, action_dim=2)
        >>> metrics = train_agent(env, total_timesteps=10000)
    """
```

## Testing Guidelines

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Aim for >90% code coverage
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`

### Example Test

```python
def test_ppo_initialization():
    """Test PPO agent initialization."""
    agent = PPO(state_dim=4, action_dim=2)
    assert agent.policy is not None
    assert agent.gamma == 0.99
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_ppo.py -v

# Run with coverage
pytest tests/ --cov=src/rl_control --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"
```

## Documentation

### Building Documentation

```bash
cd docs
make html
```

View documentation at `docs/_build/html/index.html`

### Writing Tutorials

- Place tutorials in `docs/tutorials/`
- Include code examples that users can run
- Explain the reasoning behind design decisions
- Link to relevant API documentation

## Release Process

1. Update version number in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Create a new GitHub release
4. CI/CD will automatically:
   - Run tests
   - Build documentation
   - Publish to PyPI
   - Build and push Docker images

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

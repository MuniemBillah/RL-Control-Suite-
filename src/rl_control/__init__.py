"""RL-Control-Suite: Production-ready RL and optimal control library."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from rl_control.algorithms import PPO, SAC, MPC
from rl_control.envs import make_env
from rl_control.safety import ControlBarrierFunction

__all__ = [
    "PPO",
    "SAC",
    "MPC",
    "make_env",
    "ControlBarrierFunction",
]

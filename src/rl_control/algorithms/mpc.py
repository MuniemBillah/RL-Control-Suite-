"""Model Predictive Control (MPC) implementation.

This module implements a basic MPC controller using numerical optimization.
For production use with complex dynamics, consider using CasADi backend.
"""

from typing import Callable, Optional, Any
import numpy as np
from scipy.optimize import minimize


class MPC:
    """Model Predictive Control controller.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        horizon: Prediction horizon
        cost_function: Cost function to minimize (state, action) -> cost
        dynamics_model: Optional dynamics model (state, action) -> next_state
        action_bounds: Tuple of (lower_bound, upper_bound) for actions
        dt: Time step
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 10,
        cost_function: Optional[Callable] = None,
        dynamics_model: Optional[Callable] = None,
        action_bounds: tuple = (-1.0, 1.0),
        dt: float = 0.05
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.cost_function = cost_function or self._default_cost
        self.dynamics_model = dynamics_model or self._default_dynamics
        self.action_bounds = action_bounds
        self.dt = dt
        
        # Storage for warm starting
        self.previous_solution: Optional[np.ndarray] = None
    
    def _default_cost(self, state: np.ndarray, action: np.ndarray) -> float:
        """Default quadratic cost function.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Cost value
        """
        state_cost = np.sum(state ** 2)
        action_cost = 0.01 * np.sum(action ** 2)
        return state_cost + action_cost
    
    def _default_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Default linear dynamics model (identity + action).
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Next state
        """
        # Simple integration: next_state = state + dt * action
        # This is a placeholder - in practice, you'd use a learned model
        next_state = state + self.dt * action[:len(state)]
        return next_state
    
    def _rollout_cost(self, action_sequence: np.ndarray, initial_state: np.ndarray) -> float:
        """Compute total cost for an action sequence.
        
        Args:
            action_sequence: Flattened sequence of actions
            initial_state: Starting state
            
        Returns:
            Total cost over horizon
        """
        action_sequence = action_sequence.reshape(self.horizon, self.action_dim)
        
        state = initial_state.copy()
        total_cost = 0.0
        
        for t in range(self.horizon):
            action = action_sequence[t]
            
            # Add stage cost
            total_cost += self.cost_function(state, action)
            
            # Predict next state
            state = self.dynamics_model(state, action)
        
        # Add terminal cost
        total_cost += self.cost_function(state, np.zeros(self.action_dim))
        
        return total_cost
    
    def compute_action(
        self,
        state: np.ndarray,
        method: str = 'L-BFGS-B',
        max_iter: int = 100
    ) -> np.ndarray:
        """Compute optimal action using MPC.
        
        Args:
            state: Current state
            method: Optimization method
            max_iter: Maximum optimization iterations
            
        Returns:
            Optimal action to take
        """
        # Initialize action sequence
        if self.previous_solution is not None:
            # Warm start: shift previous solution and add zero action
            initial_guess = np.vstack([
                self.previous_solution[1:],
                np.zeros((1, self.action_dim))
            ]).flatten()
        else:
            initial_guess = np.zeros(self.horizon * self.action_dim)
        
        # Define bounds
        bounds = [(self.action_bounds[0], self.action_bounds[1])] * (self.horizon * self.action_dim)
        
        # Optimize
        result = minimize(
            fun=lambda x: self._rollout_cost(x, state),
            x0=initial_guess,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        # Extract optimal action sequence
        optimal_actions = result.x.reshape(self.horizon, self.action_dim)
        self.previous_solution = optimal_actions
        
        # Return first action (receding horizon)
        return optimal_actions[0]
    
    def reset(self) -> None:
        """Reset the controller (clear previous solution)."""
        self.previous_solution = None
    
    def set_cost_function(self, cost_fn: Callable) -> None:
        """Set custom cost function.
        
        Args:
            cost_fn: Cost function (state, action) -> cost
        """
        self.cost_function = cost_fn
    
    def set_dynamics_model(self, dynamics_fn: Callable) -> None:
        """Set custom dynamics model.
        
        Args:
            dynamics_fn: Dynamics function (state, action) -> next_state
        """
        self.dynamics_model = dynamics_fn


class LinearMPC(MPC):
    """Linear MPC with quadratic cost.
    
    This variant uses analytical solutions for linear quadratic regulation.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        horizon: Prediction horizon
        A: State transition matrix
        B: Control input matrix
        Q: State cost matrix
        R: Action cost matrix
        action_bounds: Tuple of (lower_bound, upper_bound) for actions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 10,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        action_bounds: tuple = (-1.0, 1.0)
    ) -> None:
        super().__init__(state_dim, action_dim, horizon, action_bounds=action_bounds)
        
        # Linear dynamics: x_{t+1} = A * x_t + B * u_t
        self.A = A if A is not None else np.eye(state_dim)
        self.B = B if B is not None else np.eye(state_dim, action_dim)
        
        # Quadratic cost: x^T Q x + u^T R u
        self.Q = Q if Q is not None else np.eye(state_dim)
        self.R = R if R is not None else 0.01 * np.eye(action_dim)
    
    def _linear_cost(self, state: np.ndarray, action: np.ndarray) -> float:
        """Quadratic cost function.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Cost value
        """
        state_cost = state.T @ self.Q @ state
        action_cost = action.T @ self.R @ action
        return state_cost + action_cost
    
    def _linear_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Linear dynamics model.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Next state
        """
        return self.A @ state + self.B @ action
    
    def compute_action(
        self,
        state: np.ndarray,
        method: str = 'L-BFGS-B',
        max_iter: int = 100
    ) -> np.ndarray:
        """Compute optimal action using linear MPC.
        
        Args:
            state: Current state
            method: Optimization method
            max_iter: Maximum optimization iterations
            
        Returns:
            Optimal action to take
        """
        # Override cost and dynamics with linear versions
        self.cost_function = self._linear_cost
        self.dynamics_model = self._linear_dynamics
        
        return super().compute_action(state, method, max_iter)


def create_pendulum_mpc(horizon: int = 20) -> MPC:
    """Create MPC controller for inverted pendulum.
    
    Args:
        horizon: Prediction horizon
        
    Returns:
        Configured MPC controller
    """
    def pendulum_cost(state: np.ndarray, action: np.ndarray) -> float:
        """Pendulum cost function."""
        cos_theta, sin_theta, theta_dot = state
        theta = np.arctan2(sin_theta, cos_theta)
        
        # Penalize deviation from upright position and high velocity
        angle_cost = 10.0 * theta ** 2
        velocity_cost = 0.1 * theta_dot ** 2
        action_cost = 0.001 * action[0] ** 2
        
        return angle_cost + velocity_cost + action_cost
    
    def pendulum_dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Simplified pendulum dynamics."""
        cos_theta, sin_theta, theta_dot = state
        theta = np.arctan2(sin_theta, cos_theta)
        
        # Constants
        g = 10.0  # gravity
        m = 1.0   # mass
        l = 1.0   # length
        dt = 0.05
        
        # Dynamics
        torque = np.clip(action[0], -2.0, 2.0)
        theta_dot_new = theta_dot + dt * (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l ** 2) * torque)
        theta_dot_new = np.clip(theta_dot_new, -8.0, 8.0)
        theta_new = theta + dt * theta_dot_new
        
        # Wrap angle
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return np.array([np.cos(theta_new), np.sin(theta_new), theta_dot_new])
    
    mpc = MPC(
        state_dim=3,
        action_dim=1,
        horizon=horizon,
        cost_function=pendulum_cost,
        dynamics_model=pendulum_dynamics,
        action_bounds=(-2.0, 2.0)
    )
    
    return mpc

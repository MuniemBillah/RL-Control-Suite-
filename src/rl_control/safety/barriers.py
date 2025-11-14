"""Control Barrier Functions for safety-critical control.

This module implements Control Barrier Functions (CBFs) for ensuring safety constraints.
Reference: Ames et al., "Control Barrier Functions: Theory and Applications", 2019
"""

from typing import Callable, Optional, Any
import numpy as np
from scipy.optimize import minimize


class ControlBarrierFunction:
    """Control Barrier Function for safety-critical control.
    
    CBFs provide formal guarantees that a system will remain in a safe set.
    The safe set is defined as S = {x | h(x) >= 0}, where h is the barrier function.
    
    Args:
        barrier_fn: Barrier function h(state) where h(x) >= 0 defines safe set
        alpha: Class-K function parameter for safety constraint
        action_bounds: Tuple of (lower_bound, upper_bound) for actions
        gradient_fn: Optional analytical gradient of barrier function
    """
    
    def __init__(
        self,
        barrier_fn: Callable[[np.ndarray], float],
        alpha: float = 1.0,
        action_bounds: tuple = (-1.0, 1.0),
        gradient_fn: Optional[Callable] = None
    ) -> None:
        self.barrier_fn = barrier_fn
        self.alpha = alpha
        self.action_bounds = action_bounds
        self.gradient_fn = gradient_fn
        
        # Statistics
        self.num_interventions = 0
        self.total_modifications = 0.0
    
    def _numerical_gradient(self, state: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """Compute numerical gradient of barrier function.
        
        Args:
            state: Current state
            epsilon: Finite difference step size
            
        Returns:
            Gradient of barrier function
        """
        grad = np.zeros_like(state)
        h_0 = self.barrier_fn(state)
        
        for i in range(len(state)):
            state_plus = state.copy()
            state_plus[i] += epsilon
            h_plus = self.barrier_fn(state_plus)
            grad[i] = (h_plus - h_0) / epsilon
        
        return grad
    
    def _barrier_constraint(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dynamics_fn: Callable
    ) -> float:
        """Evaluate barrier function constraint.
        
        The constraint is: dh/dx * f(x, u) + alpha * h(x) >= 0
        
        Args:
            state: Current state
            action: Proposed action
            dynamics_fn: Dynamics function (state, action) -> state_dot
            
        Returns:
            Constraint value (should be >= 0)
        """
        # Compute barrier function value
        h = self.barrier_fn(state)
        
        # Compute gradient
        if self.gradient_fn is not None:
            dh_dx = self.gradient_fn(state)
        else:
            dh_dx = self._numerical_gradient(state)
        
        # Compute state derivative
        state_dot = dynamics_fn(state, action)
        
        # Barrier constraint: Lie derivative + class-K function
        constraint = np.dot(dh_dx, state_dot) + self.alpha * h
        
        return constraint
    
    def filter_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dynamics_fn: Callable,
        method: str = 'SLSQP'
    ) -> np.ndarray:
        """Filter action to ensure safety constraint.
        
        Solves: min ||u - u_nominal||^2
                s.t. CBF constraint is satisfied
        
        Args:
            state: Current state
            action: Nominal (unconstrained) action
            dynamics_fn: Dynamics function
            method: Optimization method
            
        Returns:
            Safe action
        """
        # Check if current action is already safe
        if self._barrier_constraint(state, action, dynamics_fn) >= 0:
            return action
        
        # Define optimization problem
        def objective(u: np.ndarray) -> float:
            """Minimize deviation from nominal action."""
            return np.sum((u - action) ** 2)
        
        def constraint_fn(u: np.ndarray) -> float:
            """CBF safety constraint."""
            return self._barrier_constraint(state, u, dynamics_fn)
        
        # Bounds
        action_dim = len(action)
        bounds = [(self.action_bounds[0], self.action_bounds[1])] * action_dim
        
        # Constraints
        constraints = {'type': 'ineq', 'fun': constraint_fn}
        
        # Solve
        result = minimize(
            objective,
            action,
            method=method,
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            safe_action = result.x
            self.num_interventions += 1
            self.total_modifications += np.linalg.norm(safe_action - action)
            return safe_action
        else:
            # If optimization fails, return most conservative action
            print(f"Warning: CBF optimization failed. Using conservative action.")
            return np.zeros_like(action)
    
    def wrap_agent(self, agent: Any, dynamics_fn: Optional[Callable] = None) -> 'SafeAgent':
        """Wrap an RL agent with safety filter.
        
        Args:
            agent: RL agent with get_action method
            dynamics_fn: Optional dynamics function (will use default if None)
            
        Returns:
            SafeAgent wrapper
        """
        return SafeAgent(agent, self, dynamics_fn)
    
    def is_safe(self, state: np.ndarray) -> bool:
        """Check if state is in safe set.
        
        Args:
            state: State to check
            
        Returns:
            True if state is safe (h(x) >= 0)
        """
        return self.barrier_fn(state) >= 0
    
    def safety_margin(self, state: np.ndarray) -> float:
        """Compute safety margin.
        
        Args:
            state: Current state
            
        Returns:
            Safety margin (positive means safe)
        """
        return self.barrier_fn(state)
    
    def get_statistics(self) -> dict:
        """Get safety statistics.
        
        Returns:
            Dictionary of safety statistics
        """
        return {
            "num_interventions": self.num_interventions,
            "total_modifications": self.total_modifications,
            "avg_modification": self.total_modifications / max(self.num_interventions, 1)
        }


class SafeAgent:
    """Wrapper for RL agents with safety guarantees via CBF.
    
    Args:
        agent: Base RL agent
        cbf: Control Barrier Function
        dynamics_fn: Dynamics function
    """
    
    def __init__(
        self,
        agent: Any,
        cbf: ControlBarrierFunction,
        dynamics_fn: Optional[Callable] = None
    ) -> None:
        self.agent = agent
        self.cbf = cbf
        self.dynamics_fn = dynamics_fn or self._default_dynamics
    
    def _default_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Default dynamics (single integrator)."""
        return action[:len(state)]
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get safe action.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Safe action
        """
        # Get nominal action from base agent
        nominal_action = self.agent.get_action(state, deterministic)
        
        # Filter through CBF
        safe_action = self.cbf.filter_action(state, nominal_action, self.dynamics_fn)
        
        return safe_action
    
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to base agent."""
        return getattr(self.agent, name)


def create_position_limit_cbf(
    position_index: int = 0,
    max_position: float = 2.0,
    alpha: float = 1.0
) -> ControlBarrierFunction:
    """Create CBF for position limit constraint.
    
    Args:
        position_index: Index of position in state vector
        max_position: Maximum allowed position
        alpha: CBF parameter
        
    Returns:
        Configured CBF
    """
    def barrier_fn(state: np.ndarray) -> float:
        """Barrier function: h(x) = max_pos^2 - pos^2."""
        pos = state[position_index]
        return max_position ** 2 - pos ** 2
    
    def gradient_fn(state: np.ndarray) -> np.ndarray:
        """Gradient of barrier function."""
        grad = np.zeros_like(state)
        grad[position_index] = -2 * state[position_index]
        return grad
    
    return ControlBarrierFunction(
        barrier_fn=barrier_fn,
        alpha=alpha,
        gradient_fn=gradient_fn
    )


def create_velocity_limit_cbf(
    velocity_index: int = 1,
    max_velocity: float = 8.0,
    alpha: float = 1.0
) -> ControlBarrierFunction:
    """Create CBF for velocity limit constraint.
    
    Args:
        velocity_index: Index of velocity in state vector
        max_velocity: Maximum allowed velocity
        alpha: CBF parameter
        
    Returns:
        Configured CBF
    """
    def barrier_fn(state: np.ndarray) -> float:
        """Barrier function: h(x) = max_vel^2 - vel^2."""
        vel = state[velocity_index]
        return max_velocity ** 2 - vel ** 2
    
    def gradient_fn(state: np.ndarray) -> np.ndarray:
        """Gradient of barrier function."""
        grad = np.zeros_like(state)
        grad[velocity_index] = -2 * state[velocity_index]
        return grad
    
    return ControlBarrierFunction(
        barrier_fn=barrier_fn,
        alpha=alpha,
        gradient_fn=gradient_fn
    )


class MultiCBF:
    """Multiple Control Barrier Functions for multiple constraints.
    
    Args:
        cbfs: List of individual CBFs
        action_bounds: Action bounds
    """
    
    def __init__(
        self,
        cbfs: list,
        action_bounds: tuple = (-1.0, 1.0)
    ) -> None:
        self.cbfs = cbfs
        self.action_bounds = action_bounds
    
    def filter_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dynamics_fn: Callable,
        method: str = 'SLSQP'
    ) -> np.ndarray:
        """Filter action through multiple CBFs.
        
        Args:
            state: Current state
            action: Nominal action
            dynamics_fn: Dynamics function
            method: Optimization method
            
        Returns:
            Safe action satisfying all constraints
        """
        # Define optimization problem
        def objective(u: np.ndarray) -> float:
            return np.sum((u - action) ** 2)
        
        # Create constraints for all CBFs
        constraints = []
        for cbf in self.cbfs:
            def constraint_fn(u: np.ndarray, cbf=cbf) -> float:
                return cbf._barrier_constraint(state, u, dynamics_fn)
            
            constraints.append({'type': 'ineq', 'fun': constraint_fn})
        
        # Bounds
        action_dim = len(action)
        bounds = [(self.action_bounds[0], self.action_bounds[1])] * action_dim
        
        # Solve
        result = minimize(
            objective,
            action,
            method=method,
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            return np.zeros_like(action)
    
    def is_safe(self, state: np.ndarray) -> bool:
        """Check if all constraints are satisfied.
        
        Args:
            state: State to check
            
        Returns:
            True if all constraints satisfied
        """
        return all(cbf.is_safe(state) for cbf in self.cbfs)

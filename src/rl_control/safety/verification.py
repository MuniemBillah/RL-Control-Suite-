"""Formal verification tools for safety-critical systems.

This module provides tools for verifying safety properties of control systems.
"""

from typing import Callable, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of safety verification.
    
    Attributes:
        is_safe: Whether system is verified safe
        counterexample: State that violates safety (if any)
        max_violation: Maximum safety violation found
        num_samples_checked: Number of states checked
        verification_method: Method used for verification
    """
    is_safe: bool
    counterexample: Optional[np.ndarray]
    max_violation: float
    num_samples_checked: int
    verification_method: str
    
    def __str__(self) -> str:
        """String representation."""
        status = "SAFE" if self.is_safe else "UNSAFE"
        result = f"Verification Result: {status}\n"
        result += f"Method: {self.verification_method}\n"
        result += f"Samples checked: {self.num_samples_checked}\n"
        result += f"Max violation: {self.max_violation:.6f}\n"
        
        if self.counterexample is not None:
            result += f"Counterexample found: {self.counterexample}\n"
        
        return result


class SafetyVerifier:
    """Safety verification for control systems.
    
    Args:
        safety_property: Function (state) -> bool that returns True if state is safe
        state_bounds: List of (min, max) tuples for each state dimension
    """
    
    def __init__(
        self,
        safety_property: Callable[[np.ndarray], bool],
        state_bounds: List[Tuple[float, float]]
    ) -> None:
        self.safety_property = safety_property
        self.state_bounds = state_bounds
        self.state_dim = len(state_bounds)
    
    def verify_monte_carlo(
        self,
        num_samples: int = 10000,
        seed: Optional[int] = None
    ) -> VerificationResult:
        """Verify safety using Monte Carlo sampling.
        
        Args:
            num_samples: Number of random states to check
            seed: Random seed for reproducibility
            
        Returns:
            Verification result
        """
        if seed is not None:
            np.random.seed(seed)
        
        max_violation = 0.0
        counterexample = None
        
        for _ in range(num_samples):
            # Sample random state
            state = np.array([
                np.random.uniform(low, high)
                for low, high in self.state_bounds
            ])
            
            # Check safety
            if not self.safety_property(state):
                # Found violation
                violation = self._compute_violation(state)
                if violation > max_violation:
                    max_violation = violation
                    counterexample = state.copy()
        
        is_safe = counterexample is None
        
        return VerificationResult(
            is_safe=is_safe,
            counterexample=counterexample,
            max_violation=max_violation,
            num_samples_checked=num_samples,
            verification_method="Monte Carlo"
        )
    
    def verify_grid(
        self,
        points_per_dim: int = 10
    ) -> VerificationResult:
        """Verify safety using grid-based search.
        
        Args:
            points_per_dim: Number of grid points per dimension
            
        Returns:
            Verification result
        """
        # Create grid
        grids = [
            np.linspace(low, high, points_per_dim)
            for low, high in self.state_bounds
        ]
        
        max_violation = 0.0
        counterexample = None
        num_checked = 0
        
        # Check all grid points
        for state in self._grid_iterator(grids):
            num_checked += 1
            
            if not self.safety_property(state):
                violation = self._compute_violation(state)
                if violation > max_violation:
                    max_violation = violation
                    counterexample = state.copy()
        
        is_safe = counterexample is None
        
        return VerificationResult(
            is_safe=is_safe,
            counterexample=counterexample,
            max_violation=max_violation,
            num_samples_checked=num_checked,
            verification_method="Grid Search"
        )
    
    def _grid_iterator(self, grids: List[np.ndarray]):
        """Iterate over all points in grid."""
        if len(grids) == 0:
            yield np.array([])
            return
        
        for point in grids[0]:
            if len(grids) == 1:
                yield np.array([point])
            else:
                for rest in self._grid_iterator(grids[1:]):
                    yield np.concatenate([[point], rest])
    
    def _compute_violation(self, state: np.ndarray) -> float:
        """Compute magnitude of safety violation.
        
        Args:
            state: State to check
            
        Returns:
            Violation magnitude (0 if safe)
        """
        # This is a placeholder - should be customized based on safety property
        return 1.0


class ReachabilityAnalyzer:
    """Reachability analysis for control systems.
    
    Args:
        dynamics: Dynamics function (state, action) -> next_state
        action_bounds: Action bounds
        time_horizon: Time horizon for reachability
        dt: Time step
    """
    
    def __init__(
        self,
        dynamics: Callable,
        action_bounds: Tuple[float, float],
        time_horizon: float = 1.0,
        dt: float = 0.1
    ) -> None:
        self.dynamics = dynamics
        self.action_bounds = action_bounds
        self.time_horizon = time_horizon
        self.dt = dt
        self.num_steps = int(time_horizon / dt)
    
    def compute_reachable_set(
        self,
        initial_states: np.ndarray,
        num_action_samples: int = 10
    ) -> np.ndarray:
        """Compute forward reachable set from initial states.
        
        Args:
            initial_states: Array of initial states [num_states, state_dim]
            num_action_samples: Number of actions to sample per state
            
        Returns:
            Array of reachable states
        """
        current_states = initial_states.copy()
        
        for step in range(self.num_steps):
            next_states = []
            
            for state in current_states:
                # Sample actions
                actions = np.linspace(
                    self.action_bounds[0],
                    self.action_bounds[1],
                    num_action_samples
                )
                
                # Compute next states for all actions
                for action in actions:
                    next_state = self.dynamics(state, np.array([action]))
                    next_states.append(next_state)
            
            current_states = np.array(next_states)
        
        return current_states
    
    def is_target_reachable(
        self,
        initial_state: np.ndarray,
        target_set: Callable[[np.ndarray], bool],
        num_trajectories: int = 100
    ) -> Tuple[bool, Optional[List[np.ndarray]]]:
        """Check if target set is reachable from initial state.
        
        Args:
            initial_state: Starting state
            target_set: Function (state) -> bool indicating if state is in target
            num_trajectories: Number of trajectories to simulate
            
        Returns:
            Tuple of (is_reachable, trajectory_to_target)
        """
        for _ in range(num_trajectories):
            state = initial_state.copy()
            trajectory = [state]
            
            for step in range(self.num_steps):
                # Sample random action
                action = np.random.uniform(
                    self.action_bounds[0],
                    self.action_bounds[1]
                )
                
                # Step dynamics
                state = self.dynamics(state, np.array([action]))
                trajectory.append(state)
                
                # Check if reached target
                if target_set(state):
                    return True, trajectory
        
        return False, None


class InvariantVerifier:
    """Verify invariant properties of control systems.
    
    Args:
        invariant: Invariant property (state) -> bool
        dynamics: System dynamics
        controller: Control policy (state) -> action
    """
    
    def __init__(
        self,
        invariant: Callable[[np.ndarray], bool],
        dynamics: Callable,
        controller: Callable
    ) -> None:
        self.invariant = invariant
        self.dynamics = dynamics
        self.controller = controller
    
    def verify_trajectory(
        self,
        initial_state: np.ndarray,
        num_steps: int = 100
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Verify invariant holds along trajectory.
        
        Args:
            initial_state: Starting state
            num_steps: Number of steps to simulate
            
        Returns:
            Tuple of (invariant_holds, violating_state)
        """
        state = initial_state.copy()
        
        # Check initial state
        if not self.invariant(state):
            return False, state
        
        # Simulate trajectory
        for _ in range(num_steps):
            # Get action from controller
            action = self.controller(state)
            
            # Step dynamics
            state = self.dynamics(state, action)
            
            # Check invariant
            if not self.invariant(state):
                return False, state
        
        return True, None


def verify_lyapunov_stability(
    lyapunov_fn: Callable[[np.ndarray], float],
    dynamics: Callable,
    controller: Callable,
    state_bounds: List[Tuple[float, float]],
    num_samples: int = 1000
) -> VerificationResult:
    """Verify stability using Lyapunov function.
    
    A system is stable if V(x) > 0 for x != 0 and dV/dt < 0.
    
    Args:
        lyapunov_fn: Lyapunov candidate function
        dynamics: System dynamics
        controller: Control policy
        state_bounds: Bounds on state space
        num_samples: Number of samples for verification
        
    Returns:
        Verification result
    """
    max_violation = 0.0
    counterexample = None
    epsilon = 1e-6
    
    for _ in range(num_samples):
        # Sample random state (excluding origin)
        state = np.array([
            np.random.uniform(low, high)
            for low, high in state_bounds
        ])
        
        # Skip states near origin
        if np.linalg.norm(state) < epsilon:
            continue
        
        # Check V(x) > 0
        V = lyapunov_fn(state)
        if V <= 0:
            max_violation = max(max_violation, abs(V))
            counterexample = state.copy()
            continue
        
        # Compute dV/dt
        action = controller(state)
        state_dot = dynamics(state, action)
        
        # Numerical gradient
        V_dot = 0.0
        for i in range(len(state)):
            state_plus = state.copy()
            state_plus[i] += epsilon
            V_plus = lyapunov_fn(state_plus)
            grad_i = (V_plus - V) / epsilon
            V_dot += grad_i * state_dot[i]
        
        # Check dV/dt < 0
        if V_dot >= 0:
            max_violation = max(max_violation, V_dot)
            counterexample = state.copy()
    
    is_safe = counterexample is None
    
    return VerificationResult(
        is_safe=is_safe,
        counterexample=counterexample,
        max_violation=max_violation,
        num_samples_checked=num_samples,
        verification_method="Lyapunov Stability"
    )

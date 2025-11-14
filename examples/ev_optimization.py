"""Electric Vehicle charging optimization example using MPC."""

import sys
sys.path.insert(0, '/home/claude/rl-control-suite/src')

import numpy as np
from rl_control.algorithms import MPC
import matplotlib.pyplot as plt


class EVChargingEnv:
    """Simplified EV charging environment.
    
    State: [battery_charge, time_to_departure, electricity_price]
    Action: [charging_power] (0 to max_power)
    """
    
    def __init__(
        self,
        battery_capacity: float = 60.0,  # kWh
        max_charging_power: float = 11.0,  # kW
        target_charge: float = 80.0,  # Target % charge
        hours_to_departure: int = 8
    ):
        self.battery_capacity = battery_capacity
        self.max_charging_power = max_charging_power
        self.target_charge = target_charge
        self.hours_to_departure = hours_to_departure
        
        # Time step (15 minutes)
        self.dt = 0.25  # hours
        self.max_steps = int(hours_to_departure / self.dt)
        
        # Initialize state
        self.current_charge = 0.0
        self.current_step = 0
        
        # Electricity pricing (time-of-use)
        self.base_price = 0.15  # $/kWh
    
    def reset(self):
        """Reset environment."""
        self.current_charge = np.random.uniform(10.0, 30.0)  # Start at 10-30%
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state."""
        time_remaining = (self.max_steps - self.current_step) * self.dt
        price = self._get_electricity_price()
        
        return np.array([
            self.current_charge / 100.0,  # Normalized charge %
            time_remaining / self.hours_to_departure,  # Normalized time
            price / self.base_price  # Normalized price
        ])
    
    def _get_electricity_price(self):
        """Get current electricity price based on time."""
        hour = (self.current_step * self.dt) % 24
        
        # Peak hours (expensive)
        if 16 <= hour < 21:
            return self.base_price * 2.0
        # Off-peak hours (cheap)
        elif 23 <= hour or hour < 7:
            return self.base_price * 0.5
        # Mid-peak
        else:
            return self.base_price
    
    def step(self, action):
        """Take environment step."""
        # Clip charging power
        charging_power = np.clip(action[0], 0.0, self.max_charging_power)
        
        # Update charge (simplified charging model)
        charge_added = (charging_power * self.dt / self.battery_capacity) * 100
        self.current_charge = min(100.0, self.current_charge + charge_added)
        
        # Calculate cost
        price = self._get_electricity_price()
        cost = charging_power * self.dt * price
        
        # Calculate reward
        # Penalize deviation from target and high costs
        charge_error = abs(self.current_charge - self.target_charge)
        reward = -0.1 * charge_error - cost
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Bonus for reaching target at departure
        if done and abs(self.current_charge - self.target_charge) < 5.0:
            reward += 10.0
        
        return self._get_state(), reward, done, {}
    
    @property
    def observation_space(self):
        """Observation space."""
        class Space:
            shape = (3,)
        return Space()
    
    @property
    def action_space(self):
        """Action space."""
        class Space:
            shape = (1,)
            low = np.array([0.0])
            high = np.array([11.0])
        return Space()


def optimize_ev_charging():
    """Optimize EV charging using MPC."""
    print("="*60)
    print("EV Charging Optimization with Model Predictive Control")
    print("="*60)
    
    # Create environment
    env = EVChargingEnv(
        battery_capacity=60.0,
        max_charging_power=11.0,
        target_charge=80.0,
        hours_to_departure=8
    )
    
    print("\nEV Specifications:")
    print(f"  Battery capacity: {env.battery_capacity} kWh")
    print(f"  Max charging power: {env.max_charging_power} kW")
    print(f"  Target charge: {env.target_charge}%")
    print(f"  Time to departure: {env.hours_to_departure} hours")
    
    # Define cost function for MPC
    def ev_cost_function(state, action):
        """Cost function for EV charging optimization."""
        charge_pct = state[0] * 100
        time_remaining = state[1] * env.hours_to_departure
        price_normalized = state[2]
        
        charging_power = action[0]
        
        # Costs
        # 1. Electricity cost
        electricity_cost = charging_power * price_normalized * env.base_price * env.dt
        
        # 2. Penalty for not reaching target
        target_error = (charge_pct - env.target_charge) ** 2
        target_penalty = 0.01 * target_error
        
        # 3. Penalty for overcharging
        overcharge_penalty = max(0, charge_pct - 100) ** 2
        
        # 4. Time urgency (charge faster if time is running out)
        urgency_factor = max(0, env.target_charge - charge_pct) / max(time_remaining, 0.1)
        urgency_penalty = 0.001 * urgency_factor
        
        total_cost = electricity_cost + target_penalty + overcharge_penalty - urgency_penalty
        
        return total_cost
    
    # Define dynamics model
    def ev_dynamics(state, action):
        """EV charging dynamics."""
        charge_pct, time_remaining, price = state
        charging_power = np.clip(action[0], 0.0, env.max_charging_power)
        
        # Update charge
        charge_added = (charging_power * env.dt / env.battery_capacity)
        new_charge = min(1.0, charge_pct + charge_added)
        
        # Update time
        new_time_remaining = max(0.0, time_remaining - env.dt / env.hours_to_departure)
        
        # Price might change (simplified)
        new_price = price
        
        return np.array([new_charge, new_time_remaining, new_price])
    
    # Create MPC controller
    mpc = MPC(
        state_dim=3,
        action_dim=1,
        horizon=15,  # 15-step lookahead
        cost_function=ev_cost_function,
        dynamics_model=ev_dynamics,
        action_bounds=(0.0, env.max_charging_power)
    )
    
    print("\nMPC Configuration:")
    print(f"  Horizon: {mpc.horizon} steps")
    print(f"  Time step: {env.dt} hours")
    print(f"  Lookahead: {mpc.horizon * env.dt} hours")
    
    # Run optimization
    print("\n" + "="*60)
    print("Running EV Charging Optimization...")
    print("="*60)
    
    state = env.reset()
    
    # Storage for visualization
    charges = [env.current_charge]
    actions = []
    prices = []
    times = []
    costs = []
    
    total_cost = 0.0
    total_energy = 0.0
    
    print(f"\nInitial charge: {env.current_charge:.1f}%")
    print("\nCharging schedule:")
    print(f"{'Time (h)':>10} {'Charge (%)':>12} {'Power (kW)':>12} {'Price ($/kWh)':>15} {'Step Cost ($)':>15}")
    print("-" * 70)
    
    step = 0
    done = False
    
    while not done:
        # Compute optimal action using MPC
        action = mpc.compute_action(state)
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        
        # Track metrics
        current_time = step * env.dt
        current_price = env._get_electricity_price()
        step_cost = action[0] * env.dt * current_price
        
        charges.append(env.current_charge)
        actions.append(action[0])
        prices.append(current_price)
        times.append(current_time)
        costs.append(step_cost)
        
        total_cost += step_cost
        total_energy += action[0] * env.dt
        
        # Print progress
        if step % 4 == 0:  # Every hour
            print(f"{current_time:10.2f} {env.current_charge:12.1f} {action[0]:12.2f} {current_price:15.3f} {step_cost:15.3f}")
        
        state = next_state
        step += 1
    
    print("-" * 70)
    print(f"\nFinal charge: {env.current_charge:.1f}%")
    print(f"Target charge: {env.target_charge}%")
    print(f"Charge error: {abs(env.current_charge - env.target_charge):.1f}%")
    print(f"\nTotal energy used: {total_energy:.2f} kWh")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Average price: ${total_cost/total_energy:.3f}/kWh")
    
    # Visualize results
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Charge level over time
    axes[0].plot(times, charges[:-1], 'b-', linewidth=2, label='Battery Charge')
    axes[0].axhline(y=env.target_charge, color='r', linestyle='--', label='Target Charge')
    axes[0].set_ylabel('Battery Charge (%)', fontsize=12)
    axes[0].set_title('EV Charging Optimization Results', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, env.hours_to_departure)
    
    # Plot 2: Charging power over time
    axes[1].bar(times, actions, width=env.dt*0.9, color='green', alpha=0.7, label='Charging Power')
    axes[1].set_ylabel('Charging Power (kW)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, env.hours_to_departure)
    
    # Plot 3: Electricity price over time
    axes[2].step(times, prices, 'r-', linewidth=2, where='post', label='Electricity Price')
    axes[2].set_xlabel('Time (hours)', fontsize=12)
    axes[2].set_ylabel('Price ($/kWh)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(0, env.hours_to_departure)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = 'ev_charging_optimization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Don't show plot in non-interactive environment
    # plt.show()
    
    return total_cost, total_energy


if __name__ == "__main__":
    optimize_ev_charging()

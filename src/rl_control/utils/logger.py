"""Logging utilities for training and evaluation."""

import os
import json
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


class Logger:
    """Training logger for RL experiments.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of experiment
        config: Experiment configuration dictionary
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> None:
        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save configuration
        if config is not None:
            self.save_config(config)
        
        # Initialize metrics storage
        self.metrics: Dict[str, List[Any]] = {}
        self.scalar_file = os.path.join(self.log_dir, "scalars.json")
        
        print(f"Logger initialized. Logs will be saved to: {self.log_dir}")
    
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log scalar value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if key not in self.metrics:
            self.metrics[key] = []
        
        entry = {"value": value}
        if step is not None:
            entry["step"] = step
        
        self.metrics[key].append(entry)
    
    def log_scalars(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar values.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Optional step number
        """
        for key, value in metrics_dict.items():
            self.log_scalar(key, value, step)
    
    def log_histogram(self, key: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log histogram of values.
        
        Args:
            key: Metric name
            values: Array of values
            step: Optional step number
        """
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
        
        hist_key = f"{key}_histogram"
        if hist_key not in self.metrics:
            self.metrics[hist_key] = []
        
        entry = {"stats": stats}
        if step is not None:
            entry["step"] = step
        
        self.metrics[hist_key].append(entry)
    
    def save_config(self, config: Dict) -> None:
        """Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_metrics(self) -> None:
        """Save all metrics to disk."""
        with open(self.scalar_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_object(self, obj: Any, filename: str) -> None:
        """Save Python object using pickle.
        
        Args:
            obj: Object to save
            filename: Filename (without path)
        """
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    
    def load_object(self, filename: str) -> Any:
        """Load Python object using pickle.
        
        Args:
            filename: Filename (without path)
            
        Returns:
            Loaded object
        """
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_metric(self, key: str) -> List[Any]:
        """Get metric values.
        
        Args:
            key: Metric name
            
        Returns:
            List of metric entries
        """
        return self.metrics.get(key, [])
    
    def print_summary(self) -> None:
        """Print summary of logged metrics."""
        print("\n" + "="*50)
        print(f"Experiment: {os.path.basename(self.log_dir)}")
        print("="*50)
        
        for key, values in self.metrics.items():
            if values and isinstance(values[0], dict) and "value" in values[0]:
                latest = values[-1]["value"]
                mean = np.mean([v["value"] for v in values])
                print(f"{key:30s}: Latest={latest:10.4f}, Mean={mean:10.4f}")
        
        print("="*50 + "\n")


class ConsoleLogger:
    """Simple console logger for quick experiments.
    
    Args:
        print_every: Print frequency (in steps)
    """
    
    def __init__(self, print_every: int = 100) -> None:
        self.print_every = print_every
        self.step = 0
        self.episode_rewards: List[float] = []
        self.metrics_buffer: Dict[str, List[float]] = {}
    
    def log_scalar(self, key: str, value: float) -> None:
        """Log scalar value.
        
        Args:
            key: Metric name
            value: Metric value
        """
        if key not in self.metrics_buffer:
            self.metrics_buffer[key] = []
        self.metrics_buffer[key].append(value)
        
        self.step += 1
        
        # Print periodically
        if self.step % self.print_every == 0:
            self._print_metrics()
    
    def log_episode_reward(self, reward: float) -> None:
        """Log episode reward.
        
        Args:
            reward: Episode return
        """
        self.episode_rewards.append(reward)
    
    def _print_metrics(self) -> None:
        """Print current metrics."""
        print(f"\nStep {self.step}:")
        
        for key, values in self.metrics_buffer.items():
            if values:
                mean_value = np.mean(values[-self.print_every:])
                print(f"  {key}: {mean_value:.4f}")
        
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-10:]
            print(f"  Recent Episode Rewards (last 10): {np.mean(recent_rewards):.2f}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        for key, values in self.metrics_buffer.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
        
        if self.episode_rewards:
            stats["mean_episode_reward"] = np.mean(self.episode_rewards)
            stats["std_episode_reward"] = np.std(self.episode_rewards)
        
        return stats


class TensorBoardLogger:
    """TensorBoard logger wrapper.
    
    Args:
        log_dir: Directory to save TensorBoard logs
        experiment_name: Name of experiment
    """
    
    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            if experiment_name is None:
                experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.log_dir = os.path.join(log_dir, experiment_name)
            self.writer = SummaryWriter(self.log_dir)
            print(f"TensorBoard logging to: {self.log_dir}")
            print(f"Run: tensorboard --logdir={log_dir}")
        except ImportError:
            print("Warning: TensorBoard not installed. Install with: pip install tensorboard")
            self.writer = None
    
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log scalar value to TensorBoard.
        
        Args:
            key: Metric name
            value: Metric value
            step: Step number
        """
        if self.writer is not None:
            self.writer.add_scalar(key, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple related scalars.
        
        Args:
            main_tag: Main tag name
            tag_scalar_dict: Dictionary of tags and values
            step: Step number
        """
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, key: str, values: np.ndarray, step: int) -> None:
        """Log histogram to TensorBoard.
        
        Args:
            key: Histogram name
            values: Array of values
            step: Step number
        """
        if self.writer is not None:
            self.writer.add_histogram(key, values, step)
    
    def close(self) -> None:
        """Close the writer."""
        if self.writer is not None:
            self.writer.close()

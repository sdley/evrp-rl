"""
Experiment Runner for RL Training and Evaluation.

Handles training loops, evaluation, metrics logging, and checkpointing.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

from evrp_rl.env import EVRPEnvironment
from evrp_rl.agents import BaseAgent
from .core import EnvFactory, AgentFactory, RewardModule, MaskModule


class MetricsLogger:
    """
    Logger for tracking training and evaluation metrics.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train': {
                'episodes': [],
                'rewards': [],
                'lengths': [],
                'actor_losses': [],
                'critic_losses': [],
                'entropies': [],
            },
            'eval': {
                'episodes': [],
                'rewards': [],
                'lengths': [],
                'route_lengths': [],
                'charge_visits': [],
                'success_rate': [],
            }
        }
    
    def log_train_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        metrics: Dict[str, float],
    ):
        """Log training episode metrics."""
        self.metrics['train']['episodes'].append(episode)
        self.metrics['train']['rewards'].append(reward)
        self.metrics['train']['lengths'].append(length)
        self.metrics['train']['actor_losses'].append(metrics.get('actor_loss', 0.0))
        self.metrics['train']['critic_losses'].append(metrics.get('critic_loss', 0.0))
        self.metrics['train']['entropies'].append(metrics.get('entropy', 0.0))
    
    def log_eval_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        route_length: float,
        charge_visits: int,
        success: bool,
    ):
        """Log evaluation episode metrics."""
        self.metrics['eval']['episodes'].append(episode)
        self.metrics['eval']['rewards'].append(reward)
        self.metrics['eval']['lengths'].append(length)
        self.metrics['eval']['route_lengths'].append(route_length)
        self.metrics['eval']['charge_visits'].append(charge_visits)
        self.metrics['eval']['success_rate'].append(1.0 if success else 0.0)
    
    def get_recent_stats(self, mode: str = 'train', window: int = 100) -> Dict[str, float]:
        """
        Get statistics for recent episodes.
        
        Args:
            mode: 'train' or 'eval'
            window: Number of recent episodes to consider
        
        Returns:
            Dictionary of statistics
        """
        metrics = self.metrics[mode]
        
        if len(metrics['rewards']) == 0:
            return {}
        
        recent_rewards = metrics['rewards'][-window:]
        recent_lengths = metrics['lengths'][-window:]
        
        stats = {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_length': np.mean(recent_lengths),
        }
        
        if mode == 'eval':
            stats['mean_route_length'] = np.mean(metrics['route_lengths'][-window:])
            stats['mean_charge_visits'] = np.mean(metrics['charge_visits'][-window:])
            stats['success_rate'] = np.mean(metrics['success_rate'][-window:])
        
        return stats
    
    def save(self, filename: str = 'metrics.json'):
        """Save metrics to JSON file."""
        save_path = self.log_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Rewards
        axes[0, 0].plot(self.metrics['train']['episodes'], self.metrics['train']['rewards'])
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].grid(True)
        
        # Actor loss
        axes[0, 1].plot(self.metrics['train']['episodes'], self.metrics['train']['actor_losses'])
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Actor Loss')
        axes[0, 1].grid(True)
        
        # Critic loss
        axes[1, 0].plot(self.metrics['train']['episodes'], self.metrics['train']['critic_losses'])
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].grid(True)
        
        # Entropy
        axes[1, 1].plot(self.metrics['train']['episodes'], self.metrics['train']['entropies'])
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(self.log_dir / 'training_curves.png')
        
        plt.close()


class ExperimentRunner:
    """
    Main experiment runner for training and evaluation.
    
    Features:
    - Training and evaluation modes
    - Automatic checkpointing
    - Metrics logging
    - Early stopping
    - Multi-experiment support
    """
    
    def __init__(
        self,
        env: EVRPEnvironment,
        agent: BaseAgent,
        config: Dict[str, Any],
        log_dir: str = 'results',
        checkpoint_dir: str = 'checkpoints',
    ):
        """
        Initialize experiment runner.
        
        Args:
            env: EVRP environment
            agent: RL agent
            config: Experiment configuration
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.env = env
        self.agent = agent
        self.config = config
        
        # Setup directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = config.get('run', {}).get('name', 'experiment')
        
        self.log_dir = Path(log_dir) / f"{exp_name}_{timestamp}"
        self.checkpoint_dir = Path(checkpoint_dir) / f"{exp_name}_{timestamp}"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = MetricsLogger(str(self.log_dir))
        
        # Optional reward shaping and masking
        reward_config = config.get('reward', {})
        self.reward_module = RewardModule(reward_config) if reward_config else None
        
        mask_config = config.get('mask', {})
        self.mask_module = MaskModule(mask_config) if mask_config else None
        
        # Training config
        run_config = config.get('run', {})
        self.num_epochs = run_config.get('epochs', 100)
        self.eval_frequency = run_config.get('eval_frequency', 10)
        self.save_frequency = run_config.get('save_frequency', 20)
        self.max_steps_per_episode = run_config.get('max_steps_per_episode', 200)
        self.num_eval_episodes = run_config.get('num_eval_episodes', 10)
        
        # Best model tracking
        self.best_reward = -float('inf')
        self.best_model_path = None
    
    def train_episode(self) -> Tuple[float, int, Dict[str, float]]:
        """
        Run one training episode.
        
        Returns:
            Tuple of (total_reward, episode_length, metrics)
        """
        obs, _ = self.env.reset()
        
        rollout = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
        }
        
        total_reward = 0.0
        episode_length = 0
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action, _ = self.agent.select_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Apply reward shaping if configured
            if self.reward_module:
                reward = self.reward_module(reward, action, obs, next_obs, done, info)
            
            # Store transition
            rollout['observations'].append(obs)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['next_observations'].append(next_obs)
            rollout['dones'].append(done)
            
            total_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # Update agent
        metrics = self.agent.update(rollout)
        
        return total_reward, episode_length, metrics
    
    def eval_episode(self) -> Tuple[float, int, Dict[str, Any]]:
        """
        Run one evaluation episode.
        
        Returns:
            Tuple of (total_reward, episode_length, info_dict)
        """
        obs, _ = self.env.reset()
        
        total_reward = 0.0
        episode_length = 0
        route_length = 0.0
        charge_visits = 0
        
        trajectory = []
        
        for step in range(self.max_steps_per_episode):
            # Select action deterministically
            action, _ = self.agent.select_action(obs, deterministic=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            episode_length += 1
            
            # Track metrics
            if 'distance' in info:
                route_length += info['distance']
            
            if info.get('node_type') == 'charger':
                charge_visits += 1
            
            trajectory.append({
                'state': obs,
                'action': action,
                'reward': reward,
            })
            
            obs = next_obs
            
            if done:
                break
        
        success = info.get('all_customers_visited', False)
        
        return total_reward, episode_length, {
            'route_length': route_length,
            'charge_visits': charge_visits,
            'success': success,
            'trajectory': trajectory,
        }
    
    def train(self):
        """
        Main training loop.
        """
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Log dir: {self.log_dir}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Training episode
            reward, length, metrics = self.train_episode()
            
            # Log training metrics
            self.logger.log_train_episode(epoch, reward, length, metrics)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                train_stats = self.logger.get_recent_stats('train', window=10)
                if train_stats:
                    # Get actor/loss metric if available (varies by agent type)
                    loss_value = metrics.get('actor_loss', metrics.get('policy_loss', metrics.get('q_loss', 0.0))) if metrics else 0.0
                    loss_str = f"{loss_value:.4f}" if loss_value else "N/A"
                    print(f"Epoch {epoch + 1}/{self.num_epochs} | "
                          f"Reward: {train_stats['mean_reward']:.2f} ± {train_stats['std_reward']:.2f} | "
                          f"Length: {train_stats['mean_length']:.1f} | "
                          f"Loss: {loss_str}")
            
            # Evaluation
            if (epoch + 1) % self.eval_frequency == 0:
                print(f"\nEvaluating at epoch {epoch + 1}...")
                eval_rewards = []
                eval_info = []
                
                for _ in range(self.num_eval_episodes):
                    eval_reward, eval_length, info = self.eval_episode()
                    eval_rewards.append(eval_reward)
                    eval_info.append(info)
                    
                    self.logger.log_eval_episode(
                        epoch,
                        eval_reward,
                        eval_length,
                        info['route_length'],
                        info['charge_visits'],
                        info['success'],
                    )
                
                # Print eval stats
                eval_stats = self.logger.get_recent_stats('eval', window=self.num_eval_episodes)
                if eval_stats:
                    print(f"  Eval Reward: {eval_stats['mean_reward']:.2f}")
                    print(f"  Success Rate: {eval_stats['success_rate']:.2%}")
                    print(f"  Avg Route Length: {eval_stats['mean_route_length']:.2f}")
                    print(f"  Avg Charge Visits: {eval_stats['mean_charge_visits']:.1f}")
                
                # Save best model
                mean_eval_reward = np.mean(eval_rewards)
                if mean_eval_reward > self.best_reward:
                    self.best_reward = mean_eval_reward
                    self.best_model_path = self.checkpoint_dir / 'best_model.pt'
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best model saved! (reward: {self.best_reward:.2f})")
                
                print("-" * 60)
            
            # Periodic checkpointing
            if (epoch + 1) % self.save_frequency == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch + 1}.pt'
                self.save_checkpoint(checkpoint_name)
        
        # Final save
        self.save_checkpoint('final_model.pt')
        self.logger.save()
        self.logger.plot_training_curves()
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time/60:.2f} minutes")
        print(f"Best reward: {self.best_reward:.2f}")
        print(f"Logs saved to: {self.log_dir}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        all_rewards = []
        all_lengths = []
        all_route_lengths = []
        all_charge_visits = []
        successes = []
        
        for ep in range(num_episodes):
            reward, length, info = self.eval_episode()
            
            all_rewards.append(reward)
            all_lengths.append(length)
            all_route_lengths.append(info['route_length'])
            all_charge_visits.append(info['charge_visits'])
            successes.append(1.0 if info['success'] else 0.0)
            
            if (ep + 1) % 10 == 0:
                print(f"  Progress: {ep + 1}/{num_episodes}")
        
        results = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_episode_length': np.mean(all_lengths),
            'mean_route_length': np.mean(all_route_lengths),
            'std_route_length': np.std(all_route_lengths),
            'mean_charge_visits': np.mean(all_charge_visits),
            'success_rate': np.mean(successes),
        }
        
        print("\nEvaluation Results:")
        print("-" * 60)
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        # Save results
        results_path = self.log_dir / 'eval_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        self.agent.save(str(checkpoint_path))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        self.agent.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")


def run_experiment(config: Dict[str, Any]) -> ExperimentRunner:
    """
    Run a complete experiment from configuration.
    
    Args:
        config: Experiment configuration dict
    
    Returns:
        ExperimentRunner instance with trained agent
    
    Example:
        >>> config = {
        ...     'env': {'num_customers': 20, 'num_chargers': 5},
        ...     'agent': {'type': 'sac', 'encoder': {'type': 'gat'}},
        ...     'run': {'epochs': 100}
        ... }
        >>> runner = run_experiment(config)
    """
    # Create environment
    env = EnvFactory.create(config['env'])
    
    # Create agent
    agent = AgentFactory.create(config['agent'], env.action_space.n)
    
    # Create runner
    runner = ExperimentRunner(env, agent, config)
    
    # Train
    runner.train()
    
    return runner

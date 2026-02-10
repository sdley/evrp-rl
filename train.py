"""
Training script for EVRP agents.

Usage:
    python train.py --config configs/experiment_config.yaml
    python train.py --config configs/experiment_config.yaml --device cuda

Tip: set `agent.type` inside the YAML (`a2c`, `sac`, etc.) to switch algorithms.
"""

import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any

from src.env import EVRPEnvironment
from src.agents import AgentFactory


class Trainer:
    """
    Trainer for EVRP agents.
    
    Handles:
    - Environment creation
    - Agent initialization
    - Training loop
    - Evaluation
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        config_path: str,
        device: str = 'cpu',
        seed: int = 42,
    ):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
            device: Device to use ('cpu' or 'cuda')
            seed: Random seed
        """
        self.config_path = config_path
        self.device = torch.device(device)
        self.seed = seed
        
        # Load configuration (support unified config with 'env', 'agent', 'run')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Environment configuration: prefer `env` section, fall back to legacy `training`
        env_config = self.config.get('env', self.config.get('training', {}))
        self.env = EVRPEnvironment(
            num_customers=env_config.get('num_customers', 10),
            num_chargers=env_config.get('num_chargers', 3),
            max_battery=env_config.get('battery_capacity', env_config.get('max_battery', 100.0)),
            max_cargo=env_config.get('cargo_capacity', env_config.get('max_cargo', 100.0)),
            time_limit=env_config.get('time_limit', env_config.get('max_steps_per_episode', 200)),
            seed=seed,
        )
        
        # Create agent
        action_dim = self.env.action_space.n
        self.agent = AgentFactory.create_from_dict(self.config, action_dim)
        self.agent.to(self.device)
        
        # Run/training configuration: prefer `run` section, fall back to legacy `training`
        run_config = self.config.get('run', self.config.get('training', {}))
        self.num_episodes = run_config.get('epochs', run_config.get('num_episodes', 1000))
        self.max_steps = run_config.get('max_steps_per_episode', env_config.get('time_limit', 200))
        self.eval_frequency = run_config.get('eval_frequency', 50)
        self.save_frequency = run_config.get('save_frequency', 100)

        # Resolve agent type string for conditional logic and logging
        agent_section = self.config.get('agent', {})
        if isinstance(agent_section, dict):
            self.agent_type = agent_section.get('type', 'a2c').lower()
        else:
            self.agent_type = str(agent_section).lower()
        
        # Create directories
        self.log_dir = Path(env_config.get('log_dir', 'results'))
        self.save_dir = Path(env_config.get('save_dir', 'checkpoints'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
    
    def train(self):
        """Run training loop."""
        print(f"Starting training with {self.agent_type.upper()} agent")
        print(f"Device: {self.device}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Environment: {self.env.num_customers} customers, {self.env.num_chargers} chargers")
        print("=" * 70)
        
        for episode in range(self.num_episodes):
            episode_reward, episode_length, metrics = self._run_episode(episode)
            
            # Log episode
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if metrics:
                self.training_metrics.append(metrics)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode + 1}/{self.num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Length: {episode_length} | "
                      f"Avg(10): {avg_reward:.2f}")
            
            # Evaluate
            if (episode + 1) % self.eval_frequency == 0:
                eval_reward = self._evaluate()
                print(f"  Evaluation: {eval_reward:.2f}")
            
            # Save checkpoint
            if (episode + 1) % self.save_frequency == 0:
                self._save_checkpoint(episode + 1)
        
        # Final save
        self._save_checkpoint('final')
        self._save_results()
        
        print("=" * 70)
        print("Training completed!")
        print(f"Final average reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
    
    def _run_episode(self, episode: int) -> tuple:
        """
        Run one episode.
        
        Returns:
            Tuple of (episode_reward, episode_length, training_metrics)
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # For A2C: collect rollout
        if self.agent_type == 'a2c':
            rollout = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'dones': [],
                'log_probs': [],
                'values': [],
            }
        
        for step in range(self.max_steps):
            # Select action
            action, action_info = self.agent.select_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            if self.agent_type == 'sac':
                self.agent.store_transition(obs, action, reward, next_obs, done)
            elif self.agent_type == 'a2c':
                rollout['observations'].append(obs)
                rollout['actions'].append(action)
                rollout['rewards'].append(reward)
                rollout['next_observations'].append(next_obs)
                rollout['dones'].append(done)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # Update agent
        metrics = {}
        if self.agent_type == 'a2c' and len(rollout['observations']) > 0:
            metrics = self.agent.update(rollout)
        elif self.agent_type == 'sac':
            train_freq = self.config.get('training', {}).get('train_frequency', 1)
            if episode % train_freq == 0:
                metrics = self.agent.update({})
        
        self.agent.episode_end({'reward': episode_reward, 'length': episode_length})
        
        return episode_reward, episode_length, metrics
    
    def _evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate agent.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Average evaluation reward
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                action, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def _save_checkpoint(self, episode):
        """Save agent checkpoint."""
        checkpoint_path = self.save_dir / f"agent_episode_{episode}.pt"
        self.agent.save(str(checkpoint_path))
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    def _save_results(self):
        """Save training results."""
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'config': self.config,
        }
        
        import pickle
        results_path = self.log_dir / 'training_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save as numpy arrays
        np.save(self.log_dir / 'rewards.npy', np.array(self.episode_rewards))
        np.save(self.log_dir / 'lengths.npy', np.array(self.episode_lengths))
        
        print(f"  Saved results to {self.log_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train EVRP agent')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = Trainer(
        config_path=args.config,
        device=args.device,
        seed=args.seed,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()

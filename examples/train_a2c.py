"""
Example: Train A2C agent on EVRP.

This script demonstrates training an A2C agent using the configuration file.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import EVRPEnvironment
from src.agents import AgentFactory
import torch
import numpy as np


def train_a2c_simple():
    """Simple A2C training example."""
    print("=" * 70)
    print("A2C Training Example")
    print("=" * 70)
    
    # Create environment
    env = EVRPEnvironment(num_customers=5, num_chargers=2, seed=42)
    
    # Create agent from config
    config = {
        'agent': 'a2c',
        'encoder': {
            'type': 'mlp',  # Use MLP for faster training
            'embed_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
        },
        'hyperparameters': {
            'lr': 3e-4,
            'gamma': 0.99,
            'entropy_coef': 0.01,
            'value_loss_coef': 0.5,
            'hidden_dim': 128,
        }
    }
    
    agent = AgentFactory.create_from_dict(config, env.action_space.n)
    
    # Training loop
    num_episodes = 50
    episode_rewards = []
    
    print(f"\nTraining for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        # Collect rollout
        rollout = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
        }
        
        for step in range(100):
            # Select action
            action, _ = agent.select_action(obs, deterministic=False)
            
            # Step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rollout['observations'].append(obs)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['next_observations'].append(next_obs)
            rollout['dones'].append(done)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Update agent
        if len(rollout['observations']) > 0:
            metrics = agent.update(rollout)
            
            if episode % 10 == 0:
                print(f"Episode {episode}: reward={episode_reward:.2f}, "
                      f"actor_loss={metrics.get('actor_loss', 0):.4f}, "
                      f"critic_loss={metrics.get('critic_loss', 0):.4f}")
        
        episode_rewards.append(episode_reward)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)
    
    eval_rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
    
    print(f"Evaluation reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Training reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")


if __name__ == '__main__':
    train_a2c_simple()

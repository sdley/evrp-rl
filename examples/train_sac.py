"""
Example: Train SAC agent on EVRP.

This script demonstrates training a SAC agent with replay buffer.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import EVRPEnvironment
from src.agents import AgentFactory
import torch
import numpy as np


def train_sac_simple():
    """Simple SAC training example."""
    print("=" * 70)
    print("SAC Training Example")
    print("=" * 70)
    
    # Create environment
    env = EVRPEnvironment(num_customers=5, num_chargers=2, seed=42)
    
    # Create agent from config
    config = {
        'agent': 'sac',
        'encoder': {
            'type': 'mlp',  # Use MLP for faster training
            'embed_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
        },
        'hyperparameters': {
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 'auto',
            'buffer_size': 10000,
            'batch_size': 64,
            'hidden_dim': 128,
        }
    }
    
    agent = AgentFactory.create_from_dict(config, env.action_space.n)
    
    # Training loop
    num_episodes = 100
    warmup_episodes = 10
    episode_rewards = []
    
    print(f"\nTraining for {num_episodes} episodes (warmup: {warmup_episodes})...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):
            # Select action (random during warmup)
            if episode < warmup_episodes:
                valid_actions = np.where(obs['valid_actions_mask'])[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
                action_info = {}
            else:
                action, action_info = agent.select_action(obs, deterministic=False)
            
            # Step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store in replay buffer
            agent.store_transition(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Update agent (after warmup)
        metrics = {}
        if episode >= warmup_episodes:
            metrics = agent.update({})
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: reward={episode_reward:.2f}, "
                  f"buffer_size={len(agent.replay_buffer)}, "
                  f"actor_loss={metrics.get('actor_loss', 0):.4f}")
    
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
    print(f"Final alpha: {agent.alpha:.4f}")


if __name__ == '__main__':
    train_sac_simple()

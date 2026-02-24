"""
Optimized training script for smooth S-curve learning like the orange curve example.

This script demonstrates how to achieve training curves with:
1. Steep initial learning phase (rapid improvement in first few hundred episodes)
2. Smooth plateau (gradual convergence to optimal)
3. Minimal oscillations (stable loss and reward curves)

Key optimizations:
- Reward normalization wrapper for bounded, smooth rewards
- Learning rate decay schedule (fast early, slow late)
- Larger batch collection (2048 steps instead of 512)
- Entropy decay schedule (explore → exploit)
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.evrp_env import EVRPEnvironment
from src.env.wrappers import CompositeRewardWrapper
from src.framework.training_utils import (
    exponential_decay_schedule,
    update_optimizer_lr,
    get_current_lr,
    entropy_decay_schedule
)


def create_smoothed_env(env_config: Dict, use_normalization: bool = True) -> EVRPEnvironment:
    """
    Create an EVRP environment with optional reward normalization wrapper.
    
    Args:
        env_config: Environment configuration
        use_normalization: Whether to apply reward normalization wrapper
        
    Returns:
        Environment (possibly wrapped)
    """
    env = EVRPEnvironment(**env_config)
    
    if use_normalization:
        # Apply composite wrapper: Scale → Normalize → Clip
        # This produces smooth, bounded rewards
        env = CompositeRewardWrapper(
            env,
            scale=0.1,  # Scale rewards down by 10x
            update_every=100,  # Update normalization stats every 100 episodes
            clip_min=-3.0,
            clip_max=3.0
        )
    
    return env


def train_with_optimization(
    agent_name: str = 'a2c',
    env_config: Optional[Dict] = None,
    agent_config: Optional[Dict] = None,
    max_episodes: int = 10_000,
    seed: int = 42,
    use_reward_normalization: bool = True,
    use_lr_decay: bool = True,
    use_entropy_decay: bool = True,
    batch_size: int = 2048,
    eval_interval: int = 20,
    early_stopping_patience: int = 100,
) -> Dict:
    """
    Train agent with all smoothing optimizations for S-curve learning dynamics.
    
    Args:
        agent_name: 'a2c' or 'sac'
        env_config: Environment config dict
        agent_config: Agent config dict
        max_episodes: Maximum training episodes
        seed: Random seed
        use_reward_normalization: Apply reward wrapper
        use_lr_decay: Apply learning rate decay
        use_entropy_decay: Apply entropy decay
        batch_size: Rollout batch size (larger = smoother gradients)
        eval_interval: Evaluation frequency
        early_stopping_patience: Episodes without improvement before stopping
        
    Returns:
        History dict with training metrics
    """
    
    # Default configs
    if env_config is None:
        env_config = {
            'num_customers': 20,
            'num_chargers': 5,
            'max_battery': 500.0,
            'seed': seed
        }
    
    if agent_config is None:
        agent_config = {
            'agent_type': agent_name,
            'state_dim': 128,
            'action_dim': 25,  # Will be set after env creation
            'hidden_dim': 256,
            'learning_rate': 1e-3,
            'entropy_coef': 0.002,
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environments
    train_env = create_smoothed_env(env_config, use_normalization=use_reward_normalization)
    eval_env = create_smoothed_env(env_config, use_normalization=False)  # No normalization for eval
    
    # Create agent - Import here to avoid circular dependency
    from src.agents.agent_factory import AgentFactory
    action_dim = train_env.action_space.n if hasattr(train_env.action_space, 'n') else train_env.action_space.shape[0]
    agent_config['action_dim'] = action_dim
    agent = AgentFactory.create_from_dict(agent_config, action_dim)
    agent.to(device)
    
    # Setup schedules
    lr_schedule = None
    entropy_schedule = None
    
    if use_lr_decay:
        # Exponential decay: 1e-3 → ~1e-4 over 10k episodes
        lr_schedule = exponential_decay_schedule(
            initial_lr=1e-3,
            decay_rate=0.9,  # 10% decay per 1000 episodes
            decay_steps=1000
        )
    
    if use_entropy_decay and hasattr(agent, 'entropy_coef'):
        # Entropy decay: explore early, exploit late
        entropy_schedule = entropy_decay_schedule(
            initial_entropy=0.002,
            decay_rate=0.95,
            decay_steps=500
        )
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    eval_history = []
    best_eval_reward = -float('inf')
    no_improvement_count = 0
    early_stopped = False
    
    print(f"🚀 Starting optimized training: {agent_name.upper()}")
    print(f"   Reward Normalization: {use_reward_normalization}")
    print(f"   LR Decay: {use_lr_decay}")
    print(f"   Entropy Decay: {use_entropy_decay}")
    print(f"   Batch Size: {batch_size}")
    print("-" * 70)
    
    for episode in range(max_episodes):
        # --- Collect rollout ---
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        log_prob_buffer = []
        value_buffer = []
        
        obs, _ = train_env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 200 and len(obs_buffer) < batch_size:
            # Select action
            obs_np = {k: np.array(v) if not isinstance(v, np.ndarray) else v 
                     for k, v in obs.items()} if isinstance(obs, dict) else np.array(obs)
            
            action, action_info = agent.select_action(obs_np, deterministic=False)
            action = int(action.item() if isinstance(action, torch.Tensor) else action)
            
            # Store experience
            obs_buffer.append(obs_np)
            action_buffer.append(action)
            log_prob_buffer.append(action_info.get('log_prob', 0.0))
            value_buffer.append(action_info.get('value', 0.0))
            
            # Environment step
            obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            
            reward_buffer.append(float(reward))
            done_buffer.append(done)
            episode_reward += reward
            steps += 1
        
        # --- Training update ---
        if len(obs_buffer) > 0:
            agent.train()
            
            # Update learning rate if schedule enabled
            if lr_schedule and hasattr(agent, 'optimizer'):
                new_lr = lr_schedule(episode)
                update_optimizer_lr(agent.optimizer, new_lr)
            elif lr_schedule and hasattr(agent, 'actor_optimizer'):
                new_lr = lr_schedule(episode)
                update_optimizer_lr(agent.actor_optimizer, new_lr)
            
            # Update entropy if schedule enabled
            if entropy_schedule and hasattr(agent, 'entropy_coef'):
                agent.entropy_coef = entropy_schedule(episode)
            
            # Prepare batch and update
            batch = {
                'observations': obs_buffer,
                'actions': action_buffer,
                'rewards': reward_buffer,
                'dones': done_buffer,
                'log_probs': log_prob_buffer,
                'values': value_buffer,
            }
            
            update_info = agent.update(batch)
            loss = update_info.get('total_loss', update_info.get('actor_loss', 0.0))
            episode_losses.append(float(loss))
        else:
            episode_losses.append(0.0)
        
        episode_rewards.append(episode_reward)
        
        # --- Periodic logging ---
        if (episode + 1) % 50 == 0:
            recent_reward = np.mean(episode_rewards[-50:])
            recent_loss = np.mean(episode_losses[-50:])
            current_lr = get_current_lr(agent.optimizer if hasattr(agent, 'optimizer') else agent.actor_optimizer)
            current_entropy = getattr(agent, 'entropy_coef', 0)
            print(f"Episode {episode+1:5d}: Reward={recent_reward:6.2f}, Loss={recent_loss:8.4f}, "
                  f"LR={current_lr:.6f}, Entropy={current_entropy:.5f}")
        
        # --- Evaluation ---
        if (episode + 1) % eval_interval == 0:
            agent.eval()
            eval_rewards = []
            
            for _ in range(5):
                obs, _ = eval_env.reset()
                eval_reward = 0.0
                done = False
                steps = 0
                
                while not done and steps < 200:
                    obs_np = {k: np.array(v) if not isinstance(v, np.ndarray) else v 
                             for k, v in obs.items()} if isinstance(obs, dict) else np.array(obs)
                    action, _ = agent.select_action(obs_np, deterministic=True)
                    action = int(action.item() if isinstance(action, torch.Tensor) else action)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    eval_reward += float(reward)
                    steps += 1
                
                eval_rewards.append(eval_reward)
            
            mean_eval = np.mean(eval_rewards)
            eval_history.append(mean_eval)
            
            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval
                no_improvement_count = 0
                print(f"  → New best eval: {best_eval_reward:.2f} ⭐")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if episode + 1 >= 200 and no_improvement_count >= early_stopping_patience:
                print(f"\n🛑 EARLY STOPPING at episode {episode+1}")
                early_stopped = True
                break
    
    # Return history
    return {
        'rewards': np.array(episode_rewards),
        'losses': np.array(episode_losses),
        'eval_history': eval_history,
        'best_eval_reward': float(best_eval_reward),
        'final_loss': float(episode_losses[-1]) if episode_losses else 0.0,
        'early_stopped': early_stopped,
        'total_episodes': len(episode_rewards),
        'seed': seed,
    }


if __name__ == '__main__':
    # Example usage
    history = train_with_optimization(
        agent_name='a2c',
        max_episodes=5000,
        seed=42,
        use_reward_normalization=True,
        use_lr_decay=True,
        use_entropy_decay=True,
        batch_size=2048,
    )
    
    print("\n" + "="*70)
    print("Training Summary:")
    print(f"  Episodes trained: {history['total_episodes']}")
    print(f"  Best eval reward: {history['best_eval_reward']:.2f}")
    print(f"  Final loss: {history['final_loss']:.4f}")
    print(f"  Early stopped: {history['early_stopped']}")

"""
Environment wrappers for EVRP to improve training dynamics and learning curves.

This module provides several wrappers:
1. RewardNormalizationWrapper: Normalizes rewards to bounded range for smooth learning
2. RewardScaleWrapper: Scales rewards by a fixed factor
3. RewardClipWrapper: Clips rewards to prevent gradient explosion
"""

import numpy as np
from typing import Tuple, Dict, Optional
import copy


class RewardNormalizationWrapper:
    """
    Normalize rewards using running statistics to achieve bounded, smooth reward distribution.
    
    This wrapper maintains running mean and std of rewards, then normalizes them to the range
    (-1, 1) using z-score normalization. This prevents reward explosion and enables smoother,
    more stable learning curves.
    
    The normalized curve will show:
    - Steep initial improvement (when std is small)
    - Gradual plateau (as std increases and mean stabilizes)
    """
    
    def __init__(self, env, update_every: int = 100):
        """
        Initialize reward normalization wrapper.
        
        Args:
            env: The environment to wrap
            update_every: Update running statistics every N episodes
        """
        self.env = env
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_buffer = []
        self.update_every = update_every
        self.episode_count = 0
        
        # Copy environment attributes
        self.action_space = env.action_space
        self.observation_space = env.observation_space if hasattr(env, 'observation_space') else None
        self.num_customers = env.num_customers
        self.num_chargers = env.num_chargers
        self.time_limit = env.time_limit
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment."""
        self.episode_count += 1
        
        # Update statistics if needed
        if len(self.reward_buffer) > 0 and self.episode_count % self.update_every == 0:
            rewards_array = np.array(self.reward_buffer)
            self.reward_mean = float(np.mean(rewards_array))
            self.reward_std = float(np.std(rewards_array)) + 1e-8  # Add small epsilon to prevent division by zero
            self.reward_buffer = []
        
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Step environment and normalize the reward.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (obs, normalized_reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store raw reward for statistics update
        self.reward_buffer.append(reward)
        
        # Normalize reward using z-score: (r - mean) / std
        # This centers rewards around 0 with unit variance
        normalized_reward = (reward - self.reward_mean) / self.reward_std
        
        # Clip to [-2, 2] to prevent extreme outliers (roughly ±2 std-dev)
        normalized_reward = np.clip(normalized_reward, -2.0, 2.0)
        
        return obs, float(normalized_reward), terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


class RewardScaleWrapper:
    """Scale rewards by a fixed factor to adjust reward magnitude."""
    
    def __init__(self, env, scale: float = 0.1):
        """
        Initialize reward scaling wrapper.
        
        Args:
            env: The environment to wrap
            scale: Factor to scale rewards by (e.g., 0.1 means rewards are 1/10 their original value)
        """
        self.env = env
        self.scale = scale
        
        self.action_space = env.action_space
        self.observation_space = env.observation_space if hasattr(env, 'observation_space') else None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment."""
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step environment and scale the reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale
        return obs, float(scaled_reward), terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


class RewardClipWrapper:
    """Clip rewards to prevent gradient explosion."""
    
    def __init__(self, env, min_reward: float = -10.0, max_reward: float = 10.0):
        """
        Initialize reward clipping wrapper.
        
        Args:
            env: The environment to wrap
            min_reward: Minimum reward value
            max_reward: Maximum reward value
        """
        self.env = env
        self.min_reward = min_reward
        self.max_reward = max_reward
        
        self.action_space = env.action_space
        self.observation_space = env.observation_space if hasattr(env, 'observation_space') else None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment."""
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step environment and clip the reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        clipped_reward = np.clip(reward, self.min_reward, self.max_reward)
        return obs, float(clipped_reward), terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


class CompositeRewardWrapper:
    """
    Combine multiple reward wrappers for maximum smoothness.
    
    Order: Scale → Normalize → Clip
    This ensures:
    1. Rewards are scaled down
    2. Normalized using running statistics
    3. Clipped to prevent extremes
    """
    
    def __init__(self, env, scale: float = 0.1, update_every: int = 100, 
                 clip_min: float = -3.0, clip_max: float = 3.0):
        """
        Initialize composite wrapper with all smoothing techniques.
        
        Args:
            env: The environment to wrap
            scale: Reward scale factor
            update_every: Update normalization statistics every N episodes
            clip_min: Minimum clipped reward
            clip_max: Maximum clipped reward
        """
        # Apply wrappers in order
        env = RewardScaleWrapper(env, scale=scale)
        env = RewardNormalizationWrapper(env, update_every=update_every)
        env = RewardClipWrapper(env, min_reward=clip_min, max_reward=clip_max)
        
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space if hasattr(env, 'observation_space') else None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment."""
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step environment."""
        return self.env.step(action)
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)

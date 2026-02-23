"""
Normalizers for RL training stability.

Provides running statistics tracking for reward and value normalization,
essential for stable policy gradient methods.
"""

import numpy as np
import torch
from typing import Optional


class RunningNormalizer:
    """
    Tracks running mean and std of values using Welford's online algorithm.
    
    Enables stable normalization of rewards/returns in RL training by
    maintaining statistics as data arrives, without storing all historical data.
    
    Usage:
        normalizer = RunningNormalizer(shape=())
        normalized_reward = normalizer.normalize(raw_reward)
        normalizer.update(raw_reward)
    """
    
    def __init__(self, shape: tuple = (), epsilon: float = 1e-8, momentum: float = 0.01):
        """
        Initialize running normalizer.
        
        Args:
            shape: Shape of values to normalize (e.g., () for scalar, (n,) for vector)
            epsilon: Small constant for numerical stability
            momentum: Momentum for exponential moving average (0 < momentum <= 1)
        """
        self.shape = shape
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize statistics
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4  # Small initial count for stability
    
    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with new batch of values.
        
        Uses Welford's algorithm for numerically stable online variance computation.
        
        Args:
            x: Batch of values to add to statistics (shape must broadcast with self.shape)
        """
        x = np.asarray(x)
        
        # Compute batch statistics
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > len(self.shape) else 1
        
        # Welford's online update
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update mean with momentum (exponential moving average)
        self.mean = self.momentum * batch_mean + (1 - self.momentum) * self.mean
        
        # Update variance with momentum
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = np.maximum(M2 / total_count, self.epsilon)
        
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize values using tracked statistics.
        
        Args:
            x: Values to normalize
        
        Returns:
            Normalized values with mean=0, std=1
        """
        x = np.asarray(x)
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalize values back to original scale.
        
        Args:
            x: Normalized values
        
        Returns:
            Values in original scale
        """
        x = np.asarray(x)
        return x * np.sqrt(self.var + self.epsilon) + self.mean
    
    def reset(self) -> None:
        """Reset statistics to initial state."""
        self.mean = np.zeros(self.shape, dtype=np.float32)
        self.var = np.ones(self.shape, dtype=np.float32)
        self.count = 1e-4


class RewardScaler:
    """
    Scales rewards to a target range for training stability.
    
    Most RL algorithms work best when rewards are approximately in [-1, 1].
    This module tracks reward statistics and applies linear scaling.
    
    Usage:
        scaler = RewardScaler(target_range=(0.0, 1.0))
        scaled_reward = scaler.scale(raw_reward)
        scaler.update_stats(batch_of_rewards)
    """
    
    def __init__(self, target_range: tuple = (-1.0, 1.0), momentum: float = 0.01):
        """
        Initialize reward scaler.
        
        Args:
            target_range: Tuple (min, max) for scaling target
            momentum: Momentum for exponential moving average of reward statistics
        """
        self.target_min, self.target_max = target_range
        self.target_range = target_range[1] - target_range[0]
        self.momentum = momentum
        
        # Reward statistics
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.reward_mean = 0.0
    
    def update_stats(self, rewards: np.ndarray) -> None:
        """
        Update reward statistics from batch.
        
        Args:
            rewards: Batch of reward values
        """
        rewards = np.asarray(rewards)
        
        batch_min = np.min(rewards)
        batch_max = np.max(rewards)
        batch_mean = np.mean(rewards)
        
        # Update with momentum
        self.reward_min = self.momentum * batch_min + (1 - self.momentum) * self.reward_min
        self.reward_max = self.momentum * batch_max + (1 - self.momentum) * self.reward_max
        self.reward_mean = self.momentum * batch_mean + (1 - self.momentum) * self.reward_mean
    
    def scale(self, reward: float) -> float:
        """
        Scale single reward to target range.
        
        Args:
            reward: Raw reward value
        
        Returns:
            Scaled reward
        """
        # Handle edge case where min == max
        reward_range = self.reward_max - self.reward_min
        if reward_range < 1e-6:
            return (self.target_min + self.target_max) / 2.0
        
        # Linear scaling: map [reward_min, reward_max] -> [target_min, target_max]
        normalized = (reward - self.reward_min) / reward_range
        scaled = normalized * self.target_range + self.target_min
        
        return float(scaled)
    
    def scale_batch(self, rewards: np.ndarray) -> np.ndarray:
        """
        Scale batch of rewards.
        
        Args:
            rewards: Batch of rewards
        
        Returns:
            Scaled rewards
        """
        rewards = np.asarray(rewards)
        reward_range = self.reward_max - self.reward_min
        
        if reward_range < 1e-6:
            return np.full_like(rewards, (self.target_min + self.target_max) / 2.0)
        
        normalized = (rewards - self.reward_min) / reward_range
        scaled = normalized * self.target_range + self.target_min
        
        return scaled

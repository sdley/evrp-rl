"""
Training utilities for EVRP RL agents.

Includes:
1. Learning rate schedules (exponential decay, cosine annealing, etc.)
2. Entropy schedules for exploration decay
3. Training helper functions
"""

import torch
from typing import Callable


def exponential_decay_schedule(initial_lr: float, decay_rate: float = 0.95, decay_steps: int = 1000) -> Callable:
    """
    Create exponential decay learning rate schedule.
    
    Args:
        initial_lr: Starting learning rate
        decay_rate: Decay factor per decay_steps (0.95 = 5% decay per decay_steps)
        decay_steps: Number of steps between decays
        
    Returns:
        Function that takes step number and returns learning rate
        
    Example:
        lr_schedule = exponential_decay_schedule(1e-3, decay_rate=0.95, decay_steps=1000)
        lr = lr_schedule(100)  # Learning rate at step 100
    """
    def schedule(step: int) -> float:
        return initial_lr * (decay_rate ** (step / decay_steps))
    return schedule


def cosine_annealing_schedule(initial_lr: float, total_steps: int, min_lr: float = 1e-6) -> Callable:
    """
    Create cosine annealing learning rate schedule.
    
    Gradually decreases LR from initial_lr to min_lr following cosine curve.
    Fast initial decrease, then slower near the end.
    
    Args:
        initial_lr: Starting learning rate
        total_steps: Total number of training steps
        min_lr: Minimum learning rate (at end of schedule)
        
    Returns:
        Function that takes step number and returns learning rate
    """
    def schedule(step: int) -> float:
        import math
        progress = min(step / total_steps, 1.0)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))
    return schedule


def linear_decay_schedule(initial_lr: float, final_lr: float, total_steps: int) -> Callable:
    """
    Create linear decay learning rate schedule.
    
    Args:
        initial_lr: Starting learning rate
        final_lr: Ending learning rate
        total_steps: Total number of training steps
        
    Returns:
        Function that takes step number and returns learning rate
    """
    def schedule(step: int) -> float:
        progress = min(step / total_steps, 1.0)
        return initial_lr + (final_lr - initial_lr) * progress
    return schedule


def entropy_decay_schedule(initial_entropy: float, decay_rate: float = 0.95, decay_steps: int = 1000) -> Callable:
    """
    Create entropy coefficient decay schedule for exploration.
    
    Early training: high entropy (more exploration)
    Late training: low entropy (more exploitation)
    
    Args:
        initial_entropy: Starting entropy coefficient
        decay_rate: Decay factor per decay_steps
        decay_steps: Number of steps between decays
        
    Returns:
        Function that takes step number and returns entropy coefficient
    """
    def schedule(step: int) -> float:
        return initial_entropy * (decay_rate ** (step / decay_steps))
    return schedule


def update_optimizer_lr(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    """
    Update learning rate in optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        new_lr: New learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']

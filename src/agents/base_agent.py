"""
Abstract base class for RL agents in EVRP.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class BaseAgent(ABC, nn.Module):
    """
    Abstract base class for reinforcement learning agents.
    
    All agent implementations should inherit from this class and implement
    the required methods for action selection, training, and state management.
    
    Agents are responsible for:
    1. Action selection (both training and evaluation)
    2. Learning from experience
    3. Saving and loading model checkpoints
    4. Tracking training metrics
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        config: Dict[str, Any],
    ):
        """
        Initialize base agent.
        
        Args:
            encoder: Encoder module for processing graph states
            action_dim: Dimension of action space
            config: Configuration dictionary with hyperparameters
        """
        super().__init__()
        
        self.encoder = encoder
        self.action_dim = action_dim
        self.config = config
        
        # Training metrics
        self.total_steps = 0
        self.total_episodes = 0
        
    @abstractmethod
    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action given an observation.
        
        Args:
            observation: Environment observation dictionary
            deterministic: If True, select action deterministically
        
        Returns:
            Tuple of (action, info_dict)
            - action: Selected action index
            - info_dict: Additional information (log_prob, value, etc.)
        """
        pass
    
    @abstractmethod
    def update(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Update agent from a batch of experience.
        
        Args:
            batch: Dictionary containing experience batch
                - observations: List of observations
                - actions: List of actions
                - rewards: List of rewards
                - next_observations: List of next observations
                - dones: List of done flags
        
        Returns:
            Dictionary of training metrics (losses, etc.)
        """
        pass
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics
        """
        self.total_steps += 1
        return {}
    
    def episode_end(self, episode_info: Dict[str, Any]):
        """
        Called at the end of each episode.
        
        Args:
            episode_info: Information about the completed episode
        """
        self.total_episodes += 1
    
    def save(self, path: str):
        """
        Save agent checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'encoder': self.encoder.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'config': self.config,
        }, path)
    
    def load(self, path: str):
        """
        Load agent checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        return self.config.copy()
    
    def set_training_mode(self, mode: bool = True):
        """Set training mode for all networks."""
        self.train(mode)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current training metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
        }
    
    def _prepare_observation(
        self,
        observation: Dict[str, np.ndarray],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert numpy observation to torch tensors.
        
        Args:
            observation: Observation dictionary from environment
            device: Device to place tensors on
        
        Returns:
            Dictionary of torch tensors
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Handle scalar values
        current_node = observation['current_node']
        if isinstance(current_node, (int, np.integer)):
            current_node = torch.tensor([current_node], dtype=torch.long, device=device)
        else:
            current_node = torch.from_numpy(np.array([current_node])).long().to(device)
        
        current_battery = observation['current_battery']
        if isinstance(current_battery, (float, np.floating, int, np.integer)):
            current_battery = torch.tensor([current_battery], dtype=torch.float, device=device)
        else:
            current_battery = torch.from_numpy(np.array([current_battery])).float().to(device)
        
        current_cargo = observation['current_cargo']
        if isinstance(current_cargo, (float, np.floating, int, np.integer)):
            current_cargo = torch.tensor([current_cargo], dtype=torch.float, device=device)
        else:
            current_cargo = torch.from_numpy(np.array([current_cargo])).float().to(device)
        
        return {
            'node_coords': torch.from_numpy(observation['node_coords']).unsqueeze(0).float().to(device),
            'node_demands': torch.from_numpy(observation['node_demands']).unsqueeze(0).float().to(device),
            'node_types': torch.from_numpy(observation['node_types']).unsqueeze(0).float().to(device),
            'distance_matrix': torch.from_numpy(observation['distance_matrix']).unsqueeze(0).float().to(device),
            'current_node': current_node,
            'current_battery': current_battery,
            'current_cargo': current_cargo,
            'visited_mask': torch.from_numpy(observation['visited_mask']).unsqueeze(0).float().to(device),
            'valid_actions_mask': torch.from_numpy(observation['valid_actions_mask']).unsqueeze(0).bool().to(device),
        }

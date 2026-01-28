"""
Agent factory for creating agents from configurations.
"""

import yaml
from typing import Dict, Any
import torch.nn as nn

from .base_agent import BaseAgent
from .a2c_agent import A2CAgent
from .sac_agent import SACAgent
from src.encoders import GATEncoder, MLPEncoder


class AgentFactory:
    """
    Factory for creating RL agents from YAML configurations.
    
    Supports:
    - A2C (Advantage Actor-Critic)
    - SAC (Soft Actor-Critic)
    - Custom encoder selection (GAT/MLP)
    """
    
    AGENT_REGISTRY = {
        'a2c': A2CAgent,
        'sac': SACAgent,
    }
    
    ENCODER_REGISTRY = {
        'gat': GATEncoder,
        'mlp': MLPEncoder,
    }
    
    @classmethod
    def create_from_config(
        cls,
        config_path: str,
        action_dim: int,
    ) -> BaseAgent:
        """
        Create agent from YAML configuration file.
        
        Args:
            config_path: Path to YAML config file
            action_dim: Action space dimension
        
        Returns:
            Initialized agent
        
        Example config:
            ```yaml
            agent: a2c
            encoder:
              type: gat
              embed_dim: 128
              num_layers: 3
              num_heads: 8
            hyperparameters:
              lr: 3e-4
              gamma: 0.99
              entropy_coef: 0.01
            ```
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls.create_from_dict(config, action_dim)
    
    @classmethod
    def create_from_dict(
        cls,
        config: Dict[str, Any],
        action_dim: int,
    ) -> BaseAgent:
        """
        Create agent from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            action_dim: Action space dimension
        
        Returns:
            Initialized agent
        """
        # Get agent type
        agent_type = config.get('agent', 'a2c').lower()
        if agent_type not in cls.AGENT_REGISTRY:
            raise ValueError(f"Unknown agent type: {agent_type}. "
                           f"Available: {list(cls.AGENT_REGISTRY.keys())}")
        
        # Create encoder
        encoder_config = config.get('encoder', {})
        encoder = cls._create_encoder(encoder_config)
        
        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})
        
        # Create agent
        agent_class = cls.AGENT_REGISTRY[agent_type]
        agent = agent_class(encoder, action_dim, hyperparams)
        
        return agent
    
    @classmethod
    def _create_encoder(cls, encoder_config: Dict[str, Any]) -> nn.Module:
        """
        Create encoder from configuration.
        
        Args:
            encoder_config: Encoder configuration dictionary
        
        Returns:
            Initialized encoder
        """
        encoder_type = encoder_config.get('type', 'gat').lower()
        if encoder_type not in cls.ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder type: {encoder_type}. "
                           f"Available: {list(cls.ENCODER_REGISTRY.keys())}")
        
        encoder_class = cls.ENCODER_REGISTRY[encoder_type]
        
        # Extract encoder-specific parameters
        if encoder_type == 'gat':
            encoder = encoder_class(
                embed_dim=encoder_config.get('embed_dim', 128),
                num_layers=encoder_config.get('num_layers', 3),
                num_heads=encoder_config.get('num_heads', 8),
                dropout=encoder_config.get('dropout', 0.1),
            )
        elif encoder_type == 'mlp':
            encoder = encoder_class(
                embed_dim=encoder_config.get('embed_dim', 128),
                hidden_dim=encoder_config.get('hidden_dim', 256),
                num_layers=encoder_config.get('num_layers', 3),
                dropout=encoder_config.get('dropout', 0.1),
            )
        else:
            raise NotImplementedError(f"Encoder {encoder_type} not implemented")
        
        return encoder
    
    @classmethod
    def get_available_agents(cls) -> list:
        """Return list of available agent types."""
        return list(cls.AGENT_REGISTRY.keys())
    
    @classmethod
    def get_available_encoders(cls) -> list:
        """Return list of available encoder types."""
        return list(cls.ENCODER_REGISTRY.keys())

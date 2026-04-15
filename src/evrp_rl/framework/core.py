"""
Modular RL Framework Core Components.

This module provides factory classes and utilities for building
configurable RL experiments for EVRP.
"""

from typing import Dict, Any, Optional, Callable, List
import yaml
import torch.nn as nn
from pathlib import Path

from evrp_rl.env import EVRPEnvironment
from evrp_rl.agents import BaseAgent, A2CAgent, SACAgent
from evrp_rl.encoders import GATEncoder, MLPEncoder, Encoder


class EnvFactory:
    """
    Factory for creating EVRP environments with various configurations.
    
    Supports:
    - Configurable problem size (customers, chargers)
    - Custom reward functions
    - Custom action masking strategies
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> EVRPEnvironment:
        """
        Create EVRP environment from configuration.
        
        Args:
            config: Environment configuration dict with keys:
                - num_customers: Number of customer nodes
                - num_chargers: Number of charging stations
                - battery_capacity: Vehicle battery capacity (default: 100)
                - cargo_capacity: Vehicle cargo capacity (default: 50)
                - max_demand: Maximum customer demand (default: 20)
                - grid_size: Size of coordinate grid (default: 100)
                - seed: Random seed (optional)
        
        Returns:
            Configured EVRPEnvironment
        
        Example:
            >>> config = {'num_customers': 20, 'num_chargers': 5}
            >>> env = EnvFactory.create(config)
        """
        # Extract environment parameters
        num_customers = config.get('num_customers', 10)
        num_chargers = config.get('num_chargers', 3)
        
        # Handle battery capacity naming variants
        max_battery = config.get('max_battery', config.get('battery_capacity', 100.0))
        
        # Handle cargo capacity naming variants
        max_cargo = config.get('max_cargo', config.get('cargo_capacity', 50.0))
        
        seed = config.get('seed', None)
        
        # Create environment with correct parameter names
        env_params = {
            'num_customers': num_customers,
            'num_chargers': num_chargers,
            'max_battery': max_battery,
            'max_cargo': max_cargo,
        }
        
        # Optional parameters
        for key in ['energy_consumption_rate', 'charger_cost', 'depot_revisit_cost', 'time_limit', 'render_mode']:
            if key in config:
                env_params[key] = config[key]
        
        env = EVRPEnvironment(**env_params)
        
        if seed is not None:
            env.reset(seed=seed)
        
        return env


class EncoderFactory:
    """
    Factory for creating graph encoders.
    
    Supports:
    - GAT (Graph Attention Network)
    - MLP (Multi-Layer Perceptron baseline)
    """
    
    ENCODER_REGISTRY = {
        'gat': GATEncoder,
        'mlp': MLPEncoder,
    }
    
    @staticmethod
    def create(config: Dict[str, Any]) -> Encoder:
        """
        Create encoder from configuration.
        
        Args:
            config: Encoder configuration dict with keys:
                - type: 'gat' or 'mlp'
                - embed_dim: Output embedding dimension
                - hidden_dim: Hidden layer dimension
                - num_layers: Number of layers
                - dropout: Dropout probability
                - For GAT: num_heads, edge_dim
        
        Returns:
            Configured encoder
        
        Example:
            >>> config = {'type': 'gat', 'embed_dim': 128, 'num_layers': 3}
            >>> encoder = EncoderFactory.create(config)
        """
        encoder_type = config.get('type', 'mlp').lower()
        
        if encoder_type not in EncoderFactory.ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. "
                f"Available: {list(EncoderFactory.ENCODER_REGISTRY.keys())}"
            )
        
        encoder_class = EncoderFactory.ENCODER_REGISTRY[encoder_type]
        
        # Common parameters
        embed_dim = config.get('embed_dim', 128)
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 3)
        dropout = config.get('dropout', 0.1)
        
        if encoder_type == 'gat':
            # GAT-specific parameters
            num_heads = config.get('num_heads', 8)
            negative_slope = config.get('negative_slope', 0.2)
            concat_heads = config.get('concat_heads', True)
            
            return encoder_class(
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                negative_slope=negative_slope,
                concat_heads=concat_heads,
            )
        else:  # mlp
            return encoder_class(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )


class AgentFactory:
    """
    Factory for creating RL agents with encoders.
    
    Supports:
    - A2C (Advantage Actor-Critic)
    - SAC (Soft Actor-Critic)
    """
    
    AGENT_REGISTRY = {
        'a2c': A2CAgent,
        'sac': SACAgent,
    }
    
    @staticmethod
    def create(
        config: Dict[str, Any],
        action_dim: int,
        encoder: Optional[Encoder] = None,
    ) -> BaseAgent:
        """
        Create agent from configuration.
        
        Args:
            config: Agent configuration dict with keys:
                - type: 'a2c' or 'sac'
                - encoder: Encoder config (if encoder not provided)
                - hyperparameters: Algorithm-specific hyperparameters
            action_dim: Action space dimension
            encoder: Pre-created encoder (optional, will create from config if None)
        
        Returns:
            Configured agent
        
        Example:
            >>> config = {
            ...     'type': 'a2c',
            ...     'encoder': {'type': 'gat', 'embed_dim': 128},
            ...     'hyperparameters': {'lr': 3e-4, 'gamma': 0.99}
            ... }
            >>> agent = AgentFactory.create(config, action_dim=10)
        """
        agent_type = config.get('type', 'a2c').lower()
        
        if agent_type not in AgentFactory.AGENT_REGISTRY:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available: {list(AgentFactory.AGENT_REGISTRY.keys())}"
            )
        
        # Create encoder if not provided
        if encoder is None:
            encoder_config = config.get('encoder', {})
            encoder = EncoderFactory.create(encoder_config)
        
        # Get hyperparameters
        hyperparams = config.get('hyperparameters', {})
        
        # Merge type into hyperparams for backward compatibility
        full_config = {
            'type': agent_type,
            **hyperparams
        }
        
        # Create agent
        agent_class = AgentFactory.AGENT_REGISTRY[agent_type]
        return agent_class(encoder, action_dim, full_config)

    @classmethod
    def create_from_config(cls, config_path: str, action_dim: int) -> BaseAgent:
        """
        Create agent from a YAML configuration file.

        Args:
            config_path: Path to YAML file (must contain an ``agent`` key).
            action_dim: Action space dimension.

        Returns:
            Configured agent.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls.create_from_dict(config, action_dim)

    @classmethod
    def create_from_dict(cls, config: Dict[str, Any], action_dim: int) -> BaseAgent:
        """
        Create agent from a full experiment config dictionary.

        Handles two layouts:

        * Unified layout – ``config['agent']`` is a dict with keys
          ``type``, ``encoder``, ``hyperparameters``.
        * Legacy flat layout – ``config['agent']`` is a string (e.g. ``'a2c'``)
          and ``config['encoder']`` / ``config['hyperparameters']`` are
          top-level keys.

        Args:
            config: Full experiment configuration dictionary.
            action_dim: Action space dimension.

        Returns:
            Configured agent.
        """
        agent_section = config.get("agent", {})
        if isinstance(agent_section, dict):
            agent_config = agent_section
        else:
            # Legacy: agent is a bare string like 'a2c'
            agent_config = {
                "type": str(agent_section),
                "encoder": config.get("encoder", {}),
                "hyperparameters": config.get("hyperparameters", {}),
            }
        return cls.create(agent_config, action_dim)

    @classmethod
    def get_available_agents(cls) -> list:
        """Return list of registered agent type names."""
        return list(cls.AGENT_REGISTRY.keys())

    @classmethod
    def get_available_encoders(cls) -> list:
        """Return list of registered encoder type names."""
        return list(EncoderFactory.ENCODER_REGISTRY.keys())


class RewardModule:
    """
    Custom reward shaping and penalty system for EVRP.
    
    Provides configurable penalties for:
    - Invalid actions
    - Battery depletion
    - Cargo overload
    - Long routes
    - Frequent charging
    
    Can be used to shape rewards for better learning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reward module with configuration.
        
        Args:
            config: Reward configuration dict with keys:
                - invalid_action_penalty: Penalty for invalid actions (default: -10)
                - battery_penalty_coef: Coefficient for low battery penalty (default: 0.1)
                - cargo_penalty_coef: Coefficient for overload penalty (default: 0.1)
                - distance_penalty_coef: Coefficient for distance penalty (default: 0.01)
                - charging_penalty: Penalty for each charging visit (default: -0.5)
                - completion_bonus: Bonus for completing all deliveries (default: 10)
        """
        config = config or {}
        
        self.invalid_action_penalty = config.get('invalid_action_penalty', -10.0)
        self.battery_penalty_coef = config.get('battery_penalty_coef', 0.1)
        self.cargo_penalty_coef = config.get('cargo_penalty_coef', 0.1)
        self.distance_penalty_coef = config.get('distance_penalty_coef', 0.01)
        self.charging_penalty = config.get('charging_penalty', -0.5)
        self.completion_bonus = config.get('completion_bonus', 10.0)
        
    def compute_reward(
        self,
        base_reward: float,
        action: int,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
    ) -> float:
        """
        Compute shaped reward with penalties and bonuses.
        
        Args:
            base_reward: Base reward from environment
            action: Action taken
            state: Current state dict
            next_state: Next state dict
            done: Whether episode is done
            info: Additional info from environment
        
        Returns:
            Shaped reward
        """
        reward = base_reward
        
        # Penalty for low battery
        battery_level = next_state.get('current_battery', 100.0)
        if battery_level < 20.0:
            reward -= self.battery_penalty_coef * (20.0 - battery_level)
        
        # Penalty for near cargo capacity
        cargo_level = next_state.get('current_cargo', 0.0)
        cargo_capacity = 50.0  # Should match env config
        if cargo_level > 0.8 * cargo_capacity:
            reward -= self.cargo_penalty_coef * (cargo_level - 0.8 * cargo_capacity)
        
        # Penalty for visiting charger (encourage efficiency)
        node_type = info.get('node_type', None)
        if node_type == 'charger':
            reward += self.charging_penalty
        
        # Bonus for completion
        if done and info.get('all_customers_visited', False):
            reward += self.completion_bonus
        
        return reward
    
    def __call__(self, *args, **kwargs) -> float:
        """Allow calling instance as function."""
        return self.compute_reward(*args, **kwargs)


class MaskModule:
    """
    Advanced action masking for EVRP.
    
    Provides:
    - Permanent masks: Structurally invalid actions (e.g., depot → depot)
    - Transient masks: State-dependent invalid actions (e.g., insufficient battery)
    - Soft masks: Discouraged but not forbidden actions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mask module with configuration.
        
        Args:
            config: Mask configuration dict with keys:
                - use_battery_constraint: Enforce battery constraints (default: True)
                - use_cargo_constraint: Enforce cargo constraints (default: True)
                - use_feasibility_check: Check route feasibility (default: True)
                - battery_safety_margin: Safety margin for battery (default: 0.1)
        """
        config = config or {}
        
        self.use_battery_constraint = config.get('use_battery_constraint', True)
        self.use_cargo_constraint = config.get('use_cargo_constraint', True)
        self.use_feasibility_check = config.get('use_feasibility_check', True)
        self.battery_safety_margin = config.get('battery_safety_margin', 0.1)
    
    def compute_mask(
        self,
        state: Dict[str, Any],
        env: EVRPEnvironment,
    ) -> List[bool]:
        """
        Compute action mask for current state.
        
        Args:
            state: Current state dict
            env: Environment instance for accessing problem data
        
        Returns:
            Boolean mask where True = valid action
        """
        num_nodes = len(state['node_coords'])
        mask = [True] * num_nodes
        
        current_node = state['current_node']
        current_battery = state['current_battery']
        current_cargo = state['current_cargo']
        visited_mask = state['visited_mask']
        distance_matrix = state['distance_matrix']
        node_demands = state['node_demands']
        
        # Use environment's built-in masking
        if hasattr(env, 'get_valid_actions'):
            env_mask = env.get_valid_actions()
            return env_mask
        
        # Fallback: compute mask manually
        for node in range(num_nodes):
            # Can't visit current node
            if node == current_node:
                mask[node] = False
                continue
            
            # Can't revisit customers
            if visited_mask[node] and state['node_types'][node] == 1:  # customer
                mask[node] = False
                continue
            
            # Battery constraint with safety margin
            if self.use_battery_constraint:
                distance_to_node = distance_matrix[current_node, node]
                distance_to_depot = distance_matrix[node, 0]
                required_battery = distance_to_node + distance_to_depot
                required_battery *= (1 + self.battery_safety_margin)
                
                if current_battery < required_battery:
                    # Must visit charger if not already at one
                    if state['node_types'][node] != 2:  # not a charger
                        mask[node] = False
            
            # Cargo constraint
            if self.use_cargo_constraint:
                if state['node_types'][node] == 1:  # customer
                    demand = node_demands[node]
                    if current_cargo + demand > 50.0:  # capacity
                        mask[node] = False
        
        return mask


class ConfigLoader:
    """
    Utility for loading and validating experiment configurations.
    """
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return ConfigLoader.validate(config)
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dict
        
        Returns:
            Validated configuration
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['env', 'agent']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")
        
        # Set defaults
        config.setdefault('run', {})
        config['run'].setdefault('epochs', 100)
        config['run'].setdefault('seed', 42)
        config['run'].setdefault('device', 'cpu')
        
        return config
    
    @staticmethod
    def save(config: Dict[str, Any], save_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dict
            save_path: Path to save YAML file
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_experiment_config(
    env_config: Dict[str, Any],
    agent_config: Dict[str, Any],
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Helper function to create experiment configuration programmatically.
    
    Args:
        env_config: Environment configuration
        agent_config: Agent configuration
        run_config: Run configuration (optional)
    
    Returns:
        Complete experiment configuration
    
    Example:
        >>> config = create_experiment_config(
        ...     env_config={'num_customers': 20, 'num_chargers': 5},
        ...     agent_config={'type': 'sac', 'encoder': {'type': 'gat'}},
        ...     run_config={'epochs': 100}
        ... )
    """
    config = {
        'env': env_config,
        'agent': agent_config,
        'run': run_config or {},
    }
    
    return ConfigLoader.validate(config)

"""
Modular RL Framework for EVRP.

Provides factory classes, experiment runners, and utilities for
configurable RL experiments.
"""

from .core import (
    EnvFactory,
    EncoderFactory,
    AgentFactory,
    RewardModule,
    MaskModule,
    ConfigLoader,
    create_experiment_config,
)

from .runner import (
    MetricsLogger,
    ExperimentRunner,
    run_experiment,
)

__all__ = [
    # Factories
    'EnvFactory',
    'EncoderFactory',
    'AgentFactory',
    
    # Modules
    'RewardModule',
    'MaskModule',
    
    # Config utilities
    'ConfigLoader',
    'create_experiment_config',
    
    # Runner
    'MetricsLogger',
    'ExperimentRunner',
    'run_experiment',
]

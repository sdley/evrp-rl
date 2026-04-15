"""
evrp_rl — Deep Reinforcement Learning for the Electric Vehicle Routing Problem.

Quickstart::

    from evrp_rl.env import EVRPEnvironment
    from evrp_rl.framework import EnvFactory, AgentFactory, run_experiment

    env = EVRPEnvironment(num_customers=10, num_chargers=3)
    agent = AgentFactory.create({'type': 'sac', 'encoder': {'type': 'gat'}}, action_dim=env.action_space.n)
"""

__version__ = "0.1.0"
__author__ = "EVRP-RL Contributors"

from evrp_rl.env import EVRPEnvironment
from evrp_rl.agents import A2CAgent, SACAgent, BaseAgent
from evrp_rl.encoders import GATEncoder, MLPEncoder, Encoder
from evrp_rl.framework import (
    EnvFactory,
    AgentFactory,
    EncoderFactory,
    ExperimentRunner,
    run_experiment,
)

__all__ = [
    "__version__",
    # Environment
    "EVRPEnvironment",
    # Agents
    "BaseAgent",
    "A2CAgent",
    "SACAgent",
    # Encoders
    "Encoder",
    "GATEncoder",
    "MLPEncoder",
    # Framework
    "EnvFactory",
    "AgentFactory",
    "EncoderFactory",
    "ExperimentRunner",
    "run_experiment",
]

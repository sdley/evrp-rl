"""
RL agents for EVRP training.

Provides:
- BaseAgent: Abstract base class
- A2CAgent: Advantage Actor-Critic
- SACAgent: Soft Actor-Critic

To create agents from YAML configs use :class:`evrp_rl.framework.AgentFactory`.
"""

from .base_agent import BaseAgent
from .a2c_agent import A2CAgent
from .sac_agent import SACAgent

__all__ = ["BaseAgent", "A2CAgent", "SACAgent"]

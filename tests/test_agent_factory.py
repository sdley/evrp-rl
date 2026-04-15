import pytest

from evrp_rl.framework import AgentFactory
from evrp_rl.agents import BaseAgent


def test_available_registries():
    agents = AgentFactory.get_available_agents()
    encs = AgentFactory.get_available_encoders()
    assert 'a2c' in agents
    assert 'sac' in agents
    assert 'gat' in encs
    assert 'mlp' in encs


def test_create_agent_unified_dict():
    cfg = {
        'agent': {
            'type': 'a2c',
            'encoder': {'type': 'mlp', 'embed_dim': 64, 'hidden_dim': 128, 'num_layers': 2},
            'hyperparameters': {'lr': 1e-3}
        }
    }

    agent = AgentFactory.create_from_dict(cfg, action_dim=4)
    assert isinstance(agent, BaseAgent)
    assert agent.action_dim == 4


def test_create_agent_legacy_flat():
    cfg = {
        'agent': 'sac',
        'encoder': {'type': 'gat', 'embed_dim': 32, 'num_layers': 2, 'num_heads': 4},
        'hyperparameters': {'lr': 3e-4}
    }

    agent = AgentFactory.create_from_dict(cfg, action_dim=3)
    assert isinstance(agent, BaseAgent)
    assert agent.action_dim == 3

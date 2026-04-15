"""
Quick tests for agent implementations.
"""

import torch
import numpy as np
from evrp_rl.env import EVRPEnvironment
from evrp_rl.framework import AgentFactory


def test_a2c_creation():
    """Test A2C agent creation."""
    print("Testing A2C agent creation...")

    # Create environment first to get action dim
    env = EVRPEnvironment(num_customers=3, num_chargers=1)

    config = {
        'agent': 'a2c',
        'encoder': {
            'type': 'mlp',
            'embed_dim': 32,
            'hidden_dim': 64,
            'num_layers': 2,
        },
        'hyperparameters': {
            'lr': 3e-4,
            'gamma': 0.99,
            'hidden_dim': 64,
        }
    }

    agent = AgentFactory.create_from_dict(config, action_dim=env.action_space.n)
    assert agent is not None
    print("✓ A2C agent created successfully")

    # Test forward pass
    obs, _ = env.reset()

    action, info = agent.select_action(obs, deterministic=False)
    assert isinstance(action, int)
    assert 'log_prob' in info
    assert 'value' in info
    print("✓ A2C action selection works")

    # Test update
    rollout = {
        'observations': [obs],
        'actions': [action],
        'rewards': [-1.0],
        'next_observations': [obs],
        'dones': [False],
    }

    metrics = agent.update(rollout)
    assert 'actor_loss' in metrics
    assert 'critic_loss' in metrics
    print("✓ A2C update works")


def test_sac_creation():
    """Test SAC agent creation."""
    print("\nTesting SAC agent creation...")

    # Create environment first to get action dim
    env = EVRPEnvironment(num_customers=3, num_chargers=1)

    config = {
        'agent': 'sac',
        'encoder': {
            'type': 'mlp',
            'embed_dim': 32,
            'hidden_dim': 64,
            'num_layers': 2,
        },
        'hyperparameters': {
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'buffer_size': 1000,
            'batch_size': 32,
            'hidden_dim': 64,
        }
    }

    agent = AgentFactory.create_from_dict(config, action_dim=env.action_space.n)
    assert agent is not None
    print("✓ SAC agent created successfully")

    # Test forward pass
    obs, _ = env.reset()

    action, info = agent.select_action(obs, deterministic=False)
    assert isinstance(action, int)
    assert 'probs' in info
    print("✓ SAC action selection works")

    # Test replay buffer
    agent.store_transition(obs, action, -1.0, obs, False)
    assert len(agent.replay_buffer) == 1
    print("✓ SAC replay buffer works")

    # Fill buffer for update
    for _ in range(40):
        agent.store_transition(obs, action, -1.0, obs, False)

    metrics = agent.update({})
    assert 'actor_loss' in metrics
    assert 'critic1_loss' in metrics
    print("✓ SAC update works")


def test_agent_factory():
    """Test agent factory."""
    print("\nTesting AgentFactory...")

    available_agents = AgentFactory.get_available_agents()
    assert 'a2c' in available_agents
    assert 'sac' in available_agents
    print(f"✓ Available agents: {available_agents}")

    available_encoders = AgentFactory.get_available_encoders()
    assert 'gat' in available_encoders
    assert 'mlp' in available_encoders
    print(f"✓ Available encoders: {available_encoders}")


def test_integration():
    """Test full integration."""
    print("\nTesting full integration...")

    env = EVRPEnvironment(num_customers=3, num_chargers=1)

    config = {
        'agent': 'a2c',
        'encoder': {'type': 'mlp', 'embed_dim': 32, 'hidden_dim': 64, 'num_layers': 2},
        'hyperparameters': {'lr': 3e-4, 'gamma': 0.99, 'hidden_dim': 64}
    }

    agent = AgentFactory.create_from_dict(config, env.action_space.n)

    # Run short episode
    obs, _ = env.reset()
    episode_reward = 0

    rollout = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'dones': []}

    for _ in range(5):
        action, _ = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        rollout['observations'].append(obs)
        rollout['actions'].append(action)
        rollout['rewards'].append(reward)
        rollout['next_observations'].append(next_obs)
        rollout['dones'].append(terminated or truncated)

        episode_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    metrics = agent.update(rollout)
    print(f"✓ Episode completed: reward={episode_reward:.2f}")
    print(f"✓ Training metrics: {list(metrics.keys())}")


if __name__ == '__main__':
    print("=" * 70)
    print("Agent Implementation Tests")
    print("=" * 70)

    test_a2c_creation()
    test_sac_creation()
    test_agent_factory()
    test_integration()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

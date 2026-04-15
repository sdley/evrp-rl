#!/usr/bin/env python3
"""
Test multiple training updates to verify the agent trains properly
"""

from evrp_rl.env import EVRPEnvironment
from evrp_rl.framework import AgentFactory


def main():
    print("Creating environment and agent...")
    env = EVRPEnvironment(num_customers=5, num_chargers=2)

    config = {
        'type': 'a2c',
        'encoder': {
            'type': 'mlp',
            'hidden_dim': 128,
            'num_layers': 2,
        },
        'actor_hidden_dim': 128,
        'critic_hidden_dim': 128,
        'learning_rate': 3e-4,
    }

    agent = AgentFactory.create_from_dict(config, env.action_space.n)

    print("\nTraining for 5 episodes...")
    for episode in range(5):
        obs, _ = env.reset()

        rollout = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
        }

        total_reward = 0
        for _ in range(20):
            action, _ = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rollout['observations'].append(obs)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['next_observations'].append(next_obs)
            rollout['dones'].append(done)

            total_reward += reward
            obs = next_obs

            if done:
                break

        metrics = agent.update(rollout)

        print(f"Episode {episode + 1}:")
        print(f"  Steps: {len(rollout['observations'])}, Total Reward: {total_reward:.2f}")
        print(f"  Actor Loss: {metrics['actor_loss']:.4f}, Critic Loss: {metrics['critic_loss']:.4f}")
        print(f"  Mean Value: {metrics['mean_value']:.4f}, Entropy: {metrics['entropy']:.4f}")

    print("\n✓ Training completed successfully!")


if __name__ == '__main__':
    main()

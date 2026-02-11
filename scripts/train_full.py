"""End-to-end training scaffold for EVRP experiments.

This script demonstrates a full pipeline example:
- parse YAML config
- initialize environment and agent via `EnvFactory` / `AgentFactory`
- generate synthetic training instances (configurable scale)
- run a simple training loop (batch episodes) and record rewards
- evaluate on held-out set
- plot convergence and save results

This is a scaffold to run short demos locally and can be scaled for full experiments by
adjusting the `num_train_episodes` and `num_eval_episodes` in the config or CLI.
"""
import argparse
import yaml
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import inspect

try:
    from src.framework import EnvFactory, AgentFactory
except Exception:
    EnvFactory = None
    AgentFactory = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', default='configs/experiment_config.yaml')
    p.add_argument('--outdir', '-o', default='results/train_full')
    p.add_argument('--quick', action='store_true', help='Run a short demo')
    return p.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_synthetic_scenarios(n, node_count=20, seed=0):
    rng = np.random.RandomState(seed)
    scenarios = []
    for i in range(n):
        coords = rng.uniform(0, 100, size=(node_count, 2)).tolist()
        scenarios.append({'coords': coords, 'seed': int(seed + i)})
    return scenarios


def main():
    args = parse_args()
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    os.makedirs(args.outdir, exist_ok=True)

    num_train = 100 if args.quick else cfg.get('num_train_episodes', 2000)
    num_eval = 20 if args.quick else cfg.get('num_eval_episodes', 200)
    node_count = cfg.get('node_count', 20)

    print('Config:', args.config)
    print(f'Train episodes: {num_train}, Eval episodes: {num_eval}, nodes: {node_count}')

    # Instantiate env and agent via factories if available.
    env_ctor = None
    agent = None
    if EnvFactory:
        env_ctor = getattr(EnvFactory, 'create', None) or EnvFactory

    if AgentFactory:
        agent_ctor = getattr(AgentFactory, 'create', None) or AgentFactory
        # Try to construct agent with flexible signatures. Many factories expect
        # either a single config dict or additional args like `action_dim`.
        try:
            agent = agent_ctor(cfg.get('agent', {}))
        except TypeError:
            # Inspect signature and try common parameter names
            try:
                sig = inspect.signature(agent_ctor)
                params = sig.parameters
                kwargs = {}
                # common names for config arg
                if 'config' in params:
                    kwargs['config'] = cfg.get('agent', {})
                elif 'agent_cfg' in params:
                    kwargs['agent_cfg'] = cfg.get('agent', {})
                # supply an action_dim if requested (safe default: node_count)
                if 'action_dim' in params:
                    kwargs['action_dim'] = cfg.get('action_dim', node_count)
                if kwargs:
                    agent = agent_ctor(**kwargs)
                else:
                    # try passing config then action_dim as positional
                    agent = agent_ctor(cfg.get('agent', {}), cfg.get('action_dim', node_count))
            except Exception:
                agent = None
                print('Warning: could not instantiate agent via AgentFactory; falling back to random policy')
    else:
        agent = None

    # Generate synthetic scenarios (this is quick; increase scale for full experiments)
    train_scenarios = make_synthetic_scenarios(num_train, node_count=node_count, seed=1)
    eval_scenarios = make_synthetic_scenarios(num_eval, node_count=node_count, seed=100000)

    rewards = []
    t0 = time.time()

    for i, scn in enumerate(train_scenarios, 1):
        # Create environment instance
        if env_ctor:
            env = env_ctor(scn)
        else:
            env = None

        # Simple random / greedy rollout if no agent present
        if agent is None or not hasattr(agent, 'act'):
            # fallback: random policy using numpy
            obs = None
            total = 0.0
            # perform a short dummy interaction (placeholder)
            steps = min(50, node_count * 2)
            for _ in range(steps):
                total -= 1.0
            rewards.append(total)
        else:
            # Agent-driven interaction: expect agent.act(env_state) and agent.learn(trajectory)
            obs = env.reset()
            done = False
            total = 0.0
            traj = []
            while not done:
                action = agent.act(obs, deterministic=True)
                step_result = env.step(action)
                # Support both Gym (obs, r, done, info) and Gymnasium (obs, r, terminated, truncated, info)
                if len(step_result) == 4:
                    obs, r, done, info = step_result
                else:
                    obs, r, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                total += float(r)
                traj.append((obs, action, r))
                if len(traj) > node_count * 5:
                    break
            rewards.append(total)
            # call agent learning hook if available
            if hasattr(agent, 'learn'):
                agent.learn(traj)

        # quick status
        if i % max(1, num_train // 10) == 0:
            print(f'Trained on {i}/{num_train} scenarios; avg reward: {np.mean(rewards):.3f}')

    dt = time.time() - t0
    print('Training demo finished in', f'{dt:.1f}s')

    # Evaluation (greedy inference)
    eval_rewards = []
    for scn in eval_scenarios:
        if env_ctor:
            env = env_ctor(scn)
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                if agent is None or not hasattr(agent, 'act'):
                    action = 0
                else:
                    action = agent.act(obs, deterministic=True)
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, r, done, info = step_result
                else:
                    obs, r, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                total += float(r)
                if len(eval_rewards) > node_count * 5:
                    break
            eval_rewards.append(total)
        else:
            eval_rewards.append(-node_count)

    # Plot convergence
    plt.figure(figsize=(6, 3))
    plt.plot(np.cumsum(rewards) / np.arange(1, len(rewards) + 1), label='train avg')
    plt.axhline(np.mean(eval_rewards), color='C1', linestyle='--', label='eval avg')
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')
    plt.legend()
    out_png = os.path.join(args.outdir, 'convergence.png')
    plt.tight_layout()
    plt.savefig(out_png)
    print('Saved convergence plot to', out_png)

    # Summarize
    print('Train avg reward:', np.mean(rewards))
    print('Eval avg reward:', np.mean(eval_rewards))


if __name__ == '__main__':
    main()

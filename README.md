# evrp-rl

Deep Reinforcement Learning for the **Electric Vehicle Routing Problem (EVRP)**.

Train RL agents to route battery-constrained electric vehicles across customers and charging stations — no hand-crafted heuristics required.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Package structure](#package-structure)
- [Quickstart](#quickstart)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Examples](#examples)
- [Development](#development)
- [Status & roadmap](#status--roadmap)

---

## Overview

The EVRP asks: given a fleet of electric vehicles (each with limited battery and cargo), how do you visit every customer and return to the depot while minimising total distance and managing charging stops?

This package tackles EVRP with two off-policy/on-policy deep RL algorithms:

| Algorithm | Notes |
|-----------|-------|
| **A2C** | On-policy, episode rollout, stable with entropy regularisation |
| **SAC** | Off-policy, replay buffer, automatic entropy tuning |

Both can be combined with two graph encoders:

| Encoder | Notes |
|---------|-------|
| **GAT** | Graph Attention Network — captures spatial relationships between nodes |
| **MLP** | Lightweight baseline — processes each node independently |

---

## Installation

```bash
git clone https://github.com/jnlandu/evrp-rl.git
cd evrp-rl

# Core package
pip install -e .

# With GAT encoder (torch-geometric)
pip install -e ".[gat]"

# With XAI, notebooks, and dev tools
pip install -e ".[all]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, Gymnasium ≥ 0.29.

---

## Package structure

```
evrp-rl/
├── src/
│   └── evrp_rl/                 # installable package
│       ├── __init__.py          # top-level API + __version__
│       ├── cli.py               # evrp-train / evrp-eval / evrp-config
│       ├── env/
│       │   └── evrp_env.py      # EVRPEnvironment (Gymnasium)
│       ├── agents/
│       │   ├── base_agent.py    # abstract BaseAgent
│       │   ├── a2c_agent.py     # Advantage Actor-Critic
│       │   └── sac_agent.py     # Soft Actor-Critic
│       ├── encoders/
│       │   ├── encoder.py       # abstract Encoder
│       │   ├── gat_encoder.py   # Graph Attention Network
│       │   └── mlp_encoder.py   # MLP baseline
│       ├── framework/
│       │   ├── core.py          # EnvFactory, EncoderFactory, AgentFactory, …
│       │   └── runner.py        # ExperimentRunner, MetricsLogger
│       └── xai/
│           └── attribution.py   # perturbation importance, Shapley, plots
│
├── configs/
│   ├── experiment_config.yaml   # unified SAC example
│   ├── sac_config.yaml          # SAC (legacy flat format)
│   └── a2c_config.yaml          # A2C (legacy flat format)
│
├── examples/
│   ├── train_a2c.py
│   ├── train_sac.py
│   ├── example_evrp.py
│   ├── example_encoders.py
│   └── *.ipynb                  # benchmark & ablation notebooks
│
├── tests/
│   └── test_*.py                # 84 passing tests
│
├── docs/
├── pyproject.toml
├── MANIFEST.in
└── README.md
```

---

## Quickstart

### Generate a config, then train

```bash
# 1. Generate a starter YAML
evrp-config --agent sac --encoder gat --customers 10 --episodes 500 --out my_run.yaml

# 2. Train
evrp-train --config my_run.yaml

# 3. Evaluate the best checkpoint
evrp-eval --config my_run.yaml --checkpoint checkpoints/best_model.pt --episodes 100
```

### Python

```python
from evrp_rl import EVRPEnvironment, AgentFactory, run_experiment

# One-liner experiment
runner = run_experiment({
    "env":   {"num_customers": 10, "num_chargers": 3},
    "agent": {"type": "sac", "encoder": {"type": "gat"}},
    "run":   {"epochs": 200},
})
```

---

## CLI reference

All commands accept `--help` for the full flag list.

### `evrp-config` — generate a YAML config

```bash
evrp-config [--agent {a2c,sac}] [--encoder {gat,mlp}]
            [--customers N] [--chargers N] [--episodes N]
            [--embed-dim N] [--lr LR] [--device {cpu,cuda}]
            [--out FILE]
```

```bash
# Examples
evrp-config --agent sac --encoder gat --out configs/my_sac.yaml
evrp-config --agent a2c --encoder mlp --customers 20 --episodes 1000 --out configs/a2c_large.yaml
```

---

### `evrp-train` — train an agent

```bash
evrp-train --config FILE [overrides…]
```

Every YAML field can be overridden from the command line — no need to edit files for quick experiments:

| Group | Flags |
|-------|-------|
| **Agent / encoder** | `--agent {a2c,sac}`, `--encoder {gat,mlp}`, `--embed-dim N`, `--lr LR`, `--gamma G`, `--hidden-dim N` |
| **Environment** | `--customers N`, `--chargers N`, `--max-battery B`, `--max-cargo C` |
| **Run** | `--episodes N`, `--max-steps N`, `--eval-freq N`, `--eval-episodes N`, `--save-freq N`, `--no-eval` |
| **Output** | `--name NAME`, `--log-dir DIR`, `--checkpoint-dir DIR` |
| **System** | `--device {cpu,cuda}`, `--seed N`, `--resume CKPT` |

```bash
# Basic
evrp-train --config configs/experiment_config.yaml

# Override agent and encoder without editing the YAML
evrp-train --config configs/experiment_config.yaml --agent a2c --encoder mlp

# Hyperparameter sweep — just change flags
evrp-train --config configs/experiment_config.yaml --lr 1e-3 --episodes 300 --seed 1
evrp-train --config configs/experiment_config.yaml --lr 3e-4 --episodes 300 --seed 2

# Bigger problem
evrp-train --config configs/experiment_config.yaml --customers 30 --chargers 8

# Resume a previous run
evrp-train --config configs/experiment_config.yaml --resume checkpoints/my_run/best_model.pt

# Training without evaluation (faster iteration)
evrp-train --config configs/experiment_config.yaml --no-eval --episodes 1000
```

---

### `evrp-eval` — evaluate a saved checkpoint

```bash
evrp-eval --config FILE --checkpoint CKPT [--episodes N]
          [--device {cpu,cuda}] [--seed N] [--out DIR]
          [--customers N] [--chargers N]
```

```bash
# Standard evaluation
evrp-eval --config configs/experiment_config.yaml \
          --checkpoint checkpoints/best_model.pt \
          --episodes 200 --out results/eval/

# Zero-shot generalisation — train on 10 customers, test on 20
evrp-eval --config configs/experiment_config.yaml \
          --checkpoint checkpoints/best_model.pt \
          --customers 20 --chargers 6
```

Output (printed + saved to `eval_results.json`):

```
  mean_reward            -42.1830
  std_reward               8.3210
  mean_episode_length     38.4700
  mean_route_length      312.5100
  mean_charge_visits       2.1400
  success_rate             0.8700
```

---

### `evrp-xai` — explainability

```bash
evrp-xai --env-config configs/agents_examples.yaml \
         --example a2c_example \
         --out results/xai/
```

---

## Python API

### Environment

```python
from evrp_rl.env import EVRPEnvironment

env = EVRPEnvironment(
    num_customers=10,
    num_chargers=3,
    max_battery=100.0,
    max_cargo=50.0,
    energy_consumption_rate=1.0,
    time_limit=200,
    seed=42,
)

obs, info = env.reset()
# obs keys: node_coords, distance_matrix, node_demands, node_types,
#           current_node, current_battery, current_cargo,
#           visited_mask, valid_actions_mask

obs, reward, terminated, truncated, info = env.step(action)
```

### Agents

```python
from evrp_rl.framework import AgentFactory

# From dict (programmatic)
agent = AgentFactory.create(
    config={
        "type": "sac",
        "encoder": {"type": "gat", "embed_dim": 128, "num_layers": 3},
        "hyperparameters": {"lr": 3e-4, "gamma": 0.99, "buffer_size": 100_000},
    },
    action_dim=env.action_space.n,
)

# From YAML file
agent = AgentFactory.create_from_config("configs/experiment_config.yaml", env.action_space.n)

# From full experiment dict (includes env/run sections)
agent = AgentFactory.create_from_dict(full_config, env.action_space.n)

# Inspect registries
AgentFactory.get_available_agents()    # ['a2c', 'sac']
AgentFactory.get_available_encoders()  # ['gat', 'mlp']
```

### Experiment runner

```python
from evrp_rl.framework import EnvFactory, AgentFactory, ExperimentRunner

env    = EnvFactory.create(config["env"])
agent  = AgentFactory.create(config["agent"], env.action_space.n)
runner = ExperimentRunner(env, agent, config,
                          log_dir="results", checkpoint_dir="checkpoints")
runner.train()

results = runner.evaluate(num_episodes=100)
# {'mean_reward': ..., 'success_rate': ..., ...}
```

### Encoders

```python
import torch
from evrp_rl.encoders import GATEncoder, MLPEncoder

encoder = GATEncoder(embed_dim=128, num_layers=3, num_heads=8)

graph_data = {
    "node_coords":     torch.rand(batch, N, 2),
    "node_demands":    torch.rand(batch, N),
    "node_types":      torch.rand(batch, N, 3),
    "distance_matrix": torch.rand(batch, N, N),
}
node_embeds, graph_embed = encoder(graph_data)
# node_embeds: (batch, N, 128)
# graph_embed: (batch, 128)
```

### XAI

```python
from evrp_rl.xai import perturbation_importance, approximate_shapley, plot_route_importance

# Perturbation importance
importances = perturbation_importance(
    state, feature_keys, predict_fn, perturb_fn, n_samples=50
)

# Monte-Carlo Shapley values
shapley = approximate_shapley(
    state, feature_keys, value_fn, n_permutations=100
)

# Route visualisation with importance heatmap
ax = plot_route_importance(G, route, node_importance=importances, save_path="xai.png")
```

---

## Configuration

Configs use the **unified format** (recommended):

```yaml
# configs/my_experiment.yaml

env:
  num_customers: 20
  num_chargers: 5
  battery_capacity: 100.0
  cargo_capacity: 50.0

agent:
  type: sac               # a2c | sac
  encoder:
    type: gat             # gat | mlp
    embed_dim: 128
    num_layers: 3
    num_heads: 8
    dropout: 0.1
  hyperparameters:
    lr: 3.0e-4
    gamma: 0.99
    tau: 0.005
    alpha: auto           # automatic entropy tuning
    buffer_size: 100000
    batch_size: 64
    hidden_dim: 256

run:
  epochs: 500
  eval_frequency: 50
  save_frequency: 100
  max_steps_per_episode: 200
  num_eval_episodes: 10
  device: cpu
  seed: 42
```

Load and use in Python:

```python
from evrp_rl.framework import ConfigLoader, run_experiment

config = ConfigLoader.load("configs/my_experiment.yaml")
runner = run_experiment(config)
```

Or generate one from scratch with CLI:

```bash
evrp-config --agent sac --encoder gat --customers 20 --out configs/my_experiment.yaml
```

---

## Examples

| File | Description |
|------|-------------|
| [examples/train_a2c.py](examples/train_a2c.py) | Minimal A2C training loop |
| [examples/train_sac.py](examples/train_sac.py) | SAC with warmup and replay buffer |
| [examples/example_evrp.py](examples/example_evrp.py) | Environment walkthrough |
| [examples/example_encoders.py](examples/example_encoders.py) | GAT vs MLP comparison |
| [examples/agent_benchmark_evrp_a2c_sac.ipynb](examples/agent_benchmark_evrp_a2c_sac.ipynb) | A2C vs SAC benchmark |
| [examples/ablation_study.ipynb](examples/ablation_study.ipynb) | Encoder ablation study |
| [examples/case_study.ipynb](examples/case_study.ipynb) | Full case study with XAI |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run a specific test file
pytest tests/test_framework.py -v
```

---

## Status & roadmap

### Done

- `EVRPEnvironment` — battery, cargo, charging, action masking, Gymnasium API
- **A2C** agent with entropy regularisation
- **SAC** agent with automatic entropy tuning and replay buffer
- **GAT** and **MLP** encoders
- Modular framework — `EnvFactory`, `EncoderFactory`, `AgentFactory`, `ExperimentRunner`
- CLI — `evrp-train`, `evrp-eval`, `evrp-config`, `evrp-xai`
- XAI — perturbation importance, Shapley approximation, route visualisation
- 84 passing tests

### In progress

- Hyperparameter tuning
- Transformer encoder
- PPO / DQN agents

---

## License

MIT — see [LICENSE.txt](LICENSE.txt).

## Acknowledgements

Inspired by *"Deep Reinforcement Learning for the Electric Vehicle Routing Problem With Time Windows"* (Lin et al., 2022).

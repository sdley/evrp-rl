# EVRP RL Framework

A modular reinforcement learning framework for Electric Vehicle Routing Problems (EVRP). This framework provides a config-driven system for running RL experiments with flexible environment, encoder, and agent configurations.

## Features

✨ **Key Capabilities:**

- 🏭 **Factory Pattern**: Create environments, encoders, and agents from YAML configs
- 🎯 **Multiple Algorithms**: Support for A2C and SAC agents
- 🧠 **Flexible Encoders**: MLP and GAT (Graph Attention Network) encoders
- 🎁 **Reward Shaping**: Custom reward penalties and bonuses
- 🎭 **Action Masking**: Battery-aware and cargo-aware action constraints
- 📊 **Metrics Logging**: Track rewards, losses, route metrics, success rates
- 💾 **Checkpointing**: Automatic model saving and best model tracking
- 📈 **Visualization**: Training curves and evaluation plots
- 🧪 **Well-Tested**: Comprehensive unit tests covering all components

## Architecture

```
src/framework/
├── core.py          # Factory classes and modules
│   ├── EnvFactory       # Create EVRP environments
│   ├── EncoderFactory   # Create MLP/GAT encoders
│   ├── AgentFactory     # Create A2C/SAC agents
│   ├── RewardModule     # Custom reward shaping
│   ├── MaskModule       # Action masking
│   └── ConfigLoader     # YAML configuration management
│
├── runner.py        # Experiment orchestration
│   ├── MetricsLogger    # Track and visualize metrics
│   └── ExperimentRunner # Training/evaluation loops
│
└── __init__.py      # Module exports and utilities
```

## Quick Start

### 1. Create an Experiment Configuration

```yaml
# configs/my_experiment.yaml
env:
  num_customers: 20
  num_chargers: 5
  max_battery: 100.0
  max_cargo: 50.0
  seed: 42

agent:
  type: "sac"
  encoder:
    type: "gat"
    embed_dim: 128
    num_layers: 3
    num_heads: 8
  hyperparameters:
    lr: 3e-4
    gamma: 0.99
    tau: 0.005
    alpha: "auto"
    buffer_size: 10000
    batch_size: 32

run:
  name: "sac_gat_exp1"
  epochs: 100
  eval_frequency: 10
  save_frequency: 25
  max_steps_per_episode: 100
  num_eval_episodes: 5
  device: "cuda"
  seed: 42
```

### 2. Run an Experiment

```python
from src.framework import ConfigLoader, run_experiment

# Load configuration
config = ConfigLoader.load('configs/my_experiment.yaml')

# Run experiment
run_experiment(config)
```

### 3. Programmatic Configuration

```python
from src.framework import (
    EnvFactory,
    AgentFactory,
    ExperimentRunner,
    create_experiment_config,
)

# Create config programmatically
config = create_experiment_config(
    env_config={
        'num_customers': 15,
        'num_chargers': 4,
        'max_battery': 100.0,
        'max_cargo': 50.0,
    },
    agent_config={
        'type': 'a2c',
        'encoder': {'type': 'mlp', 'embed_dim': 128},
        'hyperparameters': {
            'lr': 3e-4,
            'gamma': 0.99,
            'entropy_coef': 0.01,
        },
    },
    run_config={
        'name': 'a2c_mlp_exp',
        'epochs': 50,
        'eval_frequency': 10,
    },
)

# Create components
env = EnvFactory.create(config['env'])
agent = AgentFactory.create(config['agent'], env.action_space.n)

# Run training
runner = ExperimentRunner(env, agent, config)
runner.train()
```

## Configuration Schema

### Environment Configuration

```yaml
env:
  num_customers: 20 # Number of customers to serve
  num_chargers: 5 # Number of charging stations
  max_battery: 100.0 # Maximum battery capacity
  max_cargo: 50.0 # Maximum cargo capacity
  energy_consumption_rate: 1.0 # Energy per unit distance (optional)
  charger_cost: 0.5 # Cost penalty for charging (optional)
  depot_revisit_cost: 1.0 # Cost penalty for depot revisit (optional)
  time_limit: 100 # Max steps per episode (optional)
  seed: 42 # Random seed (optional)
```

### Agent Configuration

#### A2C Agent

```yaml
agent:
  type: "a2c"
  encoder:
    type: "mlp" # or 'gat'
    embed_dim: 128
    hidden_dim: 256 # MLP only
    num_layers: 3
    num_heads: 8 # GAT only
    dropout: 0.1
  hyperparameters:
    lr: 3e-4
    gamma: 0.99
    entropy_coef: 0.01
    hidden_dim: 256
```

#### SAC Agent

```yaml
agent:
  type: "sac"
  encoder:
    type: "gat"
    embed_dim: 128
    num_layers: 3
    num_heads: 8
    dropout: 0.1
  hyperparameters:
    lr: 3e-4
    gamma: 0.99
    tau: 0.005
    alpha: "auto" # or float value
    hidden_dim: 256
    buffer_size: 10000
    batch_size: 32
    learning_starts: 100
```

### Run Configuration

```yaml
run:
  name: "experiment_name"
  epochs: 100
  eval_frequency: 10 # Evaluate every N epochs
  save_frequency: 25 # Save checkpoint every N epochs
  max_steps_per_episode: 100
  num_eval_episodes: 5
  device: "cuda" # or 'cpu'
  seed: 42
```

### Reward Shaping (Optional)

```yaml
reward:
  battery_penalty_coef: 0.1 # Penalty for low battery
  battery_threshold: 20.0 # Battery level threshold
  cargo_penalty_coef: 0.05 # Penalty for unused cargo
  completion_bonus: 10.0 # Bonus for completing all customers
  early_termination_penalty: -5.0
```

### Action Masking (Optional)

```yaml
mask:
  mask_low_battery: true
  battery_threshold: 15.0
  mask_excess_cargo: true
  safety_margin: 1.1
```

## Metrics

The framework tracks comprehensive metrics during training and evaluation:

### Training Metrics

- Episode rewards
- Episode lengths
- Actor losses
- Critic losses (A2C) / Q losses (SAC)
- Entropy values
- Alpha values (SAC only)

### Evaluation Metrics

- Mean reward
- Success rate (% episodes completing all customers)
- Mean route length
- Mean charging station visits
- Standard deviations

## Outputs

The framework creates the following directory structure:

```
results/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── metrics.json              # All metrics in JSON format
    ├── training_curves.png       # Training visualization
    └── evaluation_curves.png     # Evaluation visualization

checkpoints/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── model_final.pt           # Final model
    ├── model_best.pt            # Best model (by eval reward)
    └── model_epoch_N.pt         # Periodic checkpoints
```

## Advanced Usage

### Custom Reward Shaping

```python
from src.framework import RewardModule

# Create custom reward module
reward_module = RewardModule(config={
    'battery_penalty_coef': 0.2,
    'battery_threshold': 25.0,
    'completion_bonus': 15.0,
})

# Use in environment wrapper
def reward_wrapper(env_reward, state, action, next_state, done, info):
    return reward_module(env_reward, state, action, next_state, done, info)
```

### Custom Action Masking

```python
from src.framework import MaskModule

# Create mask module
mask_module = MaskModule(config={
    'mask_low_battery': True,
    'battery_threshold': 20.0,
})

# Apply before action selection
valid_mask = mask_module(state, action_mask)
```

### Loading and Resuming

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/model_best.pt')

# Create agent and load weights
agent = AgentFactory.create(config['agent'], num_actions)
agent.ac_network.load_state_dict(checkpoint['model_state_dict'])

# Resume training
runner = ExperimentRunner(env, agent, config)
runner.train()
```

## Examples

See `examples/ablation_study.ipynb` for a complete ablation study comparing:

- A2C vs SAC algorithms
- MLP vs GAT encoders
- Different hyperparameter configurations

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_framework.py -v
```

All 32 tests cover:

- Factory creation and validation
- Reward and mask modules
- Configuration loading/saving
- Metrics logging
- Training and evaluation loops

## Extending the Framework

### Adding a New Agent

1. Implement `BaseAgent` interface in `src/agents/`
2. Register in `AgentFactory.AGENT_REGISTRY`
3. Add configuration validation in `ConfigLoader`

### Adding a New Encoder

1. Implement `Encoder` interface in `src/encoders/`
2. Register in `EncoderFactory.ENCODER_REGISTRY`
3. Update factory creation logic

### Adding Custom Metrics

Extend `MetricsLogger` to track additional metrics:

```python
class CustomMetricsLogger(MetricsLogger):
    def log_train_episode(self, episode, reward, length, metrics):
        super().log_train_episode(episode, reward, length, metrics)
        # Add custom metric tracking
        self.metrics['train']['custom_metric'].append(metrics['custom_value'])
```

## Best Practices

1. **Always validate configs**: Use `ConfigLoader.validate()` before running experiments
2. **Use meaningful names**: Set descriptive `run.name` for experiment tracking
3. **Save frequently**: Set appropriate `save_frequency` to avoid losing progress
4. **Monitor metrics**: Check `metrics.json` and plots regularly during training
5. **Seed everything**: Set seeds in config for reproducibility
6. **Start small**: Test with smaller problem instances before scaling up
7. **Tune hyperparameters**: Use the ablation notebook as a template for systematic tuning

## Troubleshooting

**Problem**: NaN gradients during training

- **Solution**: Reduce learning rate, check encoder initialization, increase batch size

**Problem**: Low success rate in evaluation

- **Solution**: Increase training epochs, tune reward shaping, adjust action masking

**Problem**: High memory usage

- **Solution**: Reduce buffer size (SAC), decrease batch size, use CPU device

**Problem**: Slow training

- **Solution**: Use GPU device, reduce evaluation frequency, simplify encoder

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{evrp_rl_framework,
  title={Modular RL Framework for Electric Vehicle Routing Problems},
  year={2025},
  author={Your Name},
}
```

## License

See [LICENSE.txt](../../LICENSE.txt) for details.
